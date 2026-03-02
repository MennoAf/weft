"""Session primer — a tight briefing for session startup.

The primer's job is: "what would be costly to get wrong in the first
30 seconds?"  Every line should change how the agent behaves.  Reference
material, historical summaries, and architecture facts belong in recall,
fetched when the conversation makes them relevant.

Sections (in priority order):
0. Grounding — one-line project description (orientation)
1. Rules — pinned memories only (behavioral overrides)
2. Handoff — most recent session handoff (continuity)
3. Issues — active issues (what's broken right now)
4. Decisions — closed/vetoed decisions (what NOT to suggest)
"""

from __future__ import annotations

from datetime import datetime, timezone

import asyncpg

from weft.models import Memory, MemoryStatus, MemoryType
from weft.store import list_memories
from weft.tokens import estimate_tokens

# Hard cap on decisions shown in primer.
_MAX_DECISIONS = 5

# Topic used to identify project grounding memories.
_GROUNDING_TOPIC = "project-grounding"


async def _fetch_with_globals(
    pool: asyncpg.Pool,
    project_id: str | None,
    **kwargs,
) -> list[Memory]:
    """Fetch memories matching filters, including global memories when project-scoped.

    When project_id is given, we fetch both project-specific and global
    (project_id=None) memories and merge them, deduplicating by id.
    """
    if project_id is None:
        return await list_memories(pool, **kwargs)

    project_mems = await list_memories(pool, project_id=project_id, **kwargs)
    all_mems = await list_memories(pool, **kwargs)
    global_mems = [m for m in all_mems if m.project_id is None]

    seen: set[str] = set()
    merged: list[Memory] = []
    for m in project_mems + global_mems:
        if m.id not in seen:
            seen.add(m.id)
            merged.append(m)
    return merged


async def build_primer(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    budget_tokens: int = 1500,
) -> dict:
    """Assemble a tight session briefing from memories.

    Five sections, filled in priority order within the token budget:
    0. grounding — one-line project description (orientation)
    1. rules — pinned memories (behavioral overrides, always first)
    2. handoff — last session handoff (continuity)
    3. issues — active issues (what's broken)
    4. decisions — closed decisions (what NOT to suggest)

    Everything else (architecture, patterns, reference facts, preferences)
    lives in recall — fetched on demand when the conversation needs it.
    """
    used_tokens = 0
    seen_ids: set[str] = set()
    excluded = 0
    now = datetime.now(timezone.utc)

    # --- Section 0: Project grounding (one-liner) ---
    grounding_line: str | None = None
    if project_id is not None:
        grounding_raw = await list_memories(
            pool, project_id=project_id,
            topic=_GROUNDING_TOPIC, status=MemoryStatus.active, limit=1,
        )
        if grounding_raw:
            mem = grounding_raw[0]
            cost = mem.token_count or estimate_tokens(mem.content)
            if used_tokens + cost <= budget_tokens:
                grounding_line = mem.content
                seen_ids.add(mem.id)
                used_tokens += cost
            else:
                excluded += 1

    # --- Section 1: Rules (pinned memories only) ---
    pinned_raw = await _fetch_with_globals(
        pool, project_id,
        status=MemoryStatus.active, pinned=True, limit=100,
    )
    pinned_raw.sort(key=lambda m: m.confidence, reverse=True)

    rules_section: list[dict] = []
    for mem in pinned_raw:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            rules_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # --- Section 2: Last session handoff (most recent only) ---
    handoff_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.handoff, status=MemoryStatus.active, limit=5,
    )
    # Fallback: if no typed handoffs found, check for topic "session-handoff"
    # (handles memories created before the handoff type existed or mistyped)
    if not handoff_raw:
        topic_raw = await _fetch_with_globals(
            pool, project_id,
            topic="session-handoff", status=MemoryStatus.active, limit=5,
        )
        handoff_raw = [m for m in topic_raw if "Session Handoff" in m.content]
    handoff_candidates = [m for m in handoff_raw if m.id not in seen_ids]
    handoff_candidates.sort(key=lambda m: m.created_at, reverse=True)

    handoff_section: list[dict] = []
    if handoff_candidates:
        mem = handoff_candidates[0]
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            entry = mem.to_dict()
            age_hours = (now - mem.created_at).total_seconds() / 3600
            entry["age_hours"] = round(age_hours, 1)
            handoff_section.append(entry)
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # --- Section 3: Active issues ---
    issues_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.issue, status=MemoryStatus.active, limit=20,
    )
    issue_candidates = [m for m in issues_raw if m.id not in seen_ids]
    issue_candidates.sort(key=lambda m: m.created_at, reverse=True)

    issue_items: list[dict] = []
    for mem in issue_candidates:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            issue_items.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # --- Section 4: Closed decisions (what NOT to suggest) ---
    decisions_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.decision, status=MemoryStatus.active, limit=20,
    )
    decision_candidates = [m for m in decisions_raw if m.id not in seen_ids]
    # Project-scoped first, then most recent
    decision_candidates.sort(
        key=lambda m: (
            0 if project_id and m.project_id == project_id else 1,
            -(m.created_at.timestamp()),
        ),
    )

    decisions_section: list[dict] = []
    for mem in decision_candidates:
        if len(decisions_section) >= _MAX_DECISIONS:
            excluded += len(decision_candidates) - _MAX_DECISIONS
            break
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            decisions_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    return {
        "grounding": grounding_line,
        "rules": rules_section,
        "handoff": handoff_section,
        "issues": {"count": len(issue_items), "items": issue_items},
        "decisions": decisions_section,
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
        "excluded": excluded,
    }
