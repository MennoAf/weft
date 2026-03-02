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

import asyncio
import logging
from datetime import datetime, timezone

import asyncpg

from weft.models import Memory, MemoryStatus, MemoryType
from weft.store import list_memories
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Hard cap on decisions shown in primer.
_MAX_DECISIONS = 5

# Topic used to identify project grounding memories.
_GROUNDING_TOPIC = "project-grounding"


def _newest_created_at(memories: list[dict], now: datetime) -> float | None:
    """Return age_hours of the most recently created memory, or None if empty."""
    if not memories:
        return None
    timestamps = []
    for m in memories:
        ts = m["created_at"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        timestamps.append(ts)
    newest = max(timestamps)
    return round((now - newest).total_seconds() / 3600, 1)


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

    All section fetches run in parallel via asyncio.gather, then budget
    packing happens sequentially in priority order.
    """
    now = datetime.now(timezone.utc)

    # list_memories already handles `project_id = $X OR project_id IS NULL`
    # when project_id is provided, so a single call per section suffices.
    _pid = {"project_id": project_id} if project_id else {}

    # --- Fetch all sections in parallel ---
    async def _empty() -> list[Memory]:
        return []

    grounding_coro = (
        list_memories(
            pool, project_id=project_id,
            topic=_GROUNDING_TOPIC, status=MemoryStatus.active, limit=1,
        )
        if project_id
        else _empty()
    )

    pinned_coro = list_memories(
        pool, status=MemoryStatus.active, pinned=True, limit=100, **_pid,
    )
    handoff_coro = list_memories(
        pool, memory_type=MemoryType.handoff, status=MemoryStatus.active,
        limit=5, **_pid,
    )
    issues_coro = list_memories(
        pool, memory_type=MemoryType.issue, status=MemoryStatus.active,
        limit=20, **_pid,
    )
    decisions_coro = list_memories(
        pool, memory_type=MemoryType.decision, status=MemoryStatus.active,
        limit=20, **_pid,
    )

    (
        grounding_raw,
        pinned_raw,
        handoff_raw,
        issues_raw,
        decisions_raw,
    ) = await asyncio.gather(
        grounding_coro, pinned_coro, handoff_coro, issues_coro, decisions_coro,
    )

    # --- Budget packing (sequential, in priority order) ---
    used_tokens = 0
    seen_ids: set[str] = set()
    excluded = 0

    # Section 0: Project grounding (one-liner)
    grounding_line: str | None = None
    if grounding_raw:
        mem = grounding_raw[0]
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            grounding_line = mem.content
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # Section 1: Rules (pinned memories only)
    pinned_raw.sort(key=lambda m: (m.confidence, m.created_at.timestamp()), reverse=True)

    rules_section: list[dict] = []
    for mem in pinned_raw:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            rules_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # Section 2: Last session handoff (most recent only)
    handoff_candidates = [m for m in handoff_raw if m.id not in seen_ids]
    if not handoff_candidates:
        # Fallback: check for topic "session-handoff" (handles pre-typed-handoff memories).
        # DEPRECATED: this fallback will be removed in a future version.
        topic_raw = await list_memories(
            pool, topic="session-handoff", status=MemoryStatus.active,
            limit=5, **_pid,
        )
        handoff_candidates = [m for m in topic_raw if m.id not in seen_ids]
        if handoff_candidates:
            logger.warning(
                "Handoff found via topic fallback — re-store with type=handoff "
                "to silence this warning (fallback will be removed in v0.3)",
            )
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

    # Section 3: Active issues
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

    # Section 4: Closed decisions (what NOT to suggest)
    decision_candidates = [m for m in decisions_raw if m.id not in seen_ids]
    decision_candidates.sort(
        key=lambda m: (
            0 if project_id and m.project_id == project_id else 1,
            -(m.created_at.timestamp()),
        ),
    )

    decisions_section: list[dict] = []
    for i, mem in enumerate(decision_candidates):
        if len(decisions_section) >= _MAX_DECISIONS:
            excluded += len(decision_candidates) - i
            break
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            decisions_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
        else:
            excluded += 1

    # Collect all included memories for freshness calculation
    all_included: list[dict] = (
        rules_section + handoff_section + issue_items + decisions_section
    )
    freshness_hours = _newest_created_at(all_included, now)

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
        "freshness_hours": freshness_hours,
    }
