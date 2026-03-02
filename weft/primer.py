"""Session primer — a tight briefing for session startup.

The primer's job is: "what would be costly to get wrong in the first
30 seconds?"  Every line should change how the agent behaves.  Reference
material, historical summaries, and architecture facts belong in recall,
fetched when the conversation makes them relevant.

Sections (in priority order, each with a per-section token cap):
0. Grounding — one-line project description (50 tokens)
1. Rules — pinned memories only (100 tokens)
2. Handoff — most recent session handoff (500 tokens)
3. Recent work — milestone breadcrumbs from last 72h (150 tokens)
4. Issues — active issues (200 tokens)
5. Decisions — closed/vetoed decisions (250 tokens)
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

# Per-section token caps.
_CAP_GROUNDING = 50
_CAP_RULES = 100
_CAP_HANDOFF = 500
_CAP_RECENT_WORK = 150
_CAP_ISSUES = 200
_CAP_DECISIONS = 250

# Max milestone items in recent_work section.
_MAX_RECENT_WORK = 3

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
    budget_tokens: int = 1800,
) -> dict:
    """Assemble a tight session briefing from memories.

    Six sections, filled in priority order within the token budget,
    each with its own per-section cap:
    0. grounding — one-line project description (orientation)
    1. rules — pinned memories (behavioral overrides, always first)
    2. handoff — last session handoff (continuity)
    3. recent_work — milestone breadcrumbs (what was recently done)
    4. issues — active issues (what's broken)
    5. decisions — closed decisions (what NOT to suggest)

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
    milestone_coro = list_memories(
        pool, memory_type=MemoryType.milestone, status=MemoryStatus.active,
        limit=10, **_pid,
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
        milestone_raw,
        issues_raw,
        decisions_raw,
    ) = await asyncio.gather(
        grounding_coro, pinned_coro, handoff_coro,
        milestone_coro, issues_coro, decisions_coro,
    )

    # --- Budget packing (sequential, in priority order) ---
    used_tokens = 0
    seen_ids: set[str] = set()
    excluded = 0
    section_tokens: dict[str, int] = {}

    # Section 0: Project grounding (one-liner)
    grounding_line: str | None = None
    section_used = 0
    if grounding_raw:
        mem = grounding_raw[0]
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens and cost <= _CAP_GROUNDING:
            grounding_line = mem.content
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used = cost
        else:
            excluded += 1
    section_tokens["grounding"] = section_used

    # Section 1: Rules (pinned memories only)
    pinned_raw.sort(key=lambda m: (m.confidence, m.created_at.timestamp()), reverse=True)

    rules_section: list[dict] = []
    section_used = 0
    for mem in pinned_raw:
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_RULES
        ):
            rules_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["rules"] = section_used

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
    section_used = 0
    if handoff_candidates:
        mem = handoff_candidates[0]
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and cost <= _CAP_HANDOFF
        ):
            entry = mem.to_dict()
            age_hours = (now - mem.created_at).total_seconds() / 3600
            entry["age_hours"] = round(age_hours, 1)
            handoff_section.append(entry)
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used = cost
        else:
            excluded += 1
    section_tokens["handoff"] = section_used

    # Section 3: Recent work (milestones from last 72h)
    cutoff = now.timestamp() - (72 * 3600)
    milestone_candidates = [
        m for m in milestone_raw
        if m.id not in seen_ids and m.created_at.timestamp() > cutoff
    ]
    milestone_candidates.sort(key=lambda m: m.created_at, reverse=True)

    recent_work_section: list[dict] = []
    section_used = 0
    for mem in milestone_candidates:
        if len(recent_work_section) >= _MAX_RECENT_WORK:
            excluded += 1
            continue
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_RECENT_WORK
        ):
            age_hours = (now - mem.created_at).total_seconds() / 3600
            entry = {
                "summary": mem.content,
                "age_hours": round(age_hours, 1),
                "refs": mem.topic,
                "id": mem.id,
            }
            recent_work_section.append(entry)
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["recent_work"] = section_used

    # Section 4: Active issues
    issue_candidates = [m for m in issues_raw if m.id not in seen_ids]
    issue_candidates.sort(key=lambda m: m.created_at, reverse=True)

    issue_items: list[dict] = []
    section_used = 0
    for mem in issue_candidates:
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_ISSUES
        ):
            issue_items.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["issues"] = section_used

    # Section 5: Closed decisions (what NOT to suggest)
    decision_candidates = [m for m in decisions_raw if m.id not in seen_ids]
    decision_candidates.sort(
        key=lambda m: (
            0 if project_id and m.project_id == project_id else 1,
            -(m.created_at.timestamp()),
        ),
    )

    decisions_section: list[dict] = []
    section_used = 0
    for i, mem in enumerate(decision_candidates):
        if len(decisions_section) >= _MAX_DECISIONS:
            excluded += len(decision_candidates) - i
            break
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_DECISIONS
        ):
            decisions_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["decisions"] = section_used

    # Collect all included memories for freshness calculation
    all_included: list[dict] = (
        rules_section + handoff_section + issue_items + decisions_section
    )
    freshness_hours = _newest_created_at(all_included, now)

    return {
        "grounding": grounding_line,
        "rules": rules_section,
        "handoff": handoff_section,
        "recent_work": recent_work_section,
        "issues": {"count": len(issue_items), "items": issue_items},
        "decisions": decisions_section,
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
        "excluded": excluded,
        "freshness_hours": freshness_hours,
        "section_tokens": section_tokens,
    }
