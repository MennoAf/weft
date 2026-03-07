"""Session primer — a tight briefing for session startup.

The primer's job is: "what would be costly to get wrong in the first
30 seconds?"  Every line should change how the agent behaves.  Reference
material, historical summaries, and architecture facts belong in recall,
fetched when the conversation makes them relevant.

Sections (in priority order, each with a per-section token cap):
0. Grounding — one-line project description (50 tokens)
1. Rules — pinned memories only (100 tokens)
2. Behaviors — persistent agent rules and strategies (150 tokens)
3. Handoff — most recent session handoff (800 tokens, truncated if needed)
4. Recent work — milestone breadcrumbs from last 72h (150 tokens)
5. Issues — active issues (200 tokens)
6. Decisions — closed/vetoed decisions (250 tokens)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import asyncpg

from weft.behaviors import list_behaviors, match_behaviors
from weft.models import Memory, MemoryRecall, MemoryStatus, MemoryType
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens, truncate_to_token_budget

logger = logging.getLogger(__name__)

# Hard caps on items shown in primer sections.
_MAX_DECISIONS = 5
_MAX_BEHAVIORS = 5

# Per-section token caps.
_CAP_GROUNDING = 50
_CAP_RULES = 100
_CAP_BEHAVIORS = 150
_CAP_HANDOFF = 800
_CAP_RECENT_WORK = 150
_CAP_ISSUES = 200
_CAP_DECISIONS = 250

# Max milestone items in recent_work section.
_MAX_RECENT_WORK = 3

# Topic used to identify project grounding memories.
_GROUNDING_TOPIC = "project-grounding"

# Cold-start threshold: if total primer items <= this AND no handoff, show onboarding.
_COLD_START_THRESHOLD = 2

# Query-biased search threshold — intentionally permissive (re-ranking, not filtering).
_QUERY_SIMILARITY_THRESHOLD = 0.1

# Blend weight for semantic similarity vs existing ranking signals.
_SIMILARITY_WEIGHT = 0.4

# Hints shown when a primer section is empty (disappear once populated).
_SECTION_HINTS: dict[str, str] = {
    "rules": (
        "No rules stored. Use weft_remember(type='rule', pinned=True) "
        "for persistent instructions (e.g., 'always use pytest')."
    ),
    "behaviors": (
        "No behavioral rules stored. Use weft_behavior_add(trigger_pattern, action) "
        "to teach agents persistent strategies (e.g., 'when writing tests' → 'use pytest')."
    ),
    "handoff": (
        "No session handoff found. Use weft_handoff(summary=...) "
        "before ending sessions to preserve continuity."
    ),
    "recent_work": (
        "No recent milestones. weft_learn(content=..., task_id=...) "
        "auto-creates milestones after task completion."
    ),
    "issues": (
        "No open issues. Use weft_remember(type='issue') to track "
        "bugs or blockers discovered during work."
    ),
    "decisions": (
        "No decisions recorded. Use weft_remember(type='decision') "
        "for architectural choices that shouldn't be re-proposed."
    ),
}

# Compact onboarding guide for agents seeing Weft for the first time (~180 tokens).
_ONBOARDING_TEXT = """\
Welcome to Weft — persistent memory for AI agents.

Key tools:
- weft_remember(content, type, confidence) — store knowledge \
(types: fact, decision, preference, pattern, architecture, solution, issue, rule)
- weft_recall(query) — semantic search across all memories
- weft_learn(content) — capture lessons after completing work (auto-extracts and stores)
- weft_handoff(summary, next_steps, ...) — preserve session context for the next agent
- weft_feedback(memory_id, helpful) — rate memories to improve future ranking
- weft_context(query, budget_tokens) — budget-aware retrieval for mid-session use

Tips for getting started:
- Store decisions with type='decision' so they appear in future primers and aren't re-debated
- Pin important rules with pinned=True — they always appear in the primer
- Call weft_handoff before ending sessions — the next primer surfaces it prominently
- After completing tasks, call weft_learn to capture gotchas and patterns automatically"""


def _annotate_review_after(entry: dict, now: datetime) -> dict:
    """Add review_after / review_due fields to a memory dict if applicable."""
    ra = entry.get("review_after")
    if ra is None:
        return entry
    if isinstance(ra, str):
        ra = datetime.fromisoformat(ra)
    entry["review_after"] = ra.isoformat()
    entry["review_due"] = ra <= now
    return entry


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
    agent_id: str | None = None,
    budget_tokens: int = 2400,
    query_vec: list[float] | None = None,
) -> dict:
    """Assemble a tight session briefing from memories.

    Seven sections, filled in priority order within the token budget,
    each with its own per-section cap:
    0. grounding — one-line project description (orientation)
    1. rules — pinned memories (behavioral overrides, always first)
    2. behaviors — persistent agent rules/strategies (how to act)
    3. handoff — last session handoff (continuity)
    4. recent_work — milestone breadcrumbs (what was recently done)
    5. issues — active issues (what's broken)
    6. decisions — closed decisions (what NOT to suggest)

    When *query_vec* is provided, sections 2, 4-6 (behaviors, recent_work,
    issues, decisions) use vector similarity search with blended re-ranking
    instead of plain metadata queries.  Sections 0-1, 3 are never biased.

    Everything else (architecture, patterns, reference facts, preferences)
    lives in recall — fetched on demand when the conversation needs it.

    All section fetches run in parallel via asyncio.gather, then budget
    packing happens sequentially in priority order.
    """
    now = datetime.now(timezone.utc)
    biased = query_vec is not None

    # list_memories already handles `project_id = $X OR project_id IS NULL`
    # when project_id is provided, so a single call per section suffices.
    # Same OR-NULL pattern applies to agent_id.
    _scope: dict[str, str] = {}
    if project_id:
        _scope["project_id"] = project_id
    if agent_id:
        _scope["agent_id"] = agent_id

    # --- Fetch all sections in parallel ---
    async def _empty() -> list:
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
        pool, status=MemoryStatus.active, pinned=True, limit=100, **_scope,
    )

    # Behaviors: vector match when biased, otherwise top by priority.
    if biased:
        behaviors_coro = match_behaviors(
            pool, query_vec, limit=_MAX_BEHAVIORS * 2,
            threshold=_QUERY_SIMILARITY_THRESHOLD, **_scope,
        )
    else:
        behaviors_coro = list_behaviors(
            pool, enabled=True, limit=_MAX_BEHAVIORS * 2, **_scope,
        )

    handoff_coro = list_memories(
        pool, memory_type=MemoryType.handoff, status=MemoryStatus.active,
        limit=5, **_scope,
    )

    # Biased sections: use search_by_vector when query_vec is available.
    if biased:
        milestone_coro = search_by_vector(
            pool, query_vec,
            memory_type=MemoryType.milestone, status=MemoryStatus.active,
            limit=10, threshold=_QUERY_SIMILARITY_THRESHOLD, **_scope,
        )
        issues_coro = search_by_vector(
            pool, query_vec,
            memory_type=MemoryType.issue, status=MemoryStatus.active,
            limit=20, threshold=_QUERY_SIMILARITY_THRESHOLD, **_scope,
        )
        decisions_coro = search_by_vector(
            pool, query_vec,
            memory_type=MemoryType.decision, status=MemoryStatus.active,
            limit=20, threshold=_QUERY_SIMILARITY_THRESHOLD, **_scope,
        )
    else:
        milestone_coro = list_memories(
            pool, memory_type=MemoryType.milestone, status=MemoryStatus.active,
            limit=10, **_scope,
        )
        issues_coro = list_memories(
            pool, memory_type=MemoryType.issue, status=MemoryStatus.active,
            limit=20, **_scope,
        )
        decisions_coro = list_memories(
            pool, memory_type=MemoryType.decision, status=MemoryStatus.active,
            limit=20, **_scope,
        )

    (
        grounding_raw,
        pinned_raw,
        behaviors_raw,
        handoff_raw,
        biased_milestones_raw,
        biased_issues_raw,
        biased_decisions_raw,
    ) = await asyncio.gather(
        grounding_coro, pinned_coro, behaviors_coro, handoff_coro,
        milestone_coro, issues_coro, decisions_coro,
    )

    # Unwrap MemoryRecall → (Memory, similarity) when biased, else (Memory, None).
    def _unwrap(items: list) -> list[tuple[Memory, float | None]]:
        if not items:
            return []
        if isinstance(items[0], MemoryRecall):
            return [(r.memory, r.similarity) for r in items]
        return [(m, None) for m in items]

    milestone_raw = _unwrap(biased_milestones_raw)
    issues_raw = _unwrap(biased_issues_raw)
    decisions_raw = _unwrap(biased_decisions_raw)

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
    pinned_raw.sort(key=lambda m: (m.confidence, m.usefulness_score, m.created_at.timestamp()), reverse=True)

    rules_section: list[dict] = []
    section_used = 0
    for mem in pinned_raw:
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_RULES
        ):
            rules_section.append(_annotate_review_after(mem.to_dict(), now))
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["rules"] = section_used

    # Section 2: Behaviors (persistent agent rules/strategies)
    # behaviors_raw is either list[BehaviorMatch] (biased) or list[Behavior] (unbiased)
    from weft.models import BehaviorMatch as _BM

    behaviors_section: list[dict] = []
    section_used = 0
    for item in behaviors_raw:
        if len(behaviors_section) >= _MAX_BEHAVIORS:
            break
        if isinstance(item, _BM):
            beh = item.behavior
        else:
            beh = item
        cost = beh.token_count or estimate_tokens(beh.trigger_pattern + " " + beh.action)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_BEHAVIORS
        ):
            entry = {
                "trigger": beh.trigger_pattern,
                "action": beh.action,
                "confidence": beh.confidence,
                "priority": beh.priority,
                "scope": beh.scope.value if hasattr(beh.scope, "value") else beh.scope,
                "id": beh.id,
            }
            behaviors_section.append(entry)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["behaviors"] = section_used

    # Section 3: Last session handoff (most recent only)
    handoff_candidates = [m for m in handoff_raw if m.id not in seen_ids]
    if not handoff_candidates:
        # Fallback: check for topic "session-handoff" (handles pre-typed-handoff memories).
        # DEPRECATED: this fallback will be removed in a future version.
        topic_raw = await list_memories(
            pool, topic="session-handoff", status=MemoryStatus.active,
            limit=5, **_scope,
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
        content = mem.content
        # Truncate oversized handoffs instead of dropping them.
        cap = min(_CAP_HANDOFF, budget_tokens - used_tokens)
        if cost > cap and cap > 0:
            content, cost = truncate_to_token_budget(content, cap)
        if used_tokens + cost <= budget_tokens and cap > 0:
            entry = mem.to_dict()
            entry["content"] = content
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
        (m, sim) for m, sim in milestone_raw
        if m.id not in seen_ids and m.created_at.timestamp() > cutoff
    ]
    if biased:
        # Blend similarity with recency (newer = higher score).
        _ts_range = (
            max(m.created_at.timestamp() for m, _ in milestone_candidates)
            - min(m.created_at.timestamp() for m, _ in milestone_candidates)
        ) if len(milestone_candidates) > 1 else 1.0
        milestone_candidates.sort(
            key=lambda pair: (
                _SIMILARITY_WEIGHT * (pair[1] or 0)
                + (1 - _SIMILARITY_WEIGHT) * (
                    (pair[0].created_at.timestamp() - cutoff) / max(_ts_range, 1.0)
                )
            ),
            reverse=True,
        )
    else:
        milestone_candidates.sort(key=lambda pair: pair[0].created_at, reverse=True)

    recent_work_section: list[dict] = []
    section_used = 0
    for mem, _sim in milestone_candidates:
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
    issue_candidates = [(m, sim) for m, sim in issues_raw if m.id not in seen_ids]
    if biased:
        issue_candidates.sort(
            key=lambda pair: (
                _SIMILARITY_WEIGHT * (pair[1] or 0)
                + (1 - _SIMILARITY_WEIGHT) * pair[0].usefulness_score
            ),
            reverse=True,
        )
    else:
        issue_candidates.sort(
            key=lambda pair: (pair[0].usefulness_score, pair[0].created_at.timestamp()),
            reverse=True,
        )

    issue_items: list[dict] = []
    section_used = 0
    for mem, _sim in issue_candidates:
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
    decision_candidates = [(m, sim) for m, sim in decisions_raw if m.id not in seen_ids]
    if biased:
        decision_candidates.sort(
            key=lambda pair: (
                0 if project_id and pair[0].project_id == project_id else 1,
                -(
                    _SIMILARITY_WEIGHT * (pair[1] or 0)
                    + (1 - _SIMILARITY_WEIGHT) * pair[0].usefulness_score
                ),
                -(pair[0].created_at.timestamp()),
            ),
        )
    else:
        decision_candidates.sort(
            key=lambda pair: (
                0 if project_id and pair[0].project_id == project_id else 1,
                -pair[0].usefulness_score,
                -(pair[0].created_at.timestamp()),
            ),
        )

    decisions_section: list[dict] = []
    section_used = 0
    for i, (mem, _sim) in enumerate(decision_candidates):
        if len(decisions_section) >= _MAX_DECISIONS:
            excluded += len(decision_candidates) - i
            break
        cost = mem.token_count or estimate_tokens(mem.content)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_DECISIONS
        ):
            decisions_section.append(_annotate_review_after(mem.to_dict(), now))
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

    # --- Onboarding: empty-section hints + cold-start detection ---
    hints: dict[str, str] = {}
    if not rules_section:
        hints["rules"] = _SECTION_HINTS["rules"]
    if not behaviors_section:
        hints["behaviors"] = _SECTION_HINTS["behaviors"]
    if not handoff_section:
        hints["handoff"] = _SECTION_HINTS["handoff"]
    if not recent_work_section:
        hints["recent_work"] = _SECTION_HINTS["recent_work"]
    if not issue_items:
        hints["issues"] = _SECTION_HINTS["issues"]
    if not decisions_section:
        hints["decisions"] = _SECTION_HINTS["decisions"]

    # Cold-start: no handoff AND very few memories → show onboarding guide.
    total_items = (
        len(rules_section) + len(behaviors_section) + len(handoff_section)
        + len(recent_work_section) + len(issue_items)
        + len(decisions_section)
    )
    onboarding: str | None = (
        _ONBOARDING_TEXT
        if not handoff_section and total_items <= _COLD_START_THRESHOLD
        else None
    )

    return {
        "grounding": grounding_line,
        "rules": rules_section,
        "behaviors": behaviors_section,
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
        "hints": hints,
        "onboarding": onboarding,
    }
