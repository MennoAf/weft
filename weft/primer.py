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
5b. Anti-patterns — pitfalls to avoid (150 tokens)
6. Decisions — closed/vetoed decisions (250 tokens)
7. Entities — known people, projects, tools for this project (150 tokens)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import asyncpg

from weft.behaviors import list_behaviors, match_behaviors
from weft.entities import list_entities
from weft.git_utils import get_recent_commits
from weft.models import Memory, MemoryRecall, MemorySource, MemoryStatus, MemoryType
from weft.store import get_last_handoff_timestamp, get_memory_changes_since, list_memories, search_by_vector
from weft.tokens import estimate_tokens, truncate_to_token_budget

# Overhead tokens per memory dict entry (id, type, timestamps, metadata fields).
# Accounts for JSON keys and values that to_dict() adds beyond content.
_DICT_OVERHEAD_TOKENS = 40

logger = logging.getLogger(__name__)

# Hard caps on items shown in primer sections.
_MAX_DECISIONS = 5
_MAX_BEHAVIORS = 5
_MAX_ENTITIES = 10
_MAX_ANTI_PATTERNS = 3

# Per-section token caps.
_CAP_GROUNDING = 50
_CAP_RULES = 100
_CAP_BEHAVIORS = 150
_CAP_HANDOFF = 800
_CAP_RECENT_WORK = 300
_CAP_ISSUES = 200
_CAP_ANTI_PATTERNS = 250
_CAP_DECISIONS = 250
_CAP_ENTITIES = 250
_CAP_CHANGES_SINCE_COMMITS = 20

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
- After completing tasks, call weft_learn to capture gotchas and patterns automatically

Loom integration:
- If Loom is available, run loom_create_project before decomposing work \
to avoid tasks landing in the wrong project."""


def _is_unscoped_ingest(mem: Memory, project_id: str | None) -> bool:
    """True when *mem* is a global ingested record that shouldn't appear in a
    project-scoped primer.  Intentional globals (conversation, agent, seed,
    etc.) are allowed through; only bulk-ingested data (source=ingest) with
    no project_id is filtered out."""
    if not project_id:
        return False  # no project filter → everything is fine
    if mem.project_id is not None:
        return False  # memory belongs to a project → fine
    return mem.source == MemorySource.ingest or (
        isinstance(mem.source, str) and mem.source == "ingest"
    )


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


async def _build_primer_legacy(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 2400,
    query_vec: list[float] | None = None,
    disclosure: str = "progressive",
    mode: str | None = None,
) -> dict:
    """LEGACY monolithic primer — kept as fallback. Use build_primer() instead.

    This is the original 881-line implementation, preserved intact so it can
    be called directly if the modular orchestrator has issues.  It will be
    removed once the modular version has been running in production without
    incidents for a reasonable period.

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

    When *mode* is provided, the corresponding ModeWeights are resolved and
    used to scale section caps (behavior_boost, entity_boost) and ranking
    signals (recency_bias).  vector_weight/bm25_weight are stored but not
    yet applied to primer retrieval (reserved for hybrid search integration).

    Everything else (architecture, patterns, reference facts, preferences)
    lives in recall — fetched on demand when the conversation needs it.

    All section fetches run in parallel via asyncio.gather, then budget
    packing happens sequentially in priority order.
    """
    now = datetime.now(timezone.utc)
    biased = query_vec is not None

    # Resolve mode weights (never raises — falls back to defaults)
    from weft.modes import get_active_weights
    weights = await get_active_weights(pool, mode)

    # Scale section caps by mode weights (clamped to 0)
    cap_behaviors = max(0, int(_CAP_BEHAVIORS * weights.behavior_boost))
    cap_entities = max(0, int(_CAP_ENTITIES * weights.entity_boost))

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
        anti_pattern_coro = search_by_vector(
            pool, query_vec,
            memory_type=MemoryType.anti_pattern, status=MemoryStatus.active,
            limit=10, threshold=_QUERY_SIMILARITY_THRESHOLD, **_scope,
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
        anti_pattern_coro = list_memories(
            pool, memory_type=MemoryType.anti_pattern, status=MemoryStatus.active,
            limit=10, **_scope,
        )

    entities_coro = list_entities(pool, limit=_MAX_ENTITIES, **_scope)

    (
        grounding_raw,
        pinned_raw,
        behaviors_raw,
        handoff_raw,
        biased_milestones_raw,
        biased_issues_raw,
        biased_decisions_raw,
        biased_anti_patterns_raw,
        entities_raw,
    ) = await asyncio.gather(
        grounding_coro, pinned_coro, behaviors_coro, handoff_coro,
        milestone_coro, issues_coro, decisions_coro, anti_pattern_coro, entities_coro,
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
    anti_pattern_raw = _unwrap(biased_anti_patterns_raw)

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
        cost = (mem.token_count or estimate_tokens(mem.content)) + _DICT_OVERHEAD_TOKENS
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_RULES
        ):
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "pinned": mem.pinned,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "review_after": mem.review_after,
            }
            rules_section.append(_annotate_review_after(entry, now))
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
            and section_used + cost <= cap_behaviors
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
        # Always re-estimate from content — stored token_count may be stale.
        cost = estimate_tokens(mem.content) + _DICT_OVERHEAD_TOKENS
        content = mem.content
        # Truncate oversized handoffs instead of dropping them.
        cap = min(_CAP_HANDOFF, budget_tokens - used_tokens)
        if cost > cap and cap > 0:
            content, cost = truncate_to_token_budget(content, cap)
            cost += _DICT_OVERHEAD_TOKENS
        if used_tokens + cost <= budget_tokens and cap > 0:
            age_hours = (now - mem.created_at).total_seconds() / 3600
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": content,
                "confidence": mem.confidence,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "age_hours": round(age_hours, 1),
            }
            handoff_section.append(entry)
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used = cost
        else:
            excluded += 1
    section_tokens["handoff"] = section_used

    # Section 3: Recent work (milestones from last 72h)
    cutoff = now.timestamp() - (72 * 3600)
    # Same ingest filter as issues/decisions.
    milestone_candidates = [
        (m, sim) for m, sim in milestone_raw
        if m.id not in seen_ids
        and m.created_at.timestamp() > cutoff
        and not _is_unscoped_ingest(m, project_id)
    ]
    # Compute effective similarity weight — recency_bias shifts the blend
    # toward recency.  At recency_bias=0 (default), use _SIMILARITY_WEIGHT.
    # At recency_bias=1.0, similarity gets 0 weight (pure recency sort).
    effective_sim_weight = _SIMILARITY_WEIGHT * (1.0 - weights.recency_bias)

    if biased:
        # Blend similarity with recency (newer = higher score).
        _ts_range = (
            max(m.created_at.timestamp() for m, _ in milestone_candidates)
            - min(m.created_at.timestamp() for m, _ in milestone_candidates)
        ) if len(milestone_candidates) > 1 else 1.0
        milestone_candidates.sort(
            key=lambda pair: (
                effective_sim_weight * (pair[1] or 0)
                + (1 - effective_sim_weight) * (
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
    # Filter out unscoped ingested memories when project-scoped — prevents
    # bulk-ingested data (e.g. Slack messages with project_id=None) from
    # flooding the primer.  Intentional globals (conversation, agent, etc.)
    # still surface everywhere.
    issue_candidates = [
        (m, sim) for m, sim in issues_raw
        if m.id not in seen_ids
        and not _is_unscoped_ingest(m, project_id)
    ]
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
        cost = (mem.token_count or estimate_tokens(mem.content)) + _DICT_OVERHEAD_TOKENS
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_ISSUES
        ):
            issue_items.append({
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["issues"] = section_used

    # Section 5: Anti-patterns (pitfalls to avoid)
    anti_pattern_candidates = [
        (m, sim) for m, sim in anti_pattern_raw
        if m.id not in seen_ids
        and not _is_unscoped_ingest(m, project_id)
    ]
    if biased:
        anti_pattern_candidates.sort(
            key=lambda pair: (
                _SIMILARITY_WEIGHT * (pair[1] or 0)
                + (1 - _SIMILARITY_WEIGHT) * pair[0].usefulness_score
            ),
            reverse=True,
        )
    else:
        anti_pattern_candidates.sort(
            key=lambda pair: (pair[0].usefulness_score, pair[0].created_at.timestamp()),
            reverse=True,
        )

    anti_patterns_section: list[dict] = []
    section_used = 0
    for i, (mem, _sim) in enumerate(anti_pattern_candidates):
        if len(anti_patterns_section) >= _MAX_ANTI_PATTERNS:
            excluded += len(anti_pattern_candidates) - i
            break
        cost = (mem.token_count or estimate_tokens(mem.content)) + _DICT_OVERHEAD_TOKENS
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_ANTI_PATTERNS
        ):
            anti_patterns_section.append({
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["anti_patterns"] = section_used

    # Section 6: Closed decisions (what NOT to suggest)
    # Same ingest filter as issues.
    decision_candidates = [
        (m, sim) for m, sim in decisions_raw
        if m.id not in seen_ids
        and not _is_unscoped_ingest(m, project_id)
    ]
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
        cost = (mem.token_count or estimate_tokens(mem.content)) + _DICT_OVERHEAD_TOKENS
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= _CAP_DECISIONS
        ):
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "project_id": mem.project_id,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "review_after": mem.review_after,
            }
            decisions_section.append(_annotate_review_after(entry, now))
            seen_ids.add(mem.id)
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["decisions"] = section_used

    # Section 7: Entities (known people, projects, tools)
    entities_section: list[dict] = []
    section_used = 0
    for ent in entities_raw:
        if len(entities_section) >= _MAX_ENTITIES:
            break
        ent_text = ent.name + (f": {ent.description}" if ent.description else "")
        cost = estimate_tokens(ent_text)
        if (
            used_tokens + cost <= budget_tokens
            and section_used + cost <= cap_entities
        ):
            entities_section.append({
                "name": ent.name,
                "type": ent.entity_type.value,
                "description": ent.description,
                "mention_count": ent.mention_count,
                "id": ent.id,
            })
            used_tokens += cost
            section_used += cost
        else:
            excluded += 1
    section_tokens["entities"] = section_used

    # --- Changes since last session ---
    changes_since: dict | None = None
    try:
        handoff_ts = await get_last_handoff_timestamp(pool, project_id=project_id)
        if handoff_ts is not None:
            changes = await get_memory_changes_since(
                pool, since=handoff_ts, project_id=project_id,
            )
            try:
                commits = await get_recent_commits(since=handoff_ts)
            except Exception:
                commits = []
            changes["recent_commits"] = commits[:_CAP_CHANGES_SINCE_COMMITS]
            changes_since = changes
    except Exception as exc:
        logger.warning("Failed to compute changes_since: %s", exc)

    # --- Wellness Snapshot (from check-in patterns) ---
    wellness_snapshot: dict | None = None
    try:
        from weft.check_in_patterns import analyze_all, rolling_averages, detect_streaks, trend_direction
        from weft.check_ins import list_check_ins

        check_ins = await list_check_ins(pool, limit=200)
        if check_ins:
            report = analyze_all(check_ins)
            # Build a concise snapshot: trends + streaks + rolling avg summary
            snapshot: dict = {}
            if report["trends"]["mood"] or report["trends"]["energy"] or report["trends"]["sleep"]:
                snapshot["trends"] = {
                    k: v for k, v in report["trends"].items()
                    if k != "period_days" and v is not None
                }
            if report["streaks"]["logging_streak"] > 0:
                snapshot["logging_streak"] = report["streaks"]["logging_streak"]
            if report["streaks"]["good_mood_streaks"]:
                snapshot["good_mood_streak"] = report["streaks"]["good_mood_streaks"][-1]
            if report["streaks"]["low_mood_streaks"]:
                snapshot["low_mood_streak"] = report["streaks"]["low_mood_streaks"][-1]
            # Latest rolling average point
            series = report["rolling_averages"].get("series", [])
            if series:
                latest = series[-1]
                snapshot["current_averages"] = {
                    "mood": latest["avg_mood"],
                    "energy": latest["avg_energy"],
                    "sleep": latest["avg_sleep"],
                    "window_days": report["rolling_averages"]["window_days"],
                }
            if snapshot:
                wellness_snapshot = snapshot
    except Exception as e:
        logger.debug("wellness_snapshot_fetch failed: %s", e, exc_info=True)

    # Collect all included memories for freshness calculation
    all_included: list[dict] = (
        rules_section + handoff_section + issue_items + anti_patterns_section + decisions_section
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
        + len(decisions_section) + len(entities_section)
    )
    is_cold_start = not handoff_section and total_items <= _COLD_START_THRESHOLD
    onboarding: str | None = _ONBOARDING_TEXT if is_cold_start else None

    # RLS diagnostic: if all sections are empty but the DB has rows,
    # the user_id context may be misconfigured.
    if total_items == 0 and not is_cold_start:
        try:
            approx_count = await pool.fetchval(
                "SELECT reltuples::bigint FROM pg_class WHERE relname = 'memories'"
            )
            if approx_count and approx_count > 0:
                hints["rls_diagnostic"] = (
                    f"All primer sections returned 0 items but the database "
                    f"has ~{approx_count} memories. This may indicate a "
                    f"user_id / RLS misconfiguration. Check that the "
                    f"Authorization header contains a valid JWT with the "
                    f"correct 'sub' claim, or that app.user_id is being set."
                )
        except Exception as e:
            logger.debug("rls_diagnostic_check failed: %s", e, exc_info=True)

    # Loom hint: only on cold start (new/unknown project).
    if is_cold_start:
        hints["loom"] = (
            "New project detected. If Loom is available, run "
            "loom_create_project to set up a dedicated task space "
            "before decomposing work with loom_decompose."
        )

    # --- Progressive disclosure: tier 2 sections become summaries ---
    progressive = disclosure == "progressive"
    if progressive:
        # Tier 2 sections: return counts + deferred flag instead of content.
        # Agents can call weft_focus(intent="...") to load relevant ones.
        _deferred_decisions = {
            "count": len(decisions_section),
            "deferred": True,
            "hint": "Use weft_focus(intent=...) to load relevant decisions.",
        } if decisions_section else {"count": 0, "deferred": True}

        _deferred_recent_work = {
            "count": len(recent_work_section),
            "deferred": True,
            "hint": "Use weft_focus(intent=...) to load recent work.",
        } if recent_work_section else {"count": 0, "deferred": True}

        _deferred_behaviors = {
            "count": len(behaviors_section),
            "deferred": True,
            "hint": "Use weft_focus(intent=...) to load behavioral rules.",
        } if behaviors_section else {"count": 0, "deferred": True}

        _deferred_entities = {
            "count": len(entities_section),
            "deferred": True,
            "hint": "Use weft_focus(intent=...) to load known entities.",
        } if entities_section else {"count": 0, "deferred": True}

        # Recalculate used tokens — only tier 1 sections count.
        tier1_tokens = (
            section_tokens["grounding"]
            + section_tokens["rules"]
            + section_tokens["handoff"]
            + section_tokens["issues"]
            + section_tokens["anti_patterns"]
        )

        result = {
            "grounding": grounding_line,
            "rules": rules_section,
            "behaviors": _deferred_behaviors,
            "handoff": handoff_section,
            "recent_work": _deferred_recent_work,
            "issues": {"count": len(issue_items), "items": issue_items},
            "anti_patterns": anti_patterns_section,
            "decisions": _deferred_decisions,
            "entities": _deferred_entities,
            "changes_since": changes_since,
            "total_tokens": tier1_tokens,
            "budget_tokens": budget_tokens,
            "budget_remaining": budget_tokens - tier1_tokens,
            "excluded": excluded,
            "freshness_hours": freshness_hours,
            "section_tokens": section_tokens,
            "hints": hints,
            "onboarding": onboarding,
            "disclosure": "progressive",
        }
        if wellness_snapshot:
            result["wellness_snapshot"] = wellness_snapshot
        return result

    result = {
        "grounding": grounding_line,
        "rules": rules_section,
        "behaviors": behaviors_section,
        "handoff": handoff_section,
        "recent_work": recent_work_section,
        "issues": {"count": len(issue_items), "items": issue_items},
        "anti_patterns": anti_patterns_section,
        "decisions": decisions_section,
        "entities": entities_section,
        "changes_since": changes_since,
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
        "excluded": excluded,
        "freshness_hours": freshness_hours,
        "section_tokens": section_tokens,
        "hints": hints,
        "onboarding": onboarding,
        "disclosure": "full",
    }
    if wellness_snapshot:
        result["wellness_snapshot"] = wellness_snapshot
    return result


# ---------------------------------------------------------------------------
# Modular orchestrator — calls section builders from weft.primer_sections
# ---------------------------------------------------------------------------


async def build_primer(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 2400,
    query_vec: list[float] | None = None,
    disclosure: str = "progressive",
    mode: str | None = None,
    disabled_sections: set[str] | None = None,
) -> dict:
    """Assemble a tight session briefing from memories.

    Calls modular section builders from weft.primer_sections in priority
    order, packing items within per-section and global token budgets.

    Sections (in priority order):
    0. grounding — one-line project description (orientation)
    1. rules — pinned memories (behavioral overrides, always first)
    2. behaviors — persistent agent rules/strategies (how to act)
    3. handoff — last session handoff (continuity)
    4. recent_work — milestone breadcrumbs (what was recently done)
    5. issues — active issues (what's broken)
    5b. anti_patterns — pitfalls to avoid
    6. decisions — closed decisions (what NOT to suggest)
    7. entities — known people, projects, tools
    8. autonomy — permission boundaries
    9. calibration — approval rate insights
    10. degradation — active guardrail policies
    11. triggers — proactive trigger rules
    12. cost — spending posture summary

    The legacy monolithic implementation is preserved as
    _build_primer_legacy() for fallback if needed.
    """
    from weft.modes import get_active_weights
    from weft.primer_sections.anti_patterns import build_anti_patterns_section
    from weft.primer_sections.behaviors import build_behaviors_section
    from weft.primer_sections.changes_since import build_changes_since_section
    from weft.primer_sections.context import PrimerContext, SectionResult
    from weft.primer_sections.decisions import build_decisions_section
    from weft.primer_sections.entities import build_entities_section
    from weft.primer_sections.grounding import build_grounding_section
    from weft.primer_sections.handoff import build_handoff_section
    from weft.primer_sections.issues import build_issues_section
    from weft.primer_sections.onboarding import build_onboarding_section
    from weft.primer_sections.recent_work import build_recent_work_section
    from weft.primer_sections.rules import build_rules_section
    from weft.primer_sections.autonomy import build_autonomy_section
    from weft.primer_sections.calibration import build_calibration_section
    from weft.primer_sections.cost import build_cost_section
    from weft.primer_sections.degradation import build_degradation_section
    from weft.primer_sections.triggers import build_triggers_section
    from weft.primer_sections.wellness import build_wellness_section
    from weft.primer_sections.working_memory import build_working_memory_section

    now = datetime.now(timezone.utc)

    # Resolve mode weights (never raises — falls back to defaults).
    weights = await get_active_weights(pool, mode)

    # Build context.
    ctx = PrimerContext(
        user_id="",  # Not used by sections directly (RLS handles auth).
        project_id=project_id,
        agent_id=agent_id,
        pool=pool,
        budget_tokens=budget_tokens,
        query=None,
        query_vec=query_vec,
        disclosure=disclosure,
        mode=mode,
        now=now,
        behavior_boost=weights.behavior_boost,
        entity_boost=weights.entity_boost,
        recency_bias=weights.recency_bias,
    )

    # Resolve disabled sections: explicit parameter > config > empty set.
    if disabled_sections is None:
        from weft.config import load_config
        try:
            cfg = load_config()
            disabled_sections = set(cfg.primer.disabled_sections)
        except Exception:
            disabled_sections = set()

    _skip = SectionResult(items=[], tokens_used=0, skipped=True, skip_reason="disabled")

    def _enabled(name: str) -> bool:
        return name not in disabled_sections

    # --- Phase 1: Budget-packed sections (sequential, in priority order) ---
    grounding_result = await build_grounding_section(ctx) if _enabled("grounding") else _skip
    rules_result = await build_rules_section(ctx) if _enabled("rules") else _skip
    behaviors_result = await build_behaviors_section(ctx) if _enabled("behaviors") else _skip
    handoff_result = await build_handoff_section(ctx) if _enabled("handoff") else _skip
    recent_work_result = await build_recent_work_section(ctx) if _enabled("recent_work") else _skip
    issues_result = await build_issues_section(ctx) if _enabled("issues") else _skip
    anti_patterns_result = await build_anti_patterns_section(ctx) if _enabled("anti_patterns") else _skip
    decisions_result = await build_decisions_section(ctx) if _enabled("decisions") else _skip
    entities_result = await build_entities_section(ctx) if _enabled("entities") else _skip
    autonomy_result = await build_autonomy_section(ctx) if _enabled("autonomy") else _skip
    calibration_result = await build_calibration_section(ctx) if _enabled("calibration") else _skip
    degradation_result = await build_degradation_section(ctx) if _enabled("degradation") else _skip
    triggers_result = await build_triggers_section(ctx) if _enabled("triggers") else _skip
    cost_result = await build_cost_section(ctx) if _enabled("cost") else _skip
    working_memory_result = await build_working_memory_section(ctx) if _enabled("working_memory") else _skip

    # --- Phase 2: Independent post-sections (parallel) ---
    async def _noop() -> SectionResult:
        return _skip

    changes_result, wellness_result = await asyncio.gather(
        build_changes_since_section(ctx) if _enabled("changes_since") else _noop(),
        build_wellness_section(ctx) if _enabled("wellness") else _noop(),
    )

    # --- Phase 3: Freshness calculation ---
    all_included: list[dict] = (
        rules_result.items
        + handoff_result.items
        + issues_result.items
        + anti_patterns_result.items
        + decisions_result.items
    )
    freshness_hours = _newest_created_at(all_included, now)

    # --- Phase 4: Onboarding post-processing ---
    section_counts = {
        "rules": len(rules_result.items),
        "behaviors": len(behaviors_result.items),
        "handoff": len(handoff_result.items),
        "recent_work": len(recent_work_result.items),
        "issues": len(issues_result.items),
        "decisions": len(decisions_result.items),
        "entities": len(entities_result.items),
        "autonomy": len(autonomy_result.items),
    }
    onboarding_result = await build_onboarding_section(ctx, section_counts=section_counts)
    onboarding_data = onboarding_result.items[0] if onboarding_result.items else {}
    hints = onboarding_data.get("hints", {})
    onboarding_text = onboarding_data.get("onboarding")

    # --- Phase 5: Extract section values ---
    grounding_line = (
        grounding_result.items[0]["grounding_line"]
        if grounding_result.items else None
    )
    changes_since = changes_result.items[0] if changes_result.items else None
    wellness_snapshot = wellness_result.items[0] if wellness_result.items else None

    # --- Phase 6: Progressive disclosure ---
    progressive = disclosure == "progressive"
    if progressive:
        def _deferred(section_items, hint):
            count = len(section_items)
            d = {"count": count, "deferred": True}
            if count > 0:
                d["hint"] = hint
            return d

        tier1_tokens = sum(
            ctx.section_tokens.get(s, 0)
            for s in ("grounding", "rules", "handoff", "issues", "anti_patterns")
        )

        result = {
            "grounding": grounding_line,
            "rules": rules_result.items,
            "behaviors": _deferred(
                behaviors_result.items,
                "Use weft_focus(intent=...) to load behavioral rules.",
            ),
            "handoff": handoff_result.items,
            "recent_work": _deferred(
                recent_work_result.items,
                "Use weft_focus(intent=...) to load recent work.",
            ),
            "issues": {"count": len(issues_result.items), "items": issues_result.items},
            "anti_patterns": anti_patterns_result.items,
            "decisions": _deferred(
                decisions_result.items,
                "Use weft_focus(intent=...) to load relevant decisions.",
            ),
            "entities": _deferred(
                entities_result.items,
                "Use weft_focus(intent=...) to load known entities.",
            ),
            "autonomy": _deferred(
                autonomy_result.items,
                "Use weft_autonomy_list to view autonomy policies.",
            ),
            **({"calibration": _deferred(
                calibration_result.items,
                "Use weft_calibration_summary to view calibration insights.",
            )} if not calibration_result.skipped else {}),
            **({"degradation": _deferred(
                degradation_result.items,
                "Use weft_degradation_list to view degradation policies.",
            )} if not degradation_result.skipped else {}),
            **({"triggers": _deferred(
                triggers_result.items,
                "Use weft_trigger_list to view proactive triggers.",
            )} if not triggers_result.skipped else {}),
            **({"cost": _deferred(
                cost_result.items,
                "Use weft_cost_summary to view cost details.",
            )} if not cost_result.skipped else {}),
            **({"working_memory": _deferred(
                working_memory_result.items,
                "Use weft_focus(intent=...) to load open episodes.",
            )} if not working_memory_result.skipped else {}),
            "changes_since": changes_since,
            "total_tokens": tier1_tokens,
            "budget_tokens": budget_tokens,
            "budget_remaining": budget_tokens - tier1_tokens,
            "excluded": ctx.excluded,
            "freshness_hours": freshness_hours,
            "section_tokens": ctx.section_tokens,
            "hints": hints,
            "onboarding": onboarding_text,
            "disclosure": "progressive",
        }
        if wellness_snapshot:
            result["wellness_snapshot"] = wellness_snapshot
        return result

    result = {
        "grounding": grounding_line,
        "rules": rules_result.items,
        "behaviors": behaviors_result.items,
        "handoff": handoff_result.items,
        "recent_work": recent_work_result.items,
        "issues": {"count": len(issues_result.items), "items": issues_result.items},
        "anti_patterns": anti_patterns_result.items,
        "decisions": decisions_result.items,
        "entities": entities_result.items,
        "autonomy": autonomy_result.items,
        **({"calibration": calibration_result.items} if not calibration_result.skipped else {}),
        **({"degradation": degradation_result.items} if not degradation_result.skipped else {}),
        **({"triggers": triggers_result.items} if not triggers_result.skipped else {}),
        **({"cost": cost_result.items} if not cost_result.skipped else {}),
        **({"working_memory": working_memory_result.items} if not working_memory_result.skipped else {}),
        "changes_since": changes_since,
        "total_tokens": ctx.used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - ctx.used_tokens,
        "excluded": ctx.excluded,
        "freshness_hours": freshness_hours,
        "section_tokens": ctx.section_tokens,
        "hints": hints,
        "onboarding": onboarding_text,
        "disclosure": "full",
    }
    if wellness_snapshot:
        result["wellness_snapshot"] = wellness_snapshot
    return result
