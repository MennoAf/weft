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

    # --- Phase 1: All sections in parallel ---
    # Sections are independent (read-only ctx, asyncpg pool handles concurrency).
    # Running them concurrently collapses 20-40 sequential DB round-trips into
    # a handful of concurrent batches — critical for Fly.io latency.
    async def _noop() -> SectionResult:
        return _skip

    (
        grounding_result,
        rules_result,
        behaviors_result,
        handoff_result,
        recent_work_result,
        issues_result,
        anti_patterns_result,
        decisions_result,
        entities_result,
        autonomy_result,
        calibration_result,
        degradation_result,
        triggers_result,
        cost_result,
        working_memory_result,
        changes_result,
        wellness_result,
    ) = await asyncio.gather(
        build_grounding_section(ctx) if _enabled("grounding") else _noop(),
        build_rules_section(ctx) if _enabled("rules") else _noop(),
        build_behaviors_section(ctx) if _enabled("behaviors") else _noop(),
        build_handoff_section(ctx) if _enabled("handoff") else _noop(),
        build_recent_work_section(ctx) if _enabled("recent_work") else _noop(),
        build_issues_section(ctx) if _enabled("issues") else _noop(),
        build_anti_patterns_section(ctx) if _enabled("anti_patterns") else _noop(),
        build_decisions_section(ctx) if _enabled("decisions") else _noop(),
        build_entities_section(ctx) if _enabled("entities") else _noop(),
        build_autonomy_section(ctx) if _enabled("autonomy") else _noop(),
        build_calibration_section(ctx) if _enabled("calibration") else _noop(),
        build_degradation_section(ctx) if _enabled("degradation") else _noop(),
        build_triggers_section(ctx) if _enabled("triggers") else _noop(),
        build_cost_section(ctx) if _enabled("cost") else _noop(),
        build_working_memory_section(ctx) if _enabled("working_memory") else _noop(),
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
