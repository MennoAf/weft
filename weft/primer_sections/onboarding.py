"""Onboarding section — hints and cold-start detection.

Post-processing step that runs AFTER all sections are packed.  Generates
empty-section hints, detects cold starts (no handoff + few items), and
runs an RLS diagnostic if all sections are empty but the DB has rows.

Reference: weft/primer.py lines 744-792.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import COLD_START_THRESHOLD, PrimerContext, SectionResult

logger = logging.getLogger(__name__)

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

_ONBOARDING_TEXT = """\
Welcome to Weft — persistent memory for AI agents.

Key tools:
- weft_remember(content, type, confidence) — store knowledge \
(types: fact, decision, preference, pattern, architecture, solution, issue, rule)
- When capturing a worthwhile fact, keep quantitative qualifiers that materially specify it (date, duration, amount, range, unit, period/direction); skip incidental numbers and don't save a fact solely because it has a number.
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


async def build_onboarding_section(
    ctx: PrimerContext,
    *,
    section_counts: dict[str, int] | None = None,
) -> SectionResult:
    """Compute hints, cold-start detection, and RLS diagnostic.

    *section_counts* is a dict mapping section names to their item counts
    (e.g., ``{"rules": 1, "behaviors": 0, ...}``).  The orchestrator passes
    this after all sections are packed.
    """
    counts = section_counts or {}

    hints: dict[str, str] = {}
    for section_name, hint_text in _SECTION_HINTS.items():
        if counts.get(section_name, 0) == 0:
            hints[section_name] = hint_text

    # Cold-start detection
    total_items = sum(counts.get(s, 0) for s in [
        "rules", "behaviors", "handoff", "recent_work",
        "issues", "decisions", "entities",
    ])
    has_handoff = counts.get("handoff", 0) > 0
    is_cold_start = not has_handoff and total_items <= COLD_START_THRESHOLD

    onboarding_text: str | None = _ONBOARDING_TEXT if is_cold_start else None

    # RLS diagnostic
    if total_items == 0 and not is_cold_start:
        try:
            approx_count = await ctx.pool.fetchval(
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

    # Loom hint: only on cold start.
    if is_cold_start:
        hints["loom"] = (
            "New project detected. If Loom is available, run "
            "loom_create_project to set up a dedicated task space "
            "before decomposing work with loom_decompose."
        )

    result_data = {"hints": hints, "onboarding": onboarding_text}
    return SectionResult(items=[result_data], tokens_used=0, skipped=False)
