"""Progressive disclosure — post-processing that wraps tier 2 sections.

This is NOT a peer section.  It is a post-processing step that runs AFTER
all sections are packed and converts tier 2 sections (decisions, recent_work,
behaviors, entities) into deferred summaries (count + hint).  It also
recalculates used_tokens to reflect only tier 1 content.

Reference: weft/primer.py lines 794-855.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)

# Tier 2 section names that get deferred under progressive disclosure.
TIER_2_SECTIONS = {"behaviors", "recent_work", "decisions", "entities"}

# Tier 1 sections whose tokens are counted in progressive mode.
TIER_1_SECTIONS = {"grounding", "rules", "handoff", "issues", "anti_patterns"}

# Hints for deferred sections (what to call to load them).
_DEFERRED_HINTS = {
    "behaviors": "Use weft_focus(intent=...) to load behavioral rules.",
    "recent_work": "Use weft_focus(intent=...) to load recent work.",
    "decisions": "Use weft_focus(intent=...) to load relevant decisions.",
    "entities": "Use weft_focus(intent=...) to load known entities.",
}


async def apply_progressive_disclosure(
    ctx: PrimerContext,
    *,
    section_items: dict[str, list[dict]] | None = None,
) -> SectionResult:
    """Transform tier 2 sections into deferred summaries.

    *section_items* maps section names to their packed items.
    Returns a SectionResult whose ``items`` is a single dict with
    the deferred representations for each tier 2 section, plus
    recalculated token totals.
    """
    items = section_items or {}

    deferred: dict[str, dict] = {}
    for section_name in TIER_2_SECTIONS:
        section_data = items.get(section_name, [])
        count = len(section_data)
        deferred[section_name] = {
            "count": count,
            "deferred": True,
        }
        if count > 0:
            deferred[section_name]["hint"] = _DEFERRED_HINTS.get(section_name, "")

    # Recalculate tokens for tier 1 only.
    tier1_tokens = sum(
        ctx.section_tokens.get(s, 0) for s in TIER_1_SECTIONS
    )

    result_data = {
        "deferred_sections": deferred,
        "tier1_tokens": tier1_tokens,
    }
    return SectionResult(items=[result_data], tokens_used=tier1_tokens, skipped=False)
