"""Entities section — known people, projects, and tools.

Tier 2 (deferred under progressive disclosure).  Cap is scaled by
ctx.entity_boost from mode weights.  Not query-biased.

Reference: weft/primer.py line 296 (fetch), 659-682 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.entities import list_entities
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    PrimerContext,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["entities"]
_MAX = SECTION_MAX_ITEMS["entities"]


async def build_entities_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack known entities (people, projects, tools)."""
    cap = max(0, int(_CAP * ctx.entity_boost))

    raw = await list_entities(ctx.pool, limit=_MAX, **ctx.scope)

    items: list[dict] = []
    section_used = 0
    for ent in raw:
        if len(items) >= _MAX:
            break
        ent_text = ent.name + (f": {ent.description}" if ent.description else "")
        cost = estimate_tokens(ent_text)
        if ctx.fits_or_guarantee(cost, section_used, cap):
            items.append({
                "name": ent.name,
                "type": ent.entity_type.value,
                "description": ent.description,
                "mention_count": ent.mention_count,
                "id": ent.id,
            })
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["entities"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
