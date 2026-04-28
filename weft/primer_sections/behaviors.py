"""Behaviors section — persistent agent rules and strategies.

When query-biased (ctx.biased), uses match_behaviors with vector similarity.
Otherwise, lists by priority.  Cap is scaled by ctx.behavior_boost (from
mode weights).

Reference: weft/primer.py lines 241-249 (fetch), 375-406 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.behaviors import list_behaviors, match_behaviors
from weft.models import BehaviorMatch
from weft.primer_sections.context import (
    QUERY_SIMILARITY_THRESHOLD,
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["behaviors"]
_MAX = SECTION_MAX_ITEMS["behaviors"]


async def fetch_behaviors_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch behavior rules (parallel-safe, no ctx mutation)."""
    if ctx.biased:
        raw = await match_behaviors(
            ctx.pool, ctx.query_vec, limit=_MAX * 2,
            threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_behaviors(
            ctx.pool, enabled=True, limit=_MAX * 2, **ctx.scope,
        )
    return SectionFetch(payload=raw)


def pack_behaviors_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack fetched behaviors against the budget (sequential, mutates ctx)."""
    cap = max(0, int(_CAP * ctx.behavior_boost))
    raw = fetched.payload or []
    items: list[dict] = []
    section_used = 0
    for item in raw:
        if len(items) >= _MAX:
            break
        if isinstance(item, BehaviorMatch):
            beh = item.behavior
        else:
            beh = item
        cost = beh.token_count or estimate_tokens(beh.trigger_pattern + " " + beh.action)
        if ctx.fits_or_guarantee(cost, section_used, cap):
            entry = {
                "trigger": beh.trigger_pattern,
                "action": beh.action,
                "confidence": beh.confidence,
                "priority": beh.priority,
                "scope": beh.scope.value if hasattr(beh.scope, "value") else beh.scope,
                "id": beh.id,
            }
            items.append(entry)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["behaviors"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_behaviors_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack behaviors (single-shot wrapper)."""
    return pack_behaviors_section(ctx, await fetch_behaviors_section(ctx))
