"""Rules section — pinned memories that override default agent behavior.

Always included (tier 1, never deferred).  Sorted by confidence ×
usefulness × recency.  Each entry includes review_after annotation.

Reference: weft/primer.py lines 236-238 (fetch), 347-373 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus
from weft.primer_sections.context import (
    DICT_OVERHEAD_TOKENS,
    SECTION_BUDGETS,
    PrimerContext,
    SectionFetch,
    SectionResult,
    annotate_review_after,
)
from weft.store import list_memories
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["rules"]


async def fetch_rules_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch pinned rule memories (parallel-safe, no ctx mutation)."""
    raw = await list_memories(
        ctx.pool, status=MemoryStatus.active, pinned=True, limit=100,
        **ctx.scope,
    )
    raw.sort(
        key=lambda m: (m.confidence, m.usefulness_score, m.created_at.timestamp()),
        reverse=True,
    )
    return SectionFetch(payload=raw)


def pack_rules_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack fetched rules against the budget (sequential, mutates ctx)."""
    raw = fetched.payload or []
    items: list[dict] = []
    section_used = 0
    for mem in raw:
        cost = (mem.token_count or estimate_tokens(mem.content)) + DICT_OVERHEAD_TOKENS
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "pinned": mem.pinned,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "review_after": mem.review_after,
            }
            items.append(annotate_review_after(entry, ctx.now))
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["rules"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_rules_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack pinned rule memories (single-shot wrapper)."""
    return pack_rules_section(ctx, await fetch_rules_section(ctx))
