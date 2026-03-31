"""Issues section — active bugs and blockers.

Always included (tier 1).  Filters out unscoped ingested memories.
When query-biased, uses blended similarity + usefulness ranking.

Reference: weft/primer.py lines 263-266 (fetch), 513-557 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus, MemoryType
from weft.primer_sections.context import (
    DICT_OVERHEAD_TOKENS,
    QUERY_SIMILARITY_THRESHOLD,
    SECTION_BUDGETS,
    SIMILARITY_WEIGHT,
    PrimerContext,
    SectionResult,
    is_unscoped_ingest,
    unwrap_recall,
)
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["issues"]


async def build_issues_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack active issue memories."""
    if ctx.biased:
        raw = await search_by_vector(
            ctx.pool, ctx.query_vec,
            memory_type=MemoryType.issue, status=MemoryStatus.active,
            limit=20, threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_memories(
            ctx.pool, memory_type=MemoryType.issue, status=MemoryStatus.active,
            limit=20, **ctx.scope,
        )

    pairs = unwrap_recall(raw)

    candidates = [
        (m, sim) for m, sim in pairs
        if m.id not in ctx.seen_ids
        and not is_unscoped_ingest(m, ctx.project_id)
    ]

    if ctx.biased:
        candidates.sort(
            key=lambda pair: (
                SIMILARITY_WEIGHT * (pair[1] or 0)
                + (1 - SIMILARITY_WEIGHT) * pair[0].usefulness_score
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda pair: (pair[0].usefulness_score, pair[0].created_at.timestamp()),
            reverse=True,
        )

    items: list[dict] = []
    section_used = 0
    for mem, _sim in candidates:
        cost = (mem.token_count or estimate_tokens(mem.content)) + DICT_OVERHEAD_TOKENS
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            items.append({
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            })
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["issues"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
