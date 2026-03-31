"""Anti-patterns section — pitfalls the agent should avoid.

Always included (tier 1).  Same ranking and filtering logic as issues.
Capped at 3 items.

Reference: weft/primer.py lines 273-277 (fetch), 559-602 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus, MemoryType
from weft.primer_sections.context import (
    DICT_OVERHEAD_TOKENS,
    QUERY_SIMILARITY_THRESHOLD,
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    SIMILARITY_WEIGHT,
    PrimerContext,
    SectionResult,
    is_unscoped_ingest,
    unwrap_recall,
)
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["anti_patterns"]
_MAX = SECTION_MAX_ITEMS["anti_patterns"]


async def build_anti_patterns_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack anti-pattern memories."""
    if ctx.biased:
        raw = await search_by_vector(
            ctx.pool, ctx.query_vec,
            memory_type=MemoryType.anti_pattern, status=MemoryStatus.active,
            limit=10, threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_memories(
            ctx.pool, memory_type=MemoryType.anti_pattern, status=MemoryStatus.active,
            limit=10, **ctx.scope,
        )

    pairs = unwrap_recall(raw)

    # Filter
    candidates = [
        (m, sim) for m, sim in pairs
        if m.id not in ctx.seen_ids
        and not is_unscoped_ingest(m, ctx.project_id)
    ]

    # Sort
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

    # Pack
    items: list[dict] = []
    section_used = 0
    for i, (mem, _sim) in enumerate(candidates):
        if len(items) >= _MAX:
            ctx.excluded += len(candidates) - i
            break
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

    ctx.section_tokens["anti_patterns"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
