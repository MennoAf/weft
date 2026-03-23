"""Decisions section — closed/vetoed decisions (what NOT to suggest).

Tier 2 (deferred under progressive disclosure).  Project-scoped decisions
sort before global ones.  Includes review_after annotation.

Reference: weft/primer.py lines 268-272 (fetch), 604-657 (pack).
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
    annotate_review_after,
    is_unscoped_ingest,
    unwrap_recall,
)
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["decisions"]
_MAX = SECTION_MAX_ITEMS["decisions"]


async def build_decisions_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack closed decision memories."""
    if ctx.biased:
        raw = await search_by_vector(
            ctx.pool, ctx.query_vec,
            memory_type=MemoryType.decision, status=MemoryStatus.active,
            limit=20, threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_memories(
            ctx.pool, memory_type=MemoryType.decision, status=MemoryStatus.active,
            limit=20, **ctx.scope,
        )

    pairs = unwrap_recall(raw)

    candidates = [
        (m, sim) for m, sim in pairs
        if m.id not in ctx.seen_ids
        and not is_unscoped_ingest(m, ctx.project_id)
    ]

    # Sort: project-scoped first, then by score.
    if ctx.biased:
        candidates.sort(
            key=lambda pair: (
                0 if ctx.project_id and pair[0].project_id == ctx.project_id else 1,
                -(
                    SIMILARITY_WEIGHT * (pair[1] or 0)
                    + (1 - SIMILARITY_WEIGHT) * pair[0].usefulness_score
                ),
                -(pair[0].created_at.timestamp()),
            ),
        )
    else:
        candidates.sort(
            key=lambda pair: (
                0 if ctx.project_id and pair[0].project_id == ctx.project_id else 1,
                -pair[0].usefulness_score,
                -(pair[0].created_at.timestamp()),
            ),
        )

    items: list[dict] = []
    section_used = 0
    for i, (mem, _sim) in enumerate(candidates):
        if len(items) >= _MAX:
            ctx.excluded += len(candidates) - i
            break
        cost = (mem.token_count or estimate_tokens(mem.content)) + DICT_OVERHEAD_TOKENS
        if ctx.fits(cost, section_used, _CAP):
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "confidence": mem.confidence,
                "project_id": mem.project_id,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "review_after": mem.review_after,
            }
            items.append(annotate_review_after(entry, ctx.now))
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["decisions"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
