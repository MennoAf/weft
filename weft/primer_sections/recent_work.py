"""Recent work section — milestone breadcrumbs from the last 72 hours.

When query-biased, uses blended similarity + recency ranking (with
effective_sim_weight adjusted by recency_bias from mode weights).
Filters out unscoped ingested memories.

Reference: weft/primer.py lines 258-261 (fetch), 455-511 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus, MemoryType
from weft.primer_sections.context import (
    QUERY_SIMILARITY_THRESHOLD,
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    SIMILARITY_WEIGHT,
    PrimerContext,
    SectionFetch,
    SectionResult,
    is_unscoped_ingest,
    unwrap_recall,
)
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["recent_work"]
_MAX = SECTION_MAX_ITEMS["recent_work"]

# 72-hour window for recent work.
_RECENT_HOURS = 72


async def fetch_recent_work_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch milestone memories (parallel-safe, no ctx mutation).

    seen_ids filtering is deferred to pack phase.
    """
    if ctx.biased:
        raw = await search_by_vector(
            ctx.pool, ctx.query_vec,
            memory_type=MemoryType.milestone, status=MemoryStatus.active,
            limit=10, threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_memories(
            ctx.pool, memory_type=MemoryType.milestone, status=MemoryStatus.active,
            limit=10, **ctx.scope,
        )
    return SectionFetch(payload=unwrap_recall(raw))


def pack_recent_work_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack recent milestones against the budget (sequential, mutates ctx)."""
    pairs = fetched.payload or []
    cutoff = ctx.now.timestamp() - (_RECENT_HOURS * 3600)
    candidates = [
        (m, sim) for m, sim in pairs
        if m.id not in ctx.seen_ids
        and m.created_at.timestamp() > cutoff
        and not is_unscoped_ingest(m, ctx.project_id)
    ]

    # Ranking
    effective_sim_weight = SIMILARITY_WEIGHT * (1.0 - ctx.recency_bias)

    if ctx.biased:
        _ts_range = (
            max(m.created_at.timestamp() for m, _ in candidates)
            - min(m.created_at.timestamp() for m, _ in candidates)
        ) if len(candidates) > 1 else 1.0
        candidates.sort(
            key=lambda pair: (
                effective_sim_weight * (pair[1] or 0)
                + (1 - effective_sim_weight) * (
                    (pair[0].created_at.timestamp() - cutoff) / max(_ts_range, 1.0)
                )
            ),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda pair: pair[0].created_at, reverse=True)

    items: list[dict] = []
    section_used = 0
    for mem, _sim in candidates:
        if len(items) >= _MAX:
            ctx.excluded += 1
            continue
        cost = mem.token_count or estimate_tokens(mem.content)
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            age_hours = (ctx.now - mem.created_at).total_seconds() / 3600
            entry = {
                "summary": mem.content,
                "age_hours": round(age_hours, 1),
                "refs": mem.topic,
                "id": mem.id,
            }
            items.append(entry)
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["recent_work"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_recent_work_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack recent work (single-shot wrapper)."""
    return pack_recent_work_section(ctx, await fetch_recent_work_section(ctx))
