"""Recent memories section — top-K freshly-touched memories in tier-1.

Sits next to handoff in tier-1 and answers a different question:
handoff is *the last session's narrative*; recent_memories is *what
else is alive in this user's memory right now*. The two together give
a new session enough state to land on its feet without firing
``weft_focus`` first.

Excludes types already covered by other tier-1 sections (handoff,
milestone via recent_work) and ingest noise. Ranks by a weighted
blend of recency, project_id match, and pinned status. When the
caller passes a query vector, similarity is folded in via the
mode-aware ``recency_bias`` weight, identical to recent_work's
behaviour.

Dedup against handoff (and any other tier-1 memory packed before
this one) is automatic via ``ctx.seen_ids`` — the orchestrator runs
sections sequentially in priority order and this section runs after
handoff.
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
    SectionFetch,
    SectionResult,
    annotate_review_after,
    is_unscoped_ingest,
    unwrap_recall,
)
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["recent_memories"]
_MAX = SECTION_MAX_ITEMS["recent_memories"]

# How far back the recency window stretches. 7 days covers a typical
# multi-session week without flooding the section with stale rows;
# pinned items always rise above this regardless of age.
_RECENT_HOURS = 168

# Types another section already owns. Excluded here so we don't claim
# their ids in ``ctx.seen_ids`` and starve those sections of their
# canonical content. Recent_memories is the *cross-cutting recency
# lens* — its job is to surface the leaf types (fact, preference,
# pattern, solution, architecture, user_model, relationship) that no
# other section owns. Pinned memories of any type stay in scope
# because the rules section is budget-tight and will overflow.
_EXCLUDED_TYPES: frozenset[MemoryType] = frozenset({
    MemoryType.handoff,       # handoff section
    MemoryType.milestone,     # recent_work section
    MemoryType.decision,      # decisions section
    MemoryType.issue,         # issues section
    MemoryType.anti_pattern,  # anti_patterns section
})

# Ranking weights for the unbiased path. Sum to 1.0 — the blended
# score in [0, 1] is what we sort by.
_W_RECENCY = 0.5
_W_PROJECT_MATCH = 0.3
_W_PINNED = 0.2


def _base_score(
    mem,
    *,
    project_id: str | None,
    cutoff_ts: float,
    window_seconds: float,
) -> float:
    """Recency + project-match + pinned blend in [0, 1].

    Recency is normalized inside the look-back window so a memory at
    the edge scores ~0 and one created right now scores ~1. Pinned
    memories get an additional flat boost — they're the user's
    explicit "always relevant" signal.
    """
    recency = max(
        0.0, (mem.updated_at.timestamp() - cutoff_ts) / window_seconds,
    )
    project_match = (
        1.0 if project_id and mem.project_id == project_id else 0.0
    )
    pinned = 1.0 if mem.pinned else 0.0
    return (
        _W_RECENCY * recency
        + _W_PROJECT_MATCH * project_match
        + _W_PINNED * pinned
    )


async def fetch_recent_memories_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch a wide pool of recent memories (parallel-safe, no ctx mutation).

    Pulls more than ``_MAX`` deliberately — post-fetch filtering drops
    excluded types, ingest noise, and seen ids, so we want headroom.
    """
    if ctx.biased:
        raw = await search_by_vector(
            ctx.pool, ctx.query_vec,
            status=MemoryStatus.active,
            limit=40, threshold=QUERY_SIMILARITY_THRESHOLD, **ctx.scope,
        )
    else:
        raw = await list_memories(
            ctx.pool, status=MemoryStatus.active,
            limit=40, **ctx.scope,
        )
    return SectionFetch(payload=unwrap_recall(raw))


def pack_recent_memories_section(
    ctx: PrimerContext, fetched: SectionFetch,
) -> SectionResult:
    """Filter, rank, and pack into the section budget (mutates ctx)."""
    pairs = fetched.payload or []
    cutoff_ts = ctx.now.timestamp() - (_RECENT_HOURS * 3600)
    window_seconds = float(_RECENT_HOURS * 3600)

    candidates: list[tuple[object, float | None]] = []
    for mem, sim in pairs:
        if mem.id in ctx.seen_ids:
            continue
        if mem.type in _EXCLUDED_TYPES:
            continue
        if is_unscoped_ingest(mem, ctx.project_id):
            continue
        # Pinned memories bypass the recency cutoff — they're durable
        # by definition and the user wants them surfaced.
        if not mem.pinned and mem.updated_at.timestamp() < cutoff_ts:
            continue
        candidates.append((mem, sim))

    if not candidates:
        ctx.section_tokens["recent_memories"] = 0
        return SectionResult(items=[], tokens_used=0, skipped=False)

    effective_sim_weight = SIMILARITY_WEIGHT * (1.0 - ctx.recency_bias)

    def _rank(pair) -> float:
        mem, sim = pair
        base = _base_score(
            mem,
            project_id=ctx.project_id,
            cutoff_ts=cutoff_ts,
            window_seconds=window_seconds,
        )
        if ctx.biased and sim is not None:
            return effective_sim_weight * sim + (1 - effective_sim_weight) * base
        return base

    candidates.sort(key=_rank, reverse=True)

    items: list[dict] = []
    section_used = 0
    for mem, _sim in candidates:
        if len(items) >= _MAX:
            ctx.excluded += 1
            continue
        cost = (mem.token_count or estimate_tokens(mem.content)) + DICT_OVERHEAD_TOKENS
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            age_hours = (ctx.now - mem.updated_at).total_seconds() / 3600
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": mem.content,
                "topic": mem.topic,
                "confidence": mem.confidence,
                "pinned": mem.pinned,
                "age_hours": round(age_hours, 1),
                "project_id": mem.project_id,
                "review_after": (
                    mem.review_after.isoformat() if mem.review_after else None
                ),
            }
            entry = annotate_review_after(entry, ctx.now)
            items.append(entry)
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["recent_memories"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_recent_memories_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack recent memories (single-shot wrapper)."""
    return pack_recent_memories_section(
        ctx, await fetch_recent_memories_section(ctx),
    )
