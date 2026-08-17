"""Relevance scoring engine — pure functions, no DB access.

Combines multiple signals into a unified relevance score:
  final_score = similarity * confidence * recency * frequency * usefulness * type_boost

All functions operate on Memory model fields + similarity from SearchResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from weft.models import EpisodeTurn, Memory, MemoryRecall, MemoryType


# Default per-type score multipliers.  Types not listed get 1.0.
_DEFAULT_TYPE_BOOSTS: dict[MemoryType, float] = {
    MemoryType.anti_pattern: 1.3,
}


@dataclass(frozen=True)
class RelevanceWeights:
    """Configurable weights for the scoring formula."""

    recency_half_life_days: float = 30.0
    frequency_boost_max: float = 0.2
    frequency_boost_scale: int = 10
    usefulness_floor: float = 0.5
    type_boosts: dict[MemoryType, float] | None = None  # None → use defaults


@dataclass(frozen=True)
class ScoredMemory:
    """A memory with its composite relevance score and breakdown."""

    memory: Memory
    similarity: float
    confidence_factor: float
    recency_factor: float
    frequency_factor: float
    usefulness_factor: float
    type_boost_factor: float
    score: float

    def to_dict(self) -> dict:
        d = self.memory.to_dict()
        d["similarity"] = round(self.similarity, 4)
        d["relevance_score"] = round(self.score, 4)
        d["factors"] = {
            "confidence": round(self.confidence_factor, 4),
            "recency": round(self.recency_factor, 4),
            "frequency": round(self.frequency_factor, 4),
            "usefulness": round(self.usefulness_factor, 4),
            "type_boost": round(self.type_boost_factor, 4),
        }
        return d


def confidence_factor(confidence: float) -> float:
    """Confidence contributes directly as a multiplicative factor.

    Range: [0.0, 1.0]. Higher confidence = higher score.
    """
    return max(0.0, min(1.0, confidence))


def recency_factor(
    accessed_at: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """Exponential decay based on time since last access.

    Returns 1.0 for just-accessed memories, decaying toward 0.
    Half-life controls how fast memories lose recency relevance.
    """
    now = now or datetime.now(timezone.utc)
    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - accessed_at).total_seconds() / 86400)
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, delta_days / half_life_days)


def frequency_factor(
    access_count: int,
    *,
    boost_max: float = 0.2,
    boost_scale: int = 10,
) -> float:
    """Logarithmic frequency bonus. More accesses = slightly higher score.

    Returns 1.0 for never-accessed, up to (1.0 + boost_max) for frequently accessed.
    Uses log scaling so the first few accesses matter most.
    """
    if access_count <= 0:
        return 1.0
    # log1p(count) / log1p(scale) gives a 0-1 range, capped at 1.0
    raw = min(1.0, math.log1p(access_count) / math.log1p(boost_scale))
    return 1.0 + (boost_max * raw)


def usefulness_factor(
    usefulness_score: float,
    *,
    floor: float = 0.5,
    last_boosted_at: datetime | None = None,
    now: datetime | None = None,
    decay_half_life_days: float = 30.0,
) -> float:
    """Usefulness feedback factor with time decay. Range [floor, 1.0].

    Maps usefulness_score (0.0-1.0) to a factor that penalizes
    low-usefulness memories. When last_boosted_at is provided, applies
    exponential decay (30-day half-life, matching consolidation decay)
    so stale boosts don't permanently dominate rankings.

    The floor controls the minimum factor (default 0.5 = worst memories
    get halved, not zeroed).
    """
    clamped = max(0.0, min(1.0, usefulness_score))

    # Apply time decay if we know when the memory was last boosted
    if last_boosted_at is not None and decay_half_life_days > 0:
        now = now or datetime.now(timezone.utc)
        if last_boosted_at.tzinfo is None:
            last_boosted_at = last_boosted_at.replace(tzinfo=timezone.utc)
        days_since = max(0.0, (now - last_boosted_at).total_seconds() / 86400)
        decay = math.pow(0.5, days_since / decay_half_life_days)
        clamped = clamped * decay

    return floor + (1.0 - floor) * clamped


def type_boost_factor(
    memory_type: MemoryType,
    boosts: dict[MemoryType, float] | None = None,
) -> float:
    """Per-type score multiplier.  Returns 1.0 for unlisted types."""
    mapping = _DEFAULT_TYPE_BOOSTS if boosts is None else boosts
    return mapping.get(memory_type, 1.0)


def score_memory(
    recall: MemoryRecall,
    *,
    weights: RelevanceWeights | None = None,
    now: datetime | None = None,
) -> ScoredMemory:
    """Compute the composite relevance score for a single recall result."""
    w = weights or RelevanceWeights()
    mem = recall.memory

    cf = confidence_factor(mem.confidence)
    rf = recency_factor(
        mem.accessed_at,
        now=now,
        half_life_days=w.recency_half_life_days,
    )
    ff = frequency_factor(
        mem.access_count,
        boost_max=w.frequency_boost_max,
        boost_scale=w.frequency_boost_scale,
    )
    uf = usefulness_factor(
        mem.usefulness_score,
        floor=w.usefulness_floor,
        last_boosted_at=getattr(mem, "last_boosted_at", None),
        now=now,
    )
    tb = type_boost_factor(mem.type, boosts=w.type_boosts)

    final = recall.similarity * cf * rf * ff * uf * tb

    return ScoredMemory(
        memory=mem,
        similarity=recall.similarity,
        confidence_factor=cf,
        recency_factor=rf,
        frequency_factor=ff,
        usefulness_factor=uf,
        type_boost_factor=tb,
        score=final,
    )


def rank_memories(
    recalls: list[MemoryRecall],
    *,
    weights: RelevanceWeights | None = None,
    now: datetime | None = None,
) -> list[ScoredMemory]:
    """Score and rank a list of recall results by composite relevance.

    Returns results sorted by score descending.
    """
    scored = [score_memory(r, weights=weights, now=now) for r in recalls]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# --- Turn-tier scoring (P1.A3) ---


@dataclass(frozen=True)
class ScoredTurn:
    """An episode turn with its composite relevance score and breakdown.

    Mirrors :class:`ScoredMemory` shape but drops the belief-tier-only
    factors (``confidence`` and ``type_boost``):

    * Turns are observed dialogue, not asserted beliefs — there is no
      confidence value to multiply through.
    * Turns have no ``type`` column, so per-type boosts don't apply.

    The base score is the RRF composite from
    :func:`weft.episode_turns._rrf_fuse_turn_rows` (vector + ts_rank
    fused), not a raw cosine — that's the rank-space the upstream layer
    produces.
    """

    turn: EpisodeTurn
    base_score: float
    recency_factor: float
    usefulness_factor: float
    score: float


def score_turn(
    turn: EpisodeTurn,
    base_score: float,
    *,
    weights: RelevanceWeights | None = None,
    now: datetime | None = None,
) -> ScoredTurn:
    """Compute the composite relevance score for a single turn.

    Formula:

        score = base_score * recency_factor * usefulness_factor

    where ``base_score`` is the upstream RRF composite (vector + ts_rank),
    ``recency_factor`` decays by ``occurred_at`` (the dialogue's own
    timestamp — turns don't have an ``accessed_at`` column; ``occurred_at``
    is the load-bearing temporal signal), and ``usefulness_factor`` reads
    the boost-loop columns (``usefulness_score`` + ``last_boosted_at``)
    using the same time-decayed mapping as :func:`score_memory`.

    Deliberately omits:

    * ``confidence_factor`` — turns are dialogue, not asserted beliefs.
    * ``frequency_factor`` — turns don't carry an ``access_count``;
      access tracking lives in ``turn_access_log``, and the boost loop
      already collapses that into ``usefulness_score``.
    * ``type_boost_factor`` — turns have no type column.
    """
    w = weights or RelevanceWeights()

    rf = recency_factor(
        turn.occurred_at,
        now=now,
        half_life_days=w.recency_half_life_days,
    )
    uf = usefulness_factor(
        turn.usefulness_score,
        floor=w.usefulness_floor,
        last_boosted_at=turn.last_boosted_at,
        now=now,
    )

    final = base_score * rf * uf

    return ScoredTurn(
        turn=turn,
        base_score=base_score,
        recency_factor=rf,
        usefulness_factor=uf,
        score=final,
    )


def rank_turns(
    turns_with_base: list[tuple[EpisodeTurn, float]],
    *,
    weights: RelevanceWeights | None = None,
    now: datetime | None = None,
) -> list[ScoredTurn]:
    """Score and rank a list of (turn, base_score) pairs by composite
    relevance. Returns results sorted by score descending.

    The pair shape keeps the RRF score out of the EpisodeTurn model itself
    — the model is the persistence shape, the score is a per-recall
    artifact that doesn't belong on the row.
    """
    scored = [
        score_turn(t, base, weights=weights, now=now)
        for t, base in turns_with_base
    ]
    # Secondary sort by turn ID ensures deterministic order when composite
    # scores tie — critical for reproducible A/B comparisons.
    scored.sort(key=lambda s: (-s.score, s.turn.id))
    return scored
