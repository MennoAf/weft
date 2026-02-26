"""Relevance scoring engine — pure functions, no DB access.

Combines multiple signals into a unified relevance score:
  final_score = similarity * confidence_factor * recency_factor * frequency_factor * usefulness_factor

All functions operate on Memory model fields + similarity from SearchResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from weft.models import Memory, MemoryRecall


@dataclass(frozen=True)
class RelevanceWeights:
    """Configurable weights for the scoring formula."""

    recency_half_life_days: float = 30.0
    frequency_boost_max: float = 0.2
    frequency_boost_scale: int = 10


@dataclass(frozen=True)
class ScoredMemory:
    """A memory with its composite relevance score and breakdown."""

    memory: Memory
    similarity: float
    confidence_factor: float
    recency_factor: float
    frequency_factor: float
    usefulness_factor: float
    score: float

    def to_dict(self) -> dict:
        d = self.memory.to_dict()
        d["similarity"] = round(self.similarity, 4)
        d["relevance_score"] = round(self.score, 4)
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


def usefulness_factor(usefulness_score: float) -> float:
    """Usefulness feedback factor. Range [0.5, 1.0].

    Maps usefulness_score (0.0-1.0) to a factor that modestly penalizes
    low-usefulness memories without being too aggressive.
    """
    return 0.5 + 0.5 * max(0.0, min(1.0, usefulness_score))


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
    uf = usefulness_factor(mem.usefulness_score)

    final = recall.similarity * cf * rf * ff * uf

    return ScoredMemory(
        memory=mem,
        similarity=recall.similarity,
        confidence_factor=cf,
        recency_factor=rf,
        frequency_factor=ff,
        usefulness_factor=uf,
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
