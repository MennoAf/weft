"""Tests for the relevance scoring engine — pure functions, no DB needed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import Memory, MemoryRecall, MemorySource, MemoryStatus, MemoryType
from weft.relevance import (
    RelevanceWeights,
    confidence_factor,
    frequency_factor,
    rank_memories,
    recency_factor,
    score_memory,
    usefulness_factor,
)

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)


def _make_memory(
    *,
    confidence: float = 0.9,
    accessed_at: datetime | None = None,
    access_count: int = 0,
    content: str = "test memory",
    topic: list[str] | None = None,
) -> Memory:
    return Memory(
        id="weft-test0001",
        type=MemoryType.fact,
        topic=topic or ["test"],
        content=content,
        source=MemorySource.conversation,
        confidence=confidence,
        accessed_at=accessed_at or NOW,
        access_count=access_count,
        status=MemoryStatus.active,
    )


def _make_recall(
    similarity: float = 0.8, **kwargs
) -> MemoryRecall:
    return MemoryRecall(memory=_make_memory(**kwargs), similarity=similarity)


# --- confidence_factor ---


def test_confidence_factor_clamps_to_range():
    assert confidence_factor(0.0) == 0.0
    assert confidence_factor(1.0) == 1.0
    assert confidence_factor(0.5) == 0.5
    assert confidence_factor(-0.1) == 0.0
    assert confidence_factor(1.5) == 1.0


# --- recency_factor ---


def test_recency_just_accessed():
    """Just-accessed memory should have recency ~1.0."""
    result = recency_factor(NOW, now=NOW)
    assert result == pytest.approx(1.0)


def test_recency_one_half_life():
    """After one half-life, recency should be ~0.5."""
    accessed = NOW - timedelta(days=30)
    result = recency_factor(accessed, now=NOW, half_life_days=30)
    assert result == pytest.approx(0.5, abs=0.01)


def test_recency_two_half_lives():
    """After two half-lives, recency should be ~0.25."""
    accessed = NOW - timedelta(days=60)
    result = recency_factor(accessed, now=NOW, half_life_days=30)
    assert result == pytest.approx(0.25, abs=0.01)


def test_recency_zero_half_life_returns_one():
    """Zero half-life means no decay."""
    accessed = NOW - timedelta(days=365)
    result = recency_factor(accessed, now=NOW, half_life_days=0)
    assert result == 1.0


def test_recency_naive_datetime():
    """Naive datetimes should be treated as UTC."""
    naive = NOW.replace(tzinfo=None)
    result = recency_factor(naive, now=NOW)
    assert result == pytest.approx(1.0)


# --- frequency_factor ---


def test_frequency_zero_accesses():
    """Never-accessed memory has factor 1.0 (no boost)."""
    assert frequency_factor(0) == 1.0


def test_frequency_some_accesses():
    """Some accesses should give a boost between 1.0 and 1.0+boost_max."""
    result = frequency_factor(5, boost_max=0.2, boost_scale=10)
    assert 1.0 < result < 1.2


def test_frequency_many_accesses_capped():
    """Many accesses should cap at 1.0+boost_max."""
    result = frequency_factor(1000, boost_max=0.2, boost_scale=10)
    assert result == pytest.approx(1.2, abs=0.01)


def test_frequency_negative_treated_as_zero():
    assert frequency_factor(-1) == 1.0


# --- score_memory ---


def test_score_basic():
    """Score should be similarity * confidence * recency * frequency."""
    recall = _make_recall(
        similarity=0.8,
        confidence=0.9,
        accessed_at=NOW,
        access_count=0,
    )
    scored = score_memory(recall, now=NOW)

    assert scored.similarity == 0.8
    assert scored.confidence_factor == 0.9
    assert scored.recency_factor == pytest.approx(1.0)
    assert scored.frequency_factor == 1.0
    assert scored.score == pytest.approx(0.8 * 0.9 * 1.0 * 1.0, abs=0.001)


def test_score_with_old_memory():
    """Old memories should have lower scores."""
    recent = _make_recall(similarity=0.8, confidence=0.9, accessed_at=NOW)
    old = _make_recall(
        similarity=0.8,
        confidence=0.9,
        accessed_at=NOW - timedelta(days=60),
    )

    recent_scored = score_memory(recent, now=NOW)
    old_scored = score_memory(old, now=NOW)

    assert recent_scored.score > old_scored.score


def test_score_frequency_boost():
    """Frequently accessed memories should score higher."""
    no_access = _make_recall(similarity=0.8, confidence=0.9, access_count=0)
    frequent = _make_recall(similarity=0.8, confidence=0.9, access_count=20)

    no_score = score_memory(no_access, now=NOW)
    freq_score = score_memory(frequent, now=NOW)

    assert freq_score.score > no_score.score


def test_score_deterministic():
    """Same inputs should always produce same outputs."""
    recall = _make_recall(similarity=0.75, confidence=0.85, access_count=3)
    s1 = score_memory(recall, now=NOW)
    s2 = score_memory(recall, now=NOW)
    assert s1.score == s2.score


# --- rank_memories ---


def test_rank_sorts_descending():
    """Rank should sort by score descending."""
    recalls = [
        _make_recall(similarity=0.5, confidence=0.5),
        _make_recall(similarity=0.9, confidence=0.9),
        _make_recall(similarity=0.7, confidence=0.7),
    ]
    ranked = rank_memories(recalls, now=NOW)
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_list():
    assert rank_memories([], now=NOW) == []


def test_rank_custom_weights():
    """Custom weights should affect scoring."""
    recalls = [
        _make_recall(
            similarity=0.8,
            confidence=0.9,
            accessed_at=NOW - timedelta(days=90),
            access_count=50,
        ),
    ]
    default_ranked = rank_memories(recalls, now=NOW)
    fast_decay = rank_memories(
        recalls,
        weights=RelevanceWeights(recency_half_life_days=7),
        now=NOW,
    )
    # Fast decay should penalize the old memory more
    assert fast_decay[0].score < default_ranked[0].score


def test_scored_memory_to_dict():
    recall = _make_recall(similarity=0.85, confidence=0.9)
    scored = score_memory(recall, now=NOW)
    d = scored.to_dict()
    assert d["similarity"] == 0.85
    assert "relevance_score" in d
    assert d["type"] == "fact"
