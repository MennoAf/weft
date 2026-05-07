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
    usefulness_score: float = 1.0,
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
        usefulness_score=usefulness_score,
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


def test_score_custom_usefulness_floor():
    """Custom usefulness_floor in RelevanceWeights should affect scoring."""
    recall = _make_recall(similarity=0.8, confidence=0.9, usefulness_score=0.0)

    # Default floor=0.5: usefulness_factor(0.0) = 0.5
    default_scored = score_memory(recall, now=NOW)
    assert default_scored.usefulness_factor == pytest.approx(0.5)

    # Floor=0.0: usefulness_factor(0.0) = 0.0
    zero_floor = score_memory(
        recall,
        weights=RelevanceWeights(usefulness_floor=0.0),
        now=NOW,
    )
    assert zero_floor.usefulness_factor == pytest.approx(0.0)
    assert zero_floor.score < default_scored.score

    # Floor=0.8: usefulness_factor(0.0) = 0.8
    high_floor = score_memory(
        recall,
        weights=RelevanceWeights(usefulness_floor=0.8),
        now=NOW,
    )
    assert high_floor.usefulness_factor == pytest.approx(0.8)
    assert high_floor.score > default_scored.score


def test_scored_memory_to_dict():
    recall = _make_recall(similarity=0.85, confidence=0.9)
    scored = score_memory(recall, now=NOW)
    d = scored.to_dict()
    assert d["similarity"] == 0.85
    assert "relevance_score" in d
    assert d["type"] == "fact"
    # Factor breakdowns exposed
    assert "factors" in d
    assert set(d["factors"].keys()) == {"confidence", "recency", "frequency", "usefulness", "type_boost"}
    assert d["factors"]["confidence"] == pytest.approx(0.9)
    assert d["factors"]["usefulness"] == pytest.approx(1.0)  # explicit 1.0 in helper


# --- type_boost_factor ---


def test_anti_pattern_gets_default_boost():
    """Anti-pattern memories get a 1.3x boost by default."""
    from weft.relevance import type_boost_factor

    assert type_boost_factor(MemoryType.anti_pattern) == pytest.approx(1.3)
    assert type_boost_factor(MemoryType.fact) == pytest.approx(1.0)
    assert type_boost_factor(MemoryType.decision) == pytest.approx(1.0)


def test_type_boost_disabled_with_empty_dict():
    """Passing empty boosts dict disables all boosts."""
    from weft.relevance import type_boost_factor

    assert type_boost_factor(MemoryType.anti_pattern, boosts={}) == pytest.approx(1.0)


def test_anti_pattern_ranks_higher_than_equal_fact():
    """An anti-pattern with identical signals should outscore a fact."""
    anti = MemoryRecall(
        memory=Memory(
            type=MemoryType.anti_pattern,
            content="Don't do X",
            confidence=0.9,
            accessed_at=NOW,
            usefulness_score=1.0,
            status=MemoryStatus.active,
        ),
        similarity=0.8,
    )
    fact = _make_recall(similarity=0.8, confidence=0.9)

    anti_scored = score_memory(anti, now=NOW)
    fact_scored = score_memory(fact, now=NOW)

    assert anti_scored.type_boost_factor == pytest.approx(1.3)
    assert fact_scored.type_boost_factor == pytest.approx(1.0)
    assert anti_scored.score > fact_scored.score


def test_type_boost_disabled_via_weights():
    """When type_boosts={} in weights, anti-patterns get no boost."""
    anti = MemoryRecall(
        memory=Memory(
            type=MemoryType.anti_pattern,
            content="Don't do X",
            confidence=0.9,
            accessed_at=NOW,
            usefulness_score=1.0,
            status=MemoryStatus.active,
        ),
        similarity=0.8,
    )
    weights = RelevanceWeights(type_boosts={})
    scored = score_memory(anti, weights=weights, now=NOW)

    assert scored.type_boost_factor == pytest.approx(1.0)


# --- score_turn (P1.A3) ---


from weft.models import EpisodeTurn, TurnRole  # noqa: E402
from weft.relevance import ScoredTurn, rank_turns, score_turn  # noqa: E402


def _make_turn(
    *,
    occurred_at: datetime | None = None,
    usefulness_score: float = 0.7,
    last_boosted_at: datetime | None = None,
    content: str = "test turn",
) -> EpisodeTurn:
    return EpisodeTurn(
        id="et-test00001",
        episode_id="ep-test",
        turn_index=0,
        role=TurnRole.user,
        content=content,
        occurred_at=occurred_at or NOW,
        usefulness_score=usefulness_score,
        last_boosted_at=last_boosted_at,
    )


def test_score_turn_higher_usefulness_wins():
    """All else equal, higher usefulness_score should produce a higher
    composite score."""
    low = _make_turn(usefulness_score=0.3, occurred_at=NOW)
    high = _make_turn(usefulness_score=1.0, occurred_at=NOW)

    low_scored = score_turn(low, base_score=0.5, now=NOW)
    high_scored = score_turn(high, base_score=0.5, now=NOW)

    assert high_scored.score > low_scored.score
    assert high_scored.usefulness_factor > low_scored.usefulness_factor


def test_score_turn_decays_older_turns():
    """All else equal, an older turn should have a lower recency factor
    and therefore a lower composite score."""
    recent = _make_turn(occurred_at=NOW, usefulness_score=0.7)
    old = _make_turn(occurred_at=NOW - timedelta(days=60), usefulness_score=0.7)

    recent_scored = score_turn(recent, base_score=0.5, now=NOW)
    old_scored = score_turn(old, base_score=0.5, now=NOW)

    assert recent_scored.recency_factor > old_scored.recency_factor
    assert recent_scored.score > old_scored.score


def test_score_turn_does_not_use_confidence():
    """``score_turn`` must not depend on a confidence value — turns are
    observed dialogue, not asserted beliefs. Verify by checking that
    ``ScoredTurn`` exposes no ``confidence_factor`` attribute and that
    EpisodeTurn itself has no ``confidence`` field on the model."""
    turn = _make_turn()
    scored = score_turn(turn, base_score=0.5, now=NOW)

    assert isinstance(scored, ScoredTurn)
    assert not hasattr(scored, "confidence_factor")
    assert not hasattr(scored, "type_boost_factor")
    # EpisodeTurn itself doesn't have a confidence column.
    assert not hasattr(turn, "confidence")


def test_score_turn_formula_is_base_times_recency_times_usefulness():
    """Score breakdown should match the literal multiplicative formula."""
    turn = _make_turn(occurred_at=NOW, usefulness_score=0.8)
    scored = score_turn(turn, base_score=0.4, now=NOW)

    expected = 0.4 * scored.recency_factor * scored.usefulness_factor
    assert scored.score == pytest.approx(expected)


def test_score_turn_last_boosted_at_decays_usefulness():
    """When ``last_boosted_at`` is recent, usefulness should be near full
    strength; when stale, it should decay toward the floor."""
    recent_boost = _make_turn(
        usefulness_score=1.0,
        last_boosted_at=NOW,
        occurred_at=NOW,
    )
    stale_boost = _make_turn(
        usefulness_score=1.0,
        last_boosted_at=NOW - timedelta(days=120),
        occurred_at=NOW,
    )

    fresh_scored = score_turn(recent_boost, base_score=0.5, now=NOW)
    stale_scored = score_turn(stale_boost, base_score=0.5, now=NOW)

    assert fresh_scored.usefulness_factor > stale_scored.usefulness_factor


def test_rank_turns_sorts_descending():
    pairs = [
        (_make_turn(usefulness_score=0.3, content="low"), 0.5),
        (_make_turn(usefulness_score=1.0, content="high"), 0.5),
        (_make_turn(usefulness_score=0.6, content="mid"), 0.5),
    ]
    ranked = rank_turns(pairs, now=NOW)
    contents = [s.turn.content for s in ranked]
    assert contents == ["high", "mid", "low"]


def test_rank_turns_empty_list():
    assert rank_turns([], now=NOW) == []
