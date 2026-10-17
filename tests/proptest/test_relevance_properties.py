"""Property-based tests for the relevance scoring engine — ``weft/relevance.py``.

The module is pure (imports: ``math``, ``dataclasses``, ``datetime``,
``weft.models`` — no DB code), so all properties run without fixtures or
containers.

Properties covered:

1.  Closed-form factor checks — recency at half-life == 0.5, recency(0) == 1,
    recency monotonically non-increasing in age, every factor bounded in its
    enforced range.
2.  Frequency guards — access_count <= 0 returns exactly 1.0, never
    NaN/inf; factor bounded by [1, 1 + boost_max]; cap reached at the scale.
3.  Usefulness floor/decay — output within [floor, 1] across arbitrary
    inputs, floor respected everywhere.
4.  Composite monotonicity — perturbing one factor input upward (confidence,
    access_count, usefulness_score, accessed_at, last_boosted_at,
    similarity) never decreases the composite score.
5.  NaN poisoning — composite is a finite float for all valid inputs.
    (NaN/inf similarity is rejected by the pydantic model at construction,
    and the confidence/usefulness factor functions floor non-finite input;
    see the guarded-semantics tests.)
6.  rank_memories is an honest descending sort — equal to a stable
    sorted(..., reverse=True) computed in the test; score sequence is
    permutation invariant; ties preserve input order (documented actual
    behavior: Python's stable Timsort, no tiebreaker — unlike rank_turns,
    which tiebreaks on turn id).
7.  score_turn omits exactly {confidence, frequency, type_boost} — pinned
    structurally (ScoredTurn dataclass fields) and algebraically
    (score == base_score * recency * usefulness).
8.  Boundary sweep — age 0 / half-life / 100x half-life, access_count 0 and
    huge, confidence 0 and 1: no crashes, no sign flips, in-bounds results.
9.  Determinism — same inputs produce bit-identical scores across calls.

Known behavior difference:
  * The stated type boost range [1, 1.2] does not match the actual default
    mapping, which boosts anti_pattern by 1.3 and everything else by 1.0.
    The code is authoritative; the tests pin {1.0, 1.3}.
"""

from __future__ import annotations

import math
from dataclasses import fields as dc_fields
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from weft.models import EpisodeTurn, Memory, MemoryRecall, MemoryType, TurnRole
from weft.relevance import (
    RelevanceWeights,
    ScoredMemory,
    ScoredTurn,
    confidence_factor,
    frequency_factor,
    rank_memories,
    rank_turns,
    recency_factor,
    score_memory,
    score_turn,
    type_boost_factor,
    usefulness_factor,
)

# Module-level profile per harness spec.
settings.register_profile("proptest", max_examples=200, deadline=None)
settings.load_profile("proptest")

# Fixed clock so every property is deterministic and date-independent.
NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
DEFAULT_HALF_LIFE_DAYS = 30.0
# Slack for float-ordering assertions (monotonicity across rounded products).
EPS = 1e-12
# Composite upper bound with default weights: similarity * cf(<=1) * rf(<=1)
# * ff(<=1.2) * uf(<=1) * tb(<=1.3).
COMPOSITE_MAX_PRODUCT = 1.2 * 1.3

ALL_MEMORY_TYPES = list(MemoryType)

# --- strategies ---------------------------------------------------------------

any_finite = st.floats(allow_nan=False, allow_infinity=False)
unit_float = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
half_life = st.floats(0.01, 3650.0, allow_nan=False, allow_infinity=False)
positive_half_life = st.floats(1e-3, 36500.0, allow_nan=False, allow_infinity=False)
age_delta = st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=100_000))
access_counts = st.integers(min_value=0, max_value=10**18)

weight_sets = st.builds(
    RelevanceWeights,
    recency_half_life_days=half_life,
    frequency_boost_max=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
    frequency_boost_scale=st.integers(min_value=1, max_value=1000),
    usefulness_floor=unit_float,
    type_boosts=st.none(),
)

memories = st.builds(
    Memory,
    type=st.sampled_from(ALL_MEMORY_TYPES),
    content=st.just("property-test memory"),
    confidence=unit_float,
    accessed_at=st.builds(lambda d: NOW - d, age_delta),
    access_count=access_counts,
    usefulness_score=unit_float,
    last_boosted_at=st.none() | st.builds(lambda d: NOW - d, age_delta),
)

recalls = st.builds(MemoryRecall, memory=memories, similarity=unit_float)

turns = st.builds(
    EpisodeTurn,
    id=st.just("turn-prop-0001"),
    episode_id=st.just("episode-prop"),
    turn_index=st.integers(0, 1000),
    role=st.sampled_from(list(TurnRole)),
    content=st.just("property-test turn"),
    occurred_at=st.builds(lambda d: NOW - d, age_delta),
    usefulness_score=unit_float,
    last_boosted_at=st.none() | st.builds(lambda d: NOW - d, age_delta),
)


def _rec(confidence: float, accessed: datetime, count: int, use: float,
         boost: datetime | None, sim: float, mem_id: str = "weft-prop0001") -> MemoryRecall:
    return MemoryRecall(
        memory=Memory(
            id=mem_id,
            type=MemoryType.fact,
            content="property-test memory",
            confidence=confidence,
            accessed_at=accessed,
            access_count=count,
            usefulness_score=use,
            last_boosted_at=boost,
        ),
        similarity=sim,
    )


def _sorted_ages(count: int) -> st.SearchStrategy[list[timedelta]]:
    """Draw `count` non-negative timedeltas, returned in ascending order."""
    return st.lists(age_delta, min_size=count, max_size=count).map(sorted)


# --- 1. closed-form factor checks ---------------------------------------------


@pytest.mark.parametrize("hl", [30.0, 1.0, 365.0, 0.5])
def test_recency_at_half_life_is_one_half(hl: float) -> None:
    """recency(exactly one half-life) == 0.5 within 1e-9 (catches constant drift)."""
    accessed = NOW - timedelta(days=hl)
    assert math.isclose(recency_factor(accessed, now=NOW, half_life_days=hl), 0.5, rel_tol=1e-9)


@given(hl=positive_half_life)
def test_recency_at_generated_half_life_is_one_half(hl: float) -> None:
    accessed = NOW - timedelta(days=hl)
    assert math.isclose(recency_factor(accessed, now=NOW, half_life_days=hl), 0.5, rel_tol=1e-9)


def test_recency_default_half_life_constant_is_30_days() -> None:
    """Pins the function-signature default constant: with no half_life_days
    argument, age == 30 days must yield exactly 0.5. Catches drift of the
    literal `half_life_days: float = 30.0` default (the RelevanceWeights
    default constant is pinned separately below)."""
    assert math.isclose(recency_factor(NOW - timedelta(days=30), now=NOW), 0.5, rel_tol=1e-9)
    assert not math.isclose(recency_factor(NOW - timedelta(days=31), now=NOW), 0.5, rel_tol=1e-9)


@given(hl=positive_half_life)
def test_recency_zero_age_is_exactly_one(hl: float) -> None:
    assert recency_factor(NOW, now=NOW, half_life_days=hl) == 1.0


@given(d1=age_delta, d2=age_delta)
def test_recency_monotonically_non_increasing_in_age(d1: timedelta, d2: timedelta) -> None:
    older, newer = max(d1, d2), min(d1, d2)
    assert recency_factor(NOW - older, now=NOW) <= recency_factor(NOW - newer, now=NOW) + EPS


@given(ages=_sorted_ages(3))
def test_recency_monotone_over_sorted_ages(ages: list[timedelta]) -> None:
    values = [recency_factor(NOW - a, now=NOW) for a in ages]
    assert values[0] >= values[1] >= values[2] - EPS


@pytest.mark.parametrize("hl", [0.0, -1.0, -30.0])
def test_recency_nonpositive_half_life_guard_returns_one(hl: float) -> None:
    """Pinned guard: half_life <= 0 short-circuits to 1.0 (no decay)."""
    assert recency_factor(NOW - timedelta(days=365), now=NOW, half_life_days=hl) == 1.0


@given(delta=age_delta)
def test_recency_future_access_clamped_to_one(delta: timedelta) -> None:
    assert recency_factor(NOW + delta, now=NOW) == 1.0


@given(delta=age_delta)
def test_recency_bounded_in_unit_interval(delta: timedelta) -> None:
    rf = recency_factor(NOW - delta, now=NOW)
    assert 0.0 <= rf <= 1.0


@given(delta=age_delta)
def test_recency_naive_datetime_treated_as_utc(delta: timedelta) -> None:
    """Pinned behavior: naive datetimes are interpreted as UTC."""
    aware = recency_factor(NOW - delta, now=NOW)
    naive = recency_factor((NOW - delta).replace(tzinfo=None), now=NOW)
    assert naive == aware


@pytest.mark.parametrize("hl", [30.0, 7.0])
def test_recency_at_hundred_half_lives_asymptote(hl: float) -> None:
    rf = recency_factor(NOW - timedelta(days=100 * hl), now=NOW, half_life_days=hl)
    assert math.isclose(rf, 0.5**100, rel_tol=1e-9)
    assert 0.0 < rf <= 0.5


@given(c=any_finite)
def test_confidence_factor_always_bounded(c: float) -> None:
    cf = confidence_factor(c)
    assert 0.0 <= cf <= 1.0


@given(c=unit_float)
def test_confidence_factor_identity_in_domain(c: float) -> None:
    """Within [0,1] the clamp is the identity (exact)."""
    assert confidence_factor(c) == c


@given(c1=any_finite, c2=any_finite)
def test_confidence_factor_monotone(c1: float, c2: float) -> None:
    lo, hi = sorted([c1, c2])
    assert confidence_factor(lo) <= confidence_factor(hi) + EPS


@given(t=st.sampled_from(ALL_MEMORY_TYPES))
def test_type_boost_defaults_pinned(t: MemoryType) -> None:
    """The actual default range is {1.0, 1.3}; anti_pattern receives 1.3."""
    tb = type_boost_factor(t)
    if t is MemoryType.anti_pattern:
        assert tb == 1.3
    else:
        assert tb == 1.0
    assert 1.0 <= tb <= 1.3


def test_type_boost_unlisted_type_is_one() -> None:
    assert type_boost_factor(MemoryType.fact, boosts={MemoryType.pattern: 2.0}) == 1.0


def test_type_boost_custom_mapping_respected() -> None:
    boosts = {mt: 1.5 for mt in ALL_MEMORY_TYPES}
    assert all(type_boost_factor(mt, boosts=boosts) == 1.5 for mt in ALL_MEMORY_TYPES)


def test_composite_recency_half_life_via_default_weights() -> None:
    """RelevanceWeights' default 30-day half-life shows up in the composite:
    a memory accessed exactly 30 days ago has recency factor 0.5."""
    recall = _rec(confidence=1.0, accessed=NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS),
                  count=0, use=1.0, boost=None, sim=1.0)
    scored = score_memory(recall, now=NOW)
    assert math.isclose(scored.recency_factor, 0.5, rel_tol=1e-9)
    assert math.isclose(scored.score, 0.5, rel_tol=1e-9)  # all other factors are 1.0


# --- 2. frequency guards ------------------------------------------------------


@pytest.mark.parametrize("count", [0, -1, -10**9])
def test_frequency_nonpositive_count_is_exactly_one(count: int) -> None:
    """log1p domain guard: count <= 0 short-circuits to 1.0, never NaN/inf."""
    ff = frequency_factor(count)
    assert ff == 1.0
    assert math.isfinite(ff)


@given(count=st.integers(min_value=-10**18, max_value=10**18),
       boost_max=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False))
def test_frequency_never_nan_or_inf(count: int, boost_max: float) -> None:
    assert math.isfinite(frequency_factor(count, boost_max=boost_max))


@given(count=access_counts,
       boost_max=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
       scale=st.integers(min_value=1, max_value=1000))
def test_frequency_bounded(count: int, boost_max: float, scale: int) -> None:
    ff = frequency_factor(count, boost_max=boost_max, boost_scale=scale)
    assert 1.0 <= ff <= 1.0 + boost_max + EPS


@given(count1=access_counts, count2=access_counts,
       boost_max=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False))
def test_frequency_monotone_nondecreasing(count1: int, count2: int, boost_max: float) -> None:
    lo, hi = sorted([count1, count2])
    assert frequency_factor(lo, boost_max=boost_max) <= frequency_factor(hi, boost_max=boost_max) + EPS


@given(scale=st.integers(min_value=1, max_value=1000),
       boost_max=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False))
def test_frequency_cap_reached_at_scale_and_beyond(scale: int, boost_max: float) -> None:
    """raw = log1p(count)/log1p(scale) clamps to 1.0 once count >= scale."""
    assert frequency_factor(scale, boost_max=boost_max, boost_scale=scale) == 1.0 + boost_max
    assert frequency_factor(10**18, boost_max=boost_max, boost_scale=scale) == 1.0 + boost_max


@given(count=access_counts)
def test_frequency_default_cap_is_exactly_1_2(count: int) -> None:
    """Default boost_max=0.2: capped factor is exactly 1.2."""
    ff = frequency_factor(count)
    assert ff <= 1.2 + EPS
    if count >= 10:
        assert ff == 1.2


# --- 3. usefulness floor / decay ----------------------------------------------


@given(score=any_finite, floor=unit_float, boost_delta=st.none() | age_delta,
       decay_hl=half_life)
def test_usefulness_within_floor_and_one(
    score: float, floor: float, boost_delta: timedelta | None, decay_hl: float,
) -> None:
    """Floor respected across arbitrary inputs, including decay and
    out-of-domain scores (clamped first)."""
    boost = None if boost_delta is None else NOW - boost_delta
    uf = usefulness_factor(score, floor=floor, last_boosted_at=boost, now=NOW,
                           decay_half_life_days=decay_hl)
    assert floor - EPS <= uf <= 1.0 + EPS


@pytest.mark.parametrize("score", [0.0, 1.0, -5.0, 5.0])
def test_usefulness_clamped_score_hits_exact_endpoints(score: float) -> None:
    floor = 0.5
    uf = usefulness_factor(score, floor=floor, now=NOW)
    expected = 0.5 if score <= 0.0 else 1.0
    assert math.isclose(uf, expected, rel_tol=1e-9)


def test_usefulness_zero_score_is_exactly_floor() -> None:
    assert usefulness_factor(0.0, floor=0.5, now=NOW) == 0.5


@given(floor=unit_float)
def test_usefulness_full_score_is_one(floor: float) -> None:
    assert math.isclose(usefulness_factor(1.0, floor=floor, now=NOW), 1.0, rel_tol=1e-9)


@given(score=unit_float, floor=unit_float,
       d1=age_delta, d2=age_delta, decay_hl=half_life)
def test_usefulness_monotone_in_score_and_boost_time(
    score: float, floor: float, d1: timedelta, d2: timedelta, decay_hl: float,
) -> None:
    earlier, later = max(d1, d2), min(d1, d2)
    uf_early = usefulness_factor(score, floor=floor, last_boosted_at=NOW - earlier,
                                 now=NOW, decay_half_life_days=decay_hl)
    uf_late = usefulness_factor(score, floor=floor, last_boosted_at=NOW - later,
                                now=NOW, decay_half_life_days=decay_hl)
    assert uf_late >= uf_early - EPS


@given(score1=unit_float, score2=unit_float, floor=unit_float)
def test_usefulness_monotone_in_score_no_boost(score1: float, score2: float, floor: float) -> None:
    lo, hi = sorted([score1, score2])
    assert usefulness_factor(lo, floor=floor) <= usefulness_factor(hi, floor=floor) + EPS


def test_usefulness_no_boost_beats_any_decay() -> None:
    """None (never boosted) means no decay — must be >= any boosted variant."""
    floor = 0.5
    uf_none = usefulness_factor(1.0, floor=floor, last_boosted_at=None, now=NOW)
    uf_decayed = usefulness_factor(1.0, floor=floor, last_boosted_at=NOW - timedelta(days=30),
                                   now=NOW)
    assert uf_none == 1.0
    assert math.isclose(uf_decayed, 0.75, rel_tol=1e-9)
    assert uf_decayed < uf_none


# --- 4. composite monotonicity (perturbation per factor) -----------------------


@given(base=recalls, confidences=st.lists(unit_float, min_size=2, max_size=2))
def test_composite_monotone_in_confidence(base: MemoryRecall,
                                          confidences: list[float]) -> None:
    lo, hi = sorted(confidences)
    mem = base.memory
    s_lo = score_memory(_rec(lo, mem.accessed_at, mem.access_count, mem.usefulness_score,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    s_hi = score_memory(_rec(hi, mem.accessed_at, mem.access_count, mem.usefulness_score,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    assert s_hi >= s_lo - EPS


@given(base=recalls, counts=st.lists(access_counts, min_size=2, max_size=2))
def test_composite_monotone_in_access_count(base: MemoryRecall,
                                            counts: list[int]) -> None:
    lo, hi = sorted(counts)
    mem = base.memory
    s_lo = score_memory(_rec(mem.confidence, mem.accessed_at, lo, mem.usefulness_score,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    s_hi = score_memory(_rec(mem.confidence, mem.accessed_at, hi, mem.usefulness_score,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    assert s_hi >= s_lo - EPS


@given(base=recalls, uses=st.lists(unit_float, min_size=2, max_size=2))
def test_composite_monotone_in_usefulness(base: MemoryRecall,
                                          uses: list[float]) -> None:
    lo, hi = sorted(uses)
    mem = base.memory
    s_lo = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count, lo,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    s_hi = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count, hi,
                             mem.last_boosted_at, base.similarity), now=NOW).score
    assert s_hi >= s_lo - EPS


@given(base=recalls, deltas=_sorted_ages(2))
def test_composite_monotone_in_accessed_at(base: MemoryRecall,
                                           deltas: list[timedelta]) -> None:
    older, newer = deltas[-1], deltas[0]  # larger delta = older access
    mem = base.memory
    s_older = score_memory(_rec(mem.confidence, NOW - older, mem.access_count,
                                mem.usefulness_score, mem.last_boosted_at,
                                base.similarity), now=NOW).score
    s_newer = score_memory(_rec(mem.confidence, NOW - newer, mem.access_count,
                                mem.usefulness_score, mem.last_boosted_at,
                                base.similarity), now=NOW).score
    assert s_newer >= s_older - EPS


@given(base=recalls, deltas=_sorted_ages(2))
def test_composite_monotone_in_last_boosted_at(base: MemoryRecall,
                                               deltas: list[timedelta]) -> None:
    earlier, later = max(deltas), min(deltas)
    mem = base.memory
    s_early = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count,
                                mem.usefulness_score, NOW - earlier,
                                base.similarity), now=NOW).score
    s_late = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count,
                               mem.usefulness_score, NOW - later,
                               base.similarity), now=NOW).score
    assert s_late >= s_early - EPS


@given(base=recalls, sims=st.lists(unit_float, min_size=2, max_size=2))
def test_composite_monotone_in_similarity(base: MemoryRecall,
                                          sims: list[float]) -> None:
    lo, hi = sorted(sims)
    mem = base.memory
    s_lo = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count,
                             mem.usefulness_score, mem.last_boosted_at, lo), now=NOW).score
    s_hi = score_memory(_rec(mem.confidence, mem.accessed_at, mem.access_count,
                             mem.usefulness_score, mem.last_boosted_at, hi), now=NOW).score
    assert s_hi >= s_lo - EPS


@given(base=turns, bases=st.lists(unit_float, min_size=2, max_size=2))
def test_composite_turn_monotone_in_base_score(base: EpisodeTurn,
                                               bases: list[float]) -> None:
    lo, hi = sorted(bases)
    s_lo = score_turn(base, lo, now=NOW).score
    s_hi = score_turn(base, hi, now=NOW).score
    assert s_hi >= s_lo - EPS


# --- 5. NaN poisoning / finiteness --------------------------------------------


@given(recall=recalls)
def test_composite_finite_and_in_range_for_valid_inputs(recall: MemoryRecall) -> None:
    score = score_memory(recall, now=NOW).score
    assert math.isfinite(score)
    assert -EPS <= score <= recall.similarity * COMPOSITE_MAX_PRODUCT + EPS


@given(turn=turns, base_score=unit_float)
def test_turn_score_finite_and_in_range(turn: EpisodeTurn, base_score: float) -> None:
    score = score_turn(turn, base_score, now=NOW).score
    assert math.isfinite(score)
    assert -EPS <= score <= base_score * COMPOSITE_MAX_PRODUCT + EPS


@given(recall=recalls, weights=weight_sets)
def test_composite_finite_under_arbitrary_weights(recall: MemoryRecall,
                                                  weights: RelevanceWeights) -> None:
    assert math.isfinite(score_memory(recall, weights=weights, now=NOW).score)


def test_nonfinite_similarity_rejected_and_factors_floor() -> None:
    """PINNED GUARD: MemoryRecall.similarity rejects NaN/inf at construction
    (pydantic allow_inf_nan=False), and the public factor functions floor
    non-finite input instead of letting it read as best quality: confidence
    maps to 0.0, usefulness to its floor. Poison cannot reach the composite
    and, through it, rank_memories comparisons."""
    with pytest.raises(ValidationError):
        _rec(confidence=1.0, accessed=NOW, count=0, use=1.0, boost=None,
             sim=float("nan"))
    with pytest.raises(ValidationError):
        _rec(confidence=1.0, accessed=NOW, count=0, use=1.0, boost=None,
             sim=float("inf"))
    assert confidence_factor(float("nan")) == 0.0
    assert confidence_factor(float("inf")) == 0.0
    assert usefulness_factor(float("nan"), floor=0.5) == 0.5
    assert usefulness_factor(float("inf"), floor=0.5) == 0.5


def test_nonfinite_confidence_and_usefulness_floor_to_minimum() -> None:
    """PINNED GUARD: the factor functions' isfinite check maps NaN to the
    floor, so poison reads as worst quality — the bare clamp ordering
    (max(0.0, min(1.0, nan)) keeps 1.0) would have read it as best quality."""
    assert confidence_factor(float("nan")) == 0.0
    assert usefulness_factor(float("nan"), floor=0.5) == 0.5


# --- 6. rank_memories is an honest descending sort -----------------------------


@given(items=st.lists(recalls, max_size=12))
def test_rank_memories_equals_stable_descending_sort(items: list[MemoryRecall]) -> None:
    result = rank_memories(items, now=NOW)
    expected = sorted(
        (score_memory(r, now=NOW) for r in items),
        key=lambda s: s.score,
        reverse=True,
    )
    assert result == expected
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)


@given(items=st.lists(recalls, min_size=1, max_size=10), data=st.data())
def test_rank_memories_score_sequence_permutation_invariant(
    items: list[MemoryRecall], data: st.DataObject,
) -> None:
    shuffled = data.draw(st.permutations(items))
    seq_a = [s.score for s in rank_memories(items, now=NOW)]
    seq_b = [s.score for s in rank_memories(shuffled, now=NOW)]
    assert seq_a == seq_b


@given(sim=unit_float, conf=unit_float, n=st.integers(2, 6))
def test_rank_memories_ties_preserve_input_order(
    sim: float, conf: float, n: int,
) -> None:
    """DOCUMENTED ACTUAL TIE BEHAVIOR: rank_memories uses list.sort with no
    tiebreaker, so equal scores come out in input (stable Timsort) order.
    Distinct ids make the identity check meaningful; every other scoring
    field is identical, so scores tie exactly."""
    items = [
        _rec(conf, NOW - timedelta(days=1), 3, 0.8, None, sim, mem_id=f"weft-tie{i:04d}")
        for i in range(n)
    ]
    result = rank_memories(items, now=NOW)
    assert [s.memory.id for s in result] == [r.memory.id for r in items]
    assert len({s.score for s in result}) == 1  # genuine tie, not ordered by id


@given(sims=st.lists(unit_float, min_size=2, max_size=8, unique=True),
       flip=st.booleans())
def test_rank_memories_unique_scores_fully_permutation_invariant(
    sims: list[float], flip: bool,
) -> None:
    """With all-distinct scores the output sequence must be identical under
    any input permutation — objects, not just scores."""
    items = [
        _rec(confidence=0.9, accessed=NOW - timedelta(days=1), count=0, use=0.7,
             boost=None, sim=s, mem_id=f"weft-uniq{i:04d}")
        for i, s in enumerate(sims)
    ]
    shuffled = list(reversed(items)) if flip else items
    assert rank_memories(items, now=NOW) == rank_memories(shuffled, now=NOW)


def test_rank_memories_empty_list_is_empty() -> None:
    assert rank_memories([], now=NOW) == []


@given(ids=st.lists(st.text(min_size=1, max_size=8,
                            alphabet=st.sampled_from("abcdefgh0123456789")),
                    min_size=2, max_size=2, unique=True),
       base_score=unit_float)
def test_rank_turns_tiebreak_is_turn_id_ascending(ids: list[str], base_score: float) -> None:
    """rank_turns pins a secondary sort on turn id; rank_memories does not
    (see tie test above). Same scored fields → exact tie → id order."""
    turn_items = [
        (
            EpisodeTurn(
                id=tid,
                episode_id="episode-prop",
                turn_index=i,
                role=TurnRole.user,
                content="tie",
                occurred_at=NOW - timedelta(days=1),
                usefulness_score=0.8,
                last_boosted_at=None,
            ),
            base_score,
        )
        for i, tid in enumerate(ids)
    ]
    result = rank_turns(turn_items, now=NOW)
    assert [s.turn.id for s in result] == sorted(ids)
    assert len({s.score for s in result}) == 1


# --- 7. score_turn factor omission set ----------------------------------------


def test_scored_turn_fields_are_scored_memory_minus_three_factors() -> None:
    """Pin the omission set EXACTLY: ScoredTurn drops similarity-relative
    factors {confidence_factor, frequency_factor, type_boost_factor} that
    ScoredMemory carries, keeping {recency_factor, usefulness_factor}.
    Docstring and code agree; this test freezes the contract."""
    memory_fields = {f.name for f in dc_fields(ScoredMemory)}
    turn_fields = {f.name for f in dc_fields(ScoredTurn)}
    assert memory_fields - turn_fields == {
        "memory", "similarity", "confidence_factor", "frequency_factor",
        "type_boost_factor",
    }
    assert turn_fields - memory_fields == {"turn", "base_score"}
    assert turn_fields & memory_fields == {"recency_factor", "usefulness_factor", "score"}


@given(turn=turns, base_score=unit_float, weights=weight_sets)
def test_turn_score_equals_base_times_recency_times_usefulness(
    turn: EpisodeTurn, base_score: float, weights: RelevanceWeights,
) -> None:
    """Algebraic pin: score_turn applies exactly recency (on occurred_at) and
    usefulness, nothing else, even under arbitrary weights."""
    scored = score_turn(turn, base_score, weights=weights, now=NOW)
    rf = recency_factor(turn.occurred_at, now=NOW,
                        half_life_days=weights.recency_half_life_days)
    uf = usefulness_factor(turn.usefulness_score, floor=weights.usefulness_floor,
                           last_boosted_at=turn.last_boosted_at, now=NOW)
    assert scored.recency_factor == rf
    assert scored.usefulness_factor == uf
    assert math.isclose(scored.score, base_score * rf * uf, rel_tol=1e-12)


# --- 8. boundary sweep ---------------------------------------------------------


@pytest.mark.parametrize(
    "age_days,count,confidence,usefulness,sim",
    [
        (0.0, 0, 0.0, 0.0, 0.0),
        (0.0, 10**18, 1.0, 1.0, 1.0),
        (30.0, 0, 0.0, 1.0, 1.0),
        (30.0, 10, 1.0, 0.0, 0.5),
        (3000.0, 0, 0.5, 0.5, 1.0),   # 100 x half-life asymptote
        (3000.0, 10**18, 1.0, 1.0, 1.0),
    ],
)
def test_boundary_sweep_no_crash_no_signflip_in_bounds(
    age_days: float, count: int, confidence: float, usefulness: float, sim: float,
) -> None:
    recall = _rec(confidence, NOW - timedelta(days=age_days), count, usefulness,
                  None, sim)
    scored = score_memory(recall, now=NOW)
    assert math.isfinite(scored.score)
    assert scored.score >= -EPS                      # no sign flips
    assert scored.score <= sim * COMPOSITE_MAX_PRODUCT + EPS
    assert 0.0 <= scored.confidence_factor <= 1.0
    assert 0.0 <= scored.recency_factor <= 1.0
    assert 1.0 <= scored.frequency_factor <= 1.2 + EPS
    assert 0.5 <= scored.usefulness_factor <= 1.0 + EPS
    assert 1.0 <= scored.type_boost_factor <= 1.3


def test_boundary_sweep_future_access_and_negative_similarity() -> None:
    """Future access clamps recency to 1; negative similarity keeps the
    multiplicative sign (documented: factors never flip sign, similarity can)."""
    recall = _rec(0.7, NOW + timedelta(days=5), 0, 0.2, None, -0.5)
    scored = score_memory(recall, now=NOW)
    assert scored.recency_factor == 1.0
    assert math.isfinite(scored.score)
    assert scored.score <= 0.0  # negative similarity, non-negative factors


# --- 9. determinism -------------------------------------------------------------


@given(recall=recalls)
def test_score_memory_bit_identical_across_calls(recall: MemoryRecall) -> None:
    a = score_memory(recall, now=NOW)
    b = score_memory(recall, now=NOW)
    assert a.score == b.score
    assert (a.confidence_factor, a.recency_factor, a.frequency_factor,
            a.usefulness_factor, a.type_boost_factor) == (
        b.confidence_factor, b.recency_factor, b.frequency_factor,
        b.usefulness_factor, b.type_boost_factor)


@given(items=st.lists(recalls, max_size=8))
def test_rank_memories_bit_identical_across_calls(items: list[MemoryRecall]) -> None:
    assert rank_memories(items, now=NOW) == rank_memories(items, now=NOW)


@given(turn=turns, base_score=unit_float)
def test_score_turn_bit_identical_across_calls(turn: EpisodeTurn,
                                               base_score: float) -> None:
    assert score_turn(turn, base_score, now=NOW).score == \
        score_turn(turn, base_score, now=NOW).score
