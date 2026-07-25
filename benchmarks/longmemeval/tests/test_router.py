"""Unit tests for the question-type-aware retrieval policy."""

from __future__ import annotations

from benchmarks.longmemeval.adapter import _answer_text_match
from benchmarks.longmemeval.router import diversify_temporal_turns
from weft.models import EpisodeTurn, TurnRole
from benchmarks.longmemeval.router import (
    RetrievalPolicy,
    _DEFAULT_POLICIES,
    _FALLBACK_POLICY,
    policy_for,
)


def _turn(turn_id: str, episode_id: str) -> EpisodeTurn:
    return EpisodeTurn(
        id=turn_id,
        episode_id=episode_id,
        turn_index=0,
        role=TurnRole.user,
        content=turn_id,
    )


def test_temporal_diversity_preserves_episode_coverage():
    ranked = [
        _turn("a1", "episode-a"),
        _turn("a2", "episode-a"),
        _turn("a3", "episode-a"),
        _turn("b1", "episode-b"),
        _turn("c1", "episode-c"),
    ]
    selected = diversify_temporal_turns(ranked, limit=3)
    assert [turn.id for turn in selected] == ["a1", "b1", "c1"]


def test_temporal_diversity_fills_remaining_slots_by_rank():
    ranked = [_turn("a1", "a"), _turn("a2", "a"), _turn("b1", "b")]
    selected = diversify_temporal_turns(ranked, limit=4)
    assert [turn.id for turn in selected] == ["a1", "b1", "a2"]


def test_multi_session_widens_top_k():
    """Multi-session counting questions need wider recall than single-session."""
    assert policy_for("multi-session").top_k > _FALLBACK_POLICY.top_k


def test_temporal_reasoning_widens_top_k():
    """Multi-anchor temporal questions need wider recall."""
    assert policy_for("temporal-reasoning").top_k > _FALLBACK_POLICY.top_k


def test_recall_match_accepts_numeric_gold_answers():
    """LongMemEval temporal gold answers may be numbers, not only strings."""
    assert _answer_text_match(42, "The elapsed time was 42 days.")


def test_single_session_keeps_baseline():
    """Single-session questions are well-served by top-10."""
    for qt in (
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
    ):
        assert policy_for(qt).top_k == 10


def test_knowledge_update_keeps_baseline():
    assert policy_for("knowledge-update").top_k == 10


def test_unknown_type_falls_back():
    """Unknown question types get the conservative fallback."""
    assert policy_for("not-a-real-type") == _FALLBACK_POLICY


def test_abs_suffix_stripped():
    """Abstention variants share the retrieval shape of their base type —
    only the Reader's refusal-licensed prompt differs."""
    for qt in _DEFAULT_POLICIES:
        assert policy_for(f"{qt}_abs") == policy_for(qt)


def test_policy_is_immutable():
    """RetrievalPolicy is frozen so accidental mutation in dispatch can't
    bleed across question types."""
    p = policy_for("multi-session")
    try:
        p.top_k = 999  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("RetrievalPolicy should be frozen")


def test_overfetch_multiplier_is_at_least_two():
    """Over-fetch protects against project_id=NULL global memories leaking
    into the candidate pool. A multiplier below 2 risks losing real hits
    to that filtering step."""
    for qt in _DEFAULT_POLICIES:
        policy = policy_for(qt)
        assert policy.overfetch_multiplier >= 2


def test_custom_policy_passes_through():
    """RetrievalPolicy can be constructed directly for ablations."""
    p = RetrievalPolicy(top_k=50, overfetch_multiplier=3)
    assert p.top_k == 50
    assert p.overfetch_multiplier == 3
