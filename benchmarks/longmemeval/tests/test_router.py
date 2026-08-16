"""Unit tests for the question-type-aware retrieval policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmarks.longmemeval.adapter import _answer_text_match
from weft.models import EpisodeTurn, TurnRole
from weft.turn_recall import TemporalWindow, multi_session_query_variant
from benchmarks.longmemeval.router import (
    RetrievalDiagnostics,
    RetrievalPolicy,
    _DEFAULT_POLICIES,
    _FALLBACK_POLICY,
    _retrieve_turns,
    _select_session_diverse_turns,
    _apply_centroid_selection,
    policy_for,
    retrieve,
    session_rerank_enabled_for,
    session_rerank_pool_limit_for,
)


def _async_context(value):
    class Context:
        async def __aenter__(self):
            return value

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return Context()


def _noop_async():
    async def noop():
        return None

    return noop()


def _turn(turn_id: str, episode_id: str) -> EpisodeTurn:
    return EpisodeTurn(
        id=turn_id,
        episode_id=episode_id,
        turn_index=0,
        role=TurnRole.user,
        content=turn_id,
    )


def test_retrieval_diagnostics_classifies_candidate_causes():
    diagnostics = RetrievalDiagnostics(
        path="flat",
        vector_gold_ranks={"v": 2, "both": 40},
        keyword_gold_ranks={"k": 3, "both": 41},
        vector_candidate_count=30,
        keyword_candidate_count=30,
    )

    result = diagnostics.to_dict(
        gold_turn_ids=["v", "k", "both", "absent"],
        top_k=10,
    )

    assert result["gold_vector_present"] == ["both", "v"]
    assert result["gold_keyword_present"] == ["both", "k"]
    assert result["vector_only"] == ["v"]
    assert result["keyword_only"] == ["k"]
    assert result["both_halves_absent"] == ["absent"]
    assert result["both_present_below_top_k"] == ["both"]


def test_retrieval_diagnostics_preserves_recovery_state():
    diagnostics = RetrievalDiagnostics(
        initial_empty=True,
        retry_attempted=True,
        retry_rescued=True,
        fallback_attempted=False,
        final_empty=False,
    )

    result = diagnostics.to_dict(gold_turn_ids=[], top_k=10)

    assert result["initial_empty"] is True
    assert result["retry_rescued"] is True
    assert result["fallback_attempted"] is False
    assert result["final_empty"] is False


def test_session_rerank_policy_enables_only_validated_types():
    enabled = (
        "multi-session",
        "single-session-user",
        "single-session-preference",
    )
    disabled = (
        "single-session-assistant",
        "knowledge-update",
        "temporal-reasoning",
        "unknown-type",
    )
    for question_type in enabled:
        assert session_rerank_enabled_for(question_type) is True
        assert session_rerank_pool_limit_for(question_type) == 90
        assert session_rerank_enabled_for(f"{question_type}_abs") is True
        assert session_rerank_pool_limit_for(f"{question_type}_abs") == 90
    for question_type in disabled:
        assert session_rerank_enabled_for(question_type) is False
        assert session_rerank_pool_limit_for(question_type) is None
        assert session_rerank_enabled_for(f"{question_type}_abs") is False
        assert session_rerank_pool_limit_for(f"{question_type}_abs") is None


def test_multi_session_widens_top_k():
    """Multi-session counting questions need wider recall than single-session."""
    assert policy_for("multi-session").top_k > _FALLBACK_POLICY.top_k


def test_temporal_reasoning_widens_top_k():
    """Multi-anchor temporal questions need wider recall."""
    assert policy_for("temporal-reasoning").top_k > _FALLBACK_POLICY.top_k


def test_candidate_width_is_separate_from_reader_top_k():
    baseline = RetrievalPolicy(top_k=10)
    wider = RetrievalPolicy(top_k=10, candidate_sql_limit=60, fusion_candidate_limit=30)
    assert baseline.top_k == wider.top_k
    assert wider.candidate_sql_limit == 60
    assert wider.fusion_candidate_limit == 30


def test_retrieval_diagnostics_serializes_ground_truth_sets():
    diagnostics = RetrievalDiagnostics(
        indexed_turn_ids=["distractor", "gold"],
        gold_session_turn_ids=["gold"],
    )
    payload = diagnostics.to_dict(gold_turn_ids=["gold"], top_k=10)
    assert payload["indexed_turn_count"] == 2
    assert payload["gold_session_turn_count"] == 1
    assert payload["ground_truth_derivation"]["indexed"].startswith("manifest")


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


def test_multi_session_query_variant_reduces_question_framing():
    query = "How many different times did I visit the coffee shop last year?"

    assert multi_session_query_variant(query) == "visit coffee shop"


def test_multi_session_query_variant_is_conservative_when_no_reduction():
    assert multi_session_query_variant("coffee shop visit") is None
    assert multi_session_query_variant("how many of") is None


@pytest.mark.asyncio
async def test_retrieve_forwards_multi_session_query_variant_flag(monkeypatch) -> None:
    calls = []

    async def fake_retrieve_turns(*args, **kwargs):
        calls.append(kwargs)
        return []

    async def fake_retrieve_belief(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_turns", fake_retrieve_turns,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_belief", fake_retrieve_belief,
    )

    await retrieve(
        object(), object(), question="How many visits to the shop?",
        question_type="multi-session", project_id="lme_q1", tier="turns",
        use_multi_session_query_variant=True,
    )
    assert calls[0]["use_multi_session_query_variant"] is True

    calls.clear()
    await retrieve(
        object(), object(), question="How many visits to the shop?",
        question_type="single-session-user", project_id="lme_q1", tier="turns",
        use_multi_session_query_variant=True,
    )
    assert calls[0]["use_multi_session_query_variant"] is True


@pytest.mark.asyncio
async def test_multi_session_flat_probe_isolated_from_single_session(
    monkeypatch,
) -> None:
    class Pool:
        def acquire(self):
            return _async_context(Conn())

    class Conn:
        def transaction(self):
            return _async_context(self)

    class Embedder:
        async def embed(self, query):
            return [1.0]

    calls = []

    async def fake_set_user_context_value(conn, user_id):
        return None

    async def fake_recall_turns(*args, **kwargs):
        calls.append(args[1])
        return [_turn("base", "ep1")]

    monkeypatch.setattr(
        "benchmarks.longmemeval.router.set_user_context_value",
        fake_set_user_context_value,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router.recall_turns", fake_recall_turns,
    )

    diagnostics = RetrievalDiagnostics()
    await _retrieve_turns(
        Pool(), Embedder(), question="How many visits to the shop?",
        question_type="single-session-user", project_id="lme_q1",
        policy=RetrievalPolicy(top_k=4), diagnostics=diagnostics,
        use_multi_session_query_variant=True,
    )

    assert calls == ["How many visits to the shop?"]
    assert diagnostics.query_variant_diagnostics == []


@pytest.mark.asyncio
async def test_multi_session_flat_probe_unions_deduplicates_and_caps(
    monkeypatch,
) -> None:
    class Pool:
        def acquire(self):
            return _async_context(Conn())

    class Conn:
        def transaction(self):
            return _async_context(self)

    class Embedder:
        async def embed(self, query):
            return [1.0]

    calls = []

    async def fake_set_user_context_value(conn, user_id):
        return None

    async def fake_recall_turns(*args, **kwargs):
        calls.append(args[1])
        if args[1] == "How many visits to the shop?":
            return [_turn("base-1", "ep1"), _turn("duplicate", "ep1")]
        return [
            _turn("duplicate", "ep1"),
            _turn("variant-1", "ep2"),
            _turn("variant-2", "ep3"),
        ]

    monkeypatch.setattr(
        "benchmarks.longmemeval.router.set_user_context_value",
        fake_set_user_context_value,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router.recall_turns", fake_recall_turns,
    )

    diagnostics = RetrievalDiagnostics()
    result = await _retrieve_turns(
        Pool(), Embedder(), question="How many visits to the shop?",
        question_type="multi-session", project_id="lme_q1",
        policy=RetrievalPolicy(top_k=4), diagnostics=diagnostics,
        use_multi_session_query_variant=True,
    )

    assert calls == ["How many visits to the shop?", "visits shop"]
    assert [recall.memory.id for recall in result] == [
        "base-1", "duplicate", "variant-1", "variant-2",
    ]
    assert len(result) == 4
    assert diagnostics.query_variant_diagnostics[0]["added_turn_ids"] == [
        "variant-1", "variant-2",
    ]


@pytest.mark.asyncio
async def test_multi_session_query_variant_does_not_change_temporal_path(
    monkeypatch,
) -> None:
    calls = []

    async def fake_anchor(*args, **kwargs):
        calls.append(kwargs)
        return {"anchor": []}

    monkeypatch.setattr("benchmarks.longmemeval.router.temporal_anchor", fake_anchor)
    monkeypatch.setattr(
        "benchmarks.longmemeval.router.set_user_context_value",
        lambda conn, user_id: _noop_async(),
    )

    result = await _retrieve_turns(
        type("Pool", (), {"acquire": lambda self: _async_context(
            type("Conn", (), {"transaction": lambda self: _async_context(self)})()
        )})(),
        object(), question="How many days between launch and demo?",
        question_type="temporal-reasoning", project_id="lme_q1",
        policy=RetrievalPolicy(top_k=4), use_multi_session_query_variant=True,
    )

    assert result == []
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retrieve_forwards_temporal_window(monkeypatch) -> None:
    window = TemporalWindow(
        since=datetime(2026, 7, 24, tzinfo=timezone.utc),
        until=datetime(2026, 7, 26, 23, 59, 59, tzinfo=timezone.utc),
    )
    calls = []

    async def fake_retrieve_turns(*args, **kwargs):
        calls.append(kwargs)
        return []

    async def fake_retrieve_belief(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_turns", fake_retrieve_turns,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_belief", fake_retrieve_belief,
    )

    result = await retrieve(
        object(), object(), question="What happened 10 days ago?",
        question_type="temporal-reasoning", project_id="lme_q1",
        tier="turns", temporal_window=window,
    )

    assert result == []
    assert calls[0]["temporal_window"] is window


@pytest.mark.asyncio
async def test_temporal_window_probe_unions_deduplicates_and_keeps_output_cap(
    monkeypatch,
) -> None:
    class Pool:
        def acquire(self):
            return _async_context(Conn())

    class Conn:
        def transaction(self):
            return _async_context(self)

    calls = []
    window = TemporalWindow(
        since=datetime(2026, 7, 24, tzinfo=timezone.utc),
        until=datetime(2026, 7, 26, 23, 59, 59, tzinfo=timezone.utc),
    )

    async def fake_set_user_context_value(conn, user_id):
        return None

    async def fake_anchor(pool, query, **kwargs):
        calls.append(kwargs)
        if kwargs.get("since") is None:
            return {
                "first": [_turn("base-1", "ep1"), _turn("duplicate", "ep1")],
                "second": [_turn("base-2", "ep2")],
            }
        return {
            "first": [_turn("duplicate", "ep1"), _turn("window-1", "ep1")],
            "second": [_turn("window-2", "ep2")],
        }

    monkeypatch.setattr(
        "benchmarks.longmemeval.router.set_user_context_value",
        fake_set_user_context_value,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router.temporal_anchor", fake_anchor,
    )

    result = await _retrieve_turns(
        Pool(), object(), question="What happened 10 days ago?",
        question_type="temporal-reasoning", project_id="lme_q1",
        policy=RetrievalPolicy(top_k=4), temporal_window=window,
    )

    assert [recall.memory.id for recall in result] == [
        "base-1", "duplicate", "window-1", "base-2",
    ]
    assert len(result) == 4
    assert calls[0].get("since") is None
    assert calls[0].get("until") is None
    assert calls[1]["since"] == window.since
    assert calls[1]["until"] == window.until


def test_session_selector_picks_one_best_turn_per_interleaved_session() -> None:
    turns = [
        _turn("a-best", "episode"),
        _turn("a-second", "episode"),
        _turn("b-best", "episode"),
        _turn("a-third", "episode"),
        _turn("b-second", "episode"),
        _turn("c-best", "episode"),
    ]
    mapping = {
        "a-best": "session-a",
        "a-second": "session-a",
        "b-best": "session-b",
        "a-third": "session-a",
        "b-second": "session-b",
        "c-best": "session-c",
    }

    selected = _select_session_diverse_turns(
        turns, top_k=3, turn_session_map=mapping,
    )

    assert [turn.id for turn in selected] == [
        "a-best", "b-best", "c-best",
    ]
    assert len({mapping[turn.id] for turn in selected}) == 3


def test_session_selector_preserves_reader_cap_and_rank_order() -> None:
    turns = [_turn(f"turn-{index}", "episode") for index in range(8)]
    mapping = {turn.id: f"session-{index // 2}" for index, turn in enumerate(turns)}

    selected = _select_session_diverse_turns(
        turns, top_k=4, turn_session_map=mapping,
    )

    assert len(selected) == 4
    assert len({turn.id for turn in selected}) == 4
    assert [turn.id for turn in selected] == [
        "turn-0", "turn-2", "turn-4", "turn-6",
    ]


def test_session_selector_incomplete_mapping_is_a_noop() -> None:
    turns = [_turn(f"turn-{index}", "episode") for index in range(5)]
    mapping = {turn.id: "session-a" for turn in turns[:4]}
    diagnostics = RetrievalDiagnostics()

    selected = _select_session_diverse_turns(
        turns, top_k=3, turn_session_map=mapping, diagnostics=diagnostics,
    )

    assert [turn.id for turn in selected] == ["turn-0", "turn-1", "turn-2"]
    assert diagnostics.session_selector_applied is False
    assert diagnostics.session_selector_reason == "incomplete_session_mapping"


def test_centroid_selection_restores_reader_cap_on_rejected_callback() -> None:
    turns = [_turn(f"turn-{index}", "episode") for index in range(5)]
    diagnostics = RetrievalDiagnostics(session_centroid_reason="selector_error:ValueError")

    selected = _apply_centroid_selection(turns, top_k=3, diagnostics=diagnostics)

    assert [turn.id for turn in selected] == ["turn-0", "turn-1", "turn-2"]


@pytest.mark.asyncio
async def test_type_gated_session_rerank_dispatch(monkeypatch) -> None:
    calls = []

    async def fake_retrieve_turns(*args, **kwargs):
        calls.append(kwargs)
        return []

    async def fake_retrieve_belief(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_turns", fake_retrieve_turns,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_belief", fake_retrieve_belief,
    )

    await retrieve(
        object(), object(), question="How many visits?",
        question_type="multi-session", project_id="lme_q1", tier="turns",
        turn_session_map={"turn-1": "session-1"},
    )
    assert calls[-1]["use_session_selector"] is True
    assert calls[-1]["session_selector_pool_limit"] == 90

    await retrieve(
        object(), object(), question="What happened?",
        question_type="single-session-assistant_abs", project_id="lme_q1",
        tier="turns", turn_session_map={"turn-1": "session-1"},
    )
    assert calls[-1]["use_session_selector"] is False
    assert calls[-1]["session_selector_pool_limit"] is None

    await retrieve(
        object(), object(), question="How many visits?",
        question_type="multi-session", project_id="lme_q1", tier="turns",
    )
    assert calls[-1]["use_session_selector"] is False
    assert calls[-1]["session_selector_pool_limit"] is None


@pytest.mark.asyncio
async def test_default_retrieval_does_not_enable_session_selector(monkeypatch) -> None:
    calls = []

    async def fake_retrieve_turns(*args, **kwargs):
        calls.append(kwargs)
        return []

    async def fake_retrieve_belief(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_turns", fake_retrieve_turns,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router._retrieve_belief", fake_retrieve_belief,
    )

    await retrieve(
        object(), object(), question="What happened?",
        question_type="single-session-user", project_id="lme_q1", tier="turns",
    )

    assert calls[0]["use_session_selector"] is False
    assert calls[0]["turn_session_map"] is None
    assert calls[0]["session_selector_pool_limit"] is None
