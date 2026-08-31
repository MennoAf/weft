"""Provider-free tests for Milestone A deterministic retrieval recovery."""
from __future__ import annotations

import asyncio
import json

import pytest

from weft.retrieval_recovery import (
    RecoveryConfig,
    RecoveryController,
    ScopeSnapshot,
    assess_sufficiency,
    build_recovery_shape,
    build_reformulations,
    parse_planner_response,
    unsupported_recovery,
)


def candidate(memory_id: str, content: str, **extra: object) -> dict[str, object]:
    return {"memory_id": memory_id, "content": content, **extra}


def test_shape_aware_insufficiency_is_fail_closed():
    procedural = assess_sufficiency(
        "How do I configure the export command?",
        [candidate("topic", "Export configuration overview")],
    )
    assert procedural.trigger == "procedural_topic_only"
    assert procedural.answerability == "insufficient_evidence"

    # Topic nouns must not be promoted to procedural evidence merely because
    # they overlap the query's configuration vocabulary.
    for topic_only in (
        "Export setup overview",
        "Export settings and options",
        "Export configuration pilot note: configure export settings and schedule details are archived.",
    ):
        topic_assessment = assess_sufficiency(
            "How do I configure the export command?",
            [candidate("topic", topic_only)],
        )
        assert topic_assessment.trigger == "procedural_topic_only"
        assert topic_assessment.answerability == "insufficient_evidence"

    # Operational commands and concrete operation-linked artifacts remain valid evidence.
    assert assess_sufficiency(
        "How do I configure the export command?",
        [candidate("guide", "Run export with --format json")],
    ).sufficient
    assert assess_sufficiency(
        "How do I configure the export command?",
        [candidate("guide", "Set export values in export.yaml")],
    ).sufficient

    compare = assess_sufficiency(
        "Compare Alpha and Beta",
        [candidate("a", "Alpha has a launch history")],
    )
    assert compare.trigger == "missing_required_operand"
    assert compare.covered_branches == ("branch-1",)

    temporal = assess_sufficiency(
        "How many days between the demo and the retro?",
        [candidate("d", "The demo happened on Monday")],
    )
    assert temporal.shape == "temporal"
    assert temporal.trigger == "missing_temporal_anchor"

    assert assess_sufficiency("anything", []).trigger == "zero_results"
    conflict = assess_sufficiency(
        "anything", [candidate("x", "one")], explicit_conflicts=["same date"]
    )
    assert conflict.retrieval_status == "conflict"
    assert conflict.answerability == "conflicting_evidence"


def test_direct_negative_or_unknown_evidence_is_fail_closed():
    query = "What is the pilot launch date?"
    for content in (
        "The pilot launch date is not disclosed.",
        "The pilot launch date is unknown.",
        "There is no record of the pilot launch date.",
    ):
        assessment = assess_sufficiency(query, [candidate("negative", content)])
        assert assessment.retrieval_status == "incomplete"
        assert assessment.answerability == "insufficient_evidence"
        assert assessment.trigger == "negative_or_unknown_evidence"
        assert assessment.reasons == ("evidence_denies_or_does_not_establish_requested_fact",)

    positive = assess_sufficiency(
        query, [candidate("positive", "The pilot launch date is 2026-08-24.")]
    )
    assert positive.sufficient


def test_negative_evidence_in_unrelated_sentence_does_not_veto_direct_fact():
    assessment = assess_sufficiency(
        "What is the pilot launch date?",
        [candidate("positive", "The launch date is 2026-08-24. Contact details are not available.")],
    )
    assert assessment.sufficient


def test_generic_reformulations_are_bounded_and_deduplicated():
    shape = build_recovery_shape("How do I configure the export command?")
    queries = build_reformulations("How do I configure the export command?", shape=shape, max_queries=4)
    assert len(queries) <= 4
    assert len({q.casefold() for q in queries}) == len(queries)
    assert any("setup" in q.casefold() or "command" in q.casefold() for q in queries)


def test_scope_snapshot_preserves_policy_and_filters():
    face = ScopeSnapshot.from_baseline(
        user_id="u", requested_project_id="requested", resolved_project_id="resolved",
        retrieval_mode="face", agent_id="a", topic="t", limit=7,
    )
    code = ScopeSnapshot.from_baseline(
        user_id="u", requested_project_id="requested", resolved_project_id="resolved",
        retrieval_mode="code", agent_id="a",
    )
    assert face.project_policy == "facet_boost"
    assert face.to_probe_kwargs()["project_id"] is None
    assert code.project_policy == "hard_wall"
    assert code.to_probe_kwargs()["project_id"] == "resolved"
    assert code.to_probe_kwargs()["user_id"] == "u"
    with pytest.raises(ValueError):
        ScopeSnapshot.from_baseline(user_id=None, requested_project_id=None,
                                    resolved_project_id=None, retrieval_mode="bogus")


def test_reserved_planner_parser_accepts_candidates_and_rejects_authority():
    parsed = parse_planner_response(
        {"plans": [{"branch": "branch-1", "query": "Alpha history", "source_mode": "face"}]},
        allowed_branches=["branch-1"], allowed_source_mode="face",
    )
    assert parsed[0].query == "Alpha history"
    with pytest.raises(ValueError):
        parse_planner_response({"plans": [{"branch": "branch-1", "query": "DROP TABLE x;", "source_mode": "face"}]},
                               allowed_branches=["branch-1"], allowed_source_mode="face")
    with pytest.raises(ValueError):
        parse_planner_response({"plans": [{"branch": "branch-1", "query": "x", "project_id": "other", "source_mode": "face"}]},
                               allowed_branches=["branch-1"], allowed_source_mode="face")
    with pytest.raises(ValueError):
        parse_planner_response({"plans": [{"branch": "branch-1", "query": "x", "source_mode": "code"}]},
                               allowed_branches=["branch-1"], allowed_source_mode="face")


@pytest.mark.asyncio
async def test_controller_stage_order_stop_dedup_and_provenance():
    calls: list[tuple[str, str]] = []

    async def memory_search(query, scope, limit):
        calls.append(("memory", query))
        if "Alpha" in query and "Beta" in query:
            return [candidate("same", "Alpha Beta comparison result")]
        return [candidate("same", "Alpha Beta comparison result")]

    async def alternate_search(query, scope, limit):
        calls.append(("alternate", query))
        return [candidate("same", "Alpha Beta comparison result")]

    scope = ScopeSnapshot.from_baseline(user_id="u", requested_project_id="p", resolved_project_id="p", retrieval_mode="face")
    outcome = await RecoveryController(
        config=RecoveryConfig(max_queries_per_branch=2), memory_search=memory_search, alternate_search=alternate_search
    ).recover("Compare Alpha and Beta", primary_results=(), scope=scope)
    assert [stage.stage for stage in outcome.stages] == ["primary", "deterministic_reformulation"]
    assert not any(kind == "alternate" for kind, _ in calls)
    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].authority == "candidate"
    assert outcome.candidates[0].provenance


@pytest.mark.asyncio
async def test_controller_records_timeout_and_search_error_diagnostics():
    async def timeout_search(query, scope, limit):
        await asyncio.sleep(0.05)
        return []

    timeout = await RecoveryController(
        config=RecoveryConfig(timeout_seconds=0.01), memory_search=timeout_search
    ).recover("How do I configure export?", primary_results=[])
    assert any(stage.error_category == "timeout" for stage in timeout.stages)

    async def error_search(query, scope, limit):
        raise LookupError("search unavailable")

    errored = await RecoveryController(memory_search=error_search).recover("How do I configure export?", primary_results=[])
    assert any(stage.error_category == "LookupError" for stage in errored.stages)


@pytest.mark.asyncio
async def test_controller_propagates_cancellation():
    started = asyncio.Event()

    async def blocked_search(query, scope, limit):
        started.set()
        await asyncio.sleep(10)
        return []

    task = asyncio.create_task(RecoveryController(memory_search=blocked_search).recover("How do I configure export?"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_response_redaction_byte_bound_and_unsupported_marker():
    scope = ScopeSnapshot.from_baseline(user_id="u", requested_project_id=None, resolved_project_id=None, retrieval_mode="face")
    outcome = RecoveryController().config
    assert outcome.max_response_bytes == 65_536
    # A completed outcome with an internal exact query must expose only a label/hash.
    from weft.retrieval_recovery import RetrievalStage, RecoveryOutcome
    stage = RetrievalStage(stage="primary", query_label="secret", query_hash="0" * 64,
                           exact_queries=("password=do-not-leak",), coverage={"x": "y"})
    public = RecoveryOutcome(attempted=True, scope=scope, stages=(stage,)).to_public_dict(max_bytes=2048)
    encoded = json.dumps(public).encode()
    assert len(encoded) <= 2048
    assert "do-not-leak" not in encoded.decode()
    marker = unsupported_recovery()
    assert marker["supported"] is False and marker["not_attempted"] is True
