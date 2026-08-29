from __future__ import annotations

from datetime import datetime, timezone

from weft.compositional_recall import (
    AnswerRequest,
    AnswerStatus,
    ComposedAnswer,
    adapt_recovery_to_answer,
    build_plan,
    project_recovery_evidence,
)
from weft.retrieval_recovery import RecoveryCandidate, RecoveryOutcome, ScopeSnapshot


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def _candidate(
    stable_id: str,
    *,
    branch: str,
    content: str,
    stage: str = "deterministic_reformulation",
    project_id: str = "weft",
) -> RecoveryCandidate:
    kind, identifier = stable_id.split(":", 1)
    kwargs = {
        "stable_id": stable_id,
        "stage": stage,
        "query_label": "focused recovery query",
        "query_hash": "a" * 64,
        "branch": branch,
        "source_mode": "face",
        "content": content,
        "user_id": "u",
        "project_id": project_id,
        "occurred_at": NOW,
        "supersession": "current",
        "provenance": (stage, "a" * 64),
    }
    if kind == "memory":
        kwargs["memory_id"] = identifier
    elif kind == "turn":
        kwargs["turn_id"] = identifier
    else:
        raise AssertionError(f"unsupported fixture kind: {kind}")
    return RecoveryCandidate(**kwargs)


def _outcome(request: AnswerRequest, candidates: tuple[RecoveryCandidate, ...], **updates: object) -> RecoveryOutcome:
    scope = ScopeSnapshot.from_baseline(
        user_id=request.user_id,
        requested_project_id=request.project_id,
        resolved_project_id=request.project_id,
        retrieval_mode=request.retrieval_mode,
        as_of=request.as_of,
    )
    return RecoveryOutcome(scope=scope, candidates=candidates, **updates)


def test_one_sided_comparison_remains_incomplete_after_recovery_projection() -> None:
    request = AnswerRequest(
        question="Who was first between Alpha and Beta?",
        user_id="u",
        project_id="weft",
    )
    _, plan = build_plan(request)
    answer = ComposedAnswer(
        question=request.question,
        normalized_question=plan.normalized_question,
        shape=plan.shape,
        operation=plan.operation,
        status=AnswerStatus.incomplete,
        plan=plan,
        completeness="incomplete_evidence",
        evidence_status="incomplete",
        scope={"user_id": "u", "project_id": "weft"},
        retrieval_mode="face",
        incomplete_reason="mandatory operand evidence missing or irrelevant",
    )
    outcome = _outcome(
        request,
        (_candidate("memory:alpha", branch="branch-1", content="Alpha happened first."),),
        retrieval_status="incomplete",
        answerability="insufficient_evidence",
    )

    projected = adapt_recovery_to_answer(request, answer, outcome)

    assert projected.status is AnswerStatus.incomplete
    assert projected.incomplete_reason == "mandatory operand evidence missing or irrelevant"
    assert len(projected.evidence) == 1
    assert projected.evidence[0].authority == "candidate"
    assert projected.cited_evidence_ids == ()


def test_missing_temporal_anchor_remains_incomplete() -> None:
    request = AnswerRequest(
        question="Who was first between Demo and Retro?",
        user_id="u",
        project_id="weft",
    )
    _, plan = build_plan(request)
    answer = ComposedAnswer(
        question=request.question,
        normalized_question=plan.normalized_question,
        shape=plan.shape,
        operation=plan.operation,
        status=AnswerStatus.incomplete,
        plan=plan,
        completeness="incomplete_evidence",
        evidence_status="incomplete",
        scope={"user_id": "u", "project_id": "weft"},
        retrieval_mode="face",
        incomplete_reason="mandatory operand evidence missing or irrelevant",
    )
    outcome = _outcome(
        request,
        (_candidate("turn:demo", branch="branch-1", content="The demo happened on 2026-08-01."),),
        retrieval_status="incomplete",
        answerability="insufficient_evidence",
    )

    projected = adapt_recovery_to_answer(request, answer, outcome)

    assert projected.status is AnswerStatus.incomplete
    assert projected.typed_result is None
    assert projected.cited_evidence_ids == ()
    assert projected.branch_results["recovery"]["recovered_candidate_count"] == 1


def test_identifier_candidate_projection_remains_evidence_only() -> None:
    request = AnswerRequest(
        question="Where is pilot_config.yaml configured?",
        user_id="u",
        project_id="weft",
    )
    _, plan = build_plan(request)
    outcome = _outcome(
        request,
        (_candidate("memory:distractor", branch="branch-direct", content="The pilot launch date is not disclosed."),),
    )

    evidence, metadata = project_recovery_evidence(request, plan, outcome)

    assert len(evidence) == 1
    assert evidence[0].authority == "candidate"
    assert metadata["recovered_candidate_count"] == 1


def test_recovery_rejects_candidate_owned_by_different_user_in_face_mode() -> None:
    request = AnswerRequest(
        question="Where are provider keys configured?",
        user_id="u",
        project_id="weft",
        retrieval_mode="face",
    )
    _, plan = build_plan(request)
    outcome = _outcome(
        request,
        (_candidate("memory:foreign-user", branch="branch-direct", content="Use ~/.weft/.env.").model_copy(update={"user_id": "other"}),),
    )

    evidence, metadata = project_recovery_evidence(request, plan, outcome)

    assert evidence == ()
    assert metadata["recovered_candidate_count"] == 0


def test_recovery_preserves_answer_contract_and_rejects_scope_drift() -> None:
    request = AnswerRequest(
        question="Where are provider keys configured?",
        user_id="u",
        project_id="weft",
        as_of=NOW,
        retrieval_mode="code",
    )
    _, plan = build_plan(request)
    answer = ComposedAnswer(
        question=request.question,
        normalized_question=plan.normalized_question,
        shape=plan.shape,
        operation=plan.operation,
        status=AnswerStatus.incomplete,
        plan=plan,
        completeness="incomplete_evidence",
        evidence_status="incomplete",
        scope={"user_id": "u", "project_id": "weft", "as_of": NOW.isoformat()},
        retrieval_mode="face",
    )
    outcome = _outcome(
        request,
        (_candidate("memory:other-project", branch="branch-direct", content="Use ~/.weft/.env.", project_id="other", stage="primary"),),
    )

    evidence, metadata = project_recovery_evidence(request, plan, outcome)

    assert evidence == ()
    # Candidate project metadata is not allowed to widen the authenticated request scope.
    assert metadata["recovered_candidate_count"] == 0
    assert answer.plan.schema_version == 1
    assert answer.cited_evidence_ids == ()
