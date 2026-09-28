from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.retrieval_recovery_pilot.quality_run import LiveQualityAdapter, QualityMappingError, _longmemeval_preflight
import benchmarks.retrieval_recovery_pilot.quality_run as quality_run
from benchmarks.retrieval_recovery_pilot.adapter import LivePilotAdapter, SnapshotNamespace
from benchmarks.retrieval_recovery_pilot.quality_evaluator import (
    QUALITY_REPORT_VERSION,
    AnswerQualityPilot,
    QualityCase,
    QualityReport,
    QualityValidationError,
    build_quality_report,
    load_quality_cases,
    quality_materiality,
    score_answer,
)

FIXTURE = Path(__file__).parents[1] / "quality_fixtures.json"


def case(**changes: object) -> QualityCase:
    value = {
        "case_id": "quality-case", "query": "How do I configure export?",
        "scope": {"user": "u", "project": "p", "retrieval_mode": "face"},
        "gold_evidence_ids": ["memory:gold"], "expected_outcome": "sufficient",
        "mandatory_branches": ["branch-direct"], "expected_baseline": "incorrect", "expected_treatment": "rescue",
    }
    value.update(changes)
    return QualityCase.from_mapping(value)


def answer(*, status="success", completeness="complete", evidence_status="success", cited=("memory:gold",), authority="authoritative", scope=None, typed=None, answer=None, branch="branch-direct"):
    evidence = [{"evidence_id": {"kind": "memory", "value": "gold", "rendered": "memory:gold"}, "authority": authority, "branch_id": branch, "citation_text": "bounded"}]
    return {
        "status": status, "completeness": completeness, "evidence_status": evidence_status,
        "typed_result": typed or {"value": "bounded"}, "answer": answer, "evidence": evidence,
        "cited_evidence_ids": list(cited), "scope": scope or {"user_id": "u", "project_id": "p"},
        "retrieval_mode": "face", "branch_results": {branch: {"evidence_ids": ["memory:gold"]}},
        "telemetry": {"provider_calls": 0},
    }


def test_scoring_positive_rescue_and_contract_fields() -> None:
    frozen = case()
    baseline = answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())
    treatment = answer()
    row = score_answer(frozen, "deterministic", treatment, baseline)
    assert row["contract_correct"] is True
    assert row["citation_authority_correct"] is True
    assert row["branch_complete"] is True
    assert row["provider_calls"] == 0
    assert row["legacy_drift"] == 0


def test_baseline_control_regression_and_no_change_fail_closed() -> None:
    control = case(case_id="control", expected_baseline="correct", expected_treatment="no_change")
    baseline = answer()
    changed = answer(answer="changed answer")
    base_row = score_answer(control, "off", baseline, baseline)
    treatment_row = score_answer(control, "deterministic", changed, baseline)
    assert base_row["contract_correct"] is True
    assert treatment_row["no_change"] is False
    report = build_quality_report((control,), [base_row, treatment_row])
    assert report.materiality["decision"] == "not_material"
    assert "no_change_controls_preserved" in report.materiality["reasons"]


def test_no_change_ignores_additive_recovery_branch_and_diagnostics_but_checks_answer_contract() -> None:
    control = case(case_id="recovery-metadata-control", expected_baseline="correct", expected_treatment="no_change")
    baseline = answer()
    treatment = {
        **baseline,
        "branch_results": {
            **baseline["branch_results"],
            "recovery": {"recovered_authoritative_ids": ["memory:gold"]},
            "diagnostics": {"stage_count": 2},
        },
    }
    row = score_answer(control, "deterministic", treatment, baseline)
    assert row["no_change"] is True
    assert row["legacy_drift"] == 0

    changed_answer = {**treatment, "answer": "changed answer"}
    changed_row = score_answer(control, "deterministic", changed_answer, baseline)
    assert changed_row["no_change"] is False

    changed_legacy_field = {**treatment, "shape": "compare"}
    legacy_row = score_answer(control, "deterministic", changed_legacy_field, baseline)
    assert legacy_row["no_change"] is False


def test_unknown_conflict_false_sufficiency_and_preservation() -> None:
    unknown = case(case_id="unknown", expected_outcome="unknown", gold_evidence_ids=[], mandatory_branches=[], expected_baseline="correct", expected_treatment="no_change")
    unsafe = score_answer(unknown, "deterministic", answer(), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert unsafe["false_sufficiency"] == 1
    conflict = case(case_id="conflict", expected_outcome="conflict", expected_baseline="correct", expected_treatment="no_change", mandatory_branches=[])
    safe = score_answer(conflict, "deterministic", answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert safe["incomplete_conflict_preserved"] == 1


def test_citation_authority_violation_is_not_correct() -> None:
    frozen = case()
    row = score_answer(frozen, "deterministic", answer(authority="candidate"), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert row["citation_authority_correct"] is False
    report = build_quality_report((frozen,), [score_answer(frozen, "off", answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())), row])
    assert report.materiality["decision"] == "not_material"


def test_denominator_integrity() -> None:
    frozen = case()
    baseline = score_answer(frozen, "off", answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    with pytest.raises(QualityValidationError):
        build_quality_report((frozen,), [baseline])


def test_error_envelope_retains_bounded_redacted_diagnostic_without_fake_drift() -> None:
    frozen = case(expected_outcome="unknown", expected_baseline="correct", expected_treatment="no_change", gold_evidence_ids=[], mandatory_branches=[])
    response = {
        "error": "Database unavailable",
        "detail": "postgresql://postgres:secret@127.0.0.1:55432/weftbench password=supersecret",
        "tool": "weft_answer",
        "degraded": True,
    }
    row = score_answer(frozen, "deterministic", response, response)
    diagnostic = row["diagnostic"]
    assert row["status"] == "error"
    assert row["scope_violations"] == 0
    assert row["legacy_drift"] == 0
    assert diagnostic["error"] == "Database unavailable"
    assert "secret" not in diagnostic["detail"]
    assert "[REDACTED]" in diagnostic["detail"]
    assert diagnostic["tool"] == "weft_answer"
    assert diagnostic["response_keys"] == ["degraded", "detail", "error", "tool"]


def test_recovery_rejection_metadata_is_retained_bounded() -> None:
    frozen = case(expected_outcome="unknown", expected_baseline="correct", expected_treatment="no_change", gold_evidence_ids=[], mandatory_branches=[])
    response = {
        "status": "incomplete",
        "completeness": "incomplete_evidence",
        "evidence_status": "incomplete",
        "branch_results": {"recovery": {"recovered_rejected_ids": {"memory:bounded": "foreign_project"}}},
    }
    row = score_answer(frozen, "deterministic", response, response)
    assert row["diagnostic"]["rejection_reasons"] == {"memory:bounded": "foreign_project"}


def test_fixture_metadata_has_controls_rescues_and_negative_cases() -> None:
    cases = load_quality_cases(FIXTURE)
    assert len(cases) == 12
    assert sum(case.source == "longmemeval" for case in cases) == 4
    lme = next(case for case in cases if case.source == "longmemeval")
    assert lme.longmemeval_question_id == "e47becba"
    assert lme.gold_answer == "Business Administration"
    assert lme.gold_session_ids == ("answer_280352e9",)
    assert lme.authoritative_ids_available is False
    turn_only = next(case for case in cases if case.category == "turn_only")
    assert turn_only.scope.to_dict() == {
        "user": "fixture-user", "project": "fixture-project", "retrieval_mode": "all",
    }
    turn_answer = answer(
        cited=turn_only.gold_evidence_ids,
        scope={"user_id": "fixture-user", "project_id": "fixture-project"},
    )
    turn_answer["retrieval_mode"] = "all"
    turn_answer["evidence"] = [{
        "evidence_id": {"kind": "turn", "value": "pilot-turn-gold", "rendered": "turn:pilot-turn-gold"},
        "authority": "authoritative", "branch_id": "branch-direct", "citation_text": "bounded",
    }]
    scoped_row = score_answer(turn_only, "off", turn_answer, turn_answer)
    assert scoped_row["scope_violations"] == 0
    assert scoped_row["contract_correct"] is True

    unscoped_answer = {
        **turn_answer,
        "scope": {"user_id": "fixture-user", "project_id": None},
    }
    unscoped_row = score_answer(turn_only, "off", unscoped_answer, unscoped_answer)
    assert unscoped_row["scope_violations"] == 1
    assert sum(case.expected_baseline == "correct" for case in cases) >= 3
    assert sum(case.expected_treatment == "rescue" for case in cases) >= 2
    assert {case.expected_outcome for case in cases} >= {"unknown", "incomplete", "sufficient"}


@pytest.mark.asyncio
async def test_no_provider_answer_behavior_and_paired_calls() -> None:
    frozen = case()
    calls: list[str] = []

    async def adapter(case: QualityCase, arm: str):
        calls.append(arm)
        return answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())

    report = await AnswerQualityPilot((frozen,)).evaluate(adapter)
    assert calls == ["off", "deterministic"]
    assert all(row["provider_calls"] == 0 for row in report.rows)
    assert report.denominator == {"cases": 1, "arms": 2, "rows": 2, "complete": True}


def test_longmemeval_metadata_without_authoritative_ids_blocks_materiality() -> None:
    embedded = case(
        case_id="lme-control", source="longmemeval", longmemeval_question_id="e47becba",
        gold_answer="Business Administration", gold_session_ids=("answer_280352e9",),
        authoritative_ids_available=False, expected_outcome="unknown", expected_baseline="correct",
        expected_treatment="no_change", expected_baseline_status="incomplete", expected_treatment_status="incomplete",
        gold_evidence_ids=[], mandatory_branches=[],
    )
    baseline = answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())
    rows = [score_answer(embedded, "off", baseline, baseline), score_answer(embedded, "deterministic", baseline, baseline)]
    report = build_quality_report((embedded,), rows)
    assert report.metrics["longmemeval_cases"] == 1
    assert all(row["contract_correct"] is False for row in rows)
    assert report.materiality["gates"]["longmemeval_authority_available"] is False
    assert report.materiality["decision"] == "not_material"


def test_longmemeval_materialized_expectation_requires_authoritative_gold_and_correct_answer() -> None:
    base = case(
        case_id="lme-materialized", source="longmemeval", longmemeval_question_id="e47becba",
        gold_answer="Business Administration", gold_session_ids=("answer_280352e9",),
        authoritative_ids_available=True, expected_outcome="sufficient", expected_baseline="incorrect",
        expected_treatment="rescue", gold_evidence_ids=["memory:gold"], mandatory_branches=[],
    )
    correct = answer(answer="Business Administration")
    correct_row = score_answer(base, "deterministic", correct, answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert correct_row["contract_correct"] is True
    assert correct_row["gold_answer_match"] is True

    wrong_answer = {**correct, "answer": "A different degree"}
    wrong_row = score_answer(base, "deterministic", wrong_answer, answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert wrong_row["contract_correct"] is False
    assert wrong_row["gold_answer_match"] is False

    non_authoritative = {**correct, "evidence": [{**correct["evidence"][0], "authority": "candidate"}]}
    metadata_only_row = score_answer(base, "deterministic", non_authoritative, answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()))
    assert metadata_only_row["contract_correct"] is False


def test_materialized_longmemeval_cases_reject_unverified_gold_ids(monkeypatch) -> None:
    adapter = LiveQualityAdapter(LivePilotAdapter(object(), namespace=SnapshotNamespace("u", "p", "r")))
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    adapter.longmemeval_mapping["q"] = {
        "authoritative_ids_available": True, "gold_evidence_ids": ["memory:not-verified"],
        "verified_memory_ids": ["verified-id"], "user_id": "u", "project_id": "p-longmemeval",
    }
    with pytest.raises(QualityMappingError, match="verified authoritative mapping"):
        adapter.materialize_cases((lme,))


def test_report_round_trip_hash_and_materiality_denominator_gate() -> None:
    frozen = case()
    rows = [
        score_answer(frozen, "off", answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=()), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())),
        score_answer(frozen, "deterministic", answer(), answer(status="incomplete", completeness="incomplete_evidence", evidence_status="incomplete", cited=())),
    ]
    report = build_quality_report((frozen,), rows)
    encoded = report.to_dict()
    assert QualityReport.from_dict(json.loads(json.dumps(encoded))).report_hash == encoded["report_hash"]
    assert report.version == QUALITY_REPORT_VERSION
    assert quality_materiality((frozen,), rows)["complete_denominator"] is True


def test_longmemeval_cases_are_rejected_by_synthetic_namespace() -> None:
    adapter = LiveQualityAdapter(LivePilotAdapter(object(), namespace=SnapshotNamespace("u", "p", "r")))
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    with pytest.raises(QualityMappingError, match="synthetic"):
        adapter.materialize_cases((lme,))


def test_longmemeval_preflight_blocks_absent_snapshot(tmp_path: Path) -> None:
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(QualityMappingError, match="manifest or .complete"):
        _longmemeval_preflight((lme,), dataset=dataset, snapshot=tmp_path / "missing")


def test_longmemeval_preflight_checksum_mismatch_blocks_before_materialization(tmp_path: Path) -> None:
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / ".complete").write_text("ok\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(json.dumps({"dataset_checksum": "wrong", "questions": []}), encoding="utf-8")
    with pytest.raises(QualityMappingError, match="checksum"):
        _longmemeval_preflight((lme,), dataset=dataset, snapshot=snapshot)


def test_longmemeval_preflight_materialization_is_bounded_and_selected(tmp_path: Path) -> None:
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    dataset = tmp_path / "dataset.json"
    row = {
        "question_id": "q", "question_type": "single-session-user", "question": "What?", "answer": "yes", "question_date": "2026/01/01",
        "answer_session_ids": ["s"], "haystack_session_ids": ["s", "s"],
        "haystack_dates": ["2026/01/01", "2026/01/02"],
        "haystack_sessions": [[{"role": "user", "content": "first occurrence"}], [{"role": "user", "content": "duplicate occurrence"}]],
    }
    dataset.write_text(json.dumps([row]), encoding="utf-8")
    result = _longmemeval_preflight((lme,), dataset=dataset, snapshot=tmp_path / "missing", allow_materialize=True)
    assert result["status"] == "materialize"
    assert result["source_turns"] == 1
    assert result["question_ids"] == ["q"]
    assert result["data_quality"] == {
        "haystack_dedupe_policy": "first_occurrence_wins_at_materialization",
        "haystack_dedupe_events": [{"instance_id": "q", "session_id": "s", "occurrences": 2}],
    }
    instance = quality_run._longmemeval_instances((lme,), dataset)["q"]
    assert instance.sessions[0].turns[0].content == "first occurrence"

    row["answer_session_ids"] = ["s", "s"]
    dataset.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(QualityMappingError, match="duplicate answer_session_ids"):
        quality_run._longmemeval_instances((lme,), dataset)


@pytest.mark.asyncio
async def test_longmemeval_materialization_mapping_preserves_scope_and_provenance(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{
        "question_id": "q", "question_type": "single-session-user", "question": "What?", "answer": "yes", "question_date": "2026/01/01",
        "answer_session_ids": ["s"], "haystack_session_ids": ["s", "other"], "haystack_dates": ["2026/01/01", "2026/01/02"],
        "haystack_sessions": [[{"role": "user", "content": "yes"}], [{"role": "assistant", "content": "no"}]],
    }]), encoding="utf-8")
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    base = LivePilotAdapter(object(), namespace=SnapshotNamespace("u", "p", "r"))
    adapter = LiveQualityAdapter(base)
    stored = []

    async def fake_store(pool, create, embedding):
        memory = SimpleNamespace(id=f"m{len(stored)}", project_id=create.project_id, topic=create.topic)
        stored.append(memory)
        return memory

    async def fake_get(pool, memory_id):
        return next((item for item in stored if item.id == memory_id), None)

    class Acquire:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(quality_run, "store_memory", fake_store)
    monkeypatch.setattr(quality_run, "get_memory", fake_get)
    monkeypatch.setattr(quality_run, "acquire", lambda pool: Acquire())
    mapping = await adapter.materialize_longmemeval((lme,), dataset)
    entry = mapping["q"]
    assert entry["project_id"] == "p-longmemeval"
    assert entry["user_id"] == "u"
    assert entry["authoritative_ids_available"] is True
    assert len(entry["verified_memory_ids"]) == 2
    assert len(entry["gold_evidence_ids"]) == 1
    assert entry["source_turns"][0]["session_id"] == "s"
    assert all(any(topic.startswith("longmemeval/turn:") for topic in memory.topic) for memory in stored)
    materialized = adapter.materialize_cases((lme,))
    assert materialized[0].scope.to_dict() == {"user": "u", "project": "p-longmemeval", "retrieval_mode": "all"}
    assert materialized[0].authoritative_ids_available is True
    assert materialized[0].gold_evidence_ids == ("memory:m0",)
    assert materialized[0].expected_outcome == "sufficient"
    assert materialized[0].expected_baseline == "correct"
    assert materialized[0].expected_treatment == "no_change"


def test_longmemeval_preflight_accepts_bounded_verified_mapping(tmp_path: Path) -> None:
    lme = case(source="longmemeval", longmemeval_question_id="q", gold_session_ids=("s",), gold_evidence_ids=[])
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    import hashlib
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    (snapshot / ".complete").write_text("ok\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(json.dumps({
        "dataset_checksum": checksum,
        "questions": [{"question_id": "q", "project_id": "lme_q", "turn_ids": ["et-1"], "turn_session_map": {"et-1": "s"}}],
    }), encoding="utf-8")
    result = _longmemeval_preflight((lme,), dataset=dataset, snapshot=snapshot)
    assert result["status"] == "verified"
    assert result["question_ids"] == ["q"]
