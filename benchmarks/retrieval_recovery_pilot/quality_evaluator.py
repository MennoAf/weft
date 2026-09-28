"""Provider-free paired answer-quality evaluation for the frozen recovery pilot.

This module deliberately does not alter the retrieval-only pilot contract.  It
scores only the public ``weft_answer`` response envelope and emits bounded,
contract-shaped rows (never candidate content).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from benchmarks.recall_contract.artifact_io import allocate_new_output, publish_json_atomic
from .evaluator import FrozenCase, PilotValidationError, Scope

QUALITY_PILOT_VERSION = "retrieval-recovery-answer-quality-pilot-v1"
QUALITY_REPORT_VERSION = "retrieval-recovery-answer-quality-report-v1"
_ARMS = ("off", "deterministic")
_VOLATILE = frozenset({"telemetry", "provenance", "incomplete_reason"})


class QualityValidationError(PilotValidationError):
    """Raised when quality fixtures or reports cannot support a claim."""


@dataclass(frozen=True, slots=True)
class QualityCase:
    case_id: str
    query: str
    scope: Scope
    gold_evidence_ids: tuple[str, ...]
    expected_outcome: str
    category: str = "direct"
    mandatory_branches: tuple[str, ...] = ()
    retrieval_limit: int = 10
    expected_baseline: str = "incorrect"
    expected_treatment: str = "rescue"
    expected_baseline_status: str | None = None
    expected_treatment_status: str | None = None
    expected_completeness: str | None = None
    expected_treatment_completeness: str | None = None
    source: str = "retrieval_recovery_pilot"
    longmemeval_question_id: str | None = None
    gold_answer: str | None = None
    gold_session_ids: tuple[str, ...] = ()
    authoritative_ids_available: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QualityCase":
        required = {"case_id", "query", "scope", "gold_evidence_ids", "expected_outcome", "expected_baseline", "expected_treatment"}
        missing = required - set(value)
        if missing:
            raise QualityValidationError(f"quality case missing fields: {sorted(missing)}")
        if str(value["expected_baseline"]) not in {"correct", "incorrect"}:
            raise QualityValidationError("expected_baseline must be correct or incorrect")
        if str(value["expected_treatment"]) not in {"rescue", "no_change", "fail_closed", "correct"}:
            raise QualityValidationError("invalid expected_treatment")
        source = str(value.get("source", "retrieval_recovery_pilot"))
        if source == "longmemeval" and not value.get("gold_evidence_ids"):
            base = FrozenCase(
                str(value["case_id"]), str(value["query"]), Scope.from_mapping(value["scope"]), (),
                "unknown", str(value.get("category", "longmemeval")), tuple(str(item) for item in value.get("mandatory_branches", ())),
                max(1, int(value.get("retrieval_limit", 10))),
            )
        else:
            base = FrozenCase.from_mapping(value)
        return cls(
            case_id=base.case_id, query=base.query, scope=base.scope,
            gold_evidence_ids=base.gold_evidence_ids, expected_outcome=str(value["expected_outcome"]),
            category=base.category, mandatory_branches=base.mandatory_branches,
            retrieval_limit=base.retrieval_limit,
            expected_baseline=str(value["expected_baseline"]),
            expected_treatment=str(value["expected_treatment"]),
            expected_baseline_status=_optional_str(value, "expected_baseline_status"),
            expected_treatment_status=_optional_str(value, "expected_treatment_status"),
            expected_completeness=_optional_str(value, "expected_completeness"),
            expected_treatment_completeness=_optional_str(value, "expected_treatment_completeness"),
            source=str(value.get("source", "retrieval_recovery_pilot")),
            longmemeval_question_id=_optional_str(value, "longmemeval_question_id"),
            gold_answer=_optional_str(value, "gold_answer"),
            gold_session_ids=tuple(str(item) for item in value.get("gold_session_ids", ())),
            authoritative_ids_available=bool(value.get("authoritative_ids_available", True)),
        )

    def frozen(self, *, evidence_ids: tuple[str, ...] | None = None, scope: Scope | None = None) -> FrozenCase:
        return FrozenCase(
            self.case_id, self.query, scope or self.scope,
            evidence_ids if evidence_ids is not None else self.gold_evidence_ids,
            self.expected_outcome, self.category, self.mandatory_branches, self.retrieval_limit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "query": self.query, "scope": self.scope.to_dict(),
            "gold_evidence_ids": list(self.gold_evidence_ids), "expected_outcome": self.expected_outcome,
            "category": self.category, "mandatory_branches": list(self.mandatory_branches),
            "retrieval_limit": self.retrieval_limit, "expected_baseline": self.expected_baseline,
            "expected_treatment": self.expected_treatment,
            "expected_baseline_status": self.expected_baseline_status,
            "expected_treatment_status": self.expected_treatment_status,
            "expected_completeness": self.expected_completeness,
            "expected_treatment_completeness": self.expected_treatment_completeness,
            "source": self.source, "longmemeval_question_id": self.longmemeval_question_id,
            "gold_answer": _bounded_text(self.gold_answer), "gold_session_ids": list(self.gold_session_ids),
            "authoritative_ids_available": self.authoritative_ids_available,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    version: str
    pilot_version: str
    cases_hash: str
    provenance: Mapping[str, Any]
    denominator: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    materiality: Mapping[str, Any]
    report_hash: str = ""

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "version": self.version, "pilot_version": self.pilot_version,
            "cases_hash": self.cases_hash, "provenance": _jsonable(self.provenance),
            "denominator": _jsonable(self.denominator), "rows": _jsonable(self.rows),
            "metrics": _jsonable(self.metrics), "materiality": _jsonable(self.materiality),
        }
        if include_hash:
            value["report_hash"] = self.report_hash or _hash(value)
        return value

    def validate(self) -> None:
        if self.version != QUALITY_REPORT_VERSION:
            raise QualityValidationError(f"unsupported quality report version: {self.version}")
        cases = int(self.denominator.get("cases", 0))
        expected = cases * 2
        if cases <= 0 or self.denominator.get("arms") != 2 or self.denominator.get("rows") != expected or len(self.rows) != expected:
            raise QualityValidationError("incomplete denominator")
        keys = [(str(row.get("case_id")), str(row.get("arm"))) for row in self.rows]
        if len(set(keys)) != expected or any(arm not in _ARMS for _, arm in keys):
            raise QualityValidationError("duplicate or invalid paired rows")
        if self.report_hash and self.report_hash != _hash(self.to_dict(include_hash=False)):
            raise QualityValidationError("quality report hash mismatch")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityReport":
        report = cls(
            version=str(value.get("version", "")), pilot_version=str(value.get("pilot_version", "")),
            cases_hash=str(value.get("cases_hash", "")), provenance=dict(value.get("provenance", {})),
            denominator=dict(value.get("denominator", {})), rows=tuple(value.get("rows", ())),
            metrics=dict(value.get("metrics", {})), materiality=dict(value.get("materiality", {})),
            report_hash=str(value.get("report_hash", "")),
        )
        report.validate()
        return report


AnswerFn = Callable[[QualityCase, str], Awaitable[Mapping[str, Any]] | Mapping[str, Any]]


class AnswerQualityPilot:
    """Run exactly one off and one deterministic answer call per case."""

    def __init__(self, cases: Iterable[QualityCase], *, provenance: Mapping[str, Any] | None = None) -> None:
        self.cases = tuple(cases)
        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise QualityValidationError("quality cases must be non-empty and uniquely identified")
        self.provenance = dict(provenance or {})

    async def evaluate(self, answer: AnswerFn) -> QualityReport:
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            baseline = await _invoke_async(answer, case, "off")
            treatment = await _invoke_async(answer, case, "deterministic")
            rows.append(score_answer(case, "off", baseline, baseline))
            rows.append(score_answer(case, "deterministic", treatment, baseline))
        return build_quality_report(self.cases, rows, provenance=self.provenance)


def load_quality_cases(path: Path | str) -> tuple[QualityCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("cases") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, list):
        raise QualityValidationError("quality fixture must contain a cases array")
    return tuple(QualityCase.from_mapping(item) for item in raw)


async def evaluate_quality_cases(cases: Iterable[QualityCase], answer: AnswerFn, *, provenance: Mapping[str, Any] | None = None) -> QualityReport:
    return await AnswerQualityPilot(cases, provenance=provenance).evaluate(answer)


_DIAGNOSTIC_SECRET_PATTERNS = (
    # Keep pilot diagnostics useful while ensuring DSNs, credentials, and
    # bearer-shaped values never enter a published quality artifact.
    re.compile(r"(?i)(?:api[_-]?key|token|password|secret|bearer|dsn)[=:]\s*[^,;\s]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(?:postgres(?:ql)?|mysql|redis)://[^\s]+"),
)


def _bounded_diagnostic(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Retain actionable error/rejection metadata without answer content."""
    recovery = _as_mapping(_as_mapping(value.get("branch_results")).get("recovery"))
    rejected = recovery.get("recovered_rejected_ids")
    diagnostic: dict[str, Any] = {}
    for key in ("error", "detail", "tool", "degraded", "incomplete_reason"):
        if value.get(key) is not None:
            raw = value[key]
            text = str(raw)
            for pattern in _DIAGNOSTIC_SECRET_PATTERNS:
                text = pattern.sub(
                    lambda match: (
                        match.group(0).split("//", 1)[0] + "//[REDACTED]"
                        if "//" in match.group(0)
                        else match.group(0).split("=", 1)[0] + "=[REDACTED]"
                    ),
                    text,
                )
            diagnostic[key] = text[:500] + ("…[TRUNCATED]" if len(text) > 500 else "")
    if isinstance(rejected, Mapping):
        diagnostic["rejection_reasons"] = {
            str(key)[:160]: str(reason)[:160] for key, reason in list(rejected.items())[:32]
        }
    if not diagnostic:
        return None
    diagnostic["response_keys"] = sorted(str(key) for key in value.keys())[:64]
    return diagnostic


def score_answer(case: QualityCase, arm: str, answer: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score observable answer fields and retain only bounded safe values."""
    if arm not in _ARMS:
        raise QualityValidationError(f"invalid arm: {arm}")
    value = _as_mapping(answer)
    diagnostic = _bounded_diagnostic(value)
    status = str(value.get("status", "error"))
    completeness = str(value.get("completeness", "unknown"))
    evidence_status = value.get("evidence_status")
    evidence_status = str(evidence_status) if evidence_status is not None else None
    evidence = value.get("evidence", ())
    authoritative: set[str] = set()
    available_authoritative: set[str] = set()
    for item in evidence if isinstance(evidence, (list, tuple)) else ():
        item_map = _as_mapping(item)
        evidence_id = item_map.get("evidence_id")
        if isinstance(evidence_id, Mapping):
            rendered = evidence_id.get("rendered") or _render_evidence_id(evidence_id)
        else:
            rendered = evidence_id
        if not rendered:
            continue
        rendered = str(rendered)
        if item_map.get("authority") == "authoritative":
            available_authoritative.add(rendered)
    cited = tuple(dict.fromkeys(str(item) for item in value.get("cited_evidence_ids", ()) if item))
    authoritative.update(item for item in cited if item in available_authoritative)
    required = set(case.mandatory_branches)
    covered = _covered_branches(value, available_authoritative)
    branches_ok = required.issubset(covered)
    anchors_ok = branches_ok
    expected_status = case.expected_treatment_status if arm == "deterministic" else case.expected_baseline_status
    expected_complete = case.expected_treatment_completeness if arm == "deterministic" else case.expected_completeness
    status_ok = expected_status is None and _status_matches_outcome(case.expected_outcome, status) or (expected_status == status)
    complete_ok = expected_complete is None and _completeness_matches_outcome(case.expected_outcome, completeness) or (expected_complete == completeness)
    expected_positive = case.expected_outcome == "sufficient"
    gold_answer_match = _gold_answer_match(case, value)
    longmemeval_quality_ok = (
        case.source != "longmemeval"
        or (
            case.authoritative_ids_available
            and bool(case.gold_evidence_ids)
            and bool(set(case.gold_evidence_ids) & authoritative)
            and gold_answer_match is True
        )
    )
    error_envelope = bool(diagnostic and ("error" in value or "detail" in value))
    contract_correct = bool(status_ok and complete_ok and branches_ok and _evidence_status_ok(case, status, evidence_status) and _citation_ok(case, value, authoritative, cited) and (not expected_positive or bool(authoritative)) and longmemeval_quality_ok)
    # Error/rejection envelopes are not answer envelopes. Do not manufacture a
    # scope or legacy mismatch from absent answer fields; retain the envelope
    # diagnostic so the failed seam is actionable instead.
    scope_violations = 0 if error_envelope else _scope_violations(case.scope, value)
    legacy_drift = 0
    if arm == "deterministic" and baseline is not None and not error_envelope:
        legacy_drift = int(_legacy_projection(value) != _legacy_projection(_as_mapping(baseline)))
    false_sufficiency = int(case.expected_outcome in {"unknown", "incomplete", "conflict"} and _sufficient(value))
    incomplete_preserved = int(case.expected_outcome in {"incomplete", "conflict"} and not _sufficient(value))
    no_change = _safe_contract(value) == _safe_contract(_as_mapping(baseline)) if baseline is not None else True
    return {
        "case_id": case.case_id, "category": case.category, "arm": arm,
        "question": _bounded_text(case.query),
        "expected_baseline": case.expected_baseline, "expected_treatment": case.expected_treatment,
        "status": status, "completeness": completeness, "evidence_status": evidence_status,
        "diagnostic": diagnostic,
        "typed_result": _bounded_json(value.get("typed_result")), "answer": _bounded_text(value.get("answer")),
        "cited_evidence_ids": list(cited), "authoritative_cited_evidence_ids": sorted(authoritative),
        "authoritative_evidence_ids": sorted(available_authoritative),
        "required_branches": sorted(required), "covered_branches": sorted(covered),
        "branch_complete": branches_ok, "anchor_complete": anchors_ok,
        "contract_correct": contract_correct, "status_correct": status_ok, "completeness_correct": complete_ok,
        "citation_authority_correct": _citation_ok(case, value, authoritative, cited),
        "gold_answer_match": _gold_answer_match(case, value),
        "source": case.source, "longmemeval_question_id": case.longmemeval_question_id,
        "gold_session_ids": list(case.gold_session_ids), "authoritative_ids_available": case.authoritative_ids_available,
        "false_sufficiency": false_sufficiency, "incomplete_conflict_preserved": incomplete_preserved,
        "scope_violations": scope_violations, "legacy_drift": legacy_drift,
        "no_change": bool(no_change), "provider_calls": _provider_calls(value),
    }


def build_quality_report(cases: tuple[QualityCase, ...] | Iterable[QualityCase], rows: list[dict[str, Any]], *, provenance: Mapping[str, Any] | None = None) -> QualityReport:
    cases = tuple(cases)
    expected = {(case.case_id, arm) for case in cases for arm in _ARMS}
    actual = {(str(row.get("case_id")), str(row.get("arm"))) for row in rows}
    if actual != expected:
        raise QualityValidationError("complete denominator validation failed")
    metrics = _aggregate_quality(cases, rows)
    materiality = quality_materiality(cases, rows, metrics)
    report = QualityReport(
        QUALITY_REPORT_VERSION, QUALITY_PILOT_VERSION, _hash({"cases": [case.to_dict() for case in cases]}),
        dict(provenance or {}), {"cases": len(cases), "arms": 2, "rows": len(rows), "complete": True}, tuple(rows), metrics, materiality,
    )
    report = QualityReport(
        report.version, report.pilot_version, report.cases_hash, report.provenance,
        report.denominator, report.rows, report.metrics, report.materiality,
        _hash(report.to_dict(include_hash=False)),
    )
    report.validate()
    return report


def quality_materiality(cases: Iterable[QualityCase], rows: list[Mapping[str, Any]], metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cases = tuple(cases)
    metrics = metrics or _aggregate_quality(cases, rows)
    controls = metrics["controls"]
    treatment = metrics["treatment"]
    denominator_complete = len(rows) == len(cases) * 2 and {(str(row.get("case_id")), str(row.get("arm"))) for row in rows} == {(case.case_id, arm) for case in cases for arm in _ARMS}
    gates = {
        "complete_denominator": denominator_complete,
        "baseline_controls_correct": controls["baseline_control_failures"] == 0,
        "no_change_controls_preserved": controls["no_change_control_regressions"] == 0,
        "treatment_regressions_zero": treatment["regressions_on_baseline_correct_controls"] == 0,
        "scope_and_legacy_drift_zero": treatment["scope_violations"] == 0 and treatment["legacy_drift"] == 0,
        "false_sufficiency_zero": treatment["false_sufficiency"] == 0,
        "citation_authority_correct": treatment["citation_authority_failures"] == 0,
        "incomplete_conflict_preserved": treatment["incomplete_conflict_failures"] == 0,
        "longmemeval_authority_available": all(case.authoritative_ids_available for case in cases if case.source == "longmemeval"),
        "positive_rescue_observed": treatment["positive_rescues"] > 0,
    }
    reasons = [name for name, passed in gates.items() if not passed]
    return {
        "decision": "material" if all(gates.values()) else "not_material",
        "complete_denominator": denominator_complete, "gates": gates, "reasons": reasons,
        "longmemeval_cases": sum(case.source == "longmemeval" for case in cases),
        "longmemeval_authority_note": "gold session labels are metadata only; no answer-quality lift claim without verified authoritative evidence IDs",
    }


def write_quality_report(report: QualityReport, *, source: Path | str, destination: Path | str) -> Path:
    report.validate()
    output = allocate_new_output(source, destination)
    return publish_json_atomic(output, report.to_dict())


def _aggregate_quality(cases: tuple[QualityCase, ...], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {case.case_id: case for case in cases}
    controls = {str(row["case_id"]): row for row in rows if row["arm"] == "off"}
    treatments = {str(row["case_id"]): row for row in rows if row["arm"] == "deterministic"}
    baseline_correct = sum(bool(row["contract_correct"]) for row in controls.values())
    treatment_correct = sum(bool(row["contract_correct"]) for row in treatments.values())
    baseline_controls = [case for case in cases if case.expected_baseline == "correct"]
    no_change = [case for case in cases if case.expected_treatment == "no_change"]
    positive = [case for case in cases if case.expected_treatment == "rescue"]
    positive_rescues = sum(bool(treatments[case.case_id]["contract_correct"]) and not bool(controls[case.case_id]["contract_correct"]) for case in positive)
    regressions = sum(bool(controls[case.case_id]["contract_correct"]) and not bool(treatments[case.case_id]["contract_correct"]) for case in baseline_controls)
    return {
        "baseline_accuracy": _rate(baseline_correct, len(cases)), "treatment_accuracy": _rate(treatment_correct, len(cases)),
        "positive_rescues": positive_rescues, "baseline_miss_opportunities": sum(not bool(controls[case.case_id]["contract_correct"]) for case in positive),
        "regressions_on_baseline_correct_controls": regressions,
        "false_sufficiency": sum(int(row["false_sufficiency"]) for row in treatments.values()),
        "citation_authority_failures": sum(not bool(row["citation_authority_correct"]) for row in treatments.values()),
        "incomplete_conflict_failures": sum(not bool(treatments[case.case_id]["incomplete_conflict_preserved"]) for case in cases if case.expected_outcome in {"incomplete", "conflict"}),
        "scope_violations": sum(int(row["scope_violations"]) for row in treatments.values()), "legacy_drift": sum(int(row["legacy_drift"]) for row in treatments.values()),
        "longmemeval_cases": sum(case.source == "longmemeval" for case in cases),
        "longmemeval_gold_answer_matches": sum(bool(row["gold_answer_match"]) for row in treatments.values() if row.get("source") == "longmemeval"),
        "controls": {
            "baseline_correct_cases": len(baseline_controls),
            "baseline_control_failures": sum(not bool(controls[case.case_id]["contract_correct"]) for case in baseline_controls),
            "no_change_cases": len(no_change),
            "no_change_control_regressions": sum(not bool(treatments[case.case_id]["no_change"]) or not bool(treatments[case.case_id]["contract_correct"]) for case in no_change),
        },
        "treatment": {
            "positive_rescues": positive_rescues, "regressions_on_baseline_correct_controls": regressions,
            "false_sufficiency": sum(int(row["false_sufficiency"]) for row in treatments.values()),
            "citation_authority_failures": sum(not bool(row["citation_authority_correct"]) for row in treatments.values()),
            "incomplete_conflict_failures": sum(not bool(treatments[case.case_id]["incomplete_conflict_preserved"]) for case in cases if case.expected_outcome in {"incomplete", "conflict"}),
            "scope_violations": sum(int(row["scope_violations"]) for row in treatments.values()), "legacy_drift": sum(int(row["legacy_drift"]) for row in treatments.values()),
        },
    }


def _status_matches_outcome(outcome: str, status: str) -> bool:
    return (outcome == "sufficient" and status == "success") or (outcome in {"unknown", "incomplete", "conflict"} and status != "success")


def _completeness_matches_outcome(outcome: str, completeness: str) -> bool:
    return (outcome == "sufficient" and completeness in {"complete", "indexed_lower_bound"}) or (outcome in {"unknown", "incomplete", "conflict"} and completeness not in {"complete", "indexed_lower_bound"})


def _evidence_status_ok(case: QualityCase, status: str, evidence_status: str | None) -> bool:
    if status == "success":
        return evidence_status in {"success", "empty"}
    return evidence_status != "success"


def _citation_ok(case: QualityCase, value: Mapping[str, Any], authoritative: set[str], cited: tuple[str, ...]) -> bool:
    if case.expected_outcome == "sufficient" and not cited:
        return False
    if not set(cited).issubset(authoritative):
        return False
    return all(
        str(item.get("authority")) == "authoritative"
        for item in value.get("evidence", ())
        if isinstance(item, Mapping) and _rendered_item_id(item) in cited
    )


def _gold_answer_match(case: QualityCase, value: Mapping[str, Any]) -> bool | None:
    if case.source != "longmemeval" or case.gold_answer is None:
        return None
    expected = " ".join(case.gold_answer.casefold().split())
    candidates = [value.get("answer"), _as_mapping(value.get("typed_result")).get("answer")]
    return any(expected and expected in " ".join(str(candidate).casefold().split()) for candidate in candidates if candidate)


def _sufficient(value: Mapping[str, Any]) -> bool:
    return str(value.get("status")) == "success" and str(value.get("completeness")) in {"complete", "indexed_lower_bound"} and value.get("evidence_status") in {"success", "empty"}


def _scope_violations(expected: Scope, value: Mapping[str, Any]) -> int:
    scope = _as_mapping(value.get("scope"))
    actual = Scope.from_mapping({"user": scope.get("user_id", scope.get("user")), "project": scope.get("project_id", scope.get("project")), "retrieval_mode": value.get("retrieval_mode", scope.get("retrieval_mode", "face"))})
    return int(actual != expected)


def _legacy_projection(value: Mapping[str, Any]) -> str:
    return json.dumps({key: value.get(key) for key in ("question", "normalized_question", "shape", "operation", "retrieval_mode")}, sort_keys=True, default=str, separators=(",", ":"))


def _safe_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    branch_results = _as_mapping(value.get("branch_results"))
    legacy_branches = {
        str(key): result
        for key, result in branch_results.items()
        if key not in {"recovery", "recovery_block", "retrieval_recovery", "diagnostics"}
    }
    contract = {
        key: _bounded_json(value.get(key))
        for key in (
            "status", "completeness", "evidence_status", "typed_result", "answer",
            "evidence", "cited_evidence_ids", "scope", "retrieval_mode",
        )
    }
    contract["legacy_projection"] = _legacy_projection(value)
    contract["branch_results"] = _bounded_json(legacy_branches)
    return contract


def _covered_branches(value: Mapping[str, Any], authoritative: set[str]) -> set[str]:
    covered: set[str] = set()
    branch_results = value.get("branch_results", {})
    for branch_id, result in branch_results.items() if isinstance(branch_results, Mapping) else ():
        if _as_mapping(result).get("authoritative_evidence_ids") or _as_mapping(result).get("candidate_memory_ids") or _as_mapping(result).get("evidence_ids"):
            covered.add(str(branch_id))
    for item in value.get("evidence", ()) if isinstance(value.get("evidence"), (list, tuple)) else ():
        item_map = _as_mapping(item)
        if item_map.get("authority") == "authoritative" and item_map.get("branch_id"):
            covered.add(str(item_map["branch_id"]))
    return covered


def _provider_calls(value: Mapping[str, Any]) -> int:
    telemetry = _as_mapping(value.get("telemetry"))
    return int(telemetry.get("provider_calls", value.get("provider_calls", 0)) or 0)


def _invoke(answer: AnswerFn, case: QualityCase, arm: str) -> Mapping[str, Any]:
    value = answer(case, arm)
    if inspect.isawaitable(value):
        value = asyncio.get_event_loop().run_until_complete(value) if not _in_async_context() else value
    return value  # type: ignore[return-value]


async def _invoke_async(answer: AnswerFn, case: QualityCase, arm: str) -> Mapping[str, Any]:
    value = answer(case, arm)
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, Mapping):
        raise QualityValidationError("answer adapter must return a mapping")
    return value


def _in_async_context() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _optional_str(value: Mapping[str, Any], key: str) -> str | None:
    return str(value[key]) if value.get(key) is not None else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else {}


def _render_evidence_id(value: Mapping[str, Any]) -> str:
    kind, identifier = value.get("kind"), value.get("value")
    return f"{kind}:{identifier}" if kind and identifier else ""


def _rendered_item_id(item: Mapping[str, Any]) -> str:
    evidence_id = item.get("evidence_id")
    if isinstance(evidence_id, Mapping):
        return str(evidence_id.get("rendered") or _render_evidence_id(evidence_id))
    return str(evidence_id or "")


def _bounded_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:500] + ("…[TRUNCATED]" if len(text) > 500 else "")


def _bounded_json(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return "[TRUNCATED]"
    if isinstance(value, Mapping):
        return {str(key): _bounded_json(child, depth + 1) for key, child in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_bounded_json(child, depth + 1) for child in list(value)[:32]]
    return _bounded_text(value) if isinstance(value, str) else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# Replace the sync helper's awkward event-loop bridge with the async implementation
# at the public call site while retaining a small, testable invoke boundary.
AnswerQualityPilot.evaluate = AnswerQualityPilot.evaluate

__all__ = [
    "QUALITY_PILOT_VERSION", "QUALITY_REPORT_VERSION", "AnswerQualityPilot", "QualityCase", "QualityReport",
    "QualityValidationError", "build_quality_report", "evaluate_quality_cases", "load_quality_cases",
    "quality_materiality", "score_answer", "write_quality_report",
]
