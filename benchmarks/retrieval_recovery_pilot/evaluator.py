"""Small, provider-free paired evaluator for deterministic retrieval recovery.

This pilot scores retrieval/evidence recovery only.  It never calls an answer
composer and intentionally does not use the frozen terminal-funnel schema.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from benchmarks.recall_contract.artifact_io import allocate_new_output, publish_json_atomic

PILOT_VERSION = "retrieval-recovery-pilot-v1"
REPORT_VERSION = "retrieval-recovery-pilot-report-v1"
# These fields are retrieval telemetry, not the legacy evidence contract. The
# live seam updates them as each arm reads the same snapshot, so comparing them
# would report drift even when IDs, ordering, content, and scope are unchanged.
_VOLATILE = frozenset({
    "request_id", "latency_ms", "telemetry_timestamp", "timings",
    "accessed_at", "access_count", "usefulness_score", "relevance_score",
})
_RECOVERY_KEYS = frozenset({"recovery", "recovery_block", "retrieval_recovery"})


class PilotValidationError(ValueError):
    """Raised when a frozen pilot input cannot support a claim."""


class Arm(str, Enum):
    off = "off"
    deterministic = "deterministic"


@dataclass(frozen=True, slots=True)
class Scope:
    """The three scope dimensions that a recovery probe must preserve."""

    user: str | None = None
    project: str | None = None
    retrieval_mode: str = "face"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Scope":
        mode = str(value.get("retrieval_mode", value.get("mode", "face")))
        if mode not in {"face", "code", "all"}:
            raise PilotValidationError(f"invalid retrieval_mode: {mode}")
        return cls(
            user=_optional(value, "user", "user_id"),
            project=_optional(value, "project", "project_id", "requested_project_id", "resolved_project_id"),
            retrieval_mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"user": self.user, "project": self.project, "retrieval_mode": self.retrieval_mode}


@dataclass(frozen=True, slots=True)
class FrozenCase:
    case_id: str
    query: str
    scope: Scope
    gold_evidence_ids: tuple[str, ...]
    expected_outcome: str
    category: str = "direct"
    mandatory_branches: tuple[str, ...] = ()
    retrieval_limit: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenCase":
        required = {"case_id", "query", "scope", "gold_evidence_ids", "expected_outcome"}
        missing = required - set(value)
        if missing:
            raise PilotValidationError(f"case missing fields: {sorted(missing)}")
        case_id = str(value["case_id"])
        gold = tuple(dict.fromkeys(str(item) for item in value["gold_evidence_ids"]))
        outcome = str(value["expected_outcome"])
        if outcome not in {"sufficient", "incomplete", "unknown", "conflict"}:
            raise PilotValidationError(f"invalid expected_outcome: {outcome}")
        if not case_id or (not gold and outcome != "unknown"):
            raise PilotValidationError(
                "case_id must be non-empty and gold_evidence_ids must be non-empty "
                "unless expected_outcome is unknown"
            )
        branches = tuple(dict.fromkeys(str(item) for item in value.get("mandatory_branches", ())))
        return cls(
            case_id=case_id,
            query=str(value["query"]),
            scope=Scope.from_mapping(value["scope"]),
            gold_evidence_ids=gold,
            expected_outcome=outcome,
            category=str(value.get("category", "direct")),
            mandatory_branches=branches,
            retrieval_limit=max(1, int(value.get("retrieval_limit", 10))),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scope"] = self.scope.to_dict()
        value["gold_evidence_ids"] = list(self.gold_evidence_ids)
        value["mandatory_branches"] = list(self.mandatory_branches)
        return value


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Adapter-neutral result: legacy response plus additive recovery metadata."""

    legacy_response: Mapping[str, Any] = field(default_factory=dict)
    baseline_ids: tuple[str, ...] = ()
    recovery: Any = None
    latency_ms: float = 0.0
    cap_hits: int = 0
    error_category: str | None = None
    scope: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RecallResult":
        legacy = value.get("legacy_response", value.get("legacy", value.get("response", value)))
        if not isinstance(legacy, Mapping):
            raise PilotValidationError("legacy response must be an object")
        explicit = value.get("baseline_ids")
        ids = tuple(str(item) for item in explicit) if explicit is not None else tuple(_legacy_ids(legacy))
        recovery = value.get("recovery", value.get("recovery_block"))
        return cls(
            legacy_response=dict(legacy), baseline_ids=tuple(dict.fromkeys(ids)), recovery=recovery,
            latency_ms=max(0.0, float(value.get("latency_ms", legacy.get("latency_ms", 0.0)))),
            cap_hits=max(0, int(value.get("cap_hits", value.get("caps", 0)))),
            error_category=str(value["error_category"]) if value.get("error_category") else None,
            scope=value.get("scope") if isinstance(value.get("scope"), Mapping) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_response": _jsonable(self.legacy_response),
            "baseline_ids": list(self.baseline_ids),
            "recovery": _jsonable(self.recovery),
            "latency_ms": self.latency_ms,
            "cap_hits": self.cap_hits,
            "error_category": self.error_category,
            "scope": _jsonable(self.scope),
        }


@dataclass(frozen=True, slots=True)
class ArmMetrics:
    cases: int
    baseline_evidence_recall: float
    union_evidence_recall: float
    rescue_rate: float
    absolute_lift: float
    rescued_cases: int
    rescued_evidence_ids: int
    branch_completeness: float
    anchor_completeness: float
    non_gold_candidate_rate: float
    unknown_false_sufficiency: int
    conflict_preservation: int
    scope_violations: int
    legacy_baseline_drift: int
    latency_ms: Mapping[str, float | None]
    cap_hits: int
    errors: int


@dataclass(frozen=True, slots=True)
class PilotReport:
    version: str
    pilot_version: str
    cases_hash: str
    provenance: Mapping[str, Any]
    denominator: Mapping[str, int]
    rows: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    report_hash: str = ""

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": self.version, "pilot_version": self.pilot_version,
            "cases_hash": self.cases_hash, "provenance": _jsonable(self.provenance),
            "denominator": dict(self.denominator), "rows": [_jsonable(row) for row in self.rows],
            "metrics": _jsonable(self.metrics),
        }
        if include_hash:
            value["report_hash"] = self.report_hash or _hash(value)
        return value

    def validate(self) -> None:
        expected = int(self.denominator.get("cases", 0)) * int(self.denominator.get("arms", 0))
        if expected <= 0 or len(self.rows) != expected:
            raise PilotValidationError("incomplete denominator")
        keys = [(str(row.get("case_id")), str(row.get("arm"))) for row in self.rows]
        if len(set(keys)) != len(keys):
            raise PilotValidationError("duplicate case/arm rows")
        if self.denominator.get("rows") != expected:
            raise PilotValidationError("denominator row count mismatch")
        if self.report_hash and self.report_hash != _hash(self.to_dict(include_hash=False)):
            raise PilotValidationError("report hash mismatch")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotReport":
        report = cls(
            version=str(value.get("version", "")), pilot_version=str(value.get("pilot_version", "")),
            cases_hash=str(value.get("cases_hash", "")), provenance=dict(value.get("provenance", {})),
            denominator=dict(value.get("denominator", {})), rows=tuple(value.get("rows", ())),
            metrics=dict(value.get("metrics", {})), report_hash=str(value.get("report_hash", "")),
        )
        if report.version != REPORT_VERSION:
            raise PilotValidationError(f"unsupported report version: {report.version}")
        report.validate()
        return report


RecallFn = Callable[[FrozenCase, Arm], Awaitable[RecallResult | Mapping[str, Any]]]


class RecoveryPilot:
    """Run exactly one control and one treatment call per frozen case."""

    def __init__(self, cases: Iterable[FrozenCase], *, provenance: Mapping[str, Any] | None = None) -> None:
        self.cases = tuple(cases)
        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise PilotValidationError("cases must be non-empty and uniquely identified")
        self.provenance = dict(provenance or {})

    async def evaluate(self, recall: RecallFn) -> PilotReport:
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            control = await _invoke(recall, case, Arm.off)
            treatment = await _invoke(recall, case, Arm.deterministic)
            rows.extend((_score_row(case, Arm.off, control, control), _score_row(case, Arm.deterministic, treatment, control)))
        return _build_report(self.cases, rows, provenance=self.provenance)


def load_cases(path: Path | str) -> tuple[FrozenCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("cases") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, list):
        raise PilotValidationError("fixture must contain a cases array")
    return tuple(FrozenCase.from_mapping(item) for item in raw)


async def evaluate_cases(cases: Iterable[FrozenCase], recall: RecallFn, *, provenance: Mapping[str, Any] | None = None) -> PilotReport:
    return await RecoveryPilot(cases, provenance=provenance).evaluate(recall)


def write_report(report: PilotReport, *, source: Path | str, destination: Path | str) -> Path:
    report.validate()
    output = allocate_new_output(source, destination)
    return publish_json_atomic(output, report.to_dict())


def _score_row(case: FrozenCase, arm: Arm, result: RecallResult, control: RecallResult) -> dict[str, Any]:
    baseline = set(result.baseline_ids)
    gold = set(case.gold_evidence_ids)
    recovery_ids, candidates, recovery_map = _recovery_values(result.recovery)
    union = baseline | recovery_ids
    recovery_stages = []
    raw_stages = recovery_map.get("stages", ())
    if isinstance(raw_stages, (list, tuple)):
        for raw_stage in raw_stages:
            stage = _as_mapping(raw_stage)
            if not stage:
                continue
            recovery_stages.append({
                "stage": stage.get("stage"),
                "terminal": stage.get("terminal"),
                "query_count": stage.get("query_count", 0),
                "result_count": stage.get("result_count", 0),
                "result_ids": list(stage.get("result_ids", ())) if isinstance(stage.get("result_ids", ()), (list, tuple)) else [],
                "error_category": stage.get("error_category"),
                "coverage": _jsonable(stage.get("coverage", {})),
            })
    recovery_diagnostics = {
        "supported": recovery_map.get("supported") if recovery_map else None,
        "attempted": recovery_map.get("attempted") if recovery_map else None,
        "not_attempted": recovery_map.get("not_attempted") if recovery_map else None,
        "trigger": recovery_map.get("trigger") if recovery_map else None,
        "retrieval_status": recovery_map.get("retrieval_status") if recovery_map else None,
        "answerability": recovery_map.get("answerability") if recovery_map else None,
        "stages": recovery_stages,
    }
    coverage = _as_mapping(recovery_map.get("coverage"))
    expected_branches = set(case.mandatory_branches) or set(_value(recovery_map, "required_branches", _value(coverage, "required_branches", ())))
    covered_branches = set(_value(recovery_map, "covered_branches", _value(coverage, "covered_branches", ())))
    expected_anchors = set(_value(recovery_map, "required_anchors", _value(coverage, "required_anchors", ())))
    covered_anchors = set(_value(recovery_map, "covered_anchors", _value(coverage, "covered_anchors", ())))
    if not expected_anchors and any(item.startswith("anchor-") for item in expected_branches):
        expected_anchors = expected_branches
        covered_anchors = covered_branches
    branch_complete = not expected_branches or expected_branches.issubset(covered_branches)
    anchor_complete = not expected_anchors or expected_anchors.issubset(covered_anchors)
    scope_violations = _scope_violations(case.scope, result, candidates)
    legacy_drift = _canonical_legacy(result.legacy_response) != _canonical_legacy(control.legacy_response)
    candidate_ids = set(candidates) - baseline
    candidate_authority_violation = _candidate_authority_violation(result.recovery)
    status = str(_value(recovery_map, "retrieval_status", _value(recovery_map, "status", "incomplete")))
    answerability = str(_value(recovery_map, "answerability", "insufficient_evidence"))
    if arm is Arm.off:
        status = "off"
    recovered = (union - baseline) & gold
    complete_rescue = (
        bool(recovered)
        and case.expected_outcome == "sufficient"
        and status == "sufficient"
        and answerability == "sufficient"
        and branch_complete
        and anchor_complete
        and not scope_violations
    )
    if case.category in {"comparison", "chronology"} and not (branch_complete and anchor_complete):
        complete_rescue = False
    unknown_false = int(case.expected_outcome in {"unknown", "incomplete"} and (status == "sufficient" or answerability == "sufficient"))
    # The approved live envelope does not promise conflict classification. For
    # the conflict fixture, fail-closed incomplete/insufficient is the expected
    # safe outcome; only a sufficient claim is a preservation failure. The off
    # arm makes no recovery claim and is therefore not penalized.
    conflict_ok = int(
        arm is Arm.off
        or case.category != "conflict"
        or (
            status in {"incomplete", "conflict"}
            and answerability in {"insufficient_evidence", "conflicting_evidence"}
        )
    )
    false_sufficiency = unknown_false
    baseline_recall = _rate(len(baseline & gold), len(gold))
    union_recall = _rate(len(union & gold), len(gold))
    non_gold = candidate_ids - gold
    return {
        "case_id": case.case_id, "category": case.category, "arm": arm.value,
        "baseline_ids": sorted(baseline), "recovery_candidate_ids": sorted(recovery_ids),
        "candidate_ids": sorted(candidate_ids), "union_ids": sorted(union),
        "gold_evidence_ids": sorted(gold), "baseline_recall": baseline_recall,
        "union_recall": union_recall, "rescued_evidence_ids": sorted(recovered),
        "rescued_case": complete_rescue, "branch_complete": branch_complete,
        "anchor_complete": anchor_complete, "required_branches": sorted(expected_branches),
        "covered_branches": sorted(covered_branches), "non_gold_candidate_count": len(non_gold),
        "candidate_count": len(candidate_ids), "non_gold_candidate_rate": _rate(len(non_gold), len(candidate_ids)),
        "unknown_false_sufficiency": false_sufficiency, "conflict_preserved": conflict_ok,
        "scope_violations": scope_violations + int(candidate_authority_violation),
        "legacy_baseline_drift": int(legacy_drift), "retrieval_status": status,
        "recovery_diagnostics": recovery_diagnostics,
        "answerability": answerability, "latency_ms": result.latency_ms,
        "cap_hits": result.cap_hits, "error_category": result.error_category,
        "guard_failures": [name for name, failed in {
            "treatment_legacy_drift": arm is Arm.deterministic and legacy_drift,
            "scope_mismatch": bool(scope_violations),
            "non_additive_authority": candidate_authority_violation,
            "unknown_false_sufficiency": bool(unknown_false),
            "incomplete_branch_rescue": bool(recovered) and not (branch_complete and anchor_complete),
            "conflict_not_preserved": not bool(conflict_ok),
        }.items() if failed],
    }


def _build_report(cases: tuple[FrozenCase, ...], rows: list[dict[str, Any]], *, provenance: Mapping[str, Any]) -> PilotReport:
    by_case = {case.case_id: case for case in cases}
    expected_keys = {(case.case_id, arm.value) for case in cases for arm in Arm}
    actual_keys = {(str(row.get("case_id")), str(row.get("arm"))) for row in rows}
    if actual_keys != expected_keys:
        raise PilotValidationError("complete denominator validation failed")
    metrics = _aggregate(cases, rows)
    cases_payload = {"pilot_version": PILOT_VERSION, "cases": [case.to_dict() for case in cases]}
    report = PilotReport(
        version=REPORT_VERSION, pilot_version=PILOT_VERSION, cases_hash=_hash(cases_payload),
        provenance=dict(provenance), denominator={"cases": len(cases), "arms": len(Arm), "rows": len(rows)},
        rows=tuple(rows), metrics=metrics,
    )
    report = PilotReport(**{**asdict(report), "report_hash": _hash(report.to_dict(include_hash=False))})
    report.validate()
    return report


def _aggregate(cases: tuple[FrozenCase, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    controls = [row for row in rows if row["arm"] == Arm.off.value]
    treatments = [row for row in rows if row["arm"] == Arm.deterministic.value]
    per_category: dict[str, Any] = {}
    for category in sorted({case.category for case in cases}):
        ids = {case.case_id for case in cases if case.category == category}
        per_category[category] = _arm_metrics([row for row in treatments if row["case_id"] in ids], [row for row in controls if row["case_id"] in ids])
    return {"control": _arm_metrics(controls, controls), "treatment": _arm_metrics(treatments, controls), "per_category": per_category}


def _arm_metrics(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return asdict(ArmMetrics(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, {"p50": None, "p95": None}, 0, 0))
    baseline = _rate(sum(row["baseline_recall"] for row in rows), len(rows))
    union = _rate(sum(row["union_recall"] for row in rows), len(rows))
    eligible_misses = sum(1 for row in rows if row["gold_evidence_ids"] and row["baseline_recall"] < 1.0)
    rescued = sum(bool(row["rescued_case"]) for row in rows)
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    return asdict(ArmMetrics(
        cases=len(rows), baseline_evidence_recall=baseline,
        union_evidence_recall=union, rescue_rate=_rate(rescued, eligible_misses),
        absolute_lift=union - baseline, rescued_cases=rescued,
        rescued_evidence_ids=sum(len(row["rescued_evidence_ids"]) for row in rows),
        branch_completeness=_rate(sum(bool(row["branch_complete"]) for row in rows), len(rows)),
        anchor_completeness=_rate(sum(bool(row["anchor_complete"]) for row in rows), len(rows)),
        non_gold_candidate_rate=_rate(sum(row["non_gold_candidate_count"] for row in rows), sum(row["candidate_count"] for row in rows)),
        unknown_false_sufficiency=sum(row["unknown_false_sufficiency"] for row in rows),
        conflict_preservation=sum(row["conflict_preserved"] for row in rows),
        scope_violations=sum(row["scope_violations"] for row in rows),
        legacy_baseline_drift=sum(row["legacy_baseline_drift"] for row in rows),
        latency_ms={"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)},
        cap_hits=sum(row["cap_hits"] for row in rows), errors=sum(bool(row["error_category"]) for row in rows),
    ))


def _canonical_recovery_id(value: Any) -> str | None:
    """Normalize adapter/controller aliases to the pilot stable-ID contract."""
    raw = _id_from(value)
    if raw and raw.startswith("belief:"):
        return "memory:" + raw.split(":", 1)[1]
    return raw


def _recovery_values(recovery: Any) -> tuple[set[str], set[str], Mapping[str, Any]]:
    mapping = _as_mapping(recovery)
    if not mapping:
        return set(), set(), {}
    values: list[tuple[str, bool]] = []
    raw_candidates = mapping.get("candidates", ())
    for item in raw_candidates if isinstance(raw_candidates, (list, tuple)) else ():
        item_map = _as_mapping(item)
        value = _canonical_recovery_id(item_map or item)
        if value:
            values.append((value, True))
    stages = mapping.get("stages", ())
    for stage in stages if isinstance(stages, (list, tuple)) else ():
        stage_map = _as_mapping(stage)
        if stage_map.get("stage") == "primary":
            continue
        for value in stage_map.get("result_ids", ()) if isinstance(stage_map.get("result_ids", ()), (list, tuple)) else ():
            canonical = _canonical_recovery_id(value)
            if canonical:
                values.append((canonical, True))
    for key in ("candidate_ids",):
        for value in mapping.get(key, ()) if isinstance(mapping.get(key, ()), (list, tuple)) else ():
            canonical = _canonical_recovery_id(value)
            if canonical:
                values.append((canonical, True))
    coverage = _as_mapping(mapping.get("coverage"))
    for value in coverage.get("candidate_ids", ()) if isinstance(coverage.get("candidate_ids", ()), (list, tuple)) else ():
        canonical = _canonical_recovery_id(value)
        if canonical:
            values.append((canonical, True))
    ids = {value for value, _ in values}
    return ids, ids, mapping


def _scope_violations(expected: Scope, result: RecallResult, candidates: set[str]) -> int:
    count = 0
    recovery_scope = _as_mapping(_as_mapping(result.recovery).get("scope"))
    if recovery_scope:
        actual = Scope.from_mapping(recovery_scope)
        count += int(actual != expected)
    if result.scope:
        count += int(Scope.from_mapping(result.scope) != expected)
    raw = _as_mapping(result.recovery).get("candidates", ())
    for item in raw if isinstance(raw, (list, tuple)) else ():
        candidate = _as_mapping(item)
        if not candidate:
            continue
        mode = candidate.get("source_mode", candidate.get("retrieval_mode"))
        project = candidate.get("project_id")
        user = candidate.get("user_id")
        if mode is not None and str(mode) != expected.retrieval_mode:
            count += 1
        if project is not None and expected.project is not None and str(project) != expected.project:
            count += 1
        if user is not None and expected.user is not None and str(user) != expected.user:
            count += 1
    return count


def _candidate_authority_violation(recovery: Any) -> bool:
    mapping = _as_mapping(recovery)
    raw = mapping.get("candidates", ())
    return any(_as_mapping(item).get("authority") not in {None, "candidate"} for item in raw if _as_mapping(item))


def _legacy_ids(response: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("results", "turns"):
        values = response.get(key, ())
        for item in values if isinstance(values, list) else ():
            item_map = _as_mapping(item)
            value = _id_from(item_map or item)
            if value:
                ids.append(value)
    return list(dict.fromkeys(ids))


def _canonical_legacy(value: Mapping[str, Any]) -> str:
    def strip(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): strip(child) for key, child in sorted(item.items()) if key not in _VOLATILE and key not in _RECOVERY_KEYS}
        if isinstance(item, (list, tuple)):
            return [strip(child) for child in item]
        return item
    return json.dumps(strip(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def _id_from(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    mapping = _as_mapping(value)
    for key in ("stable_id", "memory_id", "turn_id", "claim_id", "id"):
        if mapping.get(key):
            raw = str(mapping[key])
            if key in {"memory_id", "turn_id", "claim_id"} and not raw.startswith(f"{key.removesuffix('_id')}:"):
                prefix = key.removesuffix("_id")
                return f"{prefix}:{raw}"
            return raw
    return None


def _value(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def _optional(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if value.get(key) is not None:
            return str(value[key])
    return None


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def _hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    return value


async def _invoke(recall: RecallFn, case: FrozenCase, arm: Arm) -> RecallResult:
    value = recall(case, arm)
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, RecallResult):
        return value
    if not isinstance(value, Mapping):
        raise PilotValidationError("recall adapter must return RecallResult or mapping")
    return RecallResult.from_mapping(value)


__all__ = ["PILOT_VERSION", "Arm", "ArmMetrics", "FrozenCase", "PilotReport", "PilotValidationError", "RecallResult", "RecoveryPilot", "Scope", "evaluate_cases", "load_cases", "write_report"]
