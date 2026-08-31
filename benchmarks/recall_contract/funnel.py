"""Versioned terminal-funnel data model and deterministic metric aggregation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

FUNNEL_VERSION = "terminal-funnel-v1"
REQUIRED_STAGES = (
    "activation",
    "retrieval",
    "composition",
    "answer",
    "evidence",
    "terminal",
)
TERMINAL_STATUSES = frozenset({"success", "incomplete", "setup_failure", "error"})


class FunnelValidationError(ValueError):
    """Raised when a terminal artifact cannot support a funnel claim."""


_ROW_FIELDS = frozenset({
    "terminal_key", "case_id", "arm", "execution_status", "stages", "stage_records",
    "scores", "oracle", "oracle_coverage", "authority", "budgets", "provenance",
    "source", "schema_version", "scope", "plan", "disposition", "terminal",
})


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: str
    count: int = 0
    elapsed_ms: float = 0.0
    cap_hit: bool = False
    status: str = "not_recorded"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "elapsed_ms": self.elapsed_ms,
            "cap_hit": self.cap_hit,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TerminalRow:
    terminal_key: str
    case_id: str
    arm: str
    execution_status: str
    stages: Mapping[str, str] = field(default_factory=dict)
    scores: Mapping[str, Any] = field(default_factory=dict)
    oracle: Mapping[str, Any] = field(default_factory=dict)
    authority: Mapping[str, int] = field(default_factory=dict)
    budgets: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    disposition: Mapping[str, Any] = field(default_factory=dict)
    stage_records: tuple[StageRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_records:
            object.__setattr__(self, "stage_records", default_stage_records(self.stages))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TerminalRow":
        unknown = set(row) - _ROW_FIELDS
        if unknown:
            raise FunnelValidationError(f"unknown terminal row fields: {sorted(unknown)}")
        missing = {"terminal_key", "case_id", "arm", "execution_status"} - set(row)
        if missing:
            raise FunnelValidationError(f"terminal row missing fields: {sorted(missing)}")
        status = str(row["execution_status"])
        if status not in TERMINAL_STATUSES:
            raise FunnelValidationError(f"invalid terminal status: {status}")
        stages = row.get("stages", {})
        if not isinstance(stages, Mapping):
            raise FunnelValidationError("stages must be an object")
        if set(stages) != set(REQUIRED_STAGES):
            raise FunnelValidationError("terminal row stages must contain each required stage exactly")
        raw_records = row.get("stage_records", ())
        if isinstance(raw_records, Mapping):
            raw_records = tuple(raw_records.values())
        records: list[StageRecord] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping) or "stage" not in raw:
                raise FunnelValidationError("stage_records must contain stage objects")
            records.append(StageRecord(
                stage=str(raw["stage"]),
                count=max(0, int(raw.get("count", 0))),
                elapsed_ms=max(0.0, float(raw.get("elapsed_ms", 0.0))),
                cap_hit=bool(raw.get("cap_hit", False)),
                status=str(raw.get("status", "not_recorded")),
                reason=str(raw["reason"]) if raw.get("reason") is not None else None,
            ))
        record_stages = {record.stage for record in records}
        if records and record_stages != set(REQUIRED_STAGES):
            raise FunnelValidationError("stage_records must cover each required stage exactly")
        return cls(
            terminal_key=str(row["terminal_key"]),
            case_id=str(row["case_id"]),
            arm=str(row["arm"]),
            execution_status=status,
            stages=dict(stages),
            stage_records=tuple(records),
            scores=dict(row.get("scores", {})),
            oracle=dict(row.get("oracle", row.get("oracle_coverage", {}))),
            authority={str(k): int(v) for k, v in dict(row.get("authority", {})).items()},
            budgets=dict(row.get("budgets", {})),
            provenance=dict(row.get("provenance", {})),
            scope=dict(row.get("scope", {})),
            disposition=dict(row.get("disposition", {})),
        )


@dataclass(frozen=True, slots=True)
class TerminalFunnel:
    """A complete, ordered, versioned terminal artifact collection."""

    rows: tuple[TerminalRow, ...]
    expected_terminal_keys: frozenset[str]
    version: str = FUNNEL_VERSION
    stages: tuple[str, ...] = REQUIRED_STAGES
    oracle_coverage: float = 0.0
    primary_loss: str | None = None
    authority_counts: Mapping[str, int] = field(default_factory=dict)
    budgets: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.version != FUNNEL_VERSION:
            raise FunnelValidationError(f"unsupported funnel version: {self.version}")
        if tuple(self.stages) != REQUIRED_STAGES:
            raise FunnelValidationError("required stages are missing or out of order")
        keys = [row.terminal_key for row in self.rows]
        if len(keys) != len(set(keys)):
            raise FunnelValidationError("duplicate terminal keys")
        actual = set(keys)
        expected = set(self.expected_terminal_keys)
        if actual != expected:
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            raise FunnelValidationError(f"incomplete denominator (missing={missing}, extra={extra})")
        if not 0.0 <= float(self.oracle_coverage) <= 1.0:
            raise FunnelValidationError("oracle coverage must be between 0 and 1")

    @property
    def complete_denominator(self) -> bool:
        try:
            self.validate()
        except FunnelValidationError:
            return False
        return True


def expected_terminal_keys(cases: Iterable[str], arms: Iterable[str]) -> frozenset[str]:
    """Derive the denominator from declared case and arm populations."""
    case_values = tuple(str(case) for case in cases)
    arm_values = tuple(str(arm) for arm in arms)
    if not case_values or not arm_values:
        raise FunnelValidationError("declared cases and arms must be non-empty")
    if len(set(case_values)) != len(case_values) or len(set(arm_values)) != len(arm_values):
        raise FunnelValidationError("declared cases and arms must be unique")
    return frozenset(f"{case}::{arm}" for case in case_values for arm in arm_values)


def validate_terminal_sequence(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject records after a terminal denominator marker.

    A denominator marker is a control record, not a terminal row.  It must be
    last, preventing post-denominator rows from silently changing the claim.
    """
    marker_indexes = [index for index, record in enumerate(records) if record.get("type") == "terminal_denominator"]
    if marker_indexes and marker_indexes[-1] != len(records) - 1:
        raise FunnelValidationError("records found after terminal denominator marker")
    if len(marker_indexes) > 1:
        raise FunnelValidationError("multiple terminal denominator markers")


def _stage_value(row: TerminalRow, stage: str) -> str:
    return str(row.stages[stage])


def aggregate_metrics(funnel: TerminalFunnel) -> dict[str, Any]:
    """Aggregate activation, stage, loss, authority, and budget metrics."""
    funnel.validate()
    activation = Counter(_stage_value(row, "activation") for row in funnel.rows)
    stages = {stage: dict(Counter(_stage_value(row, stage) for row in funnel.rows)) for stage in REQUIRED_STAGES}
    losses = Counter(str(row.scores.get("primary_loss", "none")) for row in funnel.rows)
    authority = Counter()
    for row in funnel.rows:
        authority.update(row.authority)
    stage_counts: dict[str, dict[str, int]] = {}
    stage_elapsed_ms: dict[str, float] = {}
    stage_caps: dict[str, int] = {}
    for row in funnel.rows:
        for record in row.stage_records:
            stage_counts.setdefault(record.stage, {})[record.status] = stage_counts.setdefault(record.stage, {}).get(record.status, 0) + record.count
            stage_elapsed_ms[record.stage] = stage_elapsed_ms.get(record.stage, 0.0) + record.elapsed_ms
            stage_caps[record.stage] = stage_caps.get(record.stage, 0) + int(record.cap_hit)
    return {
        "version": funnel.version,
        "denominator": len(funnel.rows),
        "activation": dict(activation),
        "stages": stages,
        "stage_counts": stage_counts,
        "stage_elapsed_ms": stage_elapsed_ms,
        "stage_cap_hits": stage_caps,
        "primary_loss": dict(losses),
        "authority_counts": dict(authority),
        "oracle_coverage": float(funnel.oracle_coverage),
        "budgets": dict(funnel.budgets),
        "provenance": dict(funnel.provenance),
    }


def default_stage_records(stages: Mapping[str, str], *, counts: Mapping[str, int] | None = None, elapsed_ms: Mapping[str, float] | None = None, cap_reasons: Mapping[str, str] | None = None) -> tuple[StageRecord, ...]:
    counts = counts or {}
    elapsed_ms = elapsed_ms or {}
    cap_reasons = cap_reasons or {}
    return tuple(
        StageRecord(
            stage=stage,
            count=max(0, int(counts.get(stage, 0))),
            elapsed_ms=max(0.0, float(elapsed_ms.get(stage, 0.0))),
            cap_hit=stage in cap_reasons,
            status=str(stages.get(stage, "not_recorded")),
            reason=cap_reasons.get(stage),
        )
        for stage in REQUIRED_STAGES
    )


def rows_from_mappings(rows: Iterable[Mapping[str, Any]]) -> tuple[TerminalRow, ...]:
    return tuple(TerminalRow.from_mapping(row) for row in rows)
