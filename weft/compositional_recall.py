"""Opt-in compositional answering over Weft's existing retrieval primitives.

The module is intentionally independent from ``weft_recall``.  Detection,
planning, evidence selection, arithmetic, redaction, and rendering are
 deterministic and provider-free; a future provider may only phrase validated
results.  This first slice supports direct, episode-count, compare, and
chronology answers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from weft.models import Memory, MemoryStatus, MemoryType
from weft.structured_recall import (
    LEGACY_ANSWER_STATUS_TO_EVIDENCE,
    LEGACY_ANSWER_STATUS_TO_PLAN,
    LEGACY_COMPLETENESS_TO_CANONICAL,
    classify_query,
)

COMPOSITIONAL_VERSION = "compositional-v1"
logger = logging.getLogger(__name__)


class AnswerShape(str, Enum):
    direct = "direct"
    count = "count"
    compare = "compare"
    chronology = "chronology"
    enumeration = "enumeration"
    intersection = "intersection"
    longitudinal_summary = "longitudinal_summary"
    unsupported = "unsupported"


class AnswerStatus(str, Enum):
    success = "success"
    ambiguous = "ambiguous"
    unsupported = "unsupported"
    incomplete = "incomplete"
    invalid_request_or_scope = "invalid_request_or_scope"
    provider_or_schema_error = "provider_or_schema_error"


class Completeness(str, Enum):
    complete = "complete"
    indexed_lower_bound = "indexed_lower_bound"
    incomplete_evidence = "incomplete_evidence"
    cap_exhausted = "cap_exhausted"
    not_applicable = "not_applicable"


class EvidenceKind(str, Enum):
    memory = "memory"
    turn = "turn"
    claim = "claim"
    episode = "episode"


class CountUnit(str, Enum):
    episode = "episode"
    initiative = "initiative"
    benchmark_run = "benchmark_run"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompositionalBudget(StrictModel):
    max_branches: int = Field(default=4, ge=0, le=4)
    max_branch_queries: int = Field(default=8, ge=0, le=8)
    max_candidate_rows: int = Field(default=200, ge=0, le=200)
    max_selected_evidence: int = Field(default=24, ge=0, le=24)
    max_sql_probes: int = Field(default=8, ge=0, le=8)
    max_response_bytes: int = Field(default=32_000, ge=0, le=32_000)
    max_provider_calls: int = Field(default=1, ge=0, le=1)
    max_input_tokens: int = Field(default=8_000, ge=0, le=8_000)
    max_output_tokens: int = Field(default=2_000, ge=0, le=2_000)
    max_latency_ms: int = Field(default=5_000, ge=0, le=5_000)


class AnswerRequest(StrictModel):
    schema_version: Literal[1] = 1
    question: str = Field(min_length=1, max_length=8_000)
    project_id: str | None = None
    user_id: str | None = None
    as_of: datetime | None = None
    timezone: str = "UTC"
    count_unit: CountUnit | None = None
    budget: CompositionalBudget = Field(default_factory=CompositionalBudget)
    retrieval_mode: Literal["face", "code", "all"] = "face"
    enumeration_limit: int = Field(default=24, ge=1, le=24)
    limit: int = Field(default=10, ge=1, le=24)

    @field_validator("question")
    @classmethod
    def non_blank_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("invalid timezone") from exc
        return value


class DetectionResult(StrictModel):
    schema_version: Literal[1] = 1
    normalized_question: str
    shape: AnswerShape
    operation: str
    signals: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    item_category: str | None = None
    suggested_count_units: tuple[str, ...] = ()
    count_unit_state: Literal["not_applicable", "declared", "missing", "unsupported"] = "not_applicable"
    confidence: float = Field(ge=0, le=1)
    status: AnswerStatus


class BranchSpec(StrictModel):
    branch_id: str = Field(pattern=r"^branch-[a-z0-9_-]{1,64}$")
    purpose: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=1, max_length=2_000)
    operand: str | None = None
    category: str | None = None
    required_evidence: tuple[EvidenceKind, ...] = ()
    mandatory: bool = True


class CompositionPlan(StrictModel):
    schema_version: Literal[1] = 1
    version: str = COMPOSITIONAL_VERSION
    plan_id: str
    normalized_question: str
    shape: AnswerShape
    operation: str
    user_id: str | None
    project_id: str | None
    as_of: str | None
    timezone: str
    branches: tuple[BranchSpec, ...] = ()
    mandatory_branch_count: int = Field(ge=0)
    count_unit: CountUnit | None = None
    count_rule: str | None = None
    comparison_rule: str | None = None
    deduplication_rule: str | None = None
    budgets: CompositionalBudget = Field(default_factory=CompositionalBudget)
    status: AnswerStatus
    incomplete_reason: str | None = None

    @model_validator(mode="after")
    def validate_branches(self) -> "CompositionPlan":
        if len(self.branches) > self.budgets.max_branches:
            raise ValueError("branch cap exceeded")
        if self.mandatory_branch_count != sum(1 for b in self.branches if b.mandatory):
            raise ValueError("mandatory_branch_count does not match branches")
        return self


class EvidenceId(StrictModel):
    kind: EvidenceKind
    value: str = Field(min_length=1, max_length=200)

    @property
    def rendered(self) -> str:
        return f"{self.kind.value}:{self.value}"

    @classmethod
    def parse(cls, value: str) -> "EvidenceId":
        kind, sep, identifier = value.partition(":")
        if not sep or kind not in {item.value for item in EvidenceKind} or not identifier:
            raise ValueError("unsupported evidence id")
        return cls(kind=EvidenceKind(kind), value=identifier)


class AnswerEvidence(StrictModel):
    evidence_id: EvidenceId
    branch_id: str
    provenance: str = Field(min_length=1, max_length=200)
    authority: Literal["authoritative", "candidate"]
    user_id: str | None
    project_id: str | None
    occurred_at: datetime | None = None
    supersession: Literal["current", "superseded", "unknown"] = "unknown"
    match_reason: str = Field(min_length=1, max_length=300)
    citation_text: str = Field(min_length=1, max_length=500)
    content: str | None = None

    @model_validator(mode="after")
    def validate_id_mapping(self) -> "AnswerEvidence":
        if self.evidence_id.kind is EvidenceKind.claim:
            if not self.content or not self.content.strip():
                raise ValueError("claim evidence must resolve to non-empty evidence content")
        if self.evidence_id.kind is EvidenceKind.episode and self.authority == "authoritative":
            raise ValueError("episode ids are grouping metadata, not authoritative citations")
        return self


class TemporalItemValidation(StrictModel):
    valid: bool
    rejection_reason: str | None = None


class TemporalItemEvidence(StrictModel):
    """Validated item identity plus the timestamp basis used for ordering."""

    identity: str = Field(min_length=1, max_length=200)
    normalized_identity: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)
    occurred_at: datetime
    date_basis: Literal["explicit_event_date", "source_timestamp"]
    evidence: AnswerEvidence


class BudgetTelemetry(StrictModel):
    branch_queries: int = 0
    candidate_rows: int = 0
    selected_evidence: int = 0
    sql_probes: int = 0
    provider_calls: int = 0
    latency_ms: int = 0
    response_bytes: int = 0
    repeat_hash: str | None = None


class ComposedAnswer(StrictModel):
    schema_version: Literal[1] = 1
    version: str = COMPOSITIONAL_VERSION
    question: str
    normalized_question: str
    shape: AnswerShape
    operation: str
    status: AnswerStatus
    answer: str | None = None
    typed_result: dict[str, Any] | None = None
    plan: CompositionPlan
    branch_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence: tuple[AnswerEvidence, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    basis: str | None = None
    completeness: Completeness
    evidence_status: Literal["success", "empty", "incomplete", "error"] | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    retrieval_mode: Literal["face", "code", "all"] = "face"
    provenance: dict[str, Any] = Field(default_factory=dict)
    incomplete_reason: str | None = None
    telemetry: BudgetTelemetry = Field(default_factory=BudgetTelemetry)

    @model_validator(mode="after")
    def citations_subset(self) -> "ComposedAnswer":
        if self.status is AnswerStatus.success and self.evidence_status not in {"success", "empty"}:
            raise ValueError("success answer must have success or empty evidence status")
        if self.status is not AnswerStatus.success and self.evidence_status == "success":
            raise ValueError("non-success answer cannot have success evidence status")
        available = {item.evidence_id.rendered for item in self.evidence}
        cited = list(self.cited_evidence_ids)
        if len(cited) != len(set(cited)):
            raise ValueError("duplicate citations are not allowed")
        if not set(cited).issubset(available):
            raise ValueError("citations must be a subset of supplied evidence")
        return self


def normalize_question(question: str) -> str:
    return " ".join(str(question).strip().split())


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_-]?key|token|password|secret|bearer|dsn)[=:]\s*[^,;\s]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(?:postgres(?:ql)?|mysql|redis)://[^\s]+"),
)
_SAFE_KEY = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_SAFE_PATH = re.compile(r"^(?:~|/)[A-Za-z0-9._~/${} -]+$")


def redact_secret_text(value: str) -> str:
    """Remove secret-shaped values while retaining safe labels and paths."""
    redacted = str(value)
    for pattern in _SECRET_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if "://" in token:
                return token.split("://", 1)[0] + "://[REDACTED]"
            if token.casefold().startswith("bearer "):
                return "Bearer [REDACTED]"
            return token.split("=", 1)[0] + "=[REDACTED]"
        redacted = pattern.sub(_replace, redacted)
    # Never serialize a full environment dump or long opaque token.
    if len(redacted) > 8_000:
        redacted = redacted[:8_000] + "…[TRUNCATED]"
    return redacted


def safe_config_metadata(*, safe_key_name: str | None = None, config_path: str | None = None, provider_id: str | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    if safe_key_name and _SAFE_KEY.fullmatch(safe_key_name):
        result["safe_key_name"] = safe_key_name
    if config_path and _SAFE_PATH.fullmatch(config_path):
        result["config_path"] = config_path
    if provider_id and re.fullmatch(r"[a-z0-9_-]{1,80}", provider_id):
        result["provider_id"] = provider_id
    return result


def detect_answer_shape(question: str, *, count_unit: CountUnit | None = None) -> DetectionResult:
    """Legacy projection of the canonical structured classifier.

    The old DetectionResult shape is retained, but all support classification
    and stable plan semantics originate in ``structured_recall.classify_query``.
    """
    normalized = normalize_question(question)
    canonical = classify_query(normalized, user_id="__compat__", project_id="__compat__")
    lowered = normalized.casefold()
    signals: list[str] = []
    reasons: list[str] = []
    operands: tuple[str, ...] = ()
    item_category: str | None = None
    suggested_units: tuple[str, ...] = ()

    count_signal = bool(re.search(r"\b(?:how many|how often|how many times|different)\b", lowered))
    compare_signal = bool(re.search(r"\b(?:who|which|compare|versus|vs\.?)\b.*\b(?:first|better|before|after|earlier|later)\b|\b(?:before|after|earlier|later|compare|versus|vs\.?)\b", lowered))
    chronology_signal = bool(re.search(r"\b(?:who|what)\b.*\b(?:first|earliest|latest|before|after)\b|\b(?:chronolog|earliest|latest|before|after)\b", lowered))
    enumeration_signal = bool(re.search(r"\b(?:all|every|what have we tried|list|enumerate|order of|ordered list)\b", lowered))
    # An ordered list/all-history request is enumeration unless it names
    # explicit operands.  Keep temporal ordering ("earliest/latest/first")
    # reserved for chronology so two-branch comparisons remain conservative.
    enumeration_form = bool(re.search(r"\b(?:ordered list|all .* (?:visited|recorded|history)|list (?:all|every)|order of .* (?:earliest|latest)|order of .* over time)\b", lowered))
    completeness_signal = bool(re.search(r"\b(?:ever|so far|all time|over time|across sessions|since)\b", lowered))
    direct_signal = bool(re.search(r"\b(?:where|what is our preferred|how do we configure|how is .* configured|what is the solution)\b", lowered))
    if count_signal:
        signals.append("counting")
    if compare_signal:
        signals.append("comparison")
    if chronology_signal:
        signals.append("chronology")
    if enumeration_signal:
        signals.append("enumeration")
    if completeness_signal:
        signals.append("completeness")
    if direct_signal:
        signals.append("direct_lookup")

    # Named operands are deliberately conservative: quoted names, explicit
    # "between A and B", or a pair introduced by compare/versus language.
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)", normalized, re.I)
    versus = re.search(r"\b(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:[?.!,]|$)", normalized, re.I)
    quoted = re.findall(r"[\"']([^\"']{1,120})[\"']", normalized)
    if between:
        operands = tuple(part.strip() for part in between.groups() if part.strip())
    elif versus:
        operands = tuple(part.strip() for part in versus.groups() if part.strip())
    elif len(quoted) >= 2 and (compare_signal or chronology_signal):
        operands = tuple(quoted[:2])
    if operands:
        reasons.append("named_operands")

    # One bounded branch may be narrowed by an explicit item category.  Do not
    # infer a category from generic plural words: overlapping categories must
    # fail closed rather than silently widening retrieval.
    category_patterns = (
        ("sports event", r"\bsports? events?\b"),
        ("calendar event", r"\bcalendar events?\b"),
        ("event", r"(?<!calendar )\bevents?\b"),
        ("museum", r"\bmuseums?\b"),
        ("trip", r"\btrips?\b"),
    )
    matched_categories = [category for category, pattern in category_patterns if re.search(pattern, lowered)]
    specific_categories = [category for category in matched_categories if category != "event"]
    generic_event_coordinated = bool(re.search(r"\bevents?\s+(?:and|or)\s+", lowered))
    if generic_event_coordinated and specific_categories:
        reasons.append("ambiguous_item_category")
    elif len(specific_categories) == 1:
        item_category = specific_categories[0]
        signals.append("item_category")
    elif len(specific_categories) == 0 and len(matched_categories) == 1:
        item_category = matched_categories[0]
        signals.append("item_category")
    elif len(specific_categories) > 1:
        reasons.append("ambiguous_item_category")

    if count_signal:
        suggested_units = tuple(item.value for item in (CountUnit.episode, CountUnit.initiative, CountUnit.benchmark_run))
        if count_unit is None:
            return DetectionResult(normalized_question=normalized, shape=AnswerShape.count, operation="count", signals=tuple(signals), reasons=("count_unit_required",), suggested_count_units=suggested_units, count_unit_state="missing", confidence=0.99, status=AnswerStatus.ambiguous)
        if count_unit is not CountUnit.episode:
            return DetectionResult(normalized_question=normalized, shape=AnswerShape.count, operation="count", signals=tuple(signals), reasons=("count_unit_has_no_first_slice_oracle",), suggested_count_units=suggested_units, count_unit_state="unsupported", confidence=0.99, status=AnswerStatus.unsupported)
        return DetectionResult(normalized_question=normalized, shape=AnswerShape.count, operation="count", signals=tuple(signals), reasons=("episode_identity_oracle",), operands=operands, suggested_count_units=suggested_units, count_unit_state="declared", confidence=0.99, status=AnswerStatus.success)

    if (enumeration_signal or enumeration_form) and (not (compare_signal or chronology_signal) or (enumeration_form and not operands)):
        if generic_event_coordinated or len(specific_categories) > 1 or (not specific_categories and len(matched_categories) > 1):
            return DetectionResult(normalized_question=normalized, shape=AnswerShape.enumeration, operation="enumeration", signals=tuple(signals), reasons=("ambiguous_item_category",), operands=operands, confidence=0.9, status=AnswerStatus.incomplete)
        return DetectionResult(normalized_question=normalized, shape=AnswerShape.enumeration, operation="enumeration", signals=tuple(signals), reasons=("bounded_item_identity_oracle",), operands=operands, item_category=item_category, confidence=0.9, status=AnswerStatus.success)
    if (compare_signal or chronology_signal) and len(operands) >= 2:
        shape = AnswerShape.chronology if chronology_signal else AnswerShape.compare
        return DetectionResult(normalized_question=normalized, shape=shape, operation=shape.value, signals=tuple(signals), reasons=("mandatory_operand_branches",), operands=operands[:2], confidence=0.95, status=AnswerStatus.success)
    if compare_signal or chronology_signal:
        return DetectionResult(normalized_question=normalized, shape=AnswerShape.unsupported, operation="unsupported", signals=tuple(signals), reasons=("named_operands_required",), confidence=0.9, status=AnswerStatus.incomplete)
    if enumeration_signal:
        return DetectionResult(normalized_question=normalized, shape=AnswerShape.enumeration, operation="enumeration", signals=tuple(signals), reasons=("bounded_item_identity_oracle",), operands=operands, item_category=item_category, confidence=0.9, status=AnswerStatus.success)
    if direct_signal or not signals:
        return DetectionResult(normalized_question=normalized, shape=AnswerShape.direct, operation="direct", signals=tuple(signals), reasons=("single_scoped_lookup",), confidence=0.85, status=AnswerStatus.success)
    return DetectionResult(normalized_question=normalized, shape=AnswerShape.unsupported, operation="unsupported", signals=tuple(signals), reasons=("unsupported_answer_shape",), confidence=0.8, status=AnswerStatus.unsupported)


def _branch(branch_id: str, purpose: str, query: str, operand: str | None = None, category: str | None = None) -> BranchSpec:
    return BranchSpec(branch_id=branch_id, purpose=purpose, query=query, operand=operand, category=category, required_evidence=(EvidenceKind.memory, EvidenceKind.turn))


def build_plan(request: AnswerRequest) -> tuple[DetectionResult, CompositionPlan]:
    detection = detect_answer_shape(request.question, count_unit=request.count_unit)
    canonical = classify_query(
        request.question, user_id=request.user_id, project_id=request.project_id,
        as_of=request.as_of, timezone_name=request.timezone,
        retrieval_mode=request.retrieval_mode,
    )
    branches: tuple[BranchSpec, ...] = ()
    # Canonical scope/grammar status is authoritative. Legacy detection remains
    # only a shape/branch projection and cannot turn a canonical error into a
    # successful executable plan.
    status = detection.status
    # Canonical v1 owns represented structured families and hard scope errors.
    # Legacy compositional-only shapes (direct/count/compare/chronology and the
    # curated enumeration projection) remain compatibility-only; a canonical
    # `unsupported` there means "not represented in structured v1", not a
    # license to disable the existing answer contract.
    # Canonical v1 owns every query that it classified, including explicit
    # unsupported grammar.  A plain ``family=unsupported`` /
    # ``status=unsupported`` plan with no canonical reason is the legacy-only
    # compatibility escape hatch; flagged plans (ordering-only, recurring,
    # mixed/implicit time, invalid input, etc.) must fail closed even when the
    # legacy detector guessed a successful shape.
    canonical_owns_status = (
        canonical.status != "unsupported"
        or canonical.family != "unsupported"
        or bool(canonical.conflict_flags)
    )
    if canonical_owns_status:
        canonical_status = {
            "supported": AnswerStatus.success,
            "error": AnswerStatus.invalid_request_or_scope,
            "ambiguous": AnswerStatus.incomplete,
            "unsupported": AnswerStatus.unsupported,
        }[canonical.status]
        # Canonical v1 does not own the legacy count-unit contract.  Its broad
        # enumeration grammar recognizes the noun in ``how many sessions``
        # but cannot authorize the old executor's hard-coded episode count.
        # Preserve the legacy ambiguity/unsupported gate until an explicit
        # canonical counting unit is represented.
        if (
            canonical.status == "supported"
            and detection.shape is AnswerShape.count
            and detection.status is not AnswerStatus.success
        ):
            status = detection.status
        else:
            status = canonical_status
    reason = detection.reasons[0] if detection.reasons else None
    if canonical.conflict_flags:
        reason = canonical.conflict_flags[0]
    count_rule = None
    comparison_rule = None
    dedup_rule = None
    if detection.shape is AnswerShape.enumeration and status is AnswerStatus.success:
        branch_query = detection.item_category or detection.normalized_question
        branches = (_branch("branch-enumeration", "bounded item identity and typed date retrieval", branch_query, category=detection.item_category),)
        comparison_rule = "sort distinct recovered item identities by typed occurred_at"
        dedup_rule = "normalized item identity"
    elif detection.shape is AnswerShape.direct and status is AnswerStatus.success:
        branches = (_branch("branch-direct", "current scoped solution/preference/decision", detection.normalized_question),)
    elif detection.shape is AnswerShape.count and status is AnswerStatus.success:
        branches = (_branch("branch-count", "bounded episode identity retrieval", detection.normalized_question),)
        count_rule = "count distinct episode IDs; exact only when the episode oracle is complete"
        dedup_rule = "episode.id"
    elif detection.shape in {AnswerShape.compare, AnswerShape.chronology} and status is AnswerStatus.success:
        branches = tuple(_branch(f"branch-{index + 1}", "independent operand evidence", operand, operand) for index, operand in enumerate(detection.operands[:2]))
        comparison_rule = "compare typed dates/values only when every mandatory branch is supported"
    if len(branches) > request.budget.max_branches:
        status = AnswerStatus.incomplete
        reason = "max_branches"
        branches = ()
    plan = CompositionPlan(
        plan_id="",
        normalized_question=detection.normalized_question,
        shape=detection.shape,
        operation=detection.operation,
        user_id=request.user_id,
        project_id=request.project_id,
        as_of=request.as_of.isoformat() if request.as_of else None,
        timezone=request.timezone,
        branches=branches,
        mandatory_branch_count=sum(1 for branch in branches if branch.mandatory),
        count_unit=request.count_unit if detection.count_unit_state == "declared" else None,
        count_rule=count_rule,
        comparison_rule=comparison_rule,
        deduplication_rule=dedup_rule,
        budgets=request.budget,
        status=status,
        incomplete_reason=reason,
    )
    # Canonical plan identity is the source of truth. Legacy projection fields
    # remain in CompositionPlan, but never alter the canonical ID.
    canonical_id = canonical.plan_id
    return detection, plan.model_copy(update={"plan_id": canonical_id})


def _safe_content(content: str) -> str:
    return redact_secret_text(content).replace("\\x00", "")


_CATEGORY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sports event", ("sports event", "sport event")),
    ("calendar event", ("calendar event",)),
    ("museum", ("museum",)),
    ("trip", ("trip",)),
    ("event", ("event",)),
)


def validate_temporal_item(content: str, *, category: str | None = None) -> TemporalItemValidation:
    """Return stable rejection reasons for the strict temporal item parser."""
    value = str(content or "")
    prefixes = next((prefixes for name, prefixes in _CATEGORY_PREFIXES if name == category), ()) if category else tuple(prefix for _, prefixes in _CATEGORY_PREFIXES for prefix in prefixes)
    escaped = "|".join(re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True))
    match = re.search(rf"(?i)\b((?:{escaped})\s+[A-Za-z][A-Za-z0-9_-]{{0,80}})\b", value) or re.search(rf"(?i)\b([A-Z][A-Za-z0-9_-]{{0,80}}\s+(?:{escaped}))\b", value)
    if not match:
        return TemporalItemValidation(valid=False, rejection_reason="missing_typed_identity")
    identity = normalize_question(match.group(1)).casefold()
    label = identity.split()[1] if identity.split() and identity.split()[0] in {prefix for _, ps in _CATEGORY_PREFIXES for prefix in ps} and len(identity.split()) > 1 else identity.split()[0]
    if label in {"that", "has", "is", "with", "was", "where", "when", "in", "to", "and", "or", "could", "might", "again", "next"}:
        return TemporalItemValidation(valid=False, rejection_reason="prose_identity")
    explicit = re.search(r"\b(?:visited|attended|completed|occurred|held|on)\s+(?:on\s+)?(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", value, re.I)
    if explicit:
        try:
            datetime.strptime(explicit.group(1).replace("/", "-"), "%Y-%m-%d")
        except ValueError:
            return TemporalItemValidation(valid=False, rejection_reason="invalid_event_date")
    return TemporalItemValidation(valid=True)


def _extract_temporal_item(memory: Memory, *, branch_id: str, project_id: str | None, category: str | None) -> TemporalItemEvidence | None:
    """Extract a validated item and explicit date, failing closed on prose."""
    if not validate_temporal_item(memory.content or "", category=category).valid:
        return None
    content = memory.content or ""
    prefixes = next((prefixes for name, prefixes in _CATEGORY_PREFIXES if name == category), ()) if category else tuple(prefix for _, prefixes in _CATEGORY_PREFIXES for prefix in prefixes)
    escaped = "|".join(re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True))
    match = re.search(rf"(?i)\b((?:{escaped})\s+[A-Za-z][A-Za-z0-9_-]{{0,80}})\b", content)
    reverse = re.search(rf"(?i)\b([A-Z][A-Za-z0-9_-]{{0,80}}\s+(?:{escaped}))\b", content)
    if not match and not reverse:
        return None
    identity = normalize_question((match or reverse).group(1))
    normalized = identity.casefold()
    tokens = normalized.split()
    label = tokens[1] if tokens[0] in {prefix for _, prefixes in _CATEGORY_PREFIXES for prefix in prefixes} else tokens[0]
    if label in {"that", "has", "is", "with", "was", "where", "when", "in", "to", "and", "or", "could", "might", "again", "next"}:
        return None
    evidence = memory_evidence(memory, branch_id=branch_id, project_id=project_id)
    explicit = re.search(r"\b(?:visited|attended|completed|occurred|held|on)\s+(?:on\s+)?(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", content, re.I)
    occurred_at = memory.updated_at
    basis: Literal["explicit_event_date", "source_timestamp"] = "source_timestamp"
    if explicit:
        try:
            occurred_at = datetime.strptime(explicit.group(1).replace("/", "-"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            basis = "explicit_event_date"
        except ValueError:
            return None
    if occurred_at is None:
        return None
    return TemporalItemEvidence(identity=identity, normalized_identity=normalized, category=category or "event", occurred_at=occurred_at, date_basis=basis, evidence=evidence)


def memory_evidence(memory: Memory, *, branch_id: str, project_id: str | None) -> AnswerEvidence:
    current = memory.status is MemoryStatus.active
    return AnswerEvidence(
        evidence_id=EvidenceId(kind=EvidenceKind.memory, value=memory.id),
        branch_id=branch_id,
        provenance="memories",
        authority="authoritative" if current and memory.review_status == "active" else "candidate",
        user_id=None,
        project_id=memory.project_id or project_id,
        occurred_at=memory.updated_at,
        supersession="current" if current else "superseded",
        match_reason="scoped memory retrieval",
        citation_text=_safe_content(memory.content)[:500],
        content=_safe_content(memory.content),
    )


def _recovery_scope_matches(request: AnswerRequest, outcome: Any) -> bool:
    scope = getattr(outcome, "scope", None)
    if scope is None:
        return False
    if scope.user_id != request.user_id or scope.retrieval_mode != request.retrieval_mode:
        return False
    if request.retrieval_mode in {"code", "all"}:
        return scope.resolved_project_id == request.project_id
    if scope.requested_project_id not in {None, request.project_id}:
        return False
    return True


def _recovery_branch_for_candidate(candidate: Any, plan: CompositionPlan, *, content_override: str | None = None, trust_declared: bool = True) -> str | None:
    branch = str(getattr(candidate, "branch", "") or "")
    branch_ids = {item.branch_id for item in plan.branches}
    if trust_declared and branch in branch_ids:
        return branch
    content = str(content_override if content_override is not None else getattr(candidate, "content", "") or "")
    available = set(re.findall(r"[a-z0-9_./-]+", content.casefold()))
    matches: list[str] = []
    for item in plan.branches:
        if item.operand:
            terms = {term for term in re.findall(r"[a-z0-9_./-]+", item.operand.casefold()) if len(term) > 2}
        else:
            terms = {term for term in re.findall(r"[a-z0-9_./-]+", item.query.casefold()) if len(term) > 2}
        if terms and ((terms.issubset(available)) if plan.shape in {AnswerShape.compare, AnswerShape.chronology} else bool(terms & available)):
            matches.append(item.branch_id)
    if len(matches) == 1:
        return matches[0]
    if plan.shape is AnswerShape.direct and any(item.branch_id == "branch-direct" for item in plan.branches):
        return "branch-direct"
    return None


CanonicalRecoveryReader = Callable[[EvidenceId], Awaitable[Any] | Any]


def _canonical_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a canonical field from a mapping, model, or asyncpg.Record.

    ``asyncpg.Record`` supports key lookup but is not guaranteed to satisfy
    ``Mapping`` or expose columns as attributes. Canonical rereads must accept
    all three shapes because production readers and test doubles use them.
    """
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except AttributeError:
        pass
    try:
        return value[name]
    except (KeyError, IndexError, TypeError):
        return default


async def resolve_recovery_authority(
    request: AnswerRequest,
    plan: CompositionPlan,
    outcome: Any,
    reader: CanonicalRecoveryReader,
) -> tuple[tuple[AnswerEvidence, ...], dict[str, Any]]:
    """Re-read recovery IDs canonically before allowing them to answer.

    Candidate payloads are used only for stage provenance and branch hints. The
    reader must perform the authenticated/RLS-scoped lookup; all answer fields
    are copied from its returned canonical object. Fail closed on any reader
    error or validator failure.
    """
    if not _recovery_scope_matches(request, outcome):
        return (), {"recovery_scope_rejected": True, "canonical_reread_count": 0}
    evidence: list[AnswerEvidence] = []
    candidate_ids: list[str] = []
    authoritative_ids: list[str] = []
    rejected: dict[str, str] = {}
    seen: set[str] = set()
    for candidate in tuple(getattr(outcome, "candidates", ()))[: request.budget.max_selected_evidence]:
        stable_id = str(getattr(candidate, "stable_id", ""))
        if not stable_id or stable_id in seen:
            continue
        seen.add(stable_id)
        candidate_ids.append(stable_id)
        try:
            evidence_id = EvidenceId.parse(stable_id)
            canonical = reader(evidence_id)
            if isawaitable(canonical):
                canonical = await canonical
            if canonical is None:
                rejected[stable_id] = "missing_canonical_row"
                continue
            def _field(name: str, default: Any = None) -> Any:
                return _canonical_field(canonical, name, default)
            # Memory rows do not expose user_id in the canonical model; the
            # authenticated/RLS read itself establishes request ownership.
            canonical_user = _field("user_id", request.user_id)
            if canonical_user not in {request.user_id, None, "__system_global_zathras__"}:
                rejected[stable_id] = "foreign_user"
                continue
            canonical_project = _field("project_id")
            facets = {str(item).casefold() for item in (_field("project_facets", ()) or ())}
            if request.retrieval_mode in {"code", "all"} and canonical_project != request.project_id:
                rejected[stable_id] = "foreign_project"
                continue
            if request.retrieval_mode == "face" and request.project_id and canonical_project not in {None, request.project_id} and request.project_id.casefold() not in facets:
                rejected[stable_id] = "foreign_project"
                continue
            canonical_source = _field("source")
            if request.retrieval_mode == "face" and canonical_source is not None:
                source = getattr(canonical_source, "value", canonical_source)
                if source not in {"conversation", "documentation", "inference", "seed"}:
                    rejected[stable_id] = "mode_disallowed_source"
                    continue
            status = _field("status", "active")
            status_value = getattr(status, "value", status)
            if status_value != "active" or _field("review_status", "active") != "active":
                rejected[stable_id] = "stale_or_unreviewed"
                continue
            occurred_at = _field("occurred_at") or _field("updated_at")
            if request.as_of is not None and occurred_at is not None:
                cutoff = request.as_of if request.as_of.tzinfo is not None else request.as_of.replace(tzinfo=timezone.utc)
                if occurred_at > cutoff:
                    rejected[stable_id] = "post_as_of"
                    continue
            content = str(_field("content", "") or _field("value", "") or "")
            branch_id = _recovery_branch_for_candidate(candidate, plan, content_override=content, trust_declared=False)
            if branch_id is None:
                rejected[stable_id] = "ambiguous_branch"
                continue
            if not content.strip():
                rejected[stable_id] = "untyped_or_empty"
                continue
            if evidence_id.kind in {EvidenceKind.memory, EvidenceKind.turn, EvidenceKind.claim}:
                if plan.shape in {AnswerShape.compare, AnswerShape.chronology} and not _recovery_branch_for_candidate(candidate, plan, content_override=content):
                    rejected[stable_id] = "ambiguous_branch"
                    continue
            relationships = _field("relationships", ())
            def _relation_value(rel: Any) -> Any:
                value = rel.get("relation") if isinstance(rel, Mapping) else getattr(rel, "relation", None)
                return getattr(value, "value", value)
            if any(_relation_value(rel) in {"contradicts", "supersedes"} for rel in relationships):
                rejected[stable_id] = "conflicting_or_superseded"
                continue
            if plan.shape is AnswerShape.direct and not _direct_identifier_match(content, plan.normalized_question):
                rejected[stable_id] = "identifier_mismatch"
                continue
            if plan.shape is AnswerShape.direct and not _direct_memory_answerable(content, plan.normalized_question):
                rejected[stable_id] = "insufficient_direct_relevance"
                continue
            if evidence_id.kind is EvidenceKind.turn:
                occurred_at = _canonical_field(canonical, "occurred_at")
                if occurred_at is None and plan.shape in {AnswerShape.compare, AnswerShape.chronology}:
                    rejected[stable_id] = "untyped_date"
                    continue
            item = AnswerEvidence(
                evidence_id=evidence_id,
                branch_id=branch_id,
                provenance="recovery:" + ":".join(tuple(getattr(candidate, "provenance", ()))[:2]),
                authority="authoritative",
                user_id=canonical_user,
                project_id=canonical_project or request.project_id,
                occurred_at=occurred_at,
                supersession="current",
                match_reason="canonical authenticated recovery read",
                citation_text=_safe_content(content)[:500],
                content=_safe_content(content),
            )
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # fail closed; do not expose provider/DB details
            rejected[stable_id] = "canonical_read_error"
            logger.debug("recovery authority read rejected %s: %s", stable_id, exc)
            continue
        evidence.append(item)
        authoritative_ids.append(stable_id)
    return tuple(evidence[: request.budget.max_selected_evidence]), {
        "recovered_candidate_ids": candidate_ids[: request.budget.max_selected_evidence],
        "recovered_authoritative_ids": authoritative_ids[: request.budget.max_selected_evidence],
        "recovered_rejected_ids": rejected,
        "recovered_candidate_count": len(candidate_ids),
        "canonical_reread_count": len(candidate_ids),
    }


def project_recovery_evidence(
    request: AnswerRequest,
    plan: CompositionPlan,
    outcome: Any,
) -> tuple[tuple[AnswerEvidence, ...], dict[str, Any]]:
    """Project bounded recovery candidates into existing answer evidence.

    This is deliberately evidence-only: candidates stay non-authoritative and
    can never become citations or satisfy a mandatory answer branch.
    """
    if not _recovery_scope_matches(request, outcome):
        return (), {"recovery_scope_rejected": True}
    evidence: list[AnswerEvidence] = []
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for candidate in tuple(getattr(outcome, "candidates", ()))[: request.budget.max_selected_evidence]:
        stable_id = str(getattr(candidate, "stable_id", ""))
        content = str(getattr(candidate, "content", "") or "")
        if not stable_id or not content or stable_id in seen:
            continue
        if getattr(candidate, "source_mode", request.retrieval_mode) != request.retrieval_mode:
            continue
        candidate_user = getattr(candidate, "user_id", None)
        if candidate_user is not None and candidate_user != request.user_id:
            continue
        candidate_project = getattr(candidate, "project_id", None)
        if request.retrieval_mode in {"code", "all"} and candidate_project != request.project_id:
            continue
        occurred_at = getattr(candidate, "occurred_at", None)
        if request.as_of is not None and occurred_at is not None:
            cutoff = request.as_of if request.as_of.tzinfo is not None else request.as_of.replace(tzinfo=timezone.utc)
            if occurred_at > cutoff:
                continue
        branch_id = _recovery_branch_for_candidate(candidate, plan)
        if branch_id is None:
            continue
        kind_name, _, identifier = stable_id.partition(":")
        if kind_name not in {item.value for item in EvidenceKind} or not identifier:
            continue
        try:
            evidence_id = EvidenceId(kind=EvidenceKind(kind_name), value=identifier)
            item = AnswerEvidence(
                evidence_id=evidence_id,
                branch_id=branch_id,
                provenance="recovery:" + ":".join(tuple(getattr(candidate, "provenance", ()))[:2]),
                authority="candidate",
                user_id=request.user_id,
                project_id=getattr(candidate, "project_id", None) or request.project_id,
                occurred_at=occurred_at,
                supersession=(getattr(candidate, "supersession", "unknown") if getattr(candidate, "supersession", "unknown") in {"current", "superseded"} else "unknown"),
                match_reason="bounded deterministic recovery candidate; non-authoritative",
                citation_text=_safe_content(content)[:500],
                content=_safe_content(content),
            )
        except (TypeError, ValueError):
            continue
        evidence.append(item)
        candidate_ids.append(stable_id)
        seen.add(stable_id)
    return tuple(evidence[: request.budget.max_selected_evidence]), {
        "recovered_candidate_ids": candidate_ids[: request.budget.max_selected_evidence],
        "recovered_candidate_count": len(candidate_ids),
    }


async def adapt_recovery_to_answer_authoritative(
    request: AnswerRequest,
    answer: ComposedAnswer,
    outcome: Any,
    reader: CanonicalRecoveryReader,
) -> ComposedAnswer:
    """Attach only canonically re-read recovery evidence and re-run answer gates."""
    # Baseline direct retrieval has already read the persisted relationship
    # seam.  Never let deterministic recovery upgrade that explicit conflict
    # into a complete answer by selecting only one operand.
    if answer.plan.shape is AnswerShape.direct and answer.branch_results.get("explicit_conflicts"):
        return finalize_answer(answer)
    if not getattr(outcome, "supported", False):
        return answer
    evidence, metadata = await resolve_recovery_authority(request, answer.plan, outcome, reader)
    merged = list(answer.evidence)
    known = {item.evidence_id.rendered for item in merged}
    for item in evidence:
        if item.evidence_id.rendered not in known and len(merged) < request.budget.max_selected_evidence:
            merged.append(item)
            known.add(item.evidence_id.rendered)
    branch_results = dict(answer.branch_results)
    branch_results["recovery"] = metadata
    authoritative = [item for item in merged if item.authority == "authoritative"]
    mandatory = {branch.branch_id for branch in answer.plan.branches if branch.mandatory}
    supported_branches = {item.branch_id for item in authoritative}
    update: dict[str, Any] = {"evidence": tuple(merged), "branch_results": branch_results}
    if answer.plan.shape is AnswerShape.direct and authoritative:
        update.update({"status": AnswerStatus.success, "completeness": Completeness.complete,
                       "evidence_status": "success", "typed_result": {"value": authoritative[0].content},
                       "basis": "canonical deterministic recovery evidence", "incomplete_reason": None})
    elif answer.plan.shape in {AnswerShape.compare, AnswerShape.chronology}:
        if mandatory.issubset(supported_branches) and all(item.occurred_at is not None for item in authoritative if item.branch_id in mandatory):
            ordered = sorted((item for item in authoritative if item.branch_id in mandatory), key=lambda item: (item.occurred_at, item.branch_id))
            if len({item.branch_id for item in ordered}) == len(mandatory):
                dates = {item.branch_id: item.occurred_at.isoformat() for item in ordered}
                update.update({"status": AnswerStatus.success, "completeness": Completeness.complete,
                               "evidence_status": "success", "typed_result": {"answer": f"{ordered[0].branch_id} has the earliest supported evidence at {ordered[0].occurred_at.isoformat()}.", "ordered_branch_ids": [item.branch_id for item in ordered], "dates": dates},
                               "basis": "canonical deterministic recovery chronology", "incomplete_reason": None})
    if metadata.get("recovered_rejected_ids") and not authoritative:
        update["incomplete_reason"] = "recovery_authority_rejected"
    result = finalize_answer(answer.model_copy(update=update))
    if result.status is not AnswerStatus.success:
        result = result.model_copy(update={"cited_evidence_ids": ()})
    return result


def adapt_recovery_to_answer(
    request: AnswerRequest,
    answer: ComposedAnswer,
    outcome: Any,
) -> ComposedAnswer:
    """Attach validated recovery evidence without changing answer authority."""
    if not getattr(outcome, "supported", False):
        return answer
    evidence, metadata = project_recovery_evidence(request, answer.plan, outcome)
    merged = list(answer.evidence)
    known = {item.evidence_id.rendered for item in merged}
    for item in evidence:
        if item.evidence_id.rendered not in known and len(merged) < request.budget.max_selected_evidence:
            merged.append(item)
            known.add(item.evidence_id.rendered)
    branch_results = dict(answer.branch_results)
    branch_results["recovery"] = metadata
    reason = answer.incomplete_reason
    if getattr(outcome, "retrieval_status", None) == "conflict":
        reason = reason or "recovery_conflicting_evidence"
    elif evidence and answer.status is not AnswerStatus.success:
        reason = reason or "recovery_candidates_are_non_authoritative"
    return finalize_answer(answer.model_copy(update={
        "evidence": tuple(merged),
        "branch_results": branch_results,
        "incomplete_reason": reason,
    }))


def deterministic_render(answer: ComposedAnswer) -> str:
    if answer.answer:
        return answer.answer
    if answer.status is AnswerStatus.ambiguous:
        units = ", ".join(answer.plan.count_unit.value for _ in () ) if answer.plan.count_unit else "episode, initiative, or benchmark_run"
        return f"I need a counting unit before giving an exact count. Choose one: {units}."
    if answer.status in {AnswerStatus.unsupported, AnswerStatus.incomplete}:
        return "I can't establish a complete, supported answer from the available scoped evidence."
    if answer.shape is AnswerShape.count and answer.typed_result:
        count = answer.typed_result.get("count")
        unit = answer.typed_result.get("unit", "item")
        qualifier = "at least " if answer.completeness is Completeness.indexed_lower_bound else ""
        return f"{qualifier}{count} distinct {unit}(s) found in the indexed evidence."
    if answer.shape is AnswerShape.chronology and answer.typed_result:
        return str(answer.typed_result.get("answer") or "The operand chronology could not be established.")
    if answer.shape is AnswerShape.compare and answer.typed_result:
        return str(answer.typed_result.get("answer") or "The operands could not be compared from complete evidence.")
    if answer.evidence:
        return answer.evidence[0].citation_text
    return "No supported evidence was found."


def finalize_answer(answer: ComposedAnswer) -> ComposedAnswer:
    rendered = deterministic_render(answer)
    citations = tuple(item.evidence_id.rendered for item in answer.evidence if item.authority == "authoritative")
    payload = answer.model_copy(update={"answer": rendered, "cited_evidence_ids": citations})
    telemetry_hash = stable_hash({"plan": payload.plan.model_dump(mode="json"), "typed_result": payload.typed_result, "citations": citations, "answer": rendered})
    telemetry = payload.telemetry.model_copy(update={"repeat_hash": telemetry_hash, "response_bytes": len(payload.model_dump_json().encode("utf-8"))})
    return payload.model_copy(update={"telemetry": telemetry})


async def _retrieve_memories(pool: Any, query: str, *, user_id: str, project_id: str | None, limit: int, embedding_provider: Any | None, retrieval_mode: str = "face") -> list[Memory]:
    """Use the existing hybrid path when an embedder is available.

    The keyword path is a safe provider-free fallback for local/test deployments;
    it remains bounded and uses the store's existing scope filters.
    """
    from weft.store import search_by_keyword, search_hybrid
    from weft.retrieval_modes import include_agent_provenance, sources_for_mode
    from weft.db.connection import acquire

    mode = retrieval_mode if retrieval_mode in {"face", "code", "all"} else "face"
    sources = sources_for_mode(mode)
    include_agent = include_agent_provenance(mode)
    async with acquire(pool):
        if embedding_provider is not None:
            try:
                embedding = await embedding_provider.embed(query)
                rows = await search_hybrid(
                    pool, query, embedding, limit=limit, threshold=0.0,
                    status=MemoryStatus.active, project_id=project_id,
                    user_id=user_id, sources=sources,
                    include_agent_provenance=include_agent,
                )
                return [item.memory for item in rows]
            except Exception as exc:
                # A provider outage must not yield a confident answer; keyword
                # retrieval can still provide bounded evidence for a direct path.
                logger.warning("compositional embedding retrieval failed; using keyword fallback: %s", exc)
        rows = await search_by_keyword(
            pool, query, limit=limit, status=MemoryStatus.active,
            project_id=project_id, user_id=user_id, sources=sources,
            include_agent_provenance=include_agent,
        )
    return [item.memory for item in rows]


async def _explicit_contradictions(pool: Any, memories: list[Memory]) -> tuple[str, ...]:
    """Return persisted contradiction edges touching bounded retrieved memories.

    Contradictions are structural metadata only; this deliberately does not
    inspect memory prose or retrieve any additional memories.  The store query
    checks both relationship directions and applies the ambient RLS scope.
    """
    from weft.db.connection import acquire
    from weft.models import RelationType
    from weft.store import get_relationships

    relationships: list[str] = []
    memory_ids = tuple(dict.fromkeys(memory.id for memory in memories if memory.id))
    async with acquire(pool):
        for memory_id in memory_ids:
            for relationship in await get_relationships(
                pool, memory_id, relation=RelationType.contradicts,
            ):
                other_id = (
                    relationship.target_id
                    if relationship.source_id == memory_id
                    else relationship.source_id
                )
                relationships.append(f"contradicts:{memory_id}:{other_id}")
    return tuple(dict.fromkeys(relationships))


_DIRECT_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "be", "does", "do", "did", "for", "how",
    "is", "it", "my", "of", "our", "the", "this", "use", "uses", "using",
    "was", "we", "were", "what", "when", "where", "which", "who", "with",
    "configure", "configured", "configuration", "current", "preferred", "solution",
})

# Repository identifiers are identity-bearing query terms, not merely topical
# vocabulary.  Keep this deliberately narrow: filenames/extensions, paths, and
# command-line flags are stable code-memory anchors; ordinary prose words are
# not.  This mirrors the recovery reformulation vocabulary without introducing
# a fixture-specific filename allowlist.
_DIRECT_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-/])"
    r"(?:--[A-Za-z0-9][A-Za-z0-9_-]*|"
    r"/?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+(?:[.](?:py|yaml|yml|toml|json|ini|cfg|conf|sh|bash|zsh|js|ts|sql)|_[A-Za-z0-9-]+))"
    r"(?![A-Za-z0-9_.\-/])"
)


def _direct_query_identifiers(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.casefold() for match in _DIRECT_IDENTIFIER_RE.findall(query)))


def _direct_identifier_match(content: str, query: str) -> bool:
    """Require every identifier-shaped query anchor to survive canonical reread."""
    identifiers = _direct_query_identifiers(query)
    if not identifiers:
        return True
    available = {match.casefold() for match in _DIRECT_IDENTIFIER_RE.findall(content)}
    return set(identifiers).issubset(available)


def _content_terms(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.casefold()) if len(token) > 2}


_DIRECT_NEGATIVE_EVIDENCE_RE = re.compile(
    r"(?:\b(?:not|never|no longer)\s+(?:been\s+)?"
    r"(?:disclosed|known|available|revealed|published|provided|announced)\b"
    r"|\b(?:undisclosed|unknown|unavailable|unrevealed|unpublished|"
    r"unprovided|unannounced)\b"
    r"|\bno\s+(?:known|available)\s+(?:launch\s+)?(?:date|value|answer)\b)",
    re.IGNORECASE,
)


def _direct_evidence_is_negative(content: str) -> bool:
    """Recognize explicit absence/disclosure-state evidence, not every negation."""
    return bool(_DIRECT_NEGATIVE_EVIDENCE_RE.search(content))


def _direct_question_requests_negative_state(query: str) -> bool:
    """Return whether a direct question asks for disclosure/knowledge state."""
    lowered = normalize_question(query).casefold()
    if re.search(r"\b(?:do|does|did|can)\s+(?:we|you|i)\s+(?:know|disclose)\b", lowered):
        return True
    if re.search(r"\b(?:whether|if)\b[^?.!]{0,100}\b(?:disclosed|known|available|revealed|published|announced)\b", lowered):
        return True
    # A copular status question ends in the status word.  Requiring the
    # polarity term at the end avoids treating "the undisclosed launch date"
    # as a request for an absence state.
    status = re.search(
        r"\b(?:is|are|was|were|has|have)\b(?P<tail>[^?.!]{0,100})"
        r"\b(?:not\s+)?(?:disclosed|known|available|revealed|published|announced|"
        r"undisclosed|unknown|unavailable)\b\s*[?.!]?$",
        lowered,
    )
    if status and " if " not in status.group("tail"):
        return True
    # Questions such as "What remains undisclosed?" explicitly request the
    # negative state; unlike "What is the undisclosed launch date?", the
    # polarity term is the predicate rather than an adjective on the answer.
    return bool(re.search(
        r"\b(?:what|which)\b[^?.!]{0,50}\b(?:remains|is|are)\s+"
        r"(?:undisclosed|unknown|unavailable)\b\s*[?.!]?$",
        lowered,
    ))


def _direct_memory_answerable(content: str, query: str, *, topics: str = "") -> bool:
    """Require bounded lexical and answer-polarity support before direct use.

    Retrieval is intentionally broad (including provider-free keyword fallback),
    so a non-empty result is not proof that it answers the question.  Direct
    questions with multiple meaningful subject/answer terms require two shared
    terms; a one-term subject can still be answered by an exact scoped match.
    Explicit undisclosed/unknown evidence is answerable only for a question that
    asks for that disclosure state.
    """
    if _direct_evidence_is_negative(content) and not _direct_question_requests_negative_state(query):
        return False
    if not _direct_identifier_match(content, query):
        return False
    query_terms = _content_terms(query) - _DIRECT_QUERY_STOPWORDS
    if not query_terms:
        return False
    available = _content_terms(content + " " + topics)
    overlap = query_terms & available
    if len(query_terms) == 1:
        return bool(overlap)
    return len(overlap) >= 2


def _relevant_memory(memory: Memory, query: str, *, require_all_terms: bool = False) -> bool:
    terms = _content_terms(query)
    content = memory.content.casefold()
    topics = " ".join(memory.topic).casefold()
    available = _content_terms(content + " " + topics)
    if require_all_terms:
        return bool(terms) and terms.issubset(available)
    return bool(terms & available)


def _base_answer(request: AnswerRequest, plan: CompositionPlan, *, status: AnswerStatus | None = None, completeness: Completeness = Completeness.not_applicable, reason: str | None = None) -> ComposedAnswer:
    final_status = status or plan.status
    canonical_evidence = LEGACY_ANSWER_STATUS_TO_EVIDENCE.get(final_status.value, "incomplete")
    return ComposedAnswer(
        question=request.question,
        normalized_question=plan.normalized_question,
        shape=plan.shape,
        operation=plan.operation,
        status=final_status,
        plan=plan,
        completeness=completeness,
        evidence_status=canonical_evidence,
        scope={"user_id": request.user_id, "project_id": request.project_id, "as_of": request.as_of.isoformat() if request.as_of else None, "timezone": request.timezone},
        retrieval_mode=request.retrieval_mode,
        provenance={"plan_id": plan.plan_id, "source": "compositional_recall"},
        incomplete_reason=reason or plan.incomplete_reason,
        telemetry=BudgetTelemetry(),
    )


async def answer_question(
    request: AnswerRequest,
    pool: Any,
    *,
    embedding_provider: Any | None = None,
) -> ComposedAnswer:
    """Execute a bounded deterministic answer plan against existing Weft data."""
    started = datetime.now().timestamp()
    if not request.user_id:
        detection = detect_answer_shape(request.question, count_unit=request.count_unit)
        plan = CompositionPlan(
            plan_id=stable_hash(detection.model_dump(mode="json"))[:24],
            normalized_question=detection.normalized_question,
            shape=detection.shape,
            operation=detection.operation,
            user_id=None,
            project_id=request.project_id,
            as_of=request.as_of.isoformat() if request.as_of else None,
            timezone=request.timezone,
            budgets=request.budget,
            mandatory_branch_count=0,
            status=AnswerStatus.invalid_request_or_scope,
            incomplete_reason="caller user scope is required",
        )
        return finalize_answer(_base_answer(request, plan, status=AnswerStatus.invalid_request_or_scope, completeness=Completeness.incomplete_evidence, reason="caller user scope is required"))

    detection, plan = build_plan(request)
    if plan.status is not AnswerStatus.success:
        return finalize_answer(_base_answer(request, plan, completeness=Completeness.incomplete_evidence, reason=plan.incomplete_reason))
    telemetry = BudgetTelemetry()
    evidence: list[AnswerEvidence] = []
    branch_results: dict[str, dict[str, Any]] = {}

    if plan.shape is AnswerShape.enumeration and not plan.branches:
        return finalize_answer(_base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.incomplete_evidence, reason="enumeration plan has no retrieval branch"))

    if plan.shape is AnswerShape.enumeration:
        telemetry = telemetry.model_copy(update={"branch_queries": 1, "sql_probes": 1})
        branch = plan.branches[0]
        memories = await _retrieve_memories(
            pool,
            branch.query,
            user_id=request.user_id,
            project_id=request.project_id,
            limit=min(request.enumeration_limit, request.budget.max_candidate_rows),
            embedding_provider=embedding_provider,
            retrieval_mode=request.retrieval_mode,
        )
        relevant = [memory for memory in memories if memory.content and _relevant_memory(memory, branch.query, require_all_terms=bool(plan.branches[0].category or plan.branches[0].query in {"museum", "trip", "sports event", "calendar event"}))]
        normalized_items: dict[str, TemporalItemEvidence] = {}
        for memory in relevant:
            item = _extract_temporal_item(memory, branch_id=branch.branch_id, project_id=request.project_id, category=plan.branches[0].category or (branch.query if branch.query in {"museum", "trip", "sports event", "calendar event"} else None))
            if item is None:
                continue
            item = item.model_copy(update={"evidence": item.evidence.model_copy(update={"user_id": request.user_id})})
            previous = normalized_items.get(item.normalized_identity)
            if previous is None or (previous.date_basis == "source_timestamp" and item.date_basis == "explicit_event_date") or item.occurred_at < previous.occurred_at:
                normalized_items[item.normalized_identity] = item
        ordered = sorted(normalized_items.values(), key=lambda item: (item.occurred_at, item.normalized_identity))
        ordered_items = [item.normalized_identity for item in ordered]
        identity_evidence = {item.normalized_identity: item.evidence for item in ordered}
        evidence = tuple(identity_evidence[item] for item in ordered_items[: request.budget.max_selected_evidence])
        telemetry = telemetry.model_copy(update={"candidate_rows": len(memories), "selected_evidence": len(evidence)})
        if not ordered_items:
            answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.incomplete_evidence, reason="no stable item identities with typed dates")
        else:
            completeness = Completeness.indexed_lower_bound
            answer = _base_answer(request, plan, status=AnswerStatus.success, completeness=completeness)
            answer = answer.model_copy(update={
                "typed_result": {"items": ordered_items[: request.enumeration_limit], "count": len(ordered_items[: request.enumeration_limit])},
                "evidence": evidence,
                "branch_results": {branch.branch_id: {"candidate_memory_ids": [memory.id for memory in memories], "recovered_item_ids": ordered_items}},
                "basis": "distinct stable item identities sorted by memory updated_at; corpus completeness not proven",
                "telemetry": telemetry,
            })
        return finalize_answer(answer)

    if plan.shape is AnswerShape.count:
        from weft.episodes import list_episodes
        from weft.db.connection import acquire
        telemetry = telemetry.model_copy(update={"branch_queries": 1, "sql_probes": 1})
        async with acquire(pool):
            episodes = await list_episodes(pool, project_id=request.project_id, user_id=request.user_id, limit=request.budget.max_candidate_rows + 1)
        if request.as_of is not None:
            cutoff = request.as_of if request.as_of.tzinfo is not None else request.as_of.replace(tzinfo=timezone.utc)
            episodes = [episode for episode in episodes if episode.started_at <= cutoff]
        if len(episodes) > request.budget.max_candidate_rows:
            answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.cap_exhausted, reason="max_candidate_rows",)
            return finalize_answer(answer.model_copy(update={"telemetry": telemetry.model_copy(update={"candidate_rows": len(episodes)})}))
        ids = tuple(dict.fromkeys(episode.id for episode in episodes))
        branch_results["branch-count"] = {"candidate_episode_ids": ids, "count": len(ids)}
        # Episode identity is a valid counting basis, but episode IDs are
        # grouping metadata rather than citations. Supporting memory/turn
        # evidence can be added by a future resolver without changing this
        # count contract.
        completeness = Completeness.complete
        typed = {"count": len(ids), "unit": CountUnit.episode.value, "deduplication_key": "episode.id"}
        answer = _base_answer(request, plan, status=AnswerStatus.success, completeness=completeness)
        answer = answer.model_copy(update={"typed_result": typed, "branch_results": branch_results, "evidence": tuple(evidence), "basis": "distinct episode IDs from the scoped episode oracle", "telemetry": telemetry.model_copy(update={"candidate_rows": len(episodes), "selected_evidence": len(evidence)})})
        return finalize_answer(answer)

    # Direct and compare/chronology use independent memory retrieval branches.
    if plan.budgets.max_sql_probes < len(plan.branches):
        answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.cap_exhausted, reason="max_sql_probes")
        return finalize_answer(answer.model_copy(update={"telemetry": telemetry}))
    for branch in plan.branches:
        if telemetry.branch_queries >= request.budget.max_branch_queries:
            answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.cap_exhausted, reason="max_branch_queries")
            return finalize_answer(answer.model_copy(update={"telemetry": telemetry}))
        telemetry = telemetry.model_copy(update={"branch_queries": telemetry.branch_queries + 1, "sql_probes": telemetry.sql_probes + 1})
        memories = await _retrieve_memories(pool, branch.query, user_id=request.user_id, project_id=request.project_id, limit=min(request.limit, request.budget.max_candidate_rows), embedding_provider=embedding_provider, retrieval_mode=request.retrieval_mode)
        if request.as_of is not None:
            cutoff = request.as_of
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            memories = [memory for memory in memories if memory.created_at <= cutoff]
        relevant = [
            memory for memory in memories
            if (
                _direct_memory_answerable(
                    memory.content,
                    branch.query,
                    topics=" ".join(memory.topic),
                )
                if plan.shape is AnswerShape.direct
                else _relevant_memory(
                    memory,
                    branch.query,
                    require_all_terms=plan.shape in {AnswerShape.compare, AnswerShape.chronology},
                )
            )
        ]
        branch_results[branch.branch_id] = {"candidate_memory_ids": [memory.id for memory in memories], "relevant_memory_ids": [memory.id for memory in relevant]}
        existing_evidence_ids = {item.evidence_id.rendered for item in evidence}
        for memory in relevant[: max(0, request.budget.max_selected_evidence - len(evidence))]:
            evidence_item = memory_evidence(memory, branch_id=branch.branch_id, project_id=request.project_id).model_copy(update={"user_id": request.user_id})
            if evidence_item.evidence_id.rendered in existing_evidence_ids:
                continue
            evidence.append(evidence_item)
            existing_evidence_ids.add(evidence_item.evidence_id.rendered)
        telemetry = telemetry.model_copy(update={"candidate_rows": telemetry.candidate_rows + len(memories), "selected_evidence": len(evidence)})

    if plan.shape is AnswerShape.direct:
        explicit_conflicts = await _explicit_contradictions(pool, relevant)
        if explicit_conflicts:
            # A persisted contradicts edge is authoritative structural evidence
            # that the direct branch cannot establish a complete answer.  Do
            # not cite either operand, and do not infer conflicts from prose.
            branch_results["explicit_conflicts"] = {"relationship_ids": list(explicit_conflicts)}
            answer = _base_answer(
                request,
                plan,
                status=AnswerStatus.incomplete,
                completeness=Completeness.incomplete_evidence,
                reason="conflicting_evidence",
            ).model_copy(update={"branch_results": branch_results, "basis": "persisted contradiction relationship"})
        else:
            preferred = [item for item in evidence if item.authority == "authoritative"]
            # Current solution/preference/decision memories win over generic facts;
            # retrieval ordering never makes an unrelated row authoritative. The
            # single branch already contains the only retrieval result.
            preferred = [
                item for item in preferred
                if item.content and any(
                    memory.id == item.evidence_id.value
                    and memory.type in {MemoryType.solution, MemoryType.preference, MemoryType.decision}
                    for memory in memories
                )
            ] or preferred
            if not preferred:
                answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.incomplete_evidence, reason="no content-relevant current solution evidence")
            else:
                answer = _base_answer(request, plan, status=AnswerStatus.success, completeness=Completeness.complete)
                answer = answer.model_copy(update={"evidence": tuple(preferred[: request.budget.max_selected_evidence]), "branch_results": branch_results, "basis": "current scoped solution/preference/decision memory"})
    else:
        mandatory = {branch.branch_id for branch in plan.branches if branch.mandatory}
        supported = {item.branch_id for item in evidence if item.authority == "authoritative"}
        if not mandatory.issubset(supported):
            answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.incomplete_evidence, reason="mandatory operand evidence missing or irrelevant")
            answer = answer.model_copy(update={"evidence": tuple(evidence), "branch_results": branch_results, "telemetry": telemetry.model_copy(update={"selected_evidence": len(evidence)})})
        else:
            ordered = sorted((item for item in evidence if item.authority == "authoritative" and item.occurred_at is not None), key=lambda item: (item.occurred_at, item.branch_id, item.evidence_id.rendered))
            branch_dates: dict[str, datetime] = {}
            for item in ordered:
                branch_dates.setdefault(item.branch_id, item.occurred_at)
            ordered = [
                next(item for item in ordered if item.branch_id == branch_id and item.occurred_at == occurred_at)
                for branch_id, occurred_at in sorted(branch_dates.items(), key=lambda pair: (pair[1], pair[0]))
            ]
            if not ordered:
                answer = _base_answer(request, plan, status=AnswerStatus.incomplete, completeness=Completeness.incomplete_evidence, reason="typed operand dates unavailable")
            else:
                first = ordered[0]
                answer_text = f"{first.branch_id} has the earliest supported evidence at {first.occurred_at.isoformat()}."
                typed = {"answer": answer_text, "ordered_branch_ids": [item.branch_id for item in ordered], "dates": {item.branch_id: item.occurred_at.isoformat() for item in ordered}}
                answer = _base_answer(request, plan, status=AnswerStatus.success, completeness=Completeness.complete)
                answer = answer.model_copy(update={"typed_result": typed, "evidence": tuple(evidence), "branch_results": branch_results, "basis": "deterministic occurred_at ordering", "telemetry": telemetry.model_copy(update={"selected_evidence": len(evidence)})})
    latency = int((datetime.now().timestamp() - started) * 1000)
    telemetry = answer.telemetry.model_copy(update={"latency_ms": latency})
    if latency > request.budget.max_latency_ms:
        answer = answer.model_copy(update={"status": AnswerStatus.incomplete, "completeness": Completeness.cap_exhausted, "incomplete_reason": "max_latency_ms"})
    result = finalize_answer(answer.model_copy(update={"telemetry": telemetry}))
    if result.telemetry.response_bytes > request.budget.max_response_bytes:
        result = result.model_copy(update={"status": AnswerStatus.incomplete, "completeness": Completeness.cap_exhausted, "incomplete_reason": "max_response_bytes", "answer": "The answer exceeded the configured response budget."})
    return result


async def capture_provider_solution(pool: Any, *, user_id: str, project_id: str | None, provider_id: str = "weft") -> Memory:
    """Store the confirmed safe provider configuration precedence fact.

    Only structural metadata is retained. Values, bearer tokens, DSN passwords,
    and environment dumps are deliberately impossible to pass to this helper.
    """
    from weft.store import store_memory
    from weft.models import MemoryCreate, MemorySource
    from weft.db.connection import acquire

    content = (
        "Confirmed Weft provider configuration precedence: load the current-working-directory "
        "`.env` before checking credentials; existing process environment values win, cwd `.env` "
        "fills missing keys, `~/.weft/.env` fills remaining keys, and `~/.weft/config.toml` is "
        "consulted only for configuration fields it defines. Store only safe key names and paths; "
        "never store secret values or a full environment dump. Provider: " + provider_id
    )
    embedding = None
    async with acquire(pool):
        return await store_memory(pool, MemoryCreate(
            type=MemoryType.solution, content=content,
            topic=["provider-configuration", "dotenv", provider_id],
            source=MemorySource.documentation, confidence=1.0,
            project_id=project_id,
        ), embedding=embedding)


__all__ = [
    "AnswerEvidence", "AnswerRequest", "AnswerShape", "AnswerStatus",
    "BudgetTelemetry", "ComposedAnswer", "CompositionPlan", "CompositionalBudget",
    "CountUnit", "DetectionResult", "EvidenceId", "EvidenceKind", "BranchSpec", "TemporalItemEvidence",
    "Completeness", "TemporalItemValidation", "validate_temporal_item", "answer_question", "build_plan", "canonical_json",
    "capture_provider_solution", "detect_answer_shape", "deterministic_render",
    "adapt_recovery_to_answer", "project_recovery_evidence", "finalize_answer", "memory_evidence", "normalize_question", "redact_secret_text",
    "safe_config_metadata", "stable_hash",
]
