"""Bounded, scope-preserving retrieval recovery for belief recall.

Recovery is a retrieval strategist, not an answer generator.  It diagnoses why
an ordinary recall result is insufficient, probes a small number of deterministic
alternate representations, and returns additive evidence metadata.  It never
turns a candidate into an asserted fact and never changes the caller's scope.

The module is intentionally provider-free for the default deterministic path.
Search functions are injected into :class:`RecoveryController` so the gate and
budget contract can be tested without a database, embeddings, or network calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from weft.retrieval_modes import include_agent_provenance, sources_for_mode

logger = logging.getLogger(__name__)

RECOVERY_VERSION = "retrieval-recovery-v1"
RecoveryStageName = Literal[
    "primary", "deterministic_reformulation", "alternate_tier", "model_planner"
]
RecoveryMode = Literal["off", "deterministic", "model"]
RetrievalMode = Literal["face", "code", "all"]
RecoveryStatus = Literal["sufficient", "incomplete", "conflict"]
Answerability = Literal[
    "sufficient", "insufficient_evidence", "conflicting_evidence", "abstained"
]


class RecoveryModel(BaseModel):
    """Strict JSON-friendly base for the recovery envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RecoveryConfig(RecoveryModel):
    """Hard, lower-only budgets for one recovery attempt.

    The defaults are the Milestone A ceiling.  Callers may lower a value for a
    deployment or test, but cannot enlarge the recovery blast radius through a
    request parameter.
    """

    max_stages: int = Field(default=3, ge=1, le=3)
    max_queries_per_branch: int = Field(default=6, ge=1, le=6)
    max_generated_queries_per_branch: int = Field(default=4, ge=0, le=4)
    max_results: int = Field(default=24, ge=1, le=24)
    max_response_bytes: int = Field(default=65_536, ge=1_024, le=65_536)
    timeout_seconds: float = Field(default=2.0, gt=0.0, le=2.0)
    max_provider_calls: int = Field(default=0, ge=0, le=0)
    # Turn-tier caps are exposed even though the controller performs only one
    # alternate-tier stage.  They make internal probe budgets explicit.
    turn_top_k: int = Field(default=6, ge=1, le=24)
    turn_candidate_sql_limit: int = Field(default=18, ge=1, le=24)
    turn_fusion_candidate_limit: int = Field(default=18, ge=1, le=24)
    turn_result_limit: int = Field(default=24, ge=1, le=24)
    max_turn_anchor_probes: int = Field(default=4, ge=0, le=4)

    @model_validator(mode="after")
    def validate_nested_caps(self) -> "RecoveryConfig":
        if self.turn_top_k > self.max_results:
            raise ValueError("turn_top_k may not exceed max_results")
        if self.turn_result_limit > self.max_results:
            raise ValueError("turn_result_limit may not exceed max_results")
        if self.turn_candidate_sql_limit < self.turn_top_k:
            raise ValueError("turn_candidate_sql_limit must be >= turn_top_k")
        if self.turn_fusion_candidate_limit < self.turn_top_k:
            raise ValueError("turn_fusion_candidate_limit must be >= turn_top_k")
        return self


class ScopeSnapshot(RecoveryModel):
    """Exact baseline retrieval policy copied into every recovery probe."""

    user_id: str | None = None
    requested_project_id: str | None = None
    resolved_project_id: str | None = None
    project_policy: Literal["facet_boost", "hard_wall", "baseline"] = "baseline"
    ambient_rls_required: bool = False
    retrieval_mode: RetrievalMode = "face"
    source_allowlist: tuple[str, ...] = ()
    include_agent_provenance: bool = True
    agent_id: str | None = None
    memory_type: str | None = None
    status: str | None = "active"
    topic: str | None = None
    threshold: float = 0.3
    limit: int = 10
    as_of: str | None = None
    since: str | None = None
    until: str | None = None

    @classmethod
    def from_baseline(
        cls,
        *,
        user_id: str | None,
        requested_project_id: str | None,
        resolved_project_id: str | None,
        retrieval_mode: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = "active",
        topic: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
        as_of: datetime | str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        ambient_rls_required: bool = False,
    ) -> "ScopeSnapshot":
        if retrieval_mode not in {"face", "code", "all"}:
            raise ValueError("retrieval_mode must be face, code, or all")
        policy = {
            "face": "facet_boost",
            "code": "hard_wall",
            "all": "baseline",
        }[retrieval_mode]
        def _iso(value: datetime | str | None) -> str | None:
            return value.isoformat() if isinstance(value, datetime) else value
        allowlist = sources_for_mode(retrieval_mode)
        return cls(
            user_id=user_id,
            requested_project_id=requested_project_id,
            resolved_project_id=resolved_project_id,
            project_policy=policy,
            ambient_rls_required=ambient_rls_required,
            retrieval_mode=retrieval_mode,
            source_allowlist=tuple(allowlist or ()),
            include_agent_provenance=include_agent_provenance(retrieval_mode),
            agent_id=agent_id,
            memory_type=memory_type,
            status=status,
            topic=topic,
            threshold=threshold,
            limit=limit,
            as_of=_iso(as_of),
            since=_iso(since),
            until=_iso(until),
        )

    def to_probe_kwargs(self) -> dict[str, Any]:
        """Return only baseline-owned filters; no model/request field is merged."""
        project_id = (
            self.resolved_project_id if self.project_policy == "hard_wall" else None
        )
        return {
            "project_id": project_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "status": self.status,
            "memory_type": self.memory_type,
            "topic": self.topic,
            "threshold": self.threshold,
            "limit": self.limit,
            "sources": list(self.source_allowlist) or None,
            "include_agent_provenance": self.include_agent_provenance,
            "as_of": self.as_of,
            "since": self.since,
            "until": self.until,
        }


class RecoveryCandidate(RecoveryModel):
    """One additive evidence candidate with complete stage provenance."""

    stable_id: str = Field(min_length=1, max_length=200)
    memory_id: str | None = None
    turn_id: str | None = None
    claim_id: str | None = None
    stage: RecoveryStageName
    query_label: str = Field(min_length=1, max_length=160)
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch: str = "branch-direct"
    tier: Literal["belief", "turns"] = "belief"
    source_mode: RetrievalMode = "face"
    matched_signal: str = "candidate_match"
    authority: Literal["authoritative", "candidate"] = "candidate"
    model_generated_query: bool = False
    provenance: tuple[str, ...] = ()
    # Internal payload retained for the Milestone C adapter.  These fields are
    # copied from the already-scoped probe result; they never authorize a fact.
    content: str | None = Field(default=None, max_length=8_000, exclude=True)
    user_id: str | None = None
    project_id: str | None = None
    occurred_at: datetime | None = None
    supersession: Literal["current", "superseded", "unknown"] = "unknown"

    @model_validator(mode="after")
    def stable_identifier_matches_kind(self) -> "RecoveryCandidate":
        if self.memory_id and not self.stable_id.endswith(self.memory_id):
            raise ValueError("stable_id must retain memory_id provenance")
        if self.turn_id and not self.stable_id.endswith(self.turn_id):
            raise ValueError("stable_id must retain turn_id provenance")
        if self.claim_id and not self.stable_id.endswith(self.claim_id):
            raise ValueError("stable_id must retain claim_id provenance")
        return self


class RetrievalStage(RecoveryModel):
    """Bounded diagnostics for one recovery stage.

    ``exact_queries`` is internal execution state and is intentionally omitted
    from :meth:`to_public_dict` and persistence projections.
    """

    stage: RecoveryStageName
    query_label: str = Field(min_length=1, max_length=160)
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(default=1, ge=0, le=6)
    query_hashes: tuple[str, ...] = ()
    tiers: tuple[str, ...] = ("belief",)
    source_modes: tuple[RetrievalMode, ...] = ("face",)
    result_count: int = Field(default=0, ge=0, le=24)
    result_ids: tuple[str, ...] = ()
    coverage: Mapping[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(default=0, ge=0, le=10_000)
    provider_calls: int = Field(default=0, ge=0, le=1)
    terminal: str = "complete"
    error_category: str | None = Field(default=None, max_length=96)
    exact_queries: tuple[str, ...] = Field(default=(), exclude=True)

    def to_public_dict(self) -> dict[str, Any]:
        value = self.model_dump(mode="json", exclude={"exact_queries"})
        value["query_hashes"] = list(self.query_hashes)
        value["result_ids"] = list(self.result_ids)
        value["coverage"] = _bounded_json(dict(self.coverage), 4096)
        return value


class RecoveryShape(RecoveryModel):
    """Deterministic question shape and mandatory coverage branches."""

    shape: Literal["direct", "procedural", "compare", "chronology", "temporal", "unsupported"]
    operation: str
    branches: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    object_terms: tuple[str, ...] = ()
    configuration_terms: tuple[str, ...] = ()
    supported: bool = True
    reasons: tuple[str, ...] = ()


class SufficiencyAssessment(RecoveryModel):
    """Fail-closed coverage result for the current candidate set."""

    retrieval_status: RecoveryStatus
    answerability: Answerability
    trigger: str | None = None
    shape: str
    required_branches: tuple[str, ...] = ()
    covered_branches: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    conflict_flags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def sufficient(self) -> bool:
        return self.answerability == "sufficient" and self.retrieval_status == "sufficient"


class RecoveryOutcome(RecoveryModel):
    """Complete additive recovery envelope."""

    version: str = RECOVERY_VERSION
    supported: bool = True
    attempted: bool = False
    not_attempted: bool = False
    trigger: str | None = None
    retrieval_status: RecoveryStatus = "incomplete"
    answerability: Answerability = "insufficient_evidence"
    stages: tuple[RetrievalStage, ...] = ()
    candidates: tuple[RecoveryCandidate, ...] = ()
    coverage: Mapping[str, Any] = Field(default_factory=dict)
    plans_generated: int = 0
    plans_executed: int = 0
    scope: ScopeSnapshot | None = None
    error_category: str | None = None

    def to_public_dict(self, *, max_bytes: int = 65_536, include_candidates: bool = False) -> dict[str, Any]:
        if not self.supported:
            return {
                "version": self.version,
                "supported": False,
                "attempted": False,
                "not_attempted": True,
            }
        value: dict[str, Any] = {
            "version": self.version,
            "supported": self.supported,
            "attempted": self.attempted,
            "trigger": self.trigger,
            "retrieval_status": self.retrieval_status,
            "answerability": self.answerability,
            "stages": [stage.to_public_dict() for stage in self.stages],
            "coverage": _bounded_json(dict(self.coverage), 4096),
            "plans_generated": self.plans_generated,
            "plans_executed": self.plans_executed,
        }
        if include_candidates:
            value["candidates"] = [
                candidate.model_dump(mode="json", exclude={"content"})
                | ({"content": candidate.content} if candidate.content else {})
                for candidate in self.candidates
            ]
        if self.scope is not None:
            value["scope"] = self.scope.model_dump(mode="json")
        if self.error_category:
            value["error_category"] = self.error_category[:96]
        return _fit_json_bytes(value, max_bytes)


@dataclass(frozen=True, slots=True)
class _QuerySpec:
    text: str
    branch: str
    signal: str
    model_generated: bool = False


_STOPWORDS = frozenset(
    {
        "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
        "before", "but", "by", "can", "could", "did", "do", "does", "for",
        "from", "give", "has", "have", "how", "i", "in", "is", "it", "me",
        "my", "of", "on", "or", "our", "please", "should", "tell", "the",
        "their", "them", "there", "this", "to", "was", "we", "what", "when",
        "where", "which", "with", "would", "you", "your", "here",
    }
)
_ACTION_TERMS = frozenset(
    {"run", "setup", "set", "configure", "configuration", "command", "install", "build", "start", "use", "execute", "deploy", "launch", "environment"}
)
_CONFIG_TERMS = frozenset(
    {"config", "configuration", "configure", "environment", "variable", "variables", "file", "settings", "setup", "command", "flags", "option", "options"}
)
# Query-shape vocabulary intentionally includes nouns such as ``configuration``
# and ``setup``.  Those words must not, by themselves, make a topic/overview
# memory sufficient for a how-to request.  Keep the evidence vocabulary
# separate and conservative: these are operational verbs, not topic labels.
_PROCEDURAL_VERB_TERMS = frozenset(
    {"run", "set", "configure", "install", "build", "start", "use", "execute", "deploy", "launch"}
)
# Procedural evidence must be anchored to something executable or concrete;
# an operational verb in descriptive prose is not enough.  Keep these signals
# generic so the predicate does not become a fixture-specific filename allowlist.
_PROCEDURAL_COMMAND_RE = re.compile(
    r"(?:`[^`]+`|--[a-z0-9][a-z0-9_-]*|(?:^|[\n\r])\s*(?:[$>#]\s*)?[a-z0-9_.-]+\s+--[a-z0-9])",
    re.IGNORECASE,
)
_PROCEDURAL_ARTIFACT_RE = re.compile(
    r"\b[a-z0-9_./-]+\.(?:ini|cfg|conf|yaml|yml|toml|json|py|sh|bash|zsh|js|ts|sql|csv|ndjson|parquet)\b"
    r"|(?<![a-z0-9_])/(?:[a-z0-9_.-]+/)+[a-z0-9_.-]+",
    re.IGNORECASE,
)
# These markers describe evidence that does not establish a requested fact.
# Keep the vocabulary explicit and bounded: ordinary negation should not turn
# an otherwise valid candidate into a conflict.
_NEGATIVE_UNKNOWN_RE = re.compile(
    r"\b(?:unknown|undisclosed|unavailable|"
    r"not\s+(?:disclosed|available|known|provided|specified|listed|recorded|found|given)|"
    r"no\s+(?:record|records|information|data|entry|value|result|results|match|matches|evidence|documentation|details?)|"
    r"none\s+(?:found|available|provided|recorded|listed|known))\b",
    re.IGNORECASE,
)
def _has_actionable_procedural_evidence(value: str) -> bool:
    """Return whether text contains an anchored executable instruction."""
    content = str(value)
    if _PROCEDURAL_COMMAND_RE.search(content):
        return True
    artifact = _PROCEDURAL_ARTIFACT_RE.search(content)
    if artifact is None:
        return False
    # A concrete artifact only counts when tied to an operation/code context;
    # a filename mentioned as an isolated topic is still insufficient.
    before = content[max(0, artifact.start() - 80):artifact.start()]
    after = content[artifact.end():artifact.end() + 80]
    context = f"{before} {after}"
    return bool(set(_tokens(context)) & (_PROCEDURAL_VERB_TERMS | {
        "code", "data", "file", "path", "values", "settings", "using",
        "uses", "used", "configured", "command", "script", "operation",
    }))




def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_./-]+", str(value).casefold())


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for t in _tokens(value) if len(t) > 2 and t not in _STOPWORDS))


def _normalize_query(value: str) -> str:
    return " ".join(str(value).strip().split()).strip(" ?.,")


def _hash_query(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_json(value: Any, max_bytes: int) -> Any:
    """Return a JSON-safe, bounded projection without leaking arbitrary text."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return {"truncated": True}
    if len(encoded.encode("utf-8")) <= max_bytes:
        return value
    return {"truncated": True, "bytes": max_bytes}


def _fit_json_bytes(value: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Trim diagnostics only; the legacy recall payload is owned by the caller."""
    if len(json.dumps(value, separators=(",", ":"), default=str).encode()) <= max_bytes:
        return value
    clipped = dict(value)
    stages = list(clipped.get("stages", []))
    while stages and len(json.dumps(clipped, separators=(",", ":"), default=str).encode()) > max_bytes:
        stages.pop()
        clipped["stages"] = stages
    clipped["diagnostics_truncated"] = True
    # Keep the contract small even when max_bytes is unusually low.
    if len(json.dumps(clipped, separators=(",", ":"), default=str).encode()) > max_bytes:
        return {
            "version": clipped.get("version", RECOVERY_VERSION),
            "supported": True,
            "attempted": clipped.get("attempted", True),
            "trigger": clipped.get("trigger"),
            "retrieval_status": clipped.get("retrieval_status", "incomplete"),
            "answerability": clipped.get("answerability", "insufficient_evidence"),
            "diagnostics_truncated": True,
        }
    return clipped


def _redacted_label(query: str) -> str:
    """Stable, bounded label; exact query text remains internal only."""
    text = _normalize_query(query)
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret|bearer|dsn)\s*[=:]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(?:postgres(?:ql)?|mysql|redis)://[^\s]+", lambda m: m.group(0).split("://", 1)[0] + "://[REDACTED]", text)
    if len(text) > 160:
        text = text[:157] + "..."
    return text or "[empty]"


def _candidate_id(value: Any, *, tier: str = "belief") -> str | None:
    if isinstance(value, RecoveryCandidate):
        return value.stable_id
    if isinstance(value, Mapping):
        for key in ("stable_id", "id", "memory_id", "turn_id", "claim_id"):
            raw = value.get(key)
            if raw:
                raw = str(raw)
                if ":" in raw and raw.split(":", 1)[0] in {"memory", "turn", "claim"}:
                    return raw
                prefix = (
                    "claim" if key == "claim_id" else
                    "turn" if key == "turn_id" or tier == "turns" else
                    "memory"
                )
                return f"{prefix}:{raw}"
        payload = value.get("payload")
        if isinstance(payload, Mapping):
            return _candidate_id(payload, tier=tier)
    for key in ("stable_id", "id", "memory_id", "turn_id", "claim_id"):
        raw = getattr(value, key, None)
        if raw:
            raw = str(raw)
            if ":" in raw and raw.split(":", 1)[0] in {"memory", "turn", "claim"}:
                return raw
            prefix = (
                "claim" if key == "claim_id" else
                "turn" if key == "turn_id" or tier == "turns" else
                "memory"
            )
            return f"{prefix}:{raw}"
    return None


def _candidate_content(value: Any) -> str:
    if isinstance(value, Mapping):
        payload = value.get("payload")
        if isinstance(payload, Mapping):
            return str(payload.get("content", ""))
        return str(value.get("content", ""))
    if hasattr(value, "memory"):
        return str(getattr(value.memory, "content", ""))
    return str(getattr(value, "content", ""))


def _candidate_branch(value: Any, default: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get("branch") or value.get("branch_id") or default)
    return str(getattr(value, "branch", None) or getattr(value, "branch_id", None) or default)


def _candidate_authority(value: Any) -> Literal["authoritative", "candidate"]:
    raw = value.get("authority") if isinstance(value, Mapping) else getattr(value, "authority", None)
    return "authoritative" if raw == "authoritative" else "candidate"


def _candidate_project(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return value.get("project_id") or (value.get("payload") or {}).get("project_id") if isinstance(value.get("payload"), Mapping) else value.get("project_id")
    memory = getattr(value, "memory", None)
    return getattr(memory, "project_id", None) or getattr(value, "project_id", None)


def _candidate_field(value: Any, *names: str) -> Any:
    payload = value.get("payload") if isinstance(value, Mapping) else None
    memory = getattr(value, "memory", None)
    for source in (value, payload, memory):
        if isinstance(source, Mapping):
            for name in names:
                if source.get(name) is not None:
                    return source[name]
        elif source is not None:
            for name in names:
                found = getattr(source, name, None)
                if found is not None:
                    return found
    return None


def _candidate_supersession(value: Any) -> Literal["current", "superseded", "unknown"]:
    """Map canonical row status vocabulary to the recovery candidate contract."""
    raw = str(_candidate_field(value, "supersession", "status") or "unknown")
    if raw == "active":
        return "current"
    if raw in {"current", "superseded"}:
        return raw  # type: ignore[return-value]
    return "unknown"


def _candidate_payload(value: Any) -> dict[str, Any]:
    """Copy only bounded, non-authorizing evidence metadata from a probe row."""
    content = str(_candidate_field(value, "content") or "")[:8_000]
    occurred_at = _candidate_field(value, "occurred_at", "updated_at", "created_at")
    if isinstance(occurred_at, str):
        try:
            occurred_at = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError:
            occurred_at = None
    return {
        "content": content or None,
        "user_id": _candidate_field(value, "user_id"),
        "project_id": _candidate_project(value),
        "occurred_at": occurred_at,
        "supersession": _candidate_supersession(value),
    }


def _extract_operands(query: str) -> tuple[str, ...]:
    q = _normalize_query(query)
    patterns = (
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.,]|$)",
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[?.,]|$)",
        r"\b(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:[?.,]|$)",
        r"\bcompare\s+(.+?)\s+(?:and|with)\s+(.+?)(?:[?.,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            values = tuple(_normalize_query(item) for item in match.groups() if _normalize_query(item))
            if len(values) >= 2:
                return values[:2]
    quoted = tuple(item.strip() for item in re.findall(r"[\"']([^\"']{1,120})[\"']", q))
    return quoted[:2] if len(quoted) >= 2 else ()


def build_recovery_shape(
    query: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    topic: str | None = None,
    as_of: datetime | None = None,
    timezone_name: str = "UTC",
    retrieval_mode: RetrievalMode = "face",
    existing_incomplete: bool = False,
) -> RecoveryShape:
    """Classify a query for recovery without claiming an answer.

    The structured classifier remains the canonical vocabulary where it can
    classify the request.  Procedural questions are an additive recovery shape
    because the normal classifier intentionally treats them as unsupported.
    """
    normalized = _normalize_query(query)
    lowered = normalized.casefold()
    operands = _extract_operands(normalized)
    try:
        from weft.turn_recall import extract_anchors
        anchors = tuple(extract_anchors(normalized))
    except Exception:
        anchors = ()
    compare_signal = bool(re.search(r"\b(?:compare|versus|vs\.?|between)\b", lowered))
    chronology_signal = bool(re.search(r"\b(?:chronolog|earliest|latest|who came first|before|after)\b", lowered)) and bool(operands or anchors)
    procedural_signal = bool(
        re.search(
            r"\b(?:how do (?:i|we)|how can (?:i|we)|how to|setup|set up|configure|configuration|command|run|environment|benchmark|deploy|install|build)\b",
            lowered,
        )
    )
    action_terms = tuple(sorted(set(_meaningful_tokens(normalized)) & _ACTION_TERMS))
    configuration_terms = tuple(sorted(set(_meaningful_tokens(normalized)) & _CONFIG_TERMS))
    object_terms = tuple(
        token for token in _meaningful_tokens(normalized)
        if token not in _ACTION_TERMS and token not in _CONFIG_TERMS
    )[:8]
    reasons: list[str] = []
    if existing_incomplete:
        reasons.append("existing_incomplete_evidence")
    if not normalized:
        return RecoveryShape(shape="unsupported", operation="unsupported", supported=False, reasons=("empty_query",))
    if compare_signal and len(operands) < 2:
        reasons.append("named_operands_required")
        return RecoveryShape(
            shape="compare", operation="compare", operands=operands,
            branches=tuple(f"branch-{i + 1}" for i in range(2)),
            supported=False, reasons=tuple(reasons), action_terms=action_terms,
            object_terms=object_terms, configuration_terms=configuration_terms,
        )
    if chronology_signal and len(operands) >= 2:
        return RecoveryShape(
            shape="chronology", operation="chronology", operands=operands,
            branches=("branch-1", "branch-2"), anchors=anchors,
            action_terms=action_terms, object_terms=object_terms,
            configuration_terms=configuration_terms, reasons=tuple(reasons),
        )
    if compare_signal and len(operands) >= 2 and not anchors:
        return RecoveryShape(
            shape="compare", operation="compare", operands=operands,
            branches=("branch-1", "branch-2"), anchors=anchors,
            action_terms=action_terms, object_terms=object_terms,
            configuration_terms=configuration_terms, reasons=tuple(reasons),
        )
    if anchors:
        reasons.append("temporal_anchor_coverage")
        return RecoveryShape(
            shape="temporal", operation="temporal", anchors=anchors,
            branches=tuple(f"anchor-{i + 1}" for i in range(len(anchors))),
            supported=True, reasons=tuple(reasons), action_terms=action_terms,
            object_terms=object_terms, configuration_terms=configuration_terms,
        )
    if procedural_signal:
        if not action_terms:
            reasons.append("action_signal_missing")
        return RecoveryShape(
            shape="procedural", operation="procedural", branches=("branch-direct",),
            supported=True, reasons=tuple(reasons), action_terms=action_terms,
            object_terms=object_terms, configuration_terms=configuration_terms,
        )
    # Use structured_recall only as a vocabulary/compatibility signal.  Missing
    # scope must not make ordinary shape diagnostics throw or widen scope.
    try:
        from weft.structured_recall import classify_query
        plan = classify_query(
            normalized,
            user_id=user_id or "__recovery_scope__",
            project_id=project_id or "__recovery_scope__",
            topic=topic,
            as_of=as_of,
            timezone_name=timezone_name,
            retrieval_mode=retrieval_mode,
        )
        if plan.family == "multi_anchor_temporal":
            return RecoveryShape(
                shape="temporal", operation="temporal", anchors=tuple(plan.anchors),
                branches=tuple(f"anchor-{i + 1}" for i in range(len(plan.anchors))),
                reasons=tuple(reasons),
            )
    except Exception:
        logger.debug("structured recovery shape classification failed", exc_info=True)
    return RecoveryShape(
        shape="direct", operation="direct", branches=("branch-direct",),
        reasons=tuple(reasons), action_terms=action_terms,
        object_terms=object_terms, configuration_terms=configuration_terms,
    )


def _has_negative_unknown_evidence(content: str, shape: RecoveryShape) -> bool:
    """Return whether a direct candidate denies the requested fact locally."""
    requested_terms = set(shape.object_terms)
    if not requested_terms:
        return False
    text = str(content)
    for marker in _NEGATIVE_UNKNOWN_RE.finditer(text):
        start = max(
            text.rfind(separator, 0, marker.start())
            for separator in (".", "!", "?", ";", "\\n", "\\r")
        ) + 1
        boundaries = [text.find(separator, marker.end()) for separator in (".", "!", "?", ";", "\\n", "\\r")]
        end = min((boundary for boundary in boundaries if boundary >= 0), default=len(text))
        if requested_terms & set(_meaningful_tokens(text[start:end])):
            return True
    return False


def _value_matches_branch(content: str, branch: str, shape: RecoveryShape) -> bool:
    terms = set(_meaningful_tokens(content))
    if branch.startswith("branch-") and branch not in {"branch-direct"}:
        try:
            index = int(branch.rsplit("-", 1)[-1]) - 1
            operand = shape.operands[index]
        except (ValueError, IndexError):
            operand = ""
        operand_terms = set(_meaningful_tokens(operand))
        return bool(operand_terms) and operand_terms.issubset(terms)
    if branch.startswith("anchor-"):
        try:
            index = int(branch.rsplit("-", 1)[-1]) - 1
            anchor = shape.anchors[index]
        except (ValueError, IndexError):
            anchor = ""
        anchor_terms = set(_meaningful_tokens(anchor))
        return bool(anchor_terms) and anchor_terms.issubset(terms)
    query_terms = set(shape.object_terms) | set(shape.configuration_terms) | set(shape.action_terms)
    return bool(query_terms & terms) if query_terms else bool(terms)


def assess_sufficiency(
    query: str,
    results: Iterable[Any] | None = None,
    *,
    shape: RecoveryShape | None = None,
    existing_incomplete: bool = False,
    explicit_conflicts: Sequence[str] = (),
) -> SufficiencyAssessment:
    """Assess evidence coverage and return explicit incomplete/conflict states."""
    recovery_shape = shape or build_recovery_shape(query, existing_incomplete=existing_incomplete)
    values = list(results or ())
    ids = tuple(dict.fromkeys(item_id for value in values if (item_id := _candidate_id(value))))
    conflicts = tuple(dict.fromkeys(str(flag) for flag in explicit_conflicts if flag))
    if not values:
        return SufficiencyAssessment(
            retrieval_status="incomplete", answerability="insufficient_evidence",
            trigger=(recovery_shape.reasons[0] if recovery_shape.reasons else "zero_results"),
            shape=recovery_shape.shape, required_branches=recovery_shape.branches,
            candidate_ids=ids, conflict_flags=conflicts, reasons=("empty_results",),
        )
    if conflicts:
        return SufficiencyAssessment(
            retrieval_status="conflict", answerability="conflicting_evidence",
            trigger="explicit_conflict", shape=recovery_shape.shape,
            required_branches=recovery_shape.branches, candidate_ids=ids,
            conflict_flags=conflicts, reasons=("conflicting_candidates",),
        )
    covered: list[str] = []
    for branch in recovery_shape.branches:
        if any(_value_matches_branch(_candidate_content(value), branch, recovery_shape) for value in values):
            covered.append(branch)
    required = recovery_shape.branches or ("branch-direct",)
    if recovery_shape.shape == "procedural":
        # A bare topic noun such as "configuration" does not establish that
        # the result contains procedural evidence.  Require an operational
        # verb or a concrete configuration artifact instead; otherwise a
        # topic/overview summary can incorrectly satisfy a how-to query.
        # A noun like ``setup`` or ``configuration`` is a topic label, not
        # procedural evidence.  Accept only an operational verb or a concrete
        # artifact/flag syntax that can anchor instructions.
        actionable = any(_has_actionable_procedural_evidence(_candidate_content(value)) for value in values)
        if not actionable:
            return SufficiencyAssessment(
                retrieval_status="incomplete", answerability="insufficient_evidence",
                trigger="procedural_topic_only", shape=recovery_shape.shape,
                required_branches=required, covered_branches=tuple(covered), candidate_ids=ids,
                reasons=("missing_actionable_command_or_configuration_signal",),
            )
    if existing_incomplete:
        return SufficiencyAssessment(
            retrieval_status="incomplete", answerability="insufficient_evidence",
            trigger="existing_incomplete_evidence", shape=recovery_shape.shape,
            required_branches=required, covered_branches=tuple(covered), candidate_ids=ids,
        )
    if not recovery_shape.supported:
        return SufficiencyAssessment(
            retrieval_status="incomplete", answerability="insufficient_evidence",
            trigger=(recovery_shape.reasons[0] if recovery_shape.reasons else "unsupported_shape"),
            shape=recovery_shape.shape, required_branches=required,
            covered_branches=tuple(covered), candidate_ids=ids,
            reasons=recovery_shape.reasons,
        )
    if recovery_shape.shape == "direct" and any(
        _has_negative_unknown_evidence(_candidate_content(value), recovery_shape)
        for value in values
    ):
        return SufficiencyAssessment(
            retrieval_status="incomplete", answerability="insufficient_evidence",
            trigger="negative_or_unknown_evidence", shape=recovery_shape.shape,
            required_branches=required, covered_branches=tuple(covered), candidate_ids=ids,
            reasons=("evidence_denies_or_does_not_establish_requested_fact",),
        )
    if not set(required).issubset(covered):
        trigger = "missing_required_operand" if recovery_shape.shape in {"compare", "chronology"} else (
            "missing_temporal_anchor" if recovery_shape.shape == "temporal" else "weak_or_irrelevant_results"
        )
        return SufficiencyAssessment(
            retrieval_status="incomplete", answerability="insufficient_evidence",
            trigger=trigger, shape=recovery_shape.shape, required_branches=required,
            covered_branches=tuple(covered), candidate_ids=ids,
            reasons=("mandatory_coverage_missing",),
        )
    return SufficiencyAssessment(
        retrieval_status="sufficient", answerability="sufficient", trigger=None,
        shape=recovery_shape.shape, required_branches=required,
        covered_branches=tuple(covered), candidate_ids=ids,
    )


def build_reformulations(query: str, *, shape: RecoveryShape | None = None, max_queries: int = 6) -> list[str]:
    """Build conservative, deduplicated alternate representations."""
    normalized = _normalize_query(query)
    if not normalized or max_queries <= 0:
        return []
    shape = shape or build_recovery_shape(normalized)
    specs: list[_QuerySpec] = []
    def add(text: str, signal: str, branch: str = "branch-direct") -> None:
        value = _normalize_query(text)
        if len(value) < 3:
            return
        if any(existing.text.casefold() == value.casefold() for existing in specs):
            return
        specs.append(_QuerySpec(value, branch, signal))
    add(normalized, "original")
    meaningful = _meaningful_tokens(normalized)
    if meaningful:
        add(" ".join(meaningful), "meaningful_terms")
    if shape.shape == "procedural":
        object_text = " ".join(shape.object_terms) or " ".join(meaningful)
        if object_text:
            add(f"setup {object_text}", "setup")
            add(f"run {object_text}", "run_object")
            add(f"command {object_text}", "command")
            add(f"configuration environment {object_text}", "configuration")
            add(f"environment file {object_text}", "environment_file")
    elif shape.shape in {"compare", "chronology"}:
        for index, operand in enumerate(shape.operands[:2]):
            branch = f"branch-{index + 1}"
            add(operand, "operand", branch)
            add(f"{operand} first child", "relation_first_child", branch)
            add(f"{operand} became a parent", "relation_parenthood", branch)
            add(f"{operand} firstborn", "relation_firstborn", branch)
    elif shape.shape == "temporal":
        try:
            from weft.turn_recall import event_focused_anchor_query, temporal_query_variants
            for index, anchor in enumerate(shape.anchors):
                branch = f"anchor-{index + 1}"
                add(anchor, "temporal_anchor", branch)
                focused = event_focused_anchor_query(anchor)
                if focused:
                    add(focused, "event_focused_anchor", branch)
            for variant in temporal_query_variants(normalized, include_embedded_temporal_variant=True):
                add(variant, "temporal_variant")
        except Exception:
            logger.warning(
                "recovery temporal query variant expansion failed",
                exc_info=True,
            )
    # Exact repository identifiers and filenames are preserved as-is, never
    # generated from a guessed path.  This catches ``foo_bar.py`` / ``--flag``
    # misses without turning a provider into a scope authority.
    for token in re.findall(r"[A-Za-z0-9_.-]+(?:\.py|\.yaml|\.yml|\.toml|\.json|_[A-Za-z0-9_-]+)", normalized):
        add(token, "exact_identifier")
    return [spec.text for spec in specs[:max_queries]]


def _iter_result_values(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        if isinstance(result.get("results"), list):
            return list(result["results"])
        if isinstance(result.get("turns"), list):
            return list(result["turns"])
        if isinstance(result.get("candidates"), list):
            return list(result["candidates"])
        return [result]
    if isinstance(result, (list, tuple, set)):
        return list(result)
    return [result]


SearchFn = Callable[[str, ScopeSnapshot, int], Awaitable[Sequence[Any]]]


class RecoveryController:
    """Execute primary → deterministic → one alternate-tier recovery."""

    def __init__(
        self,
        *,
        config: RecoveryConfig | None = None,
        memory_search: SearchFn | None = None,
        alternate_search: SearchFn | None = None,
    ) -> None:
        self.config = config or RecoveryConfig()
        self.memory_search = memory_search
        self.alternate_search = alternate_search
        self._running = False

    async def recover(
        self,
        query: str,
        *,
        primary_results: Sequence[Any] = (),
        scope: ScopeSnapshot | None = None,
        shape: RecoveryShape | None = None,
        existing_incomplete: bool = False,
        explicit_conflicts: Sequence[str] = (),
        primary_tier: Literal["belief", "turns"] = "belief",
    ) -> RecoveryOutcome:
        if self._running:
            raise RuntimeError("recovery is not recursive")
        self._running = True
        started = time.perf_counter()
        try:
            if scope is None:
                scope = ScopeSnapshot.from_baseline(
                    user_id=None, requested_project_id=None,
                    resolved_project_id=None, retrieval_mode="face",
                )
            recovery_shape = shape or build_recovery_shape(query)
            all_values = list(primary_results)
            initial = assess_sufficiency(
                query, all_values, shape=recovery_shape,
                existing_incomplete=existing_incomplete,
                explicit_conflicts=explicit_conflicts,
            )
            stages: list[RetrievalStage] = [self._make_stage(
                "primary", [query], all_values, scope, recovery_shape, initial,
                started_at=started,
            )]
            if initial.sufficient:
                return self._outcome_from(
                    initial, stages, all_values, scope, attempted=False,
                    query=query, started=started,
                )
            if self.memory_search is not None and self.config.max_stages >= 2:
                specs = self._query_specs(query, recovery_shape)
                reformulation_values: list[Any] = []
                query_texts: list[str] = []
                deadline = started + self.config.timeout_seconds
                for spec in specs[: self.config.max_queries_per_branch]:
                    if time.perf_counter() >= deadline:
                        break
                    query_texts.append(spec.text)
                    try:
                        remaining = max(0.001, deadline - time.perf_counter())
                        result = await asyncio.wait_for(
                            self.memory_search(spec.text, scope, min(self.config.max_results, scope.limit)),
                            timeout=remaining,
                        )
                        reformulation_values.extend(_iter_result_values(result))
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        stages.append(self._make_stage(
                            "deterministic_reformulation", query_texts, reformulation_values,
                            scope, recovery_shape, initial, started_at=started,
                            terminal="timeout", error_category="timeout",
                        ))
                        break
                    except Exception as exc:  # noqa: BLE001 - stage is diagnostic
                        stages.append(self._make_stage(
                            "deterministic_reformulation", query_texts, reformulation_values,
                            scope, recovery_shape, initial, started_at=started,
                            terminal="error", error_category=type(exc).__name__[:96],
                        ))
                        break
                if query_texts and not any(stage.stage == "deterministic_reformulation" for stage in stages):
                    stages.append(self._make_stage(
                        "deterministic_reformulation", query_texts, reformulation_values,
                        scope, recovery_shape, initial, started_at=started,
                    ))
                all_values = _dedupe_values([*all_values, *reformulation_values])
                assessment = assess_sufficiency(
                    query, all_values, shape=recovery_shape,
                    existing_incomplete=existing_incomplete,
                    explicit_conflicts=explicit_conflicts,
                )
                if assessment.sufficient:
                    return self._outcome_from(
                        assessment, stages, all_values, scope, attempted=True,
                        query=query, started=started,
                    )
            else:
                assessment = initial
            if (
                self.alternate_search is not None
                and self.config.max_stages >= 3
                and not assessment.sufficient
                and time.perf_counter() - started < self.config.timeout_seconds
            ):
                alternate_values: list[Any] = []
                try:
                    remaining = max(0.001, self.config.timeout_seconds - (time.perf_counter() - started))
                    alternate_result = await asyncio.wait_for(
                        self.alternate_search(query, scope, min(self.config.max_results, scope.limit)),
                        timeout=remaining,
                    )
                    alternate_values = _iter_result_values(alternate_result)
                    stages.append(self._make_stage(
                        "alternate_tier", [query], alternate_values, scope,
                        recovery_shape, assessment, started_at=started,
                    ))
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    stages.append(self._make_stage(
                        "alternate_tier", [query], alternate_values, scope,
                        recovery_shape, assessment, started_at=started,
                        terminal="timeout", error_category="timeout",
                    ))
                except Exception as exc:  # noqa: BLE001
                    stages.append(self._make_stage(
                        "alternate_tier", [query], alternate_values, scope,
                        recovery_shape, assessment, started_at=started,
                        terminal="error", error_category=type(exc).__name__[:96],
                    ))
                all_values = _dedupe_values([*all_values, *alternate_values])
                assessment = assess_sufficiency(
                    query, all_values, shape=recovery_shape,
                    existing_incomplete=existing_incomplete,
                    explicit_conflicts=explicit_conflicts,
                )
            return self._outcome_from(
                assessment, stages, all_values, scope, attempted=True,
                query=query, started=started,
            )
        finally:
            self._running = False

    def _query_specs(self, query: str, shape: RecoveryShape) -> list[_QuerySpec]:
        normalized = _normalize_query(query).casefold()
        return [
            _QuerySpec(text, branch="branch-direct", signal="deterministic")
            for text in build_reformulations(
                query, shape=shape, max_queries=self.config.max_queries_per_branch,
            )
            if text.casefold() != normalized
        ]

    def _make_stage(
        self,
        stage: RecoveryStageName,
        queries: Sequence[str],
        values: Sequence[Any],
        scope: ScopeSnapshot,
        shape: RecoveryShape,
        assessment: SufficiencyAssessment,
        *,
        started_at: float,
        terminal: str = "complete",
        error_category: str | None = None,
    ) -> RetrievalStage:
        bounded_queries = tuple(_normalize_query(q) for q in queries[: self.config.max_queries_per_branch] if _normalize_query(q))
        hashes = tuple(_hash_query(q) for q in bounded_queries)
        ids = tuple(dict.fromkeys(item_id for item_id in (_candidate_id(value, tier="turns" if stage == "alternate_tier" else "belief") for value in values) if item_id))[: self.config.max_results]
        coverage = assess_sufficiency(
            shape=shape, query=shape.operation, results=values,
        )
        return RetrievalStage(
            stage=stage,
            query_label=_redacted_label(bounded_queries[0] if bounded_queries else "[none]"),
            query_hash=_hash_query(bounded_queries[0] if bounded_queries else "[none]"),
            query_count=len(bounded_queries),
            query_hashes=hashes,
            tiers=("turns",) if stage == "alternate_tier" else ("belief",),
            source_modes=(scope.retrieval_mode,),
            result_count=len(ids),
            result_ids=ids,
            coverage={
                "shape": coverage.shape,
                "required_branches": list(coverage.required_branches),
                "covered_branches": list(coverage.covered_branches),
                "answerability": coverage.answerability,
            },
            latency_ms=min(10_000, max(0, int((time.perf_counter() - started_at) * 1000))),
            provider_calls=0,
            terminal=terminal,
            error_category=error_category,
            exact_queries=bounded_queries,
        )

    def _outcome_from(
        self,
        assessment: SufficiencyAssessment,
        stages: Sequence[RetrievalStage],
        values: Sequence[Any],
        scope: ScopeSnapshot,
        *,
        attempted: bool,
        query: str,
        started: float,
    ) -> RecoveryOutcome:
        candidates: list[RecoveryCandidate] = []
        seen: set[str] = set()
        for stage in stages:
            for value in values:
                stable = _candidate_id(value, tier="turns" if stage.stage == "alternate_tier" else "belief")
                if not stable or stable in seen:
                    continue
                if stable not in set(stage.result_ids):
                    continue
                seen.add(stable)
                memory_id = stable.split(":", 1)[1] if stable.startswith("memory:") else None
                turn_id = stable.split(":", 1)[1] if stable.startswith("turn:") else None
                claim_id = stable.split(":", 1)[1] if stable.startswith("claim:") else None
                payload = _candidate_payload(value)
                candidates.append(RecoveryCandidate(
                    stable_id=stable, memory_id=memory_id, turn_id=turn_id, claim_id=claim_id,
                    stage=stage.stage, query_label=stage.query_label, query_hash=stage.query_hash,
                    branch=_candidate_branch(value, "branch-direct"),
                    tier="turns" if stage.stage == "alternate_tier" else "belief",
                    source_mode=scope.retrieval_mode,
                    matched_signal="recovery_candidate",
                    authority="candidate",
                    model_generated_query=False,
                    provenance=(stage.stage, stage.query_hash),
                    **payload,
                ))
        return RecoveryOutcome(
            supported=True, attempted=attempted, not_attempted=False,
            trigger=assessment.trigger,
            retrieval_status=assessment.retrieval_status,
            answerability=assessment.answerability,
            stages=tuple(stages), candidates=tuple(candidates[: self.config.max_results]),
            coverage={
                "shape": assessment.shape,
                "required_branches": list(assessment.required_branches),
                "covered_branches": list(assessment.covered_branches),
                "candidate_ids": list(assessment.candidate_ids[: self.config.max_results]),
                "conflict_flags": list(assessment.conflict_flags),
            },
            plans_generated=0, plans_executed=max(0, len(stages) - 1), scope=scope,
            error_category=("timeout" if time.perf_counter() - started >= self.config.timeout_seconds and not assessment.sufficient else None),
        )


class PlannerQuery(RecoveryModel):
    """Strict model-planner search candidate; never an answer or scope."""

    branch: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=512)
    tier: Literal["belief", "turns"] = "belief"
    source_mode: RetrievalMode = "face"
    relation_hint: str | None = Field(default=None, max_length=120)
    alias_hint: str | None = Field(default=None, max_length=120)

    @field_validator("query")
    @classmethod
    def reject_query_injection(cls, value: str) -> str:
        text = _normalize_query(value)
        if not text:
            raise ValueError("planner query must not be blank")
        if re.search(r"(?i)\b(select|insert|update|delete|drop|alter)\b|;|--|/\*|\*/", text):
            raise ValueError("planner query contains SQL-like content")
        return text


class PlannerResponse(RecoveryModel):
    plans: tuple[PlannerQuery, ...] = Field(default=())


def parse_planner_response(
    payload: str | bytes | Mapping[str, Any],
    *,
    allowed_branches: Iterable[str],
    allowed_source_mode: RetrievalMode,
    max_queries: int = 4,
) -> tuple[PlannerQuery, ...]:
    """Parse only strict search-candidate JSON; reject answers and scope fields."""
    if isinstance(payload, (str, bytes)):
        value = json.loads(payload)
    else:
        value = dict(payload)
    if not isinstance(value, Mapping):
        raise ValueError("planner response must be a JSON object")
    allowed_top = {"plans", "queries"}
    if set(value) - allowed_top:
        raise ValueError("planner response contains unknown fields")
    raw_plans = value.get("plans", value.get("queries", []))
    if not isinstance(raw_plans, list):
        raise ValueError("planner plans must be a list")
    branches = set(allowed_branches)
    parsed: list[PlannerQuery] = []
    for raw in raw_plans[: max_queries + 1]:
        if not isinstance(raw, Mapping):
            raise ValueError("planner plan must be an object")
        if set(raw) - {"branch", "query", "tier", "source_mode", "relation_hint", "alias_hint"}:
            raise ValueError("planner plan contains unknown or authoritative fields")
        plan = PlannerQuery.model_validate(raw)
        if plan.branch not in branches:
            raise ValueError("planner selected an unknown branch")
        if plan.source_mode != allowed_source_mode:
            raise ValueError("planner cannot authorize a different source mode")
        if any(existing.query.casefold() == plan.query.casefold() for existing in parsed):
            continue
        parsed.append(plan)
        if len(parsed) > max_queries:
            raise ValueError("planner query cap exceeded")
    return tuple(parsed)


def _dedupe_values(values: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        key = _candidate_id(value) or f"raw:{id(value)}"
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


async def run_recovery(
    query: str,
    *,
    primary_results: Sequence[Any] = (),
    scope: ScopeSnapshot | None = None,
    config: RecoveryConfig | None = None,
    memory_search: SearchFn | None = None,
    alternate_search: SearchFn | None = None,
    shape: RecoveryShape | None = None,
    existing_incomplete: bool = False,
    explicit_conflicts: Sequence[str] = (),
) -> RecoveryOutcome:
    """Convenience wrapper around :class:`RecoveryController`."""
    return await RecoveryController(
        config=config, memory_search=memory_search, alternate_search=alternate_search,
    ).recover(
        query, primary_results=primary_results, scope=scope, shape=shape,
        existing_incomplete=existing_incomplete, explicit_conflicts=explicit_conflicts,
    )


def unsupported_recovery() -> dict[str, Any]:
    """Bounded additive marker for legacy paths owned by another tier."""
    return {
        "version": RECOVERY_VERSION,
        "supported": False,
        "attempted": False,
        "not_attempted": True,
    }


__all__ = [
    "RECOVERY_VERSION", "RecoveryConfig", "ScopeSnapshot", "RecoveryCandidate",
    "RetrievalStage", "RecoveryShape", "SufficiencyAssessment", "RecoveryOutcome",
    "PlannerQuery", "PlannerResponse", "RecoveryController", "assess_sufficiency",
    "build_recovery_shape", "build_reformulations", "parse_planner_response",
    "run_recovery", "unsupported_recovery",
]
