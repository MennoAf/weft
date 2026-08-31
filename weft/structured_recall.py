"""Opt-N deterministic structured recall supplement.

This module is deliberately additive: it never changes the normal recall path.
It plans a query, performs only bounded deterministic probes, and returns a
versioned evidence bundle whose records retain enough provenance for benchmark
attribution. Raw memories and turns remain the factual substrate; lexical
turn matches are explicitly marked as candidates. Temporal ordering scaffolding
(e.g. ``earliest``/``latest``) is never treated as a factual anchor, and a
lexical hit must contain the meaningful terms of its declared anchor before it
can be surfaced.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg

from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.episode_turns import _row_to_turn
from weft.models import Entity, Memory, MemoryStatus
from weft.schema.versioning import SYSTEM_GLOBAL_USER_ID
from weft.store import _row_to_memory
from weft.topic_gather import gather_topic_memories
from weft.topic_resolution import resolve_topic
from weft.turn_recall import extract_anchors

OPT_N_VERSION = "opt-n-v1"
TEMPORAL_GRAMMAR_VERSION = "temporal_grammar_v1"

QueryFamily = Literal[
    "enumeration", "entity_evidence", "multi_anchor_temporal", "multi_session",
    "direct", "unsupported"
]
CanonicalShape = Literal["direct", "count", "compare", "chronology", "enumeration", "unsupported"]
# Plan classification and evidence retrieval are deliberately separate contracts.
# ``empty`` is an evidence outcome, never a plan outcome.
PlanStatus = Literal["supported", "unsupported", "ambiguous", "error"]
EvidenceStatus = Literal["success", "empty", "incomplete", "error"]
Completeness = Literal[
    "complete", "indexed_lower_bound", "incomplete_evidence",
    "cap_exhausted", "not_applicable",
]
Authority = Literal["authoritative", "candidate"]
RetrievalMode = Literal["face", "code", "all"]

# Compatibility tables are public so downstream projections do not invent their
# own interpretation of the old compositional answer contract.
LEGACY_ANSWER_STATUS_TO_PLAN: Mapping[str, PlanStatus] = {
    "success": "supported", "ambiguous": "ambiguous", "unsupported": "unsupported",
    "incomplete": "ambiguous", "invalid_request_or_scope": "error",
    "provider_or_schema_error": "error",
}
LEGACY_ANSWER_STATUS_TO_EVIDENCE: Mapping[str, EvidenceStatus] = {
    "success": "success", "ambiguous": "empty", "unsupported": "empty",
    "incomplete": "incomplete", "invalid_request_or_scope": "error",
    "provider_or_schema_error": "error",
}
LEGACY_COMPLETENESS_TO_CANONICAL: Mapping[str, Completeness] = {
    "complete": "complete",
    "indexed_lower_bound": "indexed_lower_bound",
    "incomplete_evidence": "incomplete_evidence",
    "cap_exhausted": "cap_exhausted",
    "not_applicable": "not_applicable",
}


@dataclass(frozen=True, slots=True)
class BudgetProfile:
    max_entity_candidates: int = 8
    max_anchors: int = 3
    max_variants_per_anchor: int = 2
    max_sql_probes: int = 12
    max_raw_candidates: int = 60
    max_selected_evidence: int = 24
    max_response_bytes: int = 65536

    def __post_init__(self) -> None:
        defaults = {
            "max_entity_candidates": 8,
            "max_anchors": 3,
            "max_variants_per_anchor": 2,
            "max_sql_probes": 12,
            "max_raw_candidates": 60,
            "max_selected_evidence": 24,
            "max_response_bytes": 65536,
        }
        for name, maximum in defaults.items():
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if value > maximum:
                raise ValueError(f"{name} override may only lower the default")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    id: str
    canonical_name: str
    normalized_key: str
    aliases: tuple[str, ...]
    project_id: str | None
    match_kind: Literal["name", "alias"]
    source: Literal["entities"] = "entities"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    version: str
    parser_version: str
    plan_id: str
    query: str
    normalized_query: str
    family: QueryFamily
    operation: str
    status: PlanStatus
    support_mode: str
    operands: tuple[str, ...]
    anchors: tuple[str, ...]
    temporal_predicate: Mapping[str, Any] | None
    user_id: str | None
    project_id: str | None
    timezone: str
    as_of: str | None
    entity_candidates: tuple[EntityCandidate, ...] = ()
    conflict_flags: tuple[str, ...] = ()
    budgets: BudgetProfile = field(default_factory=BudgetProfile)
    retrieval_mode: RetrievalMode = "face"
    compatibility_shape: CanonicalShape = "unsupported"
    compatibility_operands: tuple[str, ...] = ()
    compatibility_item_category: str | None = None
    compatibility_signals: tuple[str, ...] = ()
    compatibility_reasons: tuple[str, ...] = ()

    def without_plan_id(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("plan_id", None)
        return _jsonable(value)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    operation_id: str
    anchor_id: str | None
    memory_id: str | None
    turn_id: str | None
    episode_id: str | None
    source_session_id: str | None
    authority: Authority
    match_reason: str
    channel: str
    probe: str
    rank: int
    occurred_at: str | None
    source_created_at: str | None = None
    source_updated_at: str | None = None
    date_kind: str = "source_timestamp"
    timezone: str = "UTC"
    scope: Mapping[str, Any] = field(default_factory=dict)
    retrieval_mode: RetrievalMode = "face"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    complete: bool = True
    truncated: bool = False
    fallback_status: str = "not_used"
    conflict_flags: tuple[str, ...] = ()
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    version: str
    plan_id: str
    status: EvidenceStatus
    completeness: Completeness
    scope: Mapping[str, Any]
    scope_provenance: str
    retrieval_mode: RetrievalMode
    selected_evidence: tuple[EvidenceRecord, ...]
    candidate_count: int
    sql_probe_count: int
    complete: bool
    truncated: bool
    fallback_status: str
    conflict_flags: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "plan_id": self.plan_id,
            "status": self.status,
            "evidence_status": self.status,
            "completeness": self.completeness,
            "scope": _jsonable(dict(self.scope)),
            "scope_provenance": self.scope_provenance,
            "retrieval_mode": self.retrieval_mode,
            "selected_evidence": [e.to_dict() for e in self.selected_evidence],
            "candidate_count": self.candidate_count,
            "sql_probe_count": self.sql_probe_count,
            "complete": self.complete,
            "truncated": self.truncated,
            "fallback_status": self.fallback_status,
            "conflict_flags": list(self.conflict_flags),
            "error": self.error,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, BudgetProfile):
        return value.to_dict()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    """Stable JSON used by manifests, plans, and repeat hashes."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_query(query: str) -> str:
    return " ".join(query.strip().split())


def _entity_operand(query: str) -> str | None:
    match = re.search(
        r"\b(?:about|regarding|concerning)\s+(?:the\s+)?(.+?)(?=\s+(?:before|after|during|for)\b|[?.!,]|$)",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = " ".join(match.group(1).split()).strip(" ?.,")
    return value or None


def _enumeration_operand(query: str) -> str | None:
    match = re.search(
        r"\b(?:how many|list all|enumerate|all of|every)\s+(?:my\s+|the\s+)?([a-z][a-z0-9_-]*)\b",
        query,
        re.IGNORECASE,
    )
    return match.group(1).casefold() if match else None


def _multi_session(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:across|between)\s+(?:different\s+)?(?:sessions?|conversations?)\b|"
            r"\b(?:how many|enumerate|list|each)\s+(?:different\s+)?(?:sessions?|conversations?)\b|"
            r"\b(?:sessions?|conversations?)\b.*\b(?:how many|enumerate|list|each)\b",
            query,
            re.IGNORECASE,
        )
    )


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[^\s?]+)?\b")
_RECURRING = re.compile(r"\b(?:every|each)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_RELATIVE = re.compile(r"\b(?:\d+|one|two|three|a couple of)\s+(?:days?|weeks?|months?|years?)\s+ago\b", re.I)

# ``extract_anchors`` is intentionally shared with the normal turn router, so
# it can identify ordering scaffolding such as ``from earliest to latest``.
# Those words describe the requested operation, not a remembered event.  They
# must never become probes: doing so turns a valid but anchor-less query into a
# corpus-wide search for unrelated turns containing ``latest``.
_GENERIC_ORDER_ANCHORS = frozenset({
    "earliest", "latest", "first", "last", "oldest", "newest",
    "beginning", "end", "start", "finish", "initial", "final",
})


def _anchor_is_generic_ordering(anchor: str) -> bool:
    normalized = " ".join(anchor.casefold().strip(" ?.,").split())
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
    return normalized in _GENERIC_ORDER_ANCHORS


def _anchor_probe_text(anchor: str) -> str:
    """Remove ordering modifiers while retaining factual anchor terms."""
    value = " ".join(anchor.casefold().strip(" ?.,").split())
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(
        r"^(?:earliest|latest|first|last|oldest|newest|beginning|end|start|finish|initial|final)\s+",
        "",
        value,
    )
    return value.strip()



_TURN_MATCH_STOPWORDS = frozenset({
    "a", "about", "after", "an", "and", "at", "before", "between",
    "but", "by", "did", "do", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "the", "to", "was", "we",
    "what", "when", "with", "you", "your",
})


def _anchor_content_match(content: str, anchor: str) -> bool:
    """Require the returned turn to contain the factual anchor terms.

    Full-text ranking is a candidate generator, not a relevance proof.  This
    deterministic gate avoids promoting turns that merely share temporal
    scaffolding with a query.  A phrase match is preferred; the token fallback
    handles natural punctuation and articles while still requiring every
    meaningful anchor term.
    """
    content_normalized = " ".join(str(content).casefold().split())
    anchor_normalized = " ".join(anchor.casefold().strip(" ?.,").split())
    if not anchor_normalized or not content_normalized:
        return False
    if _anchor_is_generic_ordering(anchor_normalized):
        return False
    if anchor_normalized in content_normalized:
        return True
    probe_text = _anchor_probe_text(anchor_normalized)
    terms = [
        token for token in re.findall(r"[a-z0-9]+", probe_text)
        if len(token) > 2 and token not in _TURN_MATCH_STOPWORDS
    ]
    if not terms:
        return False
    content_terms = set(re.findall(r"[a-z0-9]+", content_normalized))
    return all(term in content_terms for term in terms)


def _temporal_predicate(query: str, anchors: list[str], as_of: datetime | None) -> tuple[dict[str, Any] | None, tuple[str, ...], PlanStatus]:
    flags: list[str] = []
    if _RECURRING.search(query):
        return None, ("recurring_weekday",), "unsupported"
    if len(anchors) > 3:
        return None, ("too_many_anchors",), "ambiguous"
    has_iso = bool(_ISO_DATE.search(query))
    has_relative = bool(_RELATIVE.search(query))
    if has_iso and has_relative:
        return None, ("mixed_absolute_relative_time",), "unsupported"
    if anchors and len(anchors) != 2:
        return None, ("anchor_count_not_two",), "unsupported"
    if len(anchors) == 2:
        if not re.search(r"\b(?:between|from|after|before)\b", query, re.I):
            return None, ("unrecognized_anchor_grammar",), "unsupported"
        return {
            "kind": "occurred_at_anchor_pair",
            "grammar": TEMPORAL_GRAMMAR_VERSION,
            "anchor_count": 2,
            "as_of": as_of.isoformat() if as_of else None,
        }, (), "supported"
    if has_iso:
        return {
            "kind": "occurred_at_explicit_date",
            "grammar": TEMPORAL_GRAMMAR_VERSION,
            "dates": tuple(_ISO_DATE.findall(query)),
            "as_of": as_of.isoformat() if as_of else None,
        }, (), "supported"
    if has_relative:
        # Relative quantities are only meaningful with an explicit fixed as_of.
        if as_of is None:
            return None, ("missing_as_of",), "error"
        return {
            "kind": "occurred_at_relative_date",
            "grammar": TEMPORAL_GRAMMAR_VERSION,
            "as_of": as_of.isoformat(),
        }, (), "supported"
    if re.search(r"\b(?:when did|what day|what date|how long ago)\b", query, re.I):
        return None, ("event_time_inference",), "unsupported"
    return None, tuple(flags), "supported"


def classify_query(
    query: str,
    *,
    user_id: str | None,
    project_id: str | None,
    topic: str | None = None,
    as_of: datetime | None = None,
    timezone_name: str = "UTC",
    retrieval_mode: RetrievalMode = "face",
    budgets: BudgetProfile | None = None,
) -> QueryPlan:
    """Canonical classifier entry point; aliases plan_query for one active parser."""
    plan = plan_query(query, user_id=user_id, project_id=project_id, topic=topic,
                      as_of=as_of, timezone_name=timezone_name,
                      retrieval_mode=retrieval_mode, budgets=budgets)
    if plan.retrieval_mode != retrieval_mode:
        plan = replace(plan, retrieval_mode=retrieval_mode)
        plan = replace(plan, plan_id=stable_hash(plan.without_plan_id())[:24])
    return plan


def plan_query(
    query: str,
    *,
    user_id: str | None,
    project_id: str | None,
    topic: str | None = None,
    as_of: datetime | None = None,
    timezone_name: str = "UTC",
    retrieval_mode: RetrievalMode = "face",
    budgets: BudgetProfile | None = None,
) -> QueryPlan:
    """Build a deterministic, DB-free plan.  No heuristic enters evidence."""
    budget = budgets or BudgetProfile()
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return _make_plan(query, user_id, project_id, "unsupported", "unsupported", (), (), None, ("invalid_timezone",), timezone_name, as_of, budget)
    normalized = _normalized_query(query) if isinstance(query, str) else ""
    if not normalized:
        return _make_plan(query if isinstance(query, str) else "", user_id, project_id, "unsupported", "unsupported", (), (), None, ("empty_query",), timezone_name, as_of, budget)
    if not user_id or not project_id:
        return _make_plan(normalized, user_id, project_id, "unsupported", "error", (), (), None, ("missing_scope",), timezone_name, as_of, budget)

    extracted_anchors = extract_anchors(normalized)
    # The shared extractor also sees ordering scaffolding (for example
    # ``from earliest to latest``).  Those are not factual operands and must
    # never be sent to the corpus as probes.
    anchors = [anchor for anchor in extracted_anchors if not _anchor_is_generic_ordering(anchor)]
    if extracted_anchors and not anchors:
        return _make_plan(
            normalized, user_id, project_id, "unsupported", "unsupported", (), (),
            None, ("ordering_terms_without_factual_anchors",), timezone_name,
            as_of, budget,
        )
    if len(anchors) > budget.max_anchors:
        return _make_plan(normalized, user_id, project_id, "multi_anchor_temporal", "multi_anchor_temporal", (), tuple(anchors[: budget.max_anchors]), None, ("anchor_cap_exhausted",), timezone_name, as_of, budget, status="ambiguous")
    enum = topic or _enumeration_operand(normalized)
    entity = _entity_operand(normalized)
    temporal_language = bool(re.search(r"\b(?:when did|what day|what date|how long ago|every|each)\b", normalized, re.I))
    if anchors:
        family, operation, support, operands = "multi_anchor_temporal", "multi_anchor_temporal", "anchor_turn_evidence", ()
        predicate, flags, status = _temporal_predicate(normalized, anchors, as_of)
    elif temporal_language or _RELATIVE.search(normalized) or _ISO_DATE.search(normalized):
        family, operation, support, operands = "unsupported", "unsupported", "unsupported", ()
        predicate, flags, status = _temporal_predicate(normalized, [], as_of)
    elif _multi_session(normalized) or (entity and re.search(r"\blist\b", normalized, re.I)):
        family, operation, support, operands = "multi_session", "multi_session", "lexical_turn_candidate", ()
        predicate, flags, status = _temporal_predicate(normalized, [], as_of)
    elif enum:
        family, operation, support, operands = "enumeration", "enumeration", "authoritative_memory", (enum,)
        predicate, flags, status = None, (), "supported"
    elif entity:
        family, operation, support, operands = "entity_evidence", "entity_evidence", "authoritative_memory", (entity,)
        predicate, flags, status = None, (), "supported"
    else:
        family, operation, support, operands = "unsupported", "unsupported", "unsupported", ()
        predicate, flags, status = None, (), "unsupported"
    return _make_plan(normalized, user_id, project_id, family, status, operands, tuple(anchors), predicate, flags, timezone_name, as_of, budget, operation=operation, support_mode=support)


def _compatibility_fields(query: str, family: QueryFamily, operands: tuple[str, ...], flags: tuple[str, ...]) -> tuple[CanonicalShape, tuple[str, ...], str | None, tuple[str, ...], tuple[str, ...]]:
    lowered = query.casefold()
    if re.search(r"\b(?:how many|how often|how many times)\b", lowered):
        shape: CanonicalShape = "count"
    elif family == "enumeration":
        shape = "enumeration"
    elif family == "multi_anchor_temporal":
        shape = "chronology" if re.search(r"\b(?:first|earliest|latest|chronolog)\b", lowered) else "compare"
    elif family == "direct":
        shape = "direct"
    else:
        shape = "unsupported"
    return shape, operands, None, (), flags


def _make_plan(
    query: str,
    user_id: str | None,
    project_id: str | None,
    family: QueryFamily,
    status: PlanStatus,
    operands: tuple[str, ...],
    anchors: tuple[str, ...],
    predicate: Mapping[str, Any] | None,
    flags: tuple[str, ...],
    timezone_name: str,
    as_of: datetime | None,
    budget: BudgetProfile,
    *,
    operation: str | None = None,
    support_mode: str | None = None,
    retrieval_mode: RetrievalMode = "face",
) -> QueryPlan:
    compatibility_shape, compatibility_operands, compatibility_item_category, compatibility_signals, compatibility_reasons = _compatibility_fields(query, family, operands, flags)
    draft = QueryPlan(
        version=OPT_N_VERSION,
        parser_version=TEMPORAL_GRAMMAR_VERSION,
        plan_id="",
        query=query,
        normalized_query=_normalized_query(query),
        family=family,
        operation=operation or family,
        status=status,
        support_mode=support_mode or ("unsupported" if family == "unsupported" else ""),
        operands=operands,
        anchors=anchors,
        temporal_predicate=predicate,
        user_id=user_id,
        project_id=project_id,
        timezone=timezone_name,
        as_of=as_of.isoformat() if as_of else None,
        conflict_flags=flags,
        budgets=budget,
        retrieval_mode=retrieval_mode,
        compatibility_shape=compatibility_shape,
        compatibility_operands=compatibility_operands,
        compatibility_item_category=compatibility_item_category,
        compatibility_signals=compatibility_signals,
        compatibility_reasons=compatibility_reasons,
    )
    return replace(draft, plan_id=stable_hash(draft.without_plan_id())[:24])


async def resolve_exact_entities(
    pool: asyncpg.Pool,
    operand: str,
    *,
    user_id: str,
    project_id: str,
    limit: int = 8,
) -> list[EntityCandidate]:
    """Resolve exact canonical names/aliases without arbitrary tie breaking."""
    limit = min(limit, BudgetProfile().max_entity_candidates)
    normalized = " ".join(operand.casefold().split())
    async with acquire(pool) as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, aliases, project_id
              FROM entities
             WHERE status = 'active'
               AND (user_id = $1 OR user_id = $2)
               AND (project_id = $3 OR project_id IS NULL)
               AND (lower(name) = $4
                    OR EXISTS (
                      SELECT 1 FROM unnest(COALESCE(aliases, ARRAY[]::text[])) alias
                       WHERE lower(alias) = $4
                    ))
             ORDER BY id
             LIMIT $5
            """,
            user_id,
            SYSTEM_GLOBAL_USER_ID,
            project_id,
            normalized,
            limit + 1,
        )
    candidates: list[EntityCandidate] = []
    for row in rows:
        name = str(row["name"])
        aliases = tuple(str(a) for a in (row["aliases"] or []))
        kind: Literal["name", "alias"] = "name" if name.casefold() == normalized else "alias"
        candidates.append(EntityCandidate(str(row["id"]), name, normalized, aliases, row["project_id"], kind))
    return candidates


def _scope_dict(plan: QueryPlan) -> dict[str, Any]:
    return {"user_id": plan.user_id, "project_id": plan.project_id, "timezone": plan.timezone, "as_of": plan.as_of}


def make_evidence_bundle(
    plan: QueryPlan,
    *,
    status: EvidenceStatus,
    completeness: Completeness,
    selected_evidence: tuple[EvidenceRecord, ...] = (),
    candidate_count: int = 0,
    sql_probe_count: int = 0,
    truncated: bool = False,
    conflict_flags: tuple[str, ...] = (),
    error: str | None = None,
) -> EvidenceBundle:
    """Single fail-closed constructor for every terminal evidence outcome."""
    if status == "success" and not selected_evidence and completeness == "complete":
        # Empty success is not a valid evidence terminal state.
        status, completeness = "empty", "not_applicable"
    if truncated:
        status, completeness = "incomplete", "cap_exhausted"
    elif status == "error":
        completeness = "incomplete_evidence"
    elif status == "incomplete" and completeness == "complete":
        completeness = "incomplete_evidence"
    complete = completeness == "complete" and status == "success" and not truncated
    return EvidenceBundle(
        version=OPT_N_VERSION,
        plan_id=plan.plan_id,
        status=status,
        completeness=completeness,
        scope=_scope_dict(plan),
        scope_provenance="explicit_user_and_project_scope" if plan.user_id and plan.project_id else "missing_scope",
        retrieval_mode=plan.retrieval_mode,
        selected_evidence=selected_evidence,
        candidate_count=candidate_count,
        sql_probe_count=sql_probe_count,
        complete=complete,
        truncated=truncated,
        fallback_status="not_used",
        conflict_flags=conflict_flags or plan.conflict_flags,
        error=error,
    )


def _empty_bundle(plan: QueryPlan, *, status: EvidenceStatus = "empty", error: str | None = None, flags: tuple[str, ...] = ()) -> EvidenceBundle:
    return make_evidence_bundle(
        plan,
        status=status,
        completeness="incomplete_evidence" if status == "error" else "not_applicable",
        conflict_flags=flags,
        error=error,
    )


def _memory_record(memory: Memory, plan: QueryPlan, rank: int, reason: str, authority: Authority = "authoritative") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"memory:{memory.id}", operation_id=plan.operation, anchor_id=None,
        memory_id=memory.id, turn_id=None, episode_id=None, source_session_id=None,
        authority=authority, match_reason=reason, channel="entity_link" if authority == "authoritative" else "lexical",
        probe=plan.operands[0] if plan.operands else plan.query, rank=rank,
        occurred_at=memory.created_at.isoformat(),
        source_created_at=memory.created_at.isoformat() if memory.created_at else None,
        source_updated_at=memory.updated_at.isoformat() if memory.updated_at else None,
        date_kind="source_timestamp", timezone=plan.timezone,
        scope=_scope_dict(plan), retrieval_mode=plan.retrieval_mode,
        provenance={"source": "memories", "memory_id": memory.id, "authority": authority},
        complete=True, truncated=False, fallback_status="not_used", content=memory.content,
    )


def _turn_record(turn: Any, plan: QueryPlan, rank: int, anchor: str, probe: str, reason: str, authority: Authority) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"turn:{turn.id}", operation_id=plan.operation, anchor_id=stable_hash(anchor)[:16],
        memory_id=None, turn_id=turn.id, episode_id=turn.episode_id, source_session_id=turn.source_session_id,
        authority=authority, match_reason=reason, channel="occurred_at_turn", probe=probe, rank=rank,
        occurred_at=turn.occurred_at.isoformat(),
        source_created_at=None, source_updated_at=None,
        date_kind="event_timestamp", timezone=plan.timezone,
        scope=_scope_dict(plan), retrieval_mode=plan.retrieval_mode,
        provenance={"source": "episode_turns", "turn_id": turn.id, "episode_id": turn.episode_id},
        complete=True, truncated=False, fallback_status="not_used", content=turn.content,
    )


async def _entity_memories(pool: asyncpg.Pool, entity_id: str, *, user_id: str, project_id: str, limit: int) -> list[Memory]:
    async with acquire(pool) as conn:
        rows = await conn.fetch(
            """
            SELECT m.* FROM memories m
            JOIN entity_mentions em ON em.memory_id = m.id
            WHERE em.entity_id = $1 AND m.status = 'active'
              AND (m.user_id = $2 OR m.user_id = $3)
              AND (m.project_id = $4 OR m.project_id IS NULL)
            ORDER BY em.mentioned_at DESC, m.id
            LIMIT $5
            """,
            entity_id, user_id, SYSTEM_GLOBAL_USER_ID, project_id, limit,
        )
    return [_row_to_memory(row) for row in rows]


async def _lexical_turns(pool: asyncpg.Pool, probe: str, *, user_id: str, project_id: str, limit: int) -> list[Any]:
    async with acquire(pool) as conn:
        try:
            await conn.execute("SAVEPOINT opt_n_fts_probe")
            rows = await conn.fetch(
                """
                SELECT t.* FROM episode_turns t
                JOIN episodes e ON e.id = t.episode_id
                WHERE e.project_id = $1
                  AND (t.user_id = $2 OR t.user_id = $3
                       OR t.user_id IS NULL)
                  AND t.search_tsv @@ websearch_to_tsquery('english', $4)
                ORDER BY ts_rank(
                    t.search_tsv,
                    websearch_to_tsquery('english', $4)
                ) DESC, t.id
                LIMIT $5
                """,
                project_id, user_id, SYSTEM_GLOBAL_USER_ID, probe, limit,
            )
            await conn.execute("RELEASE SAVEPOINT opt_n_fts_probe")
        except asyncpg.UndefinedColumnError:
            # Older local/test schemas predate the generated FTS column. Roll
            # back only this failed probe, then keep the compatibility path
            # scoped and deterministic. Durable materializations use the
            # indexed query above.
            await conn.execute("ROLLBACK TO SAVEPOINT opt_n_fts_probe")
            await conn.execute("RELEASE SAVEPOINT opt_n_fts_probe")
            rows = await conn.fetch(
                """
                SELECT t.* FROM episode_turns t
                JOIN episodes e ON e.id = t.episode_id
                WHERE e.project_id = $1
                  AND (t.user_id = $2 OR t.user_id = $3
                       OR t.user_id IS NULL)
                  AND t.content ILIKE '%' || $4 || '%'
                ORDER BY t.occurred_at ASC, t.id
                LIMIT $5
                """,
                project_id, user_id, SYSTEM_GLOBAL_USER_ID, probe, limit,
            )
    return [_row_to_turn(row) for row in rows]


async def _anchor_turns(
    pool: asyncpg.Pool,
    anchor: str,
    probes: list[str],
    *,
    user_id: str,
    project_id: str,
    limit: int,
    max_sql_probes: int = 12,
) -> tuple[list[Any], int]:
    seen: dict[str, Any] = {}
    sql_count = 0
    for probe in probes[:2]:
        if sql_count >= max_sql_probes:
            break
        rows = await _lexical_turns(pool, probe, user_id=user_id, project_id=project_id, limit=limit)
        sql_count += 1
        for turn in rows:
            # FTS/ILIKE only establishes a candidate.  Require the turn to
            # contain the meaningful terms for this specific anchor before it
            # can be surfaced in the structured bundle.  This prevents a
            # generic probe hit from turning into apparent temporal evidence.
            if _anchor_content_match(turn.content, anchor):
                seen.setdefault(turn.id, turn)
    return sorted(seen.values(), key=lambda t: (t.occurred_at, t.id))[:limit], sql_count


async def build_opt_n_supplement(
    pool: asyncpg.Pool,
    query: str,
    *,
    user_id: str | None,
    project_id: str | None,
    topic: str | None = None,
    as_of: datetime | None = None,
    timezone_name: str = "UTC",
    retrieval_mode: RetrievalMode = "face",
    budgets: BudgetProfile | None = None,
    enumeration_result: tuple[list[str], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Build the opt-in sidecar.  Baseline recall is never called here."""
    plan = classify_query(query, user_id=user_id, project_id=project_id, topic=topic, as_of=as_of, timezone_name=timezone_name, retrieval_mode=retrieval_mode, budgets=budgets)
    if plan.status == "error":
        return {"version": OPT_N_VERSION, "query_plan": plan.to_dict(), "evidence": _empty_bundle(plan, status="error", error="missing_scope").to_dict()}
    if plan.status != "supported":
        return {"version": OPT_N_VERSION, "query_plan": plan.to_dict(), "evidence": _empty_bundle(plan, status="empty").to_dict()}

    try:
        if plan.family == "enumeration":
            if enumeration_result is None:
                from weft.enumeration_router import gather_enumeration
                tags, gathered = await gather_enumeration(
                    pool,
                    plan.operands[0],
                    plan.user_id or "",
                    project_id=plan.project_id,
                )
            else:
                tags, gathered = enumeration_result
            if gathered is None:
                bundle = _empty_bundle(plan, status="error", error="enumeration_gather_failed")
                result = {"version": OPT_N_VERSION, "query_plan": plan.to_dict(), "evidence": bundle.to_dict()}
                result["repeat_hash"] = stable_hash(result)
                return result
            memories = list(gathered["memories"])
            records = tuple(_memory_record(m, plan, i, "topic_membership") for i, m in enumerate(memories[: plan.budgets.max_selected_evidence]))
            bundle = make_evidence_bundle(
                plan,
                status="success" if records and gathered["complete"] else ("empty" if not records else "incomplete"),
                completeness="complete" if gathered["complete"] else "incomplete_evidence",
                selected_evidence=records,
                candidate_count=len(memories),
                sql_probe_count=1,
                truncated=gathered["truncated"],
                error=None if gathered["complete"] else "incomplete_gather",
            )
        elif plan.family == "entity_evidence":
            candidates = await resolve_exact_entities(pool, plan.operands[0], user_id=plan.user_id or "", project_id=plan.project_id or "", limit=plan.budgets.max_entity_candidates)
            if not candidates:
                bundle = _empty_bundle(plan, status="empty", flags=("no_entity_match",))
            elif len(candidates) > 1:
                plan = replace(plan, status="ambiguous", entity_candidates=tuple(candidates), conflict_flags=("alias_collision",))
                alias_flags = ("alias_collision", "alias_cap_exhausted") if len(candidates) > plan.budgets.max_entity_candidates else ("alias_collision",)
                bundle = _empty_bundle(plan, status="empty", flags=alias_flags)
            else:
                plan = replace(plan, entity_candidates=tuple(candidates))
                memories = await _entity_memories(pool, candidates[0].id, user_id=plan.user_id or "", project_id=plan.project_id or "", limit=plan.budgets.max_selected_evidence + 1)
                authoritative_memories = memories[: plan.budgets.max_selected_evidence]
                entity_truncated = len(memories) > plan.budgets.max_selected_evidence
                records: list[EvidenceRecord] = [_memory_record(m, plan, i, "explicit_entity_link") for i, m in enumerate(authoritative_memories)]
                # Lexical turn probes are candidates only and cannot satisfy an authoritative entity lift.
                if len(records) < plan.budgets.max_selected_evidence and plan.budgets.max_sql_probes:
                    turns = await _lexical_turns(pool, candidates[0].canonical_name, user_id=plan.user_id or "", project_id=plan.project_id or "", limit=min(plan.budgets.max_raw_candidates, plan.budgets.max_selected_evidence - len(records)))
                    records.extend(_turn_record(t, plan, len(records), candidates[0].canonical_name, candidates[0].canonical_name, "canonical_name_text_probe", "candidate") for t in turns if f"turn:{t.id}" not in {r.evidence_id for r in records})
                has_authoritative = bool(authoritative_memories)
                bundle = make_evidence_bundle(
                    plan,
                    status="success" if has_authoritative and not entity_truncated else ("empty" if not has_authoritative else "incomplete"),
                    completeness="complete" if has_authoritative and not entity_truncated else "incomplete_evidence",
                    selected_evidence=tuple(records[: plan.budgets.max_selected_evidence]),
                    candidate_count=len(records),
                    sql_probe_count=1,
                    truncated=entity_truncated,
                )
        elif plan.family in {"multi_anchor_temporal", "multi_session"}:
            if plan.family == "multi_session" and not plan.project_id:
                bundle = _empty_bundle(plan, status="error", error="missing_scope")
            else:
                records: list[EvidenceRecord] = []
                probe_query = plan.query
                if plan.family == "multi_session":
                    match = re.search(r"\babout\s+(.+?)(?:[?.!,]|$)", plan.query, re.I)
                    probe_query = match.group(1).strip() if match else plan.query
                candidate_count = 0
                probes = 0
                anchor_inputs = plan.anchors
                if not anchor_inputs:
                    anchor_inputs = (probe_query if plan.family == "multi_session" else plan.query,)
                for anchor in anchor_inputs:
                    probe_anchor = _anchor_probe_text(anchor)
                    variants = [
                        value for value in (probe_anchor, anchor, re.sub(r"^the\s+", "", anchor, flags=re.I))
                        if value
                    ]
                    # Preserve deterministic order while removing duplicates.
                    variants = list(dict.fromkeys(variants))
                    if plan.family == "multi_session":
                        variants = [
                            probe_query,
                            plan.query,
                        ]
                    turns, used = await _anchor_turns(
                        pool,
                        anchor,
                        variants[: plan.budgets.max_variants_per_anchor],
                        user_id=plan.user_id or "",
                        project_id=plan.project_id or "",
                        limit=plan.budgets.max_raw_candidates,
                        max_sql_probes=max(0, plan.budgets.max_sql_probes - probes),
                    )
                    probes += used
                    candidate_count += len(turns)
                    records.extend(_turn_record(t, plan, len(records), anchor, variants[0], "declared_anchor_occurred_at_probe", "candidate") for t in turns if f"turn:{t.id}" not in {r.evidence_id for r in records})
                evidence_truncated = candidate_count > plan.budgets.max_selected_evidence
                records = records[: plan.budgets.max_selected_evidence]
                bundle = make_evidence_bundle(
                    plan,
                    status="success" if records and not evidence_truncated else ("empty" if not records else "incomplete"),
                    completeness="indexed_lower_bound" if records and not evidence_truncated else "incomplete_evidence",
                    selected_evidence=tuple(records),
                    candidate_count=candidate_count,
                    sql_probe_count=probes,
                    truncated=evidence_truncated,
                )
        else:
            bundle = _empty_bundle(plan)
    except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, ConnectionError) as exc:
        bundle = _empty_bundle(plan, status="error", error=type(exc).__name__)
    result = {"version": OPT_N_VERSION, "query_plan": plan.to_dict(), "evidence": bundle.to_dict()}
    raw = canonical_json(result).encode("utf-8")
    if len(raw) > plan.budgets.max_response_bytes:
        trimmed = tuple(bundle.selected_evidence[: max(0, plan.budgets.max_selected_evidence // 2)])
        bundle = make_evidence_bundle(
            plan,
            status="incomplete",
            completeness="cap_exhausted",
            selected_evidence=trimmed,
            candidate_count=bundle.candidate_count,
            sql_probe_count=bundle.sql_probe_count,
            truncated=True,
            conflict_flags=tuple(bundle.conflict_flags) + ("response_bytes_cap_exhausted",),
        )
        result["evidence"] = bundle.to_dict()
        result["response_truncated"] = True
    result["repeat_hash"] = stable_hash(result)
    return result


__all__ = [
    "BudgetProfile", "EntityCandidate", "EvidenceBundle", "EvidenceRecord", "QueryPlan",
    "EvidenceStatus", "Completeness", "CanonicalShape", "RetrievalMode",
    "LEGACY_ANSWER_STATUS_TO_PLAN", "LEGACY_ANSWER_STATUS_TO_EVIDENCE", "LEGACY_COMPLETENESS_TO_CANONICAL",
    "OPT_N_VERSION", "TEMPORAL_GRAMMAR_VERSION", "build_opt_n_supplement",
    "canonical_json", "classify_query", "make_evidence_bundle", "plan_query", "resolve_exact_entities", "stable_hash",
]
