from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.compositional_recall import (
    AnswerRequest,
    AnswerStatus,
    EvidenceId,
    adapt_recovery_to_answer_authoritative,
    build_plan,
)
from weft.models import Memory, MemorySource, MemoryStatus, MemoryType
from weft.retrieval_recovery import (
    RecoveryCandidate,
    RecoveryOutcome,
    ScopeSnapshot,
    _candidate_id,
    _candidate_payload,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def test_candidate_payload_maps_canonical_active_status_to_current() -> None:
    payload = _candidate_payload({"status": "active", "content": "bounded", "project_id": "weft"})
    assert payload["supersession"] == "current"


def test_primary_memory_candidate_id_uses_canonical_authority_kind() -> None:
    assert _candidate_id({"id": "recovered"}) == "memory:recovered"
    assert _candidate_id({"id": "recovered"}, tier="turns") == "turn:recovered"
    assert _candidate_id({"claim_id": "assertion"}) == "claim:assertion"


def _candidate(stable_id: str, content: str, *, branch: str = "branch-direct") -> RecoveryCandidate:
    kind, identifier = stable_id.split(":", 1)
    fields = dict(
        stable_id=stable_id, stage="deterministic_reformulation", query_label="provider keys",
        query_hash="a" * 64, branch=branch, source_mode="face", content=content,
        user_id="u", project_id="weft", occurred_at=NOW, supersession="current",
        provenance=("deterministic_reformulation", "a" * 64),
    )
    fields[f"{kind}_id"] = identifier
    return RecoveryCandidate(**fields)


class _RecordLike:
    """Minimal asyncpg.Record-shaped object: key lookup, no attributes."""

    def __init__(self, **values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]


@pytest.mark.asyncio
async def test_canonical_reread_accepts_record_like_rows() -> None:
    request = AnswerRequest(question="Where are provider keys configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:recovered", "candidate payload")
    canonical = _RecordLike(
        id="recovered", content="Provider keys are configured in pilot_config.yaml.",
        user_id="u", project_id="weft", source="documentation", status="active",
        review_status="active", updated_at=NOW, project_facets=(), relationships=(),
    )
    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )
    assert result.status is AnswerStatus.success
    assert result.cited_evidence_ids == ("memory:recovered",)


@pytest.mark.asyncio
async def test_canonical_turn_reread_accepts_record_like_occurred_at() -> None:
    request = AnswerRequest(question="Who was first between Alpha and Beta?", user_id="u", project_id="weft")
    candidate = _candidate(
        "turn:recovered", "Alpha happened first on 2026-08-01.", branch="branch-1"
    ).model_copy(update={"tier": "turns", "source_mode": "face"})
    canonical = _RecordLike(
        id="recovered", content=candidate.content, user_id="u", project_id="weft",
        source="conversation", status="active", review_status="active",
        occurred_at=NOW, updated_at=NOW, project_facets=(), relationships=(),
    )
    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )
    rejected = result.branch_results["recovery"]["recovered_rejected_ids"]
    assert rejected.get("turn:recovered") not in {"canonical_read_error", "untyped_date"}


def _outcome(request: AnswerRequest, *candidates: RecoveryCandidate) -> RecoveryOutcome:
    return RecoveryOutcome(
        supported=True,
        scope=ScopeSnapshot.from_baseline(
            user_id=request.user_id, requested_project_id=request.project_id,
            resolved_project_id=request.project_id, retrieval_mode=request.retrieval_mode,
            as_of=request.as_of,
        ),
        candidates=tuple(candidates),
    )


def _incomplete(request: AnswerRequest):
    _, plan = build_plan(request)
    from weft.compositional_recall import ComposedAnswer
    return ComposedAnswer(
        question=request.question, normalized_question=plan.normalized_question,
        shape=plan.shape, operation=plan.operation, status=AnswerStatus.incomplete,
        plan=plan, completeness="incomplete_evidence", evidence_status="incomplete",
        scope={"user_id": request.user_id, "project_id": request.project_id},
        retrieval_mode=request.retrieval_mode,
    )


@pytest.mark.asyncio
async def test_canonical_direct_recovery_promotes_typed_answer_and_citation() -> None:
    request = AnswerRequest(question="Where are provider keys configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:recovered", "Provider keys are configured in ~/.weft/.env.")
    canonical = Memory(
        id="recovered", type=MemoryType.solution, content="Canonical provider keys live in ~/.weft/.env.",
        source=MemorySource.documentation, project_id="weft", updated_at=NOW,
    )
    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )
    assert result.status is AnswerStatus.success
    assert result.typed_result == {"value": canonical.content}
    assert result.cited_evidence_ids == ("memory:recovered",)
    assert result.branch_results["recovery"]["recovered_authoritative_ids"] == ["memory:recovered"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "canonical, reason",
    [
        (None, "missing_canonical_row"),
        (Memory(id="stale", type=MemoryType.fact, content="x", status=MemoryStatus.archived), "stale_or_unreviewed"),
        ({"id": "foreign", "user_id": "other", "content": "x", "status": "active"}, "foreign_user"),
        (Memory(id="other-project", type=MemoryType.fact, content="x", project_id="other"), "foreign_project"),
        (Memory(id="code", type=MemoryType.fact, content="x", source=MemorySource.code), "mode_disallowed_source"),
    ],
)
async def test_canonical_rejections_remain_incomplete_and_uncited(canonical, reason: str) -> None:
    request = AnswerRequest(question="Where are provider keys configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:recovered", "candidate payload must not authorize")
    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )
    assert result.status is AnswerStatus.incomplete
    assert result.cited_evidence_ids == ()
    assert result.branch_results["recovery"]["recovered_rejected_ids"][candidate.stable_id] == reason


@pytest.mark.asyncio
async def test_canonical_read_error_fails_closed() -> None:
    request = AnswerRequest(question="Where are provider keys configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:recovered", "x")
    def broken(_id: EvidenceId):
        raise RuntimeError("database unavailable")
    result = await adapt_recovery_to_answer_authoritative(request, _incomplete(request), _outcome(request, candidate), broken)
    assert result.status is AnswerStatus.incomplete
    assert result.cited_evidence_ids == ()
    assert result.branch_results["recovery"]["recovered_rejected_ids"][candidate.stable_id] == "canonical_read_error"


@pytest.mark.asyncio
async def test_identifier_distractor_is_rejected_after_canonical_reread() -> None:
    request = AnswerRequest(question="Where is pilot_config.yaml configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:distractor", "The pilot launch date is not disclosed.")
    canonical = Memory(
        id="distractor", type=MemoryType.fact, content=candidate.content,
        source=MemorySource.documentation, project_id="weft", updated_at=NOW,
    )

    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )

    assert result.status is AnswerStatus.incomplete
    assert result.cited_evidence_ids == ()
    assert result.branch_results["recovery"]["recovered_rejected_ids"][candidate.stable_id] == "identifier_mismatch"


@pytest.mark.asyncio
async def test_identifier_exact_match_remains_authoritative() -> None:
    request = AnswerRequest(question="Where is pilot_config.yaml configured?", user_id="u", project_id="weft")
    candidate = _candidate("memory:exact", "pilot_config.yaml is configured in deployment code.")
    canonical = Memory(
        id="exact", type=MemoryType.solution, content=candidate.content,
        source=MemorySource.documentation, project_id="weft", updated_at=NOW,
    )

    result = await adapt_recovery_to_answer_authoritative(
        request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical,
    )

    assert result.status is AnswerStatus.success
    assert result.cited_evidence_ids == ("memory:exact",)
    assert result.typed_result == {"value": canonical.content}


@pytest.mark.asyncio
async def test_conflict_and_as_of_rows_are_rejected() -> None:
    request = AnswerRequest(question="Where are provider keys configured?", user_id="u", project_id="weft", as_of=NOW)
    candidate = _candidate("memory:recovered", "candidate payload")
    conflicting = Memory(id="recovered", type=MemoryType.fact, content="Canonical answer", project_id="weft", updated_at=NOW).model_dump()
    conflicting["relationships"] = [{"relation": "contradicts"}]
    result = await adapt_recovery_to_answer_authoritative(request, _incomplete(request), _outcome(request, candidate), lambda _id: conflicting)
    assert result.status is AnswerStatus.incomplete
    assert result.cited_evidence_ids == ()
    assert result.branch_results["recovery"]["recovered_rejected_ids"][candidate.stable_id] == "conflicting_or_superseded"

    post_cutoff = Memory(id="recovered", type=MemoryType.fact, content="Canonical answer", project_id="weft", updated_at=NOW.replace(year=2027))
    result = await adapt_recovery_to_answer_authoritative(request, _incomplete(request), _outcome(request, candidate), lambda _id: post_cutoff)
    assert result.status is AnswerStatus.incomplete
    assert result.branch_results["recovery"]["recovered_rejected_ids"][candidate.stable_id] == "post_as_of"


@pytest.mark.asyncio
async def test_compare_requires_both_canonical_branches() -> None:
    request = AnswerRequest(question="Who was first between Alpha and Beta?", user_id="u", project_id="weft")
    _, plan = build_plan(request)
    candidate = _candidate("memory:alpha", "Alpha occurred on 2026-08-01.", branch="branch-1")
    canonical = Memory(id="alpha", type=MemoryType.fact, content=candidate.content, project_id="weft", updated_at=NOW)
    result = await adapt_recovery_to_answer_authoritative(request, _incomplete(request), _outcome(request, candidate), lambda _id: canonical)
    assert result.status is AnswerStatus.incomplete
    assert result.typed_result is None
    assert result.cited_evidence_ids == ()
    assert set(result.branch_results["recovery"]["recovered_authoritative_ids"]) == {"memory:alpha"}


@pytest.mark.asyncio
async def test_live_weft_answer_uses_production_canonical_reader_for_recovery(pool) -> None:
    """The MCP seam must resolve recovered IDs through the authenticated reader."""
    from tests.test_mcp_tools import FakeEmbeddingProvider, _make_ctx
    from weft.compositional_recall import adapt_recovery_to_answer_authoritative as production_adapter
    from weft.mcp.tools import weft_answer
    from weft.models import MemorySource
    from weft.retrieval_recovery import RecoveryConfig

    app = AppContext(
        pool=pool, cache=NullCache(), embedding=FakeEmbeddingProvider(), config=WeftConfig()
    )
    ctx = _make_ctx(app)
    request = AnswerRequest(
        question="Where are provider keys configured?", user_id="authenticated-user", project_id="weft"
    )
    candidate = _candidate("memory:recovered", "candidate payload must not authorize")
    canonical = Memory(
        id="recovered", type=MemoryType.solution,
        content="Canonical provider keys live in ~/.weft/.env.",
        source=MemorySource.documentation, project_id="weft", updated_at=NOW,
    )
    outcome = _outcome(request, candidate)
    observations: dict[str, object] = {}
    resolver_only = AsyncMock(return_value={"content": candidate.content})

    async def canonical_get_memory(pool_arg, memory_id):
        observations.setdefault("canonical_calls", []).append((pool_arg, memory_id))
        observations["canonical_pool"] = pool_arg
        observations["canonical_id"] = memory_id
        observations["canonical_user"] = current_user_id.get()
        return canonical.model_dump()

    class FakeRecoveryController:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["config"], RecoveryConfig)

        async def recover(self, *args, **kwargs):
            return outcome

    async def adapter_spy(request_arg, answer_arg, outcome_arg, reader):
        observations["reader"] = reader
        # Invoke the callback handed to the adapter; a resolver-only API is not
        # sufficient because it would bypass this production canonical reader.
        resolved = await reader(EvidenceId.parse(candidate.stable_id))
        observations["resolved_content"] = resolved["content"]
        return await production_adapter(request_arg, answer_arg, outcome_arg, reader)

    with (
        patch("weft.compositional_recall.answer_question", new_callable=AsyncMock, return_value=_incomplete(request)) as answer_question,
        patch("weft.retrieval_recovery.RecoveryController", FakeRecoveryController),
        patch("weft.compositional_recall.adapt_recovery_to_answer_authoritative", side_effect=adapter_spy) as adapter,
        patch("weft.store.get_memory", side_effect=canonical_get_memory) as get_memory,
        patch("weft.mcp.tools.get_relationships", new_callable=AsyncMock, return_value=[]),
    ):
        token = current_user_id.set("authenticated-user")
        try:
            response = await weft_answer(
                ctx, question=request.question, project_id="weft", recovery_mode="deterministic"
            )
        finally:
            current_user_id.reset(token)

    answer_question.assert_awaited_once()
    adapter.assert_awaited_once()
    assert get_memory.await_count == 2  # wiring probe + adapter's real authority reread
    assert all(call.args == (pool, "recovered") for call in get_memory.await_args_list)
    resolver_only.assert_not_awaited()
    assert callable(observations["reader"])
    assert observations["canonical_calls"] == [(pool, "recovered"), (pool, "recovered")]
    assert observations["canonical_pool"] is pool
    assert observations["canonical_id"] == "recovered"
    assert observations["canonical_user"] == "authenticated-user"
    assert observations["resolved_content"] == canonical.content
    assert response["status"] == "success"
    assert response["cited_evidence_ids"] == ["memory:recovered"]
    assert response["branch_results"]["recovery"]["recovered_authoritative_ids"] == ["memory:recovered"]
