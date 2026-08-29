"""Milestone A recovery telemetry projections and best-effort persistence."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.mcp.tools import (
    _recall_query_was_inserted,
    _recovery_telemetry_projection,
    _write_recovery_telemetry,
)
from weft.retrieval_recovery import RecoveryConfig, RecoveryOutcome, RetrievalStage, ScopeSnapshot
from tests.test_mcp_tools import FakeEmbeddingProvider, _make_ctx


@pytest.fixture
def recovery_objects():
    scope = ScopeSnapshot.from_baseline(
        user_id="telemetry-user",
        requested_project_id="requested-project",
        resolved_project_id="resolved-project",
        retrieval_mode="face",
    )
    stage = RetrievalStage(
        stage="deterministic_reformulation",
        query_label="redacted query label",
        query_hash="a" * 64,
        result_ids=tuple(f"memory:{i}" for i in range(40)),
        coverage={"answerability": "insufficient_evidence", "padding": "x" * 5000},
        latency_ms=10000,
        error_category="e" * 96,
        exact_queries=("DO NOT STORE THIS PROMPT",),
    )
    outcome = RecoveryOutcome(
        attempted=True,
        trigger="trigger-" + "x" * 200,
        answerability="insufficient_evidence",
        stages=(stage,),
        scope=scope,
    )
    return stage, outcome, scope


def test_projection_is_bounded_and_redacted(recovery_objects):
    stage, outcome, scope = recovery_objects
    row = _recovery_telemetry_projection(stage, outcome, scope, "rq-parent")

    assert row["parent_query_id"] == "rq-parent"
    assert len(row["attempt_id"]) <= 80
    assert len(row["trigger"]) <= 96
    assert len(row["result_ids"]) <= 24
    assert len(json.dumps(row["coverage"]).encode()) <= 4096
    assert row["latency_ms"] == 10000
    assert len(row["error_category"]) <= 96
    assert "DO NOT STORE THIS PROMPT" not in json.dumps(row)
    assert "query_text" not in row


@pytest.mark.asyncio
async def test_writer_is_best_effort_on_db_failure(recovery_objects):
    stage, outcome, scope = recovery_objects
    with patch("weft.mcp.tools.acquire", side_effect=RuntimeError("db down")):
        row = await _write_recovery_telemetry(
            object(), user_id="telemetry-user", stage=stage, outcome=outcome,
            scope=scope,
        )
    assert row["parent_query_id"] is None


@pytest.mark.asyncio
async def test_parent_observation_requires_inserted_row(pool):
    assert await _recall_query_was_inserted(pool, "telemetry-user", "rq-not-inserted") is False
    await pool.execute(
        "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text) "
        "VALUES ($1, $2, 'recall', $3)",
        "rq-inserted", "telemetry-user", "query must not be persisted in recovery row",
    )
    assert await _recall_query_was_inserted(pool, "telemetry-user", "rq-inserted") is True


@pytest.mark.asyncio
async def test_writer_persists_v72_row_with_null_parent_on_unavailable_logger(pool, recovery_objects):
    stage, outcome, scope = recovery_objects
    await _write_recovery_telemetry(
        pool, user_id="telemetry-user", stage=stage, outcome=outcome,
        scope=scope, parent_query_id=None,
    )
    row = await pool.fetchrow("SELECT * FROM weft_recovery_attempts")
    assert row["parent_query_id"] is None
    assert row["provider_calls"] == 0
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    assert row["query_hash"] == "a" * 64
    assert "DO NOT STORE THIS PROMPT" not in json.dumps(dict(row), default=str)


@pytest.mark.asyncio
async def test_writer_links_parent_only_when_explicitly_observed(pool, recovery_objects):
    stage, outcome, scope = recovery_objects
    await pool.execute(
        "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text) "
        "VALUES ($1, $2, 'recall', $3)",
        "rq-linked", "telemetry-user", "parent query",
    )
    assert await _recall_query_was_inserted(pool, "telemetry-user", "rq-linked") is True
    await _write_recovery_telemetry(
        pool, user_id="telemetry-user", stage=stage, outcome=outcome,
        scope=scope, parent_query_id="rq-linked",
    )
    row = await pool.fetchrow("SELECT parent_query_id FROM weft_recovery_attempts")
    assert row["parent_query_id"] == "rq-linked"


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


class _SupportedRecoveryController:
    def __init__(self, *, outcome):
        self.outcome = outcome

    async def recover(self, *args, **kwargs):
        return self.outcome


@pytest.mark.asyncio
async def test_supported_recovery_observes_parent_before_writing(ctx, recovery_objects, monkeypatch):
    from weft.mcp.tools import weft_recall

    stage, outcome, _ = recovery_objects
    events = []

    async def fake_logger(*args, **kwargs):
        events.append("logger")
        return kwargs["query_id"]

    async def fake_observation(pool, user_id, query_id):
        events.append("observed")
        return True

    async def fake_writer(*args, parent_query_id=None, **kwargs):
        events.append(("write", parent_query_id))
        return {"parent_query_id": parent_query_id}

    monkeypatch.setattr("weft.mcp.tools._resolve_project_id", AsyncMock(return_value=None))
    with (
        patch("weft.retrieval_recovery.RecoveryController", lambda **_: _SupportedRecoveryController(outcome=outcome)),
        patch("weft.mcp.tools.log_recall_query", side_effect=fake_logger),
        patch("weft.mcp.tools._recall_query_was_inserted", side_effect=fake_observation),
        patch("weft.mcp.tools._write_recovery_telemetry", side_effect=fake_writer),
        patch("weft.mcp.tools.search_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        response = await weft_recall(
            ctx, query="supported recovery", mode="keyword", tier="belief",
            recovery_mode="deterministic",
        )

    assert response["recovery"]["supported"] is True
    assert events[0:2] == ["logger", "observed"]
    assert events[2][0] == "write"
    assert events[2][1].startswith("rq-")


@pytest.mark.asyncio
async def test_logger_result_without_inserted_parent_uses_null_fallback(
    ctx, recovery_objects, monkeypatch,
):
    from weft.mcp.tools import weft_recall

    _, outcome, _ = recovery_objects
    written_parents = []

    async def fake_logger(*args, **kwargs):
        return kwargs["query_id"]

    async def fake_writer(*args, parent_query_id=None, **kwargs):
        written_parents.append(parent_query_id)
        return {"parent_query_id": parent_query_id}

    monkeypatch.setattr("weft.mcp.tools._resolve_project_id", AsyncMock(return_value=None))
    with (
        patch("weft.retrieval_recovery.RecoveryController", lambda **_: _SupportedRecoveryController(outcome=outcome)),
        patch("weft.mcp.tools.log_recall_query", side_effect=fake_logger),
        patch("weft.mcp.tools._recall_query_was_inserted", new_callable=AsyncMock, return_value=False),
        patch("weft.mcp.tools._write_recovery_telemetry", side_effect=fake_writer),
        patch("weft.mcp.tools.search_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        await weft_recall(
            ctx, query="unobserved recovery", mode="keyword", tier="belief",
            recovery_mode="deterministic",
        )

    assert written_parents == [None]


@pytest.mark.asyncio
async def test_logger_timeout_is_cancelled_and_drained_before_null_fallback(
    ctx, recovery_objects, monkeypatch,
):
    from weft.mcp.tools import weft_recall

    real_recovery_config = RecoveryConfig
    _, outcome, _ = recovery_objects
    logger_cancelled = False
    written_parents = []

    async def hanging_logger(*args, **kwargs):
        nonlocal logger_cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger_cancelled = True
            raise

    async def fake_writer(*args, parent_query_id=None, **kwargs):
        written_parents.append(parent_query_id)
        return {"parent_query_id": parent_query_id}

    monkeypatch.setattr("weft.mcp.tools._resolve_project_id", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "weft.retrieval_recovery.RecoveryConfig",
        lambda: real_recovery_config(timeout_seconds=0.01),
    )
    with (
        patch("weft.retrieval_recovery.RecoveryController", lambda **_: _SupportedRecoveryController(outcome=outcome)),
        patch("weft.mcp.tools.log_recall_query", side_effect=hanging_logger),
        patch("weft.mcp.tools._recall_query_was_inserted", new_callable=AsyncMock, return_value=True),
        patch("weft.mcp.tools._write_recovery_telemetry", side_effect=fake_writer),
        patch("weft.mcp.tools.search_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        await weft_recall(
            ctx, query="timed out recovery", mode="keyword", tier="belief",
            recovery_mode="deterministic",
        )

    assert logger_cancelled is True
    assert written_parents == [None]
    assert not [
        task for task in asyncio.all_tasks()
        if task.get_name() == "weft-recall-query-log" and not task.done()
    ]
