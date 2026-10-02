"""Regression tests for resilient primer behavior during partial DB failures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.primer import build_primer
from weft.store import store_memory


class _Embedding:
    provider_name = "test"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]


def _app(pool, *, prime_timeout: float = 60.0) -> AppContext:
    config = WeftConfig()
    config.database.prime_timeout = prime_timeout
    app = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_Embedding(),
        config=config,
    )
    # Avoid starting incidental background work in handler-level tests.
    app.spawn_background_task = lambda awaitable, *, name: awaitable.close()
    return app


def _ctx(app: AppContext):
    return SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=app),
        transport="stdio",
    )


async def _store(pool, content: str, memory_type: MemoryType, **kwargs):
    return await store_memory(
        pool,
        MemoryCreate(
            type=memory_type,
            content=content,
            source=MemorySource.conversation,
            confidence=0.9,
            **kwargs,
        ),
    )


def test_prime_timeout_config_default_and_environment_override(monkeypatch):
    from weft.config import DatabaseConfig, load_config

    assert DatabaseConfig().prime_timeout == 60.0
    monkeypatch.setenv("WEFT_DB_PRIME_TIMEOUT", "1.25")
    assert load_config().database.prime_timeout == 1.25


@pytest.mark.asyncio
async def test_failed_section_preserves_survivors_and_suppresses_empty_hints(
    pool, monkeypatch
):
    """A failed section does not erase valid data or emit false empty hints."""
    await _store(pool, "Real surviving issue", MemoryType.issue)

    calls = 0

    async def fail_rules_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary database read failure")

    monkeypatch.setattr(
        "weft.primer_sections.rules.list_memories", fail_rules_query
    )

    result = await build_primer(pool, disclosure="full")

    assert calls == 2  # exactly one retry
    assert result["degraded"] is True
    assert result["incomplete_evidence"] is True
    assert result["section_status"]["rules"] == "failed"
    assert [failure["section"] for failure in result["failed_sections"]] == ["rules"]
    assert result["issues"]["count"] == 1
    assert result["issues"]["items"][0]["content"] == "Real surviving issue"
    rendered = repr(result)
    assert "No rules stored" not in rendered
    assert "Welcome to Weft" not in rendered
    assert result["hints"] == {
        "degraded": (
            "degraded: some sections failed (see failed_sections) — verify any "
            "'nothing found' conclusion via weft_recall before acting."
        )
    }
    assert result["onboarding"] is None
    assert all(
        status in {"ok", "empty", "failed"}
        for status in result["section_status"].values()
    )


@pytest.mark.asyncio
async def test_database_error_fallback_has_honest_degraded_contract(pool, monkeypatch):
    import asyncpg

    from weft.mcp import tools

    async def fail_build(*args, **kwargs):
        raise asyncpg.InterfaceError("database connection failed")

    monkeypatch.setattr("weft.primer.build_primer", fail_build)
    result = await tools.weft_prime(_ctx(_app(pool)), project_id="db-failure")

    assert result["degraded"] is True
    assert result["incomplete_evidence"] is True
    assert result["failed_sections"][0]["section"] == "database"
    assert result["hints"] == {
        "degraded": (
            "degraded: some sections failed (see failed_sections) — verify any "
            "'nothing found' conclusion via weft_recall before acting."
        )
    }
    assert result["onboarding"] is None
    assert "No rules stored" not in repr(result)
    assert "Welcome to Weft" not in repr(result)


@pytest.mark.asyncio
async def test_prime_handler_budget_returns_degraded_payload(pool, monkeypatch):
    from weft.mcp import tools

    app = _app(pool, prime_timeout=0.01)
    query_started = asyncio.Event()

    async def never_returns_query(*args, **kwargs):
        query_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "weft.primer_sections.rules.list_memories", never_returns_query
    )
    monkeypatch.setattr(tools, "_owner_scoped_canary_health", AsyncMock(return_value=None))

    result = await asyncio.wait_for(
        tools.weft_prime(_ctx(app), project_id="budget-test"), timeout=0.5
    )

    assert query_started.is_set()
    assert result["degraded"] is True
    assert result["incomplete_evidence"] is True
    assert result["failed_sections"] == [{
        "section": "prime_budget_exceeded",
        "error": "prime wall-clock budget exceeded",
    }]
    assert result["section_status"]["prime_budget_exceeded"] == "failed"
    assert result["onboarding"] is None
    assert "No rules stored" not in repr(result)


@pytest.mark.asyncio
async def test_successful_section_runs_once_without_retry(pool, monkeypatch):
    from weft.primer_sections.rules import list_memories as original_list_memories

    calls = 0

    async def count_rules_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_list_memories(*args, **kwargs)

    monkeypatch.setattr("weft.primer_sections.rules.list_memories", count_rules_query)

    result = await build_primer(pool, disclosure="full")

    assert calls == 1
    assert result["section_status"]["rules"] == "empty"
    assert result["degraded"] is False
    assert result["failed_sections"] == []


@pytest.mark.asyncio
async def test_section_retries_once_then_succeeds(pool, monkeypatch):
    from weft.primer_sections.rules import list_memories as original_list_memories

    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("single transient failure")
        return await original_list_memories(*args, **kwargs)

    monkeypatch.setattr("weft.primer_sections.rules.list_memories", fail_once)

    result = await build_primer(pool, disclosure="full")

    assert calls == 2
    assert result["section_status"]["rules"] == "empty"
    assert result["degraded"] is False
    assert result["failed_sections"] == []


@pytest.mark.asyncio
async def test_near_miss_project_key_suggests_case_insensitive_match(pool, monkeypatch):
    from weft.mcp import tools

    for index in range(4):
        await _store(
            pool,
            f"Project memory {index}",
            MemoryType.fact,
            project_id="weft",
        )

    app = _app(pool)
    monkeypatch.setattr(tools, "_owner_scoped_canary_health", AsyncMock(return_value=None))

    near_miss = await tools.weft_prime(_ctx(app), project_id="Weft")
    canonical = await tools.weft_prime(_ctx(app), project_id="weft")

    assert near_miss["project_warning"]
    assert near_miss["suggestions"] == [{
        "project_id": "weft",
        "active_memory_count": 4,
    }]
    assert "project_warning" not in canonical
    assert "suggestions" not in canonical


@pytest.mark.asyncio
async def test_closed_pool_reports_independent_section_failures(pool):
    await pool.close()

    result = await build_primer(pool, project_id="closed-pool-test", disclosure="full")

    failed_names = [failure["section"] for failure in result["failed_sections"]]
    assert result["degraded"] is True
    assert result["incomplete_evidence"] is True
    expected_failed_sections = {
        "grounding", "rules", "behaviors", "handoff", "recent_memories",
        "recent_work", "issues", "anti_patterns", "decisions", "entities",
        "autonomy", "calibration", "degradation", "triggers", "cost",
        "working_memory", "changes_since", "wellness",
    }
    assert len(failed_names) == len(set(failed_names))
    assert set(failed_names) == expected_failed_sections
    assert all(failure["error"] for failure in result["failed_sections"])
    assert result["hints"] == {
        "degraded": (
            "degraded: some sections failed (see failed_sections) — verify any "
            "'nothing found' conclusion via weft_recall before acting."
        )
    }
    assert result["onboarding"] is None


@pytest.mark.asyncio
async def test_project_census_failure_is_nonfatal(pool, monkeypatch):
    from weft.mcp import tools

    app = _app(pool)
    monkeypatch.setattr(tools, "_owner_scoped_canary_health", AsyncMock(return_value=None))
    monkeypatch.setattr(
        tools,
        "_project_memory_census",
        AsyncMock(side_effect=RuntimeError("census unavailable")),
    )

    result = await tools.weft_prime(_ctx(app), project_id="unknown-project")

    assert result["degraded"] is False
    assert result["project_warning"]
    assert result["suggestions"] == []
