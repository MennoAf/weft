from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.config import WeftConfig
from weft.db import connection
from weft.mcp import server


@pytest.mark.asyncio
async def test_pool_creation_passes_explicit_connection_timeout(monkeypatch) -> None:
    pool = object()
    create = AsyncMock(return_value=pool)
    monkeypatch.setattr(connection.asyncpg, "create_pool", create)

    result = await connection.create_pool(WeftConfig(), connect_timeout=2.5)

    assert result is pool
    assert create.await_args.kwargs["timeout"] == 2.5


@pytest.mark.asyncio
async def test_retry_ladder_fails_deterministically_at_total_deadline() -> None:
    calls: list[float] = []

    async def unavailable(timeout: float) -> None:
        calls.append(timeout)
        raise ConnectionError("database unavailable")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.035
    started = loop.time()
    with pytest.raises(TimeoutError, match="startup deadline exhausted"):
        await server._connect_with_retry(
            unavailable,
            "Postgres",
            max_retries=5,
            base_delay=0.01,
            deadline=deadline,
            attempt_timeout=0.02,
        )

    assert calls
    assert len(calls) < 6
    assert all(0 < timeout <= 0.02 for timeout in calls)
    assert loop.time() - started < 0.1


@pytest.mark.asyncio
async def test_retry_attempt_is_cancelled_within_its_connection_timeout() -> None:
    calls = 0

    async def stalled(timeout: float) -> None:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await server._connect_with_retry(
            stalled,
            "Postgres",
            max_retries=4,
            base_delay=0.01,
            deadline=asyncio.get_running_loop().time() + 0.03,
            attempt_timeout=0.01,
        )
    assert calls <= 3
    assert time.monotonic() - started < 0.08


@pytest.mark.asyncio
async def test_embedding_validation_timeout_fails_before_readiness() -> None:
    class HangingEmbedding:
        provider_name = "test-provider"

        async def embed(self, _text: str) -> list[float]:
            await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="test-provider.*validation exceeded"):
        await server._validate_embedding_provider(HangingEmbedding(), timeout=0.01)


@pytest.mark.asyncio
async def test_shutdown_degrades_inside_shared_deadline(monkeypatch, caplog) -> None:
    release = asyncio.Event()
    monkeypatch.setattr(server, "SHUTDOWN_TIMEOUT_SECONDS", 0.03)

    async def slow_drain(_pool):
        await release.wait()
        return {"shutdown_drained": False}

    monkeypatch.setattr(server.tool_usage_middleware, "drain", slow_drain)
    ctx = SimpleNamespace(
        pool=object(),
        cache=None,
        embedding=None,
        episode_embedding=None,
        owned_resources=(),
        subprocesses=(),
        _background_tasks=set(),
    )

    started = time.monotonic()
    errors = await server._shutdown_app(ctx)
    elapsed = time.monotonic() - started
    assert elapsed < 0.08
    assert errors and isinstance(errors[0], TimeoutError)
    assert "telemetry drain failed" in caplog.text
    release.set()
    await asyncio.sleep(0)


def test_fly_health_and_kill_budgets_cover_cleanup_margin() -> None:
    repo = Path(__file__).resolve().parents[1]
    configs = (
        repo / "deploy/examples/fly/fly.example.toml",
        repo / "deploy/examples/fly/fly.staging.example.toml",
    )
    for config in configs:
        content = config.read_text(encoding="utf-8")
        assert 'kill_timeout = "15s"' in content

    assert server.STARTUP_READINESS_TIMEOUT_SECONDS + server.STARTUP_CLEANUP_TIMEOUT_SECONDS < 30
    assert server.SHUTDOWN_TIMEOUT_SECONDS < 15
    assert 'grace_period = "30s"' in configs[-1].read_text(encoding="utf-8")
