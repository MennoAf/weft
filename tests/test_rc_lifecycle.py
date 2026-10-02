"""Deterministic RC-FL-12 lifecycle and worker-fencing contracts."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.mcp import server
from weft.scheduler import scheduler_loop


class CloseProbe:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.closed = False
        self.error = error

    async def aclose(self) -> None:
        self.closed = True
        if self.error:
            raise self.error


class ProcessProbe:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


@pytest.mark.asyncio
async def test_lifecycle_budgets_are_explicit() -> None:
    assert server.STARTUP_READINESS_TIMEOUT_SECONDS <= 45
    assert server.SHUTDOWN_TIMEOUT_SECONDS <= 10


@pytest.mark.asyncio
async def test_cleanup_closes_nested_provider_client() -> None:
    client = CloseProbe()
    provider = SimpleNamespace(_client=client)

    errors = await server._cleanup_resources((provider,), timeout=0.1)

    assert errors == []
    assert client.closed


@pytest.mark.asyncio
async def test_cleanup_closes_all_owned_resources_and_surfaces_errors(caplog) -> None:
    pool = CloseProbe()
    redis = CloseProbe()
    embedding = CloseProbe()
    text_provider = CloseProbe(error=RuntimeError("text close failed"))
    process = ProcessProbe()

    started = time.monotonic()
    errors = await server._cleanup_resources(
        (pool, redis, embedding, text_provider, process),
        timeout=server.SHUTDOWN_TIMEOUT_SECONDS,
    )
    elapsed = time.monotonic() - started

    assert elapsed <= 10
    assert all(resource.closed for resource in (pool, redis, embedding))
    assert text_provider.closed
    assert process.terminated and process.waited
    assert any("text close failed" in str(error) for error in errors)
    assert any("text close failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_cleanup_kills_subprocess_after_terminate_timeout() -> None:
    class StubbornProcess(ProcessProbe):
        async def wait(self) -> int:
            if not self.killed:
                raise asyncio.TimeoutError()
            self.waited = True
            return self.returncode or 0

    process = StubbornProcess()
    errors = await server._cleanup_resources((process,), timeout=0.5)

    assert errors == []
    assert process.terminated and process.killed and process.waited


@pytest.mark.asyncio
async def test_cleanup_subprocess_post_kill_wait_uses_remaining_budget() -> None:
    class PostKillStallProcess(ProcessProbe):
        def terminate(self) -> None:
            self.terminated = True

        async def wait(self) -> int:
            await asyncio.sleep(60)
            self.waited = True
            return self.returncode or 0

    process = PostKillStallProcess()
    started = time.monotonic()
    errors = await server._cleanup_resources((process,), timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.08
    assert process.terminated and process.killed
    assert any(isinstance(error, TimeoutError) for error in errors)


@pytest.mark.asyncio
async def test_shutdown_preserves_primary_failure_and_closes_resources() -> None:
    pool = CloseProbe()
    redis = CloseProbe()
    embedding = CloseProbe()
    ctx = SimpleNamespace(
        pool=pool,
        cache=SimpleNamespace(_redis=redis),
        embedding=embedding,
        episode_embedding=None,
        owned_resources=(CloseProbe(),),
        subprocesses=(ProcessProbe(),),
        _background_tasks=set(),
    )

    with pytest.raises(ValueError, match="primary"):
        await server._shutdown_app(
            ctx,
            redis=redis,
            lifespan_tasks=(),
            primary_error=ValueError("primary"),
        )
    assert pool.closed and redis.closed and embedding.closed
    assert ctx.owned_resources[0].closed
    assert ctx.subprocesses[0].waited


@pytest.mark.asyncio
async def test_startup_failure_cleanup_closes_resources() -> None:
    pool = CloseProbe()
    redis = CloseProbe()
    provider = CloseProbe()

    with pytest.raises(RuntimeError, match="startup"):
        await server._run_startup_cleanup(
            RuntimeError("startup"),
            resources=(pool, redis, provider),
        )

    assert pool.closed and redis.closed and provider.closed


@pytest.mark.asyncio
async def test_cancel_background_tasks_is_bounded_and_observed() -> None:
    finished = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    task = asyncio.create_task(worker())
    await asyncio.sleep(0)
    started = time.monotonic()
    await server._cancel_background_tasks((task,), timeout=server.SHUTDOWN_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    assert elapsed <= 10
    assert task.done() and task.cancelled()
    assert finished.is_set()


@pytest.mark.asyncio
async def test_lease_loss_prevents_scheduler_side_effects(monkeypatch) -> None:
    class LostLease:
        async def acquire(self):
            return object()

        async def assert_owner(self):
            from weft.worker_lease import LeaseLostError

            raise LeaseLostError("fenced")

        async def release(self):
            return True

    poll = AsyncMock()
    dispatch = AsyncMock()
    monkeypatch.setattr(server, "scheduler_loop", scheduler_loop)
    monkeypatch.setattr("weft.scheduler.poll_due_alerts", poll)
    monkeypatch.setattr("weft.scheduler.dispatch_alert", dispatch)

    await scheduler_loop(AsyncMock(), lease=LostLease(), interval=0)

    poll.assert_not_awaited()
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_lease_release_error_is_visible_without_masking_cancel(caplog) -> None:
    class BrokenReleaseLease:
        async def acquire(self):
            return object()

        async def assert_owner(self):
            raise asyncio.CancelledError()

        async def release(self):
            raise RuntimeError("release failed")

    with pytest.raises(asyncio.CancelledError):
        await scheduler_loop(AsyncMock(), lease=BrokenReleaseLease(), interval=0)
    assert any("release failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_scheduler_default_has_no_lease_behavior(monkeypatch) -> None:
    calls = 0

    async def poll(_pool, *, batch_size):
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    monkeypatch.setattr("weft.scheduler.poll_due_alerts", poll)
    with pytest.raises(asyncio.CancelledError):
        await scheduler_loop(AsyncMock(), interval=0)
    assert calls == 1
