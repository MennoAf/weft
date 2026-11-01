"""Shutdown cleanup must not misclassify non-process resources as processes.

Redis clients expose a ``wait`` method (the Redis WAIT command, which
requires ``num_replicas`` and ``timeout`` arguments). The cleanup helper
must only take the process wait/terminate path for process-like resources
(callable ``terminate``), so database clients fall through to ``aclose``.
Regression coverage for the fresh-install feedback: MCP stdio shutdown
crashed with ``ManagementCommands.wait() missing 2 required positional
arguments`` on redis-py 8.x.
"""

from __future__ import annotations

import asyncio

import pytest

from weft.mcp.server import _close_one_resource


class _RedisLikeClient:
    """Redis-async-client-shaped: ``wait`` is the Redis WAIT command."""

    def __init__(self):
        self.closed = False

    async def wait(self, num_replicas, timeout):  # pragma: no cover - guard
        raise AssertionError("wait() must not be called without arguments")

    async def aclose(self):
        self.closed = True


class _AsyncProcessLike:
    """asyncio-subprocess-shaped resource: wait()/terminate(), no returncode yet."""

    def __init__(self):
        self.terminated = False
        self.wait_calls = 0

    def terminate(self):
        self.terminated = True

    @property
    def returncode(self):
        return None if self.wait_calls == 0 else 0

    async def wait(self):
        self.wait_calls += 1


@pytest.mark.asyncio
async def test_redis_client_uses_close_path_not_process_wait():
    client = _RedisLikeClient()
    deadline = asyncio.get_running_loop().time() + 5
    await _close_one_resource(client, deadline)
    assert client.closed is True


@pytest.mark.asyncio
async def test_process_like_resource_uses_terminate_and_wait():
    proc = _AsyncProcessLike()
    deadline = asyncio.get_running_loop().time() + 5
    await _close_one_resource(proc, deadline)
    assert proc.terminated is True
    assert proc.wait_calls >= 1
