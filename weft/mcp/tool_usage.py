"""FastMCP middleware that records tool names without request payloads."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import asyncpg
import mcp.types as mt
from fastmcp.server.middleware import Middleware, MiddlewareContext

from weft.tool_usage import record_tool_usage, record_tool_usage_heartbeat

logger = logging.getLogger(__name__)

ToolUsageRecorder = Callable[[asyncpg.Pool, str], Awaitable[None]]


class ToolUsageMiddleware(Middleware):
    """Record attempted MCP tool calls using fire-and-forget persistence."""

    def __init__(
        self,
        pool_getter: Callable[[], asyncpg.Pool | None],
        recorder: ToolUsageRecorder = record_tool_usage,
    ) -> None:
        self._pool_getter = pool_getter
        self._recorder = recorder
        self._tasks: set[asyncio.Task] = set()
        self._failure_count = 0
        self._reported_failure_count = 0

    @property
    def pending_count(self) -> int:
        """Number of telemetry writes still retained by the middleware."""
        return len(self._tasks)

    @property
    def failure_count(self) -> int:
        """Number of retained recorder tasks that completed with an error."""
        return self._failure_count

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next,
    ):
        pool = self._pool_getter()
        if pool is not None:
            task = asyncio.create_task(
                self._recorder(pool, context.message.name),
                name=f"weft-tool-usage-{context.message.name}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._task_done)
        return await call_next(context)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        if (error := task.exception()) is not None:
            self._failure_count += 1
            logger.warning("tool usage telemetry task failed: %s", error)

    async def heartbeat(self, pool: asyncpg.Pool) -> None:
        """Record that this recorder version is alive, even on a quiet day."""
        await record_tool_usage_heartbeat(pool)

    async def drain(self, pool: asyncpg.Pool) -> dict:
        """Wait for retained writes and persist final failure/drain state."""
        pending = list(self._tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        drained = not self._tasks
        failures = self._failure_count
        failure_delta = failures - self._reported_failure_count
        try:
            await record_tool_usage_heartbeat(
                pool,
                failure_count=failure_delta,
                shutdown_drained=drained,
            )
            self._reported_failure_count = failures
        except (OSError, asyncpg.PostgresError, RuntimeError) as exc:
            logger.warning("tool usage shutdown marker failed: %s", exc)
            drained = False
        return {
            "pending_before_drain": len(pending),
            "pending_after_drain": len(self._tasks),
            "failure_count": failures,
            "shutdown_drained": drained,
        }
