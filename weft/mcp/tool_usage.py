"""FastMCP middleware that records tool names without request payloads."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import asyncpg
import mcp.types as mt
from fastmcp.server.middleware import Middleware, MiddlewareContext

from weft.tool_usage import record_tool_usage

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
            task.add_done_callback(self._report_task_failure)
        return await call_next(context)

    @staticmethod
    def _report_task_failure(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if (error := task.exception()) is not None:
            logger.warning("tool usage telemetry task failed: %s", error)
