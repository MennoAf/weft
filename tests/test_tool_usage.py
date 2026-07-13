"""Tests for durable MCP tool usage telemetry and middleware."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from weft.mcp.tool_usage import ToolUsageMiddleware
from weft.tool_usage import get_tool_usage_summary, record_tool_usage

pytestmark = pytest.mark.asyncio


async def test_up_next_is_marked_deprecated_in_mcp_metadata():
    from weft.mcp import mcp

    tools = await mcp.list_tools()
    up_next = next(tool for tool in tools if tool.name == "weft_up_next")

    assert "deprecated" in up_next.tags
    assert "weft_board" in (up_next.description or "")


async def test_record_tool_usage_accumulates_by_day(pool):
    observed = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

    await record_tool_usage(pool, "weft_up_next", called_at=observed)
    await record_tool_usage(pool, "weft_up_next", called_at=observed)
    await record_tool_usage(pool, "weft_board", called_at=observed)

    summary = await get_tool_usage_summary(
        pool,
        days=1,
        today=date(2026, 7, 13),
    )

    assert summary["tools_used"] == 2
    assert summary["total_calls"] == 3
    assert summary["tools"][0]["tool_name"] == "weft_up_next"
    assert summary["tools"][0]["call_count"] == 2


async def test_usage_summary_rejects_non_positive_window(pool):
    with pytest.raises(ValueError, match="days must be positive"):
        await get_tool_usage_summary(pool, days=0)


async def test_middleware_records_tool_before_dispatch():
    calls: list[tuple[object, str]] = []

    async def recorder(pool, tool_name):
        calls.append((pool, tool_name))

    async def call_next(context):
        return {"ok": True}

    pool = object()
    middleware = ToolUsageMiddleware(lambda: pool, recorder)
    context = SimpleNamespace(message=SimpleNamespace(name="weft_board"))

    result = await middleware.on_call_tool(context, call_next)
    await asyncio.sleep(0)

    assert result == {"ok": True}
    assert calls == [(pool, "weft_board")]


async def test_middleware_skips_recording_before_lifespan():
    called = False

    async def recorder(_pool, _tool_name):
        nonlocal called
        called = True

    async def call_next(_context):
        return "ok"

    middleware = ToolUsageMiddleware(lambda: None, recorder)
    context = SimpleNamespace(message=SimpleNamespace(name="weft_board"))

    assert await middleware.on_call_tool(context, call_next) == "ok"
    assert called is False
