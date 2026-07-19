"""Tests for durable MCP tool usage telemetry and middleware."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from weft.mcp.tool_usage import ToolUsageMiddleware
from weft.tool_usage import (
    RECORDER_VERSION,
    get_tool_usage_summary,
    record_tool_usage,
    record_tool_usage_heartbeat,
)

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


async def test_usage_summary_default_window_uses_utc_day(pool):
    observed = datetime(2026, 7, 19, 0, 5, tzinfo=timezone.utc)
    await record_tool_usage_heartbeat(pool, observed_at=observed, failure_count=1)

    with patch("weft.tool_usage.datetime") as clock:
        clock.now.return_value = observed
        summary = await get_tool_usage_summary(pool, days=1)

    assert summary["through"] == "2026-07-19"
    assert summary["coverage"]["failure_total"] == 1


async def test_offset_aware_writes_are_bucketed_by_utc_day(pool):
    pacific = timezone(timedelta(hours=-7))
    observed = datetime(2026, 7, 18, 20, 30, tzinfo=pacific)

    await record_tool_usage(pool, "weft_board", called_at=observed)
    summary = await get_tool_usage_summary(
        pool, days=1, today=date(2026, 7, 19)
    )

    assert summary["total_calls"] == 1
    assert summary["tools"][0]["first_called_at"].startswith("2026-07-19T03:30:00")
    assert summary["coverage"]["valid_days"] == 1


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


async def test_failed_background_task_does_not_abort_sibling_cleanup(caplog):
    from weft.mcp.server import _cancel_background_tasks

    async def fail():
        raise RuntimeError("background boom")

    sibling_started = asyncio.Event()
    sibling_cleaned = asyncio.Event()

    async def sibling():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cleaned.set()

    failed_task = asyncio.create_task(fail())
    sibling_task = asyncio.create_task(sibling())
    await sibling_started.wait()
    await asyncio.sleep(0)

    await _cancel_background_tasks((failed_task, sibling_task))

    assert sibling_cleaned.is_set()
    assert failed_task.done()
    assert sibling_task.done()
    assert "background task failed before shutdown cleanup" in caplog.text


async def test_usage_recording_task_retained_and_drained(pool):
    release = asyncio.Event()

    async def recorder(_pool, _tool_name):
        await release.wait()

    async def call_next(_context):
        return "ok"

    middleware = ToolUsageMiddleware(lambda: pool, recorder)
    context = SimpleNamespace(message=SimpleNamespace(name="weft_board"))
    assert await middleware.on_call_tool(context, call_next) == "ok"
    assert middleware.pending_count == 1

    release.set()
    report = await middleware.drain(pool)
    assert report == {
        "pending_before_drain": 1,
        "pending_after_drain": 0,
        "failure_count": 0,
        "shutdown_drained": True,
    }


async def test_usage_recorder_failure_counted(pool):
    async def recorder(_pool, _tool_name):
        raise RuntimeError("telemetry database unavailable")

    async def call_next(_context):
        return "tool-result"

    middleware = ToolUsageMiddleware(lambda: pool, recorder)
    context = SimpleNamespace(message=SimpleNamespace(name="weft_board"))
    assert await middleware.on_call_tool(context, call_next) == "tool-result"
    report = await middleware.drain(pool)
    assert report["pending_after_drain"] == 0
    assert report["failure_count"] == 1
    assert report["shutdown_drained"] is True

    summary = await get_tool_usage_summary(pool, days=1)
    assert summary["coverage"]["failure_total"] == 1
    assert summary["coverage"]["valid_days"] == 0
    assert summary["zero_use_classification"] == "coverage-incomplete"


async def test_usage_heartbeat_records_valid_quiet_day(pool):
    observed = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    await record_tool_usage_heartbeat(pool, observed_at=observed)

    summary = await get_tool_usage_summary(
        pool,
        days=1,
        today=date(2026, 7, 13),
    )
    assert summary["total_calls"] == 0
    assert summary["coverage"] == {
        "recorder_version": RECORDER_VERSION,
        "expected_days": 1,
        "valid_days": 1,
        "gap_days": 0,
        "gap_dates": [],
        "failure_total": 0,
        "successful_writes": 0,
        "shutdown_not_drained_dates": [],
        "complete": True,
    }
    assert summary["zero_use_classification"] == "observed-zero"
    assert summary["deprecation_eligible"] is False


async def test_coverage_gap_blocks_zero_use_classification(pool):
    await record_tool_usage_heartbeat(
        pool,
        observed_at=datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc),
    )
    summary = await get_tool_usage_summary(
        pool,
        days=2,
        today=date(2026, 7, 13),
    )
    assert summary["coverage"]["valid_days"] == 1
    assert summary["coverage"]["gap_dates"] == ["2026-07-13"]
    assert summary["zero_use_classification"] == "coverage-incomplete"
    assert summary["deprecation_eligible"] is False


async def test_failed_shutdown_drain_invalidates_coverage_day(pool):
    observed = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    await record_tool_usage_heartbeat(
        pool,
        observed_at=observed,
        shutdown_drained=False,
    )
    summary = await get_tool_usage_summary(
        pool,
        days=1,
        today=date(2026, 7, 13),
    )
    assert summary["coverage"]["valid_days"] == 0
    assert summary["coverage"]["shutdown_not_drained_dates"] == ["2026-07-13"]
    assert summary["deprecation_eligible"] is False


async def test_failed_shutdown_marker_is_sticky_for_the_day(pool):
    observed = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    await record_tool_usage_heartbeat(
        pool, observed_at=observed, shutdown_drained=False,
    )
    await record_tool_usage_heartbeat(
        pool,
        observed_at=observed + timedelta(hours=1),
        shutdown_drained=True,
    )

    summary = await get_tool_usage_summary(
        pool, days=1, today=date(2026, 7, 13),
    )
    assert summary["coverage"]["valid_days"] == 0
    assert summary["coverage"]["shutdown_not_drained_dates"] == ["2026-07-13"]


async def test_repeated_drain_does_not_double_count_failures(pool):
    async def recorder(_pool, _tool_name):
        raise RuntimeError("telemetry database unavailable")

    middleware = ToolUsageMiddleware(lambda: pool, recorder)
    context = SimpleNamespace(message=SimpleNamespace(name="weft_board"))
    await middleware.on_call_tool(context, lambda _context: asyncio.sleep(0))
    await middleware.drain(pool)
    await middleware.drain(pool)

    summary = await get_tool_usage_summary(pool, days=1)
    assert summary["coverage"]["failure_total"] == 1


async def test_deprecation_requires_thirty_valid_coverage_days(pool):
    start = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
    for offset in range(30):
        await record_tool_usage_heartbeat(
            pool,
            observed_at=start + timedelta(days=offset),
        )

    summary = await get_tool_usage_summary(
        pool,
        days=30,
        today=date(2026, 7, 13),
    )
    assert summary["coverage"]["valid_days"] == 30
    assert summary["coverage"]["complete"] is True
    assert summary["deprecation_eligible"] is True
