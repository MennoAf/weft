"""Tests for alert MCP tools — weft_alert_create, weft_alert_list, weft_alert_dismiss."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


class FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# ── Registration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alert_tools_registered():
    from weft.mcp.tools import mcp

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "weft_alert_create" in tool_names
    assert "weft_alert_list" in tool_names
    assert "weft_alert_dismiss" in tool_names


# ── weft_alert_create ───────────────────────────────────────────────


class TestAlertCreate:
    @pytest.mark.asyncio
    async def test_create_minimal(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="custom",
            title="Test alert",
            trigger_at=trigger,
        )
        assert result["success"] is True
        alert = result["alert"]
        assert alert["title"] == "Test alert"
        assert alert["alert_type"] == "custom"
        assert alert["channel"] == "log"
        assert alert["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_with_all_fields(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="due_task",
            title="Task overdue",
            trigger_at=trigger,
            body="Your task is overdue",
            channel="slack",
            channel_target="#alerts",
            payload={"task_id": "loom-123"},
        )
        assert result["success"] is True
        alert = result["alert"]
        assert alert["channel"] == "slack"
        assert alert["channel_target"] == "#alerts"
        assert alert["payload"] == {"task_id": "loom-123"}
        assert alert["body"] == "Your task is overdue"

    @pytest.mark.asyncio
    async def test_create_past_trigger_allowed(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="follow_up",
            title="Past alert",
            trigger_at=trigger,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_naive_datetime_becomes_utc(self, ctx):
        from weft.mcp.tools import weft_alert_create

        result = await weft_alert_create(
            ctx,
            alert_type="custom",
            title="Naive dt",
            trigger_at="2026-12-01T10:00:00",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_invalid_alert_type(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = datetime.now(timezone.utc).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="nonexistent",
            title="Bad type",
            trigger_at=trigger,
        )
        assert "error" in result
        assert "nonexistent" in result["detail"]

    @pytest.mark.asyncio
    async def test_create_invalid_channel(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = datetime.now(timezone.utc).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="custom",
            title="Bad channel",
            trigger_at=trigger,
            channel="email",
        )
        assert "error" in result
        assert "email" in result["detail"]

    @pytest.mark.asyncio
    async def test_create_slack_without_target(self, ctx):
        from weft.mcp.tools import weft_alert_create

        trigger = datetime.now(timezone.utc).isoformat()
        result = await weft_alert_create(
            ctx,
            alert_type="custom",
            title="No target",
            trigger_at=trigger,
            channel="slack",
        )
        assert "error" in result
        assert "channel_target" in result["detail"]

    @pytest.mark.asyncio
    async def test_create_invalid_trigger_at(self, ctx):
        from weft.mcp.tools import weft_alert_create

        result = await weft_alert_create(
            ctx,
            alert_type="custom",
            title="Bad time",
            trigger_at="not-a-date",
        )
        assert "error" in result
        assert "ISO8601" in result["detail"]

    @pytest.mark.asyncio
    async def test_create_appears_in_list(self, ctx):
        from weft.mcp.tools import weft_alert_create, weft_alert_list

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        await weft_alert_create(
            ctx,
            alert_type="custom",
            title="Listable alert",
            trigger_at=trigger,
        )
        result = await weft_alert_list(ctx)
        assert result["count"] == 1
        assert result["alerts"][0]["title"] == "Listable alert"


# ── weft_alert_list ─────────────────────────────────────────────────


class TestAlertList:
    @pytest.mark.asyncio
    async def test_list_empty(self, ctx):
        from weft.mcp.tools import weft_alert_list

        result = await weft_alert_list(ctx)
        assert result["count"] == 0
        assert result["alerts"] == []

    @pytest.mark.asyncio
    async def test_list_filters_by_status(self, ctx, pool):
        from weft.mcp.tools import weft_alert_create, weft_alert_dismiss, weft_alert_list

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        r1 = await weft_alert_create(ctx, alert_type="custom", title="Pending", trigger_at=trigger)
        r2 = await weft_alert_create(ctx, alert_type="custom", title="To dismiss", trigger_at=trigger)
        await weft_alert_dismiss(ctx, alert_id=r2["alert"]["id"])

        pending = await weft_alert_list(ctx, status="pending")
        assert pending["count"] == 1
        assert pending["alerts"][0]["title"] == "Pending"

        dismissed = await weft_alert_list(ctx, status="dismissed")
        assert dismissed["count"] == 1
        assert dismissed["alerts"][0]["title"] == "To dismiss"

    @pytest.mark.asyncio
    async def test_list_invalid_status(self, ctx):
        from weft.mcp.tools import weft_alert_list

        result = await weft_alert_list(ctx, status="invalid")
        assert "error" in result
        assert "invalid" in result["detail"]

    @pytest.mark.asyncio
    async def test_list_with_limit(self, ctx):
        from weft.mcp.tools import weft_alert_create, weft_alert_list

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        for i in range(5):
            await weft_alert_create(ctx, alert_type="custom", title=f"Alert {i}", trigger_at=trigger)

        result = await weft_alert_list(ctx, limit=3)
        assert result["count"] == 3


# ── weft_alert_dismiss ──────────────────────────────────────────────


class TestAlertDismiss:
    @pytest.mark.asyncio
    async def test_dismiss_success(self, ctx):
        from weft.mcp.tools import weft_alert_create, weft_alert_dismiss

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        r = await weft_alert_create(ctx, alert_type="custom", title="Dismiss me", trigger_at=trigger)
        alert_id = r["alert"]["id"]

        result = await weft_alert_dismiss(ctx, alert_id=alert_id)
        assert result["success"] is True
        assert result["dismissed"] == alert_id

    @pytest.mark.asyncio
    async def test_dismiss_nonexistent(self, ctx):
        from weft.mcp.tools import weft_alert_dismiss

        result = await weft_alert_dismiss(ctx, alert_id="weft-nonexistent")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_dismiss_invalid_id(self, ctx):
        from weft.mcp.tools import weft_alert_dismiss

        result = await weft_alert_dismiss(ctx, alert_id="bad-id")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_dismiss_idempotent(self, ctx):
        from weft.mcp.tools import weft_alert_create, weft_alert_dismiss

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        r = await weft_alert_create(ctx, alert_type="custom", title="Double dismiss", trigger_at=trigger)
        alert_id = r["alert"]["id"]

        await weft_alert_dismiss(ctx, alert_id=alert_id)
        result = await weft_alert_dismiss(ctx, alert_id=alert_id)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_dismissed_shows_in_list(self, ctx):
        from weft.mcp.tools import weft_alert_create, weft_alert_dismiss, weft_alert_list

        trigger = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        r = await weft_alert_create(ctx, alert_type="custom", title="Will dismiss", trigger_at=trigger)
        await weft_alert_dismiss(ctx, alert_id=r["alert"]["id"])

        result = await weft_alert_list(ctx, status="dismissed")
        assert result["count"] == 1
        assert result["alerts"][0]["status"] == "dismissed"
