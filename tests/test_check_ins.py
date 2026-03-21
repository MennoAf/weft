"""Tests for check-in system — models, store, and MCP tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.check_ins import create_check_in, get_check_in_stats, list_check_ins
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.models import CheckIn, CheckInCreate


# ── Model tests ─────────────────────────────────────────────────────


class TestCheckInCreate:
    def test_minimal(self):
        c = CheckInCreate(mood=3)
        assert c.mood == 3
        assert c.sleep_hours is None

    def test_all_fields(self):
        c = CheckInCreate(mood=4, sleep_hours=7.5, energy=3, notes="Good day")
        assert c.sleep_hours == 7.5

    def test_mood_out_of_range(self):
        with pytest.raises(ValueError, match="mood must be 1-5"):
            CheckInCreate(mood=6)

    def test_energy_out_of_range(self):
        with pytest.raises(ValueError, match="energy must be 1-5"):
            CheckInCreate(energy=0)

    def test_sleep_out_of_range(self):
        with pytest.raises(ValueError, match="sleep_hours must be 0-24"):
            CheckInCreate(sleep_hours=25)

    def test_notes_only(self):
        c = CheckInCreate(notes="Just a note")
        assert c.mood is None


class TestCheckInModel:
    def test_to_dict(self):
        c = CheckIn(mood=3, sleep_hours=7.0, energy=4, trigger_at=datetime.now(timezone.utc))
        d = c.to_dict()
        assert d["mood"] == 3
        assert d["sleep_hours"] == 7.0


# ── Store tests (real DB) ───────────────────────────────────────────


class TestCheckInStore:
    @pytest.mark.asyncio
    async def test_create_and_list(self, pool):
        await create_check_in(pool, CheckInCreate(mood=4, sleep_hours=7, energy=3))
        await create_check_in(pool, CheckInCreate(mood=3, energy=2, notes="Tired"))

        results = await list_check_ins(pool)
        assert len(results) == 2
        assert results[0].mood in (3, 4)

    @pytest.mark.asyncio
    async def test_create_with_logged_at(self, pool):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        ci = await create_check_in(pool, CheckInCreate(mood=5, logged_at=past))
        assert ci.logged_at.date() == past.date()

    @pytest.mark.asyncio
    async def test_list_ordering(self, pool):
        old = datetime.now(timezone.utc) - timedelta(days=2)
        new = datetime.now(timezone.utc) - timedelta(hours=1)
        await create_check_in(pool, CheckInCreate(mood=2, logged_at=old))
        await create_check_in(pool, CheckInCreate(mood=5, logged_at=new))

        results = await list_check_ins(pool)
        assert results[0].mood == 5  # newest first
        assert results[1].mood == 2

    @pytest.mark.asyncio
    async def test_list_with_limit(self, pool):
        for i in range(5):
            await create_check_in(pool, CheckInCreate(mood=i + 1))
        results = await list_check_ins(pool, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_stats(self, pool):
        await create_check_in(pool, CheckInCreate(mood=2, sleep_hours=6, energy=2))
        await create_check_in(pool, CheckInCreate(mood=4, sleep_hours=8, energy=4))

        stats = await get_check_in_stats(pool)
        assert stats["total"] == 2
        assert stats["avg_mood"] == 3.0
        assert stats["avg_sleep"] == 7.0
        assert stats["avg_energy"] == 3.0

    @pytest.mark.asyncio
    async def test_stats_empty(self, pool):
        stats = await get_check_in_stats(pool)
        assert stats["total"] == 0
        assert stats["avg_mood"] is None


# ── MCP tool tests ──────────────────────────────────────────────────


class FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool, cache=NullCache(), embedding=FakeEmbeddingProvider(), config=WeftConfig()
    )


@pytest.fixture
def ctx(app):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


class TestCheckInTool:
    @pytest.mark.asyncio
    async def test_log_check_in(self, ctx):
        from weft.mcp.tools import weft_check_in

        result = await weft_check_in(ctx, mood=4, sleep_hours=7.5, energy=3)
        assert result["success"] is True
        assert result["check_in"]["mood"] == 4
        assert result["check_in"]["sleep_hours"] == 7.5

    @pytest.mark.asyncio
    async def test_notes_only(self, ctx):
        from weft.mcp.tools import weft_check_in

        result = await weft_check_in(ctx, notes="Feeling off today")
        assert result["success"] is True
        assert result["check_in"]["notes"] == "Feeling off today"

    @pytest.mark.asyncio
    async def test_no_fields_errors(self, ctx):
        from weft.mcp.tools import weft_check_in

        result = await weft_check_in(ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_mood_out_of_range(self, ctx):
        from weft.mcp.tools import weft_check_in

        result = await weft_check_in(ctx, mood=7)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_logged_at(self, ctx):
        from weft.mcp.tools import weft_check_in

        result = await weft_check_in(ctx, mood=3, logged_at="2026-03-20T08:00:00+00:00")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_history(self, ctx):
        from weft.mcp.tools import weft_check_in, weft_check_in_history

        await weft_check_in(ctx, mood=3, energy=2)
        await weft_check_in(ctx, mood=4, sleep_hours=8)

        result = await weft_check_in_history(ctx)
        assert result["count"] == 2
        assert result["stats_30d"]["total"] == 2


@pytest.mark.asyncio
async def test_check_in_tools_registered():
    from weft.mcp.tools import mcp

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "weft_check_in" in tool_names
    assert "weft_check_in_history" in tool_names


# ── DB migration tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_ins_table_exists(pool):
    exists = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'check_ins')"
    )
    assert exists


@pytest.mark.asyncio
async def test_check_ins_rls_enabled(pool):
    rls = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'check_ins'"
    )
    assert rls is True
