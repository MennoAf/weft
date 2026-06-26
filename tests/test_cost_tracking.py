"""Tests for cost tracking — models, store, migration, and MCP tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.cost_tracking import (
    BudgetStatus,
    CostEntry,
    CostEntryCreate,
    CostEntryType,
    CostSummary,
    SpendTrend,
    check_budget,
    get_cost_summary,
    get_spend_trend,
    list_cost_entries,
    record_cost,
)
from weft.mcp.server import AppContext


# --- Model tests ---


def test_cost_entry_type_values():
    assert CostEntryType.session.value == "session"
    assert CostEntryType.task.value == "task"
    assert CostEntryType.tool_call.value == "tool_call"
    # Tier-2 topic-digest synthesis fire/abstain telemetry (weft-d58f7350).
    assert CostEntryType.topic_synthesis.value == "topic_synthesis"


def test_cost_entry_create_defaults():
    ce = CostEntryCreate()
    assert ce.entry_type == CostEntryType.session
    assert ce.reference_id is None
    assert ce.model is None
    assert ce.input_tokens == 0
    assert ce.output_tokens == 0
    assert ce.total_tokens == 0
    assert ce.estimated_cost_usd == 0.0
    assert ce.metadata == {}


def test_cost_entry_defaults():
    ce = CostEntry()
    assert ce.id.startswith("weft-")
    assert ce.entry_type == CostEntryType.session
    assert ce.total_tokens == 0
    assert ce.estimated_cost_usd == 0.0


def test_cost_entry_to_dict():
    ce = CostEntry(
        entry_type=CostEntryType.task,
        reference_id="loom-123",
        model="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.015,
    )
    d = ce.to_dict()
    assert d["entry_type"] == "task"
    assert d["reference_id"] == "loom-123"
    assert d["total_tokens"] == 1500


def test_budget_status_to_dict():
    bs = BudgetStatus(
        daily_limit_usd=10.0,
        daily_spent_usd=3.14159,
        within_budget=True,
        pct_used=31.4159,
        remaining_usd=6.85841,
    )
    d = bs.to_dict()
    assert d["pct_used"] == 31.42
    assert d["within_budget"] is True


# --- Migration tests ---


async def test_cost_entries_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'cost_entries'
        )
        """
    )
    assert exists is True


async def test_cost_entries_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'cost_entries'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "entry_type" in columns
    assert "reference_id" in columns
    assert "model" in columns
    assert "input_tokens" in columns
    assert "output_tokens" in columns
    assert "total_tokens" in columns
    assert "estimated_cost_usd" in columns
    assert "metadata" in columns
    assert "project_id" in columns
    assert "agent_id" in columns
    assert "user_id" in columns
    assert "created_at" in columns

    assert columns["entry_type"]["is_nullable"] == "NO"
    assert columns["input_tokens"]["is_nullable"] == "NO"
    assert columns["reference_id"]["is_nullable"] == "YES"
    assert columns["model"]["is_nullable"] == "YES"


async def test_cost_entries_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'cost_entries'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "cost_entries_pkey" in index_names
    assert "idx_cost_entries_type" in index_names
    assert "idx_cost_entries_reference" in index_names
    assert "idx_cost_entries_user" in index_names
    assert "idx_cost_entries_project" in index_names
    assert "idx_cost_entries_created" in index_names


# --- Store tests ---


class TestRecordCost:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, pool):
        create = CostEntryCreate(
            entry_type=CostEntryType.task,
            reference_id="loom-abc",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.015,
        )
        entry = await record_cost(pool, create)
        assert entry.id.startswith("weft-")
        assert entry.entry_type == CostEntryType.task
        assert entry.reference_id == "loom-abc"
        assert entry.total_tokens == 1500

    @pytest.mark.asyncio
    async def test_record_with_metadata(self, pool):
        create = CostEntryCreate(
            metadata={"tool": "weft_recall", "query": "test"},
        )
        entry = await record_cost(pool, create)
        assert entry.metadata == {"tool": "weft_recall", "query": "test"}


class TestGetCostSummary:
    @pytest.mark.asyncio
    async def test_summary_empty(self, pool):
        summary = await get_cost_summary(pool)
        assert summary.total_entries == 0
        assert summary.total_tokens == 0
        assert summary.total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_summary_aggregates(self, pool):
        for i in range(3):
            await record_cost(pool, CostEntryCreate(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                estimated_cost_usd=0.01,
            ))
        summary = await get_cost_summary(pool)
        assert summary.total_entries == 3
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150
        assert summary.total_tokens == 450
        assert abs(summary.total_cost_usd - 0.03) < 0.001

    @pytest.mark.asyncio
    async def test_summary_filter_by_type(self, pool):
        await record_cost(pool, CostEntryCreate(
            entry_type=CostEntryType.task, total_tokens=100,
        ))
        await record_cost(pool, CostEntryCreate(
            entry_type=CostEntryType.session, total_tokens=200,
        ))
        summary = await get_cost_summary(pool, entry_type=CostEntryType.task)
        assert summary.total_entries == 1
        assert summary.total_tokens == 100


class TestGetSpendTrend:
    """7d-vs-prior-7d trend computation. Backdates created_at via raw SQL
    because record_cost stamps with now() unconditionally.
    """

    async def _insert_at(
        self,
        pool,
        *,
        days_ago: float,
        cost: float,
        anchor: datetime | None = None,
    ) -> None:
        base = anchor if anchor is not None else datetime.now(timezone.utc)
        ts = base - timedelta(days=days_ago)
        await pool.execute(
            """
            INSERT INTO cost_entries (
                id, entry_type, total_tokens, estimated_cost_usd, metadata,
                created_at, user_id
            )
            VALUES (
                $1, 'session', 0, $2, '{}'::jsonb, $3,
                nullif(current_setting('app.user_id', true), '')
            )
            """,
            f"weft-{days_ago}-{cost}",
            cost,
            ts,
        )

    @pytest.mark.asyncio
    async def test_insufficient_data(self, pool):
        trend = await get_spend_trend(pool)
        assert isinstance(trend, SpendTrend)
        assert trend.trend == "insufficient data"
        assert trend.recent_total_usd == 0.0
        assert trend.today_usd == 0.0

    @pytest.mark.asyncio
    async def test_rising_trend(self, pool):
        # Prior 7d: $1 total. Recent 7d: $10 total. Should rise.
        await self._insert_at(pool, days_ago=10, cost=1.0)
        await self._insert_at(pool, days_ago=3, cost=5.0)
        await self._insert_at(pool, days_ago=1, cost=5.0)

        trend = await get_spend_trend(pool)
        assert trend.recent_total_usd == pytest.approx(10.0)
        assert trend.prior_total_usd == pytest.approx(1.0)
        assert trend.trend == "rising ↑"

    @pytest.mark.asyncio
    async def test_improving_trend(self, pool):
        # Prior 7d: $20. Recent 7d: $2. Should improve.
        await self._insert_at(pool, days_ago=10, cost=10.0)
        await self._insert_at(pool, days_ago=8, cost=10.0)
        await self._insert_at(pool, days_ago=2, cost=2.0)

        trend = await get_spend_trend(pool)
        assert trend.trend == "improving ↓"

    @pytest.mark.asyncio
    async def test_stable_trend(self, pool):
        # Both windows ~equal.
        await self._insert_at(pool, days_ago=10, cost=5.0)
        await self._insert_at(pool, days_ago=2, cost=5.0)

        trend = await get_spend_trend(pool)
        assert trend.trend == "stable →"

    @pytest.mark.asyncio
    async def test_today_usd_subset_of_recent(self, pool):
        # Pin as_of to a fixed mid-UTC-day moment so "today" (UTC midnight →
        # as_of) deterministically contains the ~1h-ago entry, regardless of
        # what UTC hour the test runs at.
        as_of = datetime.now(timezone.utc).replace(
            hour=12, minute=0, second=0, microsecond=0,
        )
        await self._insert_at(pool, days_ago=0.05, cost=2.5, anchor=as_of)
        await self._insert_at(pool, days_ago=3, cost=4.0, anchor=as_of)
        # Need prior data so we don't hit insufficient_data short-circuit
        await self._insert_at(pool, days_ago=10, cost=1.0, anchor=as_of)

        trend = await get_spend_trend(pool, as_of=as_of)
        assert trend.today_usd == pytest.approx(2.5)
        assert trend.recent_total_usd == pytest.approx(6.5)


class TestCheckBudget:
    @pytest.mark.asyncio
    async def test_within_budget(self, pool):
        await record_cost(pool, CostEntryCreate(estimated_cost_usd=3.0))
        status = await check_budget(pool, daily_limit_usd=10.0)
        assert status.within_budget is True
        assert status.pct_used == pytest.approx(30.0, abs=1.0)
        assert status.remaining_usd == pytest.approx(7.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_over_budget(self, pool):
        await record_cost(pool, CostEntryCreate(estimated_cost_usd=15.0))
        status = await check_budget(pool, daily_limit_usd=10.0)
        assert status.within_budget is False
        assert status.pct_used > 100.0
        assert status.remaining_usd == 0.0

    @pytest.mark.asyncio
    async def test_zero_budget_empty(self, pool):
        status = await check_budget(pool, daily_limit_usd=10.0)
        assert status.within_budget is True
        assert status.pct_used == 0.0
        assert status.remaining_usd == 10.0


class TestListCostEntries:
    @pytest.mark.asyncio
    async def test_list_by_reference(self, pool):
        await record_cost(pool, CostEntryCreate(
            reference_id="ref-1", total_tokens=100,
        ))
        await record_cost(pool, CostEntryCreate(
            reference_id="ref-2", total_tokens=200,
        ))
        entries = await list_cost_entries(pool, reference_id="ref-1")
        assert len(entries) == 1
        assert entries[0].reference_id == "ref-1"


# --- MCP tool tests ---


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


@pytest.mark.asyncio
async def test_cost_tools_registered():
    from weft.mcp.tools import mcp

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "weft_cost_record" in tool_names
    assert "weft_cost_summary" in tool_names
    assert "weft_budget_check" in tool_names


class TestCostRecordTool:
    @pytest.mark.asyncio
    async def test_record_minimal(self, ctx):
        from weft.mcp.tools import weft_cost_record

        result = await weft_cost_record(ctx)
        assert result["success"] is True
        assert result["entry"]["entry_type"] == "session"

    @pytest.mark.asyncio
    async def test_record_full(self, ctx):
        from weft.mcp.tools import weft_cost_record

        result = await weft_cost_record(
            ctx,
            entry_type="task",
            reference_id="loom-123",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.015,
        )
        assert result["success"] is True
        entry = result["entry"]
        assert entry["entry_type"] == "task"
        assert entry["reference_id"] == "loom-123"
        assert entry["total_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_record_invalid_type(self, ctx):
        from weft.mcp.tools import weft_cost_record

        result = await weft_cost_record(ctx, entry_type="bogus")
        assert "error" in result


class TestCostSummaryTool:
    @pytest.mark.asyncio
    async def test_summary_empty(self, ctx):
        from weft.mcp.tools import weft_cost_summary

        result = await weft_cost_summary(ctx)
        assert result["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_summary_after_recording(self, ctx):
        from weft.mcp.tools import weft_cost_record, weft_cost_summary

        await weft_cost_record(
            ctx, total_tokens=1000, estimated_cost_usd=0.01,
        )
        result = await weft_cost_summary(ctx)
        assert result["total_entries"] >= 1
        assert result["total_tokens"] >= 1000


class TestBudgetCheckTool:
    @pytest.mark.asyncio
    async def test_budget_check_within(self, ctx):
        from weft.mcp.tools import weft_budget_check

        result = await weft_budget_check(ctx, daily_limit_usd=10.0)
        assert result["within_budget"] is True
        assert "pct_used" in result
        assert "remaining_usd" in result

    @pytest.mark.asyncio
    async def test_budget_check_negative_limit(self, ctx):
        from weft.mcp.tools import weft_budget_check

        result = await weft_budget_check(ctx, daily_limit_usd=-1.0)
        assert "error" in result
