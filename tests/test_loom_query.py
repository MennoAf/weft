"""Tests for weft.loom_query — Loom SQL abstraction layer."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tests.test_loom_alerts import _LOOM_SCHEMA, _create_project, _create_task

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def loom_pool(pool):
    """Pool with Loom schema tables created."""
    await pool.execute(_LOOM_SCHEMA)
    yield pool
    await pool.execute("DROP TABLE IF EXISTS tasks CASCADE")
    await pool.execute("DROP TABLE IF EXISTS projects CASCADE")


# ---------------------------------------------------------------------------
# Module import & basic contract
# ---------------------------------------------------------------------------


class TestModuleContract:
    """Verify loom_query exists and exports the expected interface."""

    def test_module_importable(self):
        import weft.loom_query  # noqa: F401

    def test_exports_exception(self):
        from weft.loom_query import LoomQueryError

        assert issubclass(LoomQueryError, Exception)

    def test_exports_query_functions(self):
        from weft.loom_query import (
            get_blocked_pile_ups,
            get_completable_epics,
            get_ready_tasks,
            get_stale_claimed_tasks,
            loom_tables_exist,
        )

        for fn in [
            loom_tables_exist,
            get_stale_claimed_tasks,
            get_completable_epics,
            get_blocked_pile_ups,
            get_ready_tasks,
        ]:
            assert inspect.iscoroutinefunction(fn)

    def test_pool_is_first_param(self):
        """Every public query function takes pool as first parameter."""
        from weft import loom_query

        for name in dir(loom_query):
            if name.startswith("_"):
                continue
            obj = getattr(loom_query, name)
            if inspect.iscoroutinefunction(obj):
                params = list(inspect.signature(obj).parameters)
                assert params[0] == "pool", f"{name} first param should be 'pool'"


# ---------------------------------------------------------------------------
# None pool guard
# ---------------------------------------------------------------------------


class TestNonePoolGuard:
    @pytest.mark.asyncio
    async def test_stale_claims_raises_on_none_pool(self):
        from weft.loom_query import LoomQueryError, get_stale_claimed_tasks

        with pytest.raises(LoomQueryError, match="pool"):
            await get_stale_claimed_tasks(None)

    @pytest.mark.asyncio
    async def test_completable_epics_raises_on_none_pool(self):
        from weft.loom_query import LoomQueryError, get_completable_epics

        with pytest.raises(LoomQueryError, match="pool"):
            await get_completable_epics(None)

    @pytest.mark.asyncio
    async def test_blocked_pile_ups_raises_on_none_pool(self):
        from weft.loom_query import LoomQueryError, get_blocked_pile_ups

        with pytest.raises(LoomQueryError, match="pool"):
            await get_blocked_pile_ups(None)

    @pytest.mark.asyncio
    async def test_ready_tasks_raises_on_none_pool(self):
        from weft.loom_query import LoomQueryError, get_ready_tasks

        with pytest.raises(LoomQueryError, match="pool"):
            await get_ready_tasks(None)

    @pytest.mark.asyncio
    async def test_tables_exist_raises_on_none_pool(self):
        from weft.loom_query import LoomQueryError, loom_tables_exist

        with pytest.raises(LoomQueryError, match="pool"):
            await loom_tables_exist(None)


# ---------------------------------------------------------------------------
# loom_tables_exist
# ---------------------------------------------------------------------------


class TestLoomTablesExist:
    @pytest.mark.asyncio
    async def test_true_when_tables_present(self, loom_pool):
        from weft.loom_query import loom_tables_exist

        assert await loom_tables_exist(loom_pool) is True

    @pytest.mark.asyncio
    async def test_false_when_no_tables(self, pool):
        from weft.loom_query import loom_tables_exist

        assert await loom_tables_exist(pool) is False


# ---------------------------------------------------------------------------
# get_stale_claimed_tasks
# ---------------------------------------------------------------------------


class TestGetStaleClaimedTasks:
    @pytest.mark.asyncio
    async def test_returns_stale_tasks(self, loom_pool):
        from weft.loom_query import get_stale_claimed_tasks

        pid = await _create_project(loom_pool)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=50)
        await _create_task(
            loom_pool, pid, task_id="stale-1", title="Stale Task",
            status="claimed", assignee="agent-1", claimed_at=stale_time,
        )
        result = await get_stale_claimed_tasks(loom_pool)
        assert len(result) == 1
        assert result[0]["id"] == "stale-1"
        assert result[0]["title"] == "Stale Task"
        assert result[0]["assignee"] == "agent-1"
        assert "project_name" in result[0]
        assert "claimed_at" in result[0]

    @pytest.mark.asyncio
    async def test_excludes_recent_claims(self, loom_pool):
        from weft.loom_query import get_stale_claimed_tasks

        pid = await _create_project(loom_pool)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await _create_task(
            loom_pool, pid, status="claimed", claimed_at=recent_time,
        )
        result = await get_stale_claimed_tasks(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_custom_threshold(self, loom_pool):
        from weft.loom_query import get_stale_claimed_tasks

        pid = await _create_project(loom_pool)
        # 25 hours ago — stale at 24h threshold but not at default 48h
        time_25h = datetime.now(timezone.utc) - timedelta(hours=25)
        await _create_task(
            loom_pool, pid, status="claimed", claimed_at=time_25h,
        )
        assert len(await get_stale_claimed_tasks(loom_pool, threshold_hours=48)) == 0
        assert len(await get_stale_claimed_tasks(loom_pool, threshold_hours=24)) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_matches(self, loom_pool):
        from weft.loom_query import get_stale_claimed_tasks

        result = await get_stale_claimed_tasks(loom_pool)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_plain_dicts(self, loom_pool):
        from weft.loom_query import get_stale_claimed_tasks

        pid = await _create_project(loom_pool)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=50)
        await _create_task(
            loom_pool, pid, status="claimed", claimed_at=stale_time,
        )
        result = await get_stale_claimed_tasks(loom_pool)
        assert isinstance(result[0], dict)


# ---------------------------------------------------------------------------
# get_completable_epics
# ---------------------------------------------------------------------------


class TestGetCompletableEpics:
    @pytest.mark.asyncio
    async def test_returns_epics_with_all_children_done(self, loom_pool):
        from weft.loom_query import get_completable_epics

        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", title="My Epic", status="pending")
        await _create_task(loom_pool, pid, task_id="c-1", status="done", parent_id="epic-1")
        await _create_task(loom_pool, pid, task_id="c-2", status="done", parent_id="epic-1")

        result = await get_completable_epics(loom_pool)
        assert len(result) == 1
        assert result[0]["id"] == "epic-1"
        assert result[0]["child_count"] == 2
        assert "project_name" in result[0]

    @pytest.mark.asyncio
    async def test_excludes_epics_with_pending_children(self, loom_pool):
        from weft.loom_query import get_completable_epics

        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", status="pending")
        await _create_task(loom_pool, pid, task_id="c-1", status="done", parent_id="epic-1")
        await _create_task(loom_pool, pid, task_id="c-2", status="pending", parent_id="epic-1")

        result = await get_completable_epics(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_cancelled_children_count_as_done(self, loom_pool):
        from weft.loom_query import get_completable_epics

        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", status="pending")
        await _create_task(loom_pool, pid, task_id="c-1", status="done", parent_id="epic-1")
        await _create_task(loom_pool, pid, task_id="c-2", status="cancelled", parent_id="epic-1")

        result = await get_completable_epics(loom_pool)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_epics(self, loom_pool):
        from weft.loom_query import get_completable_epics

        result = await get_completable_epics(loom_pool)
        assert result == []


# ---------------------------------------------------------------------------
# get_blocked_pile_ups
# ---------------------------------------------------------------------------


class TestGetBlockedPileUps:
    @pytest.mark.asyncio
    async def test_returns_projects_above_threshold(self, loom_pool):
        from weft.loom_query import get_blocked_pile_ups

        pid = await _create_project(loom_pool, name="Blocked Project")
        for i in range(5):
            await _create_task(
                loom_pool, pid, task_id=f"b-{i}", status="blocked",
            )

        result = await get_blocked_pile_ups(loom_pool, threshold=5)
        assert len(result) == 1
        assert result[0]["project_name"] == "Blocked Project"
        assert result[0]["blocked_count"] == 5

    @pytest.mark.asyncio
    async def test_excludes_below_threshold(self, loom_pool):
        from weft.loom_query import get_blocked_pile_ups

        pid = await _create_project(loom_pool)
        for i in range(4):
            await _create_task(
                loom_pool, pid, task_id=f"b-{i}", status="blocked",
            )

        result = await get_blocked_pile_ups(loom_pool, threshold=5)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_custom_threshold(self, loom_pool):
        from weft.loom_query import get_blocked_pile_ups

        pid = await _create_project(loom_pool)
        for i in range(3):
            await _create_task(
                loom_pool, pid, task_id=f"b-{i}", status="blocked",
            )

        assert len(await get_blocked_pile_ups(loom_pool, threshold=3)) == 1
        assert len(await get_blocked_pile_ups(loom_pool, threshold=4)) == 0


# ---------------------------------------------------------------------------
# get_ready_tasks
# ---------------------------------------------------------------------------


class TestGetReadyTasks:
    @pytest.mark.asyncio
    async def test_returns_pending_tasks(self, loom_pool):
        from weft.loom_query import get_ready_tasks

        pid = await _create_project(loom_pool, name="My Project")
        await _create_task(
            loom_pool, pid, task_id="t-1", title="Ready Task",
            status="pending",
        )

        result = await get_ready_tasks(loom_pool)
        assert len(result) == 1
        assert result[0]["title"] == "Ready Task"
        assert result[0]["priority"] == "p1"
        assert "project_name" in result[0]

    @pytest.mark.asyncio
    async def test_excludes_done_tasks(self, loom_pool):
        from weft.loom_query import get_ready_tasks

        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, status="done")

        result = await get_ready_tasks(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_contract_matches_daily_brief(self, loom_pool):
        """get_ready_tasks returns dicts with 'title' and 'priority' keys,
        matching the shape daily_brief.py expects from the old CLI JSON."""
        from weft.loom_query import get_ready_tasks

        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, title="Test", status="pending")

        result = await get_ready_tasks(loom_pool)
        assert len(result) >= 1
        row = result[0]
        assert "title" in row
        assert "priority" in row

    @pytest.mark.asyncio
    async def test_respects_limit(self, loom_pool):
        from weft.loom_query import get_ready_tasks

        pid = await _create_project(loom_pool)
        for i in range(5):
            await _create_task(
                loom_pool, pid, task_id=f"t-{i}", status="pending",
            )

        result = await get_ready_tasks(loom_pool, limit=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Architectural boundary enforcement
# ---------------------------------------------------------------------------


class TestArchitecturalBoundary:
    """After refactor, loom_alerts.py and health_check.py should not contain
    raw SQL touching Loom tables."""

    def test_loom_alerts_no_raw_sql(self):
        import weft.loom_alerts as mod

        source = inspect.getsource(mod)
        assert "FROM tasks" not in source, (
            "loom_alerts.py should not contain raw SQL — use loom_query instead"
        )

    def test_health_check_no_raw_sql(self):
        import weft.health_check as mod

        source = inspect.getsource(mod)
        assert "FROM tasks" not in source, (
            "health_check.py should not contain raw SQL — use loom_query instead"
        )

    def test_daily_brief_no_subprocess(self):
        import weft.daily_brief as mod

        source = inspect.getsource(mod)
        assert "subprocess" not in source, (
            "daily_brief.py should not use subprocess — use loom_query instead"
        )
