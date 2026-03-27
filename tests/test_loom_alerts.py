"""Tests for Loom task awareness alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.loom_alerts import (
    LoomAlertConfig,
    check_blocked_pile_up,
    check_epic_completion,
    check_stale_claims,
    evaluate_loom_alerts,
)

# Use config defaults as test constants
_cfg = LoomAlertConfig()
_BLOCKED_PILE_UP_THRESHOLD = _cfg.blocked_pile_up_threshold
_STALE_CLAIM_HOURS = _cfg.stale_claim_hours
from weft.models import AlertType

# Minimal Loom schema for tests (only what the queries need)
_LOOM_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    priority TEXT NOT NULL DEFAULT 'p1',
    assignee TEXT,
    parent_id TEXT REFERENCES tasks(id),
    context JSONB NOT NULL DEFAULT '{}',
    output JSONB NOT NULL DEFAULT '{}',
    done_when TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    done_at TIMESTAMPTZ,
    claim_expires_at TIMESTAMPTZ,
    retry_count INT NOT NULL DEFAULT 0,
    last_failed_at TIMESTAMPTZ
);
"""


@pytest.fixture
async def loom_pool(pool):
    """Pool with Loom schema tables created."""
    await pool.execute(_LOOM_SCHEMA)
    yield pool
    # Clean up Loom tables after test
    await pool.execute("DROP TABLE IF EXISTS tasks CASCADE")
    await pool.execute("DROP TABLE IF EXISTS projects CASCADE")


async def _create_project(pool, name: str = "test-project") -> str:
    """Create a Loom project and return its UUID."""
    row = await pool.fetchrow(
        "INSERT INTO projects (name) VALUES ($1) RETURNING id",
        name,
    )
    return str(row["id"])


async def _create_task(
    pool,
    project_id: str,
    *,
    task_id: str = "loom-test-1",
    title: str = "Test task",
    status: str = "pending",
    assignee: str | None = None,
    parent_id: str | None = None,
    claimed_at: datetime | None = None,
    claim_expires_at: datetime | None = None,
) -> str:
    await pool.execute(
        """
        INSERT INTO tasks (id, project_id, title, status, assignee, parent_id, claimed_at, claim_expires_at)
        VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8)
        """,
        task_id,
        project_id,
        title,
        status,
        assignee,
        parent_id,
        claimed_at,
        claim_expires_at,
    )
    return task_id


# --- Stale Claims ---


class TestStaleClaimsAlert:
    @pytest.mark.asyncio
    async def test_fires_on_stale_claim(self, loom_pool):
        pid = await _create_project(loom_pool)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=_STALE_CLAIM_HOURS + 1)
        await _create_task(
            loom_pool,
            pid,
            status="claimed",
            assignee="agent-1",
            claimed_at=stale_time,
        )
        result = await check_stale_claims(loom_pool)
        assert len(result) == 1
        assert result[0]["alert_type"] == AlertType.loom_stale_claim.value

    @pytest.mark.asyncio
    async def test_no_alert_for_recent_claim(self, loom_pool):
        pid = await _create_project(loom_pool)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await _create_task(
            loom_pool,
            pid,
            status="claimed",
            assignee="agent-1",
            claimed_at=recent_time,
        )
        result = await check_stale_claims(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_alert_for_done_task(self, loom_pool):
        pid = await _create_project(loom_pool)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=_STALE_CLAIM_HOURS + 1)
        await _create_task(
            loom_pool,
            pid,
            status="done",
            claimed_at=stale_time,
        )
        result = await check_stale_claims(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_dedup_suppresses_duplicate(self, loom_pool):
        pid = await _create_project(loom_pool)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=_STALE_CLAIM_HOURS + 1)
        await _create_task(
            loom_pool,
            pid,
            status="claimed",
            assignee="agent-1",
            claimed_at=stale_time,
        )
        result1 = await check_stale_claims(loom_pool)
        assert len(result1) == 1
        result2 = await check_stale_claims(loom_pool)
        assert len(result2) == 0


# --- Epic Completion ---


class TestEpicCompletionAlert:
    @pytest.mark.asyncio
    async def test_fires_when_all_children_done(self, loom_pool):
        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", title="Epic", status="pending")
        await _create_task(
            loom_pool, pid, task_id="child-1", title="Child 1",
            status="done", parent_id="epic-1",
        )
        await _create_task(
            loom_pool, pid, task_id="child-2", title="Child 2",
            status="done", parent_id="epic-1",
        )
        result = await check_epic_completion(loom_pool)
        assert len(result) == 1
        assert result[0]["alert_type"] == AlertType.loom_epic_ready.value

    @pytest.mark.asyncio
    async def test_no_alert_when_children_pending(self, loom_pool):
        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", title="Epic", status="pending")
        await _create_task(
            loom_pool, pid, task_id="child-1", title="Child 1",
            status="done", parent_id="epic-1",
        )
        await _create_task(
            loom_pool, pid, task_id="child-2", title="Child 2",
            status="pending", parent_id="epic-1",
        )
        result = await check_epic_completion(loom_pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_cancelled_children_count_as_done(self, loom_pool):
        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", title="Epic", status="pending")
        await _create_task(
            loom_pool, pid, task_id="child-1", title="Child 1",
            status="done", parent_id="epic-1",
        )
        await _create_task(
            loom_pool, pid, task_id="child-2", title="Child 2",
            status="cancelled", parent_id="epic-1",
        )
        result = await check_epic_completion(loom_pool)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_alert_when_epic_already_done(self, loom_pool):
        pid = await _create_project(loom_pool)
        await _create_task(loom_pool, pid, task_id="epic-1", title="Epic", status="done")
        await _create_task(
            loom_pool, pid, task_id="child-1", title="Child 1",
            status="done", parent_id="epic-1",
        )
        result = await check_epic_completion(loom_pool)
        assert len(result) == 0


# --- Blocked Pile-Up ---


class TestBlockedPileUpAlert:
    @pytest.mark.asyncio
    async def test_fires_on_threshold(self, loom_pool):
        pid = await _create_project(loom_pool)
        for i in range(_BLOCKED_PILE_UP_THRESHOLD):
            await _create_task(
                loom_pool, pid, task_id=f"blocked-{i}",
                title=f"Blocked {i}", status="blocked",
            )
        result = await check_blocked_pile_up(loom_pool)
        assert len(result) == 1
        assert result[0]["alert_type"] == AlertType.loom_blocked_pile_up.value

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, loom_pool):
        pid = await _create_project(loom_pool)
        for i in range(_BLOCKED_PILE_UP_THRESHOLD - 1):
            await _create_task(
                loom_pool, pid, task_id=f"blocked-{i}",
                title=f"Blocked {i}", status="blocked",
            )
        result = await check_blocked_pile_up(loom_pool)
        assert len(result) == 0


# --- evaluate_loom_alerts (integration) ---


class TestEvaluateLoomAlerts:
    @pytest.mark.asyncio
    async def test_no_loom_tables(self, pool):
        """Should gracefully skip when Loom tables don't exist."""
        result = await evaluate_loom_alerts(pool)
        assert result == []

    @pytest.mark.asyncio
    async def test_combined_check(self, loom_pool):
        """All checks run and results are combined."""
        pid = await _create_project(loom_pool)
        # Stale claim
        stale_time = datetime.now(timezone.utc) - timedelta(hours=_STALE_CLAIM_HOURS + 1)
        await _create_task(
            loom_pool, pid, task_id="stale-1",
            status="claimed", assignee="agent", claimed_at=stale_time,
        )
        # Epic ready to close
        await _create_task(loom_pool, pid, task_id="epic-1", title="Epic", status="pending")
        await _create_task(
            loom_pool, pid, task_id="child-1", title="Child",
            status="done", parent_id="epic-1",
        )
        result = await evaluate_loom_alerts(loom_pool)
        types = {a["alert_type"] for a in result}
        assert AlertType.loom_stale_claim.value in types
        assert AlertType.loom_epic_ready.value in types
