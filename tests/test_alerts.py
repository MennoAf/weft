"""Tests for Alert system — DB schema, Pydantic models, and RLS."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weft.models import Alert, AlertChannel, AlertCreate, AlertStatus, AlertType


# ── Pydantic model tests (no DB needed) ──────────────────────────────


class TestAlertType:
    def test_values(self):
        assert AlertType.due_task.value == "due_task"
        assert AlertType.stale_decision.value == "stale_decision"
        assert AlertType.follow_up.value == "follow_up"
        assert AlertType.custom.value == "custom"


class TestAlertChannel:
    def test_values(self):
        assert AlertChannel.log.value == "log"
        assert AlertChannel.slack.value == "slack"


class TestAlertStatus:
    def test_values(self):
        assert AlertStatus.pending.value == "pending"
        assert AlertStatus.fired.value == "fired"
        assert AlertStatus.dismissed.value == "dismissed"


class TestAlertCreate:
    def test_minimal(self):
        ac = AlertCreate(
            alert_type=AlertType.custom,
            title="Test alert",
            trigger_at=datetime.now(timezone.utc),
        )
        assert ac.channel == AlertChannel.log
        assert ac.payload == {}
        assert ac.channel_target is None

    def test_with_slack(self):
        ac = AlertCreate(
            alert_type=AlertType.due_task,
            title="Task due",
            trigger_at=datetime.now(timezone.utc),
            channel=AlertChannel.slack,
            channel_target="#alerts",
            payload={"task_id": "loom-123"},
        )
        assert ac.channel == AlertChannel.slack
        assert ac.channel_target == "#alerts"

    def test_rejects_naive_datetime(self):
        with pytest.raises(Exception):
            AlertCreate(
                alert_type=AlertType.custom,
                title="Bad",
                trigger_at=datetime(2026, 1, 1),  # naive
            )


class TestAlertModel:
    def test_round_trip(self):
        now = datetime.now(timezone.utc)
        a = Alert(
            alert_type=AlertType.follow_up,
            title="Check back",
            trigger_at=now,
            user_id="user-1",
        )
        d = a.model_dump(mode="json")
        a2 = Alert.model_validate(d)
        assert a2.title == "Check back"
        assert a2.status == AlertStatus.pending

    def test_defaults(self):
        a = Alert(
            alert_type=AlertType.custom,
            title="Test",
            trigger_at=datetime.now(timezone.utc),
        )
        assert a.status == AlertStatus.pending
        assert a.fired_at is None
        assert a.channel == AlertChannel.log
        assert a.payload == {}

    def test_to_dict(self):
        a = Alert(
            alert_type=AlertType.due_task,
            title="Overdue",
            trigger_at=datetime.now(timezone.utc),
            user_id="user-1",
        )
        d = a.to_dict()
        assert d["alert_type"] == "due_task"
        assert d["status"] == "pending"
        assert d["channel"] == "log"


# ── DB migration tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alerts_table_exists(pool):
    exists = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'alerts')"
    )
    assert exists


@pytest.mark.asyncio
async def test_alerts_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'alerts'
        ORDER BY ordinal_position
        """
    )
    cols = {r["column_name"]: r["data_type"] for r in rows}
    assert "id" in cols
    assert "user_id" in cols
    assert "alert_type" in cols
    assert "title" in cols
    assert "body" in cols
    assert "trigger_at" in cols
    assert "status" in cols
    assert "channel" in cols
    assert "channel_target" in cols
    assert "payload" in cols
    assert "fired_at" in cols
    assert "created_at" in cols


@pytest.mark.asyncio
async def test_alerts_poll_index_exists(pool):
    """Partial index on (user_id, status, trigger_at) WHERE status='pending'."""
    idx = await pool.fetchval(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'alerts' AND indexname = 'idx_alerts_poll'"
    )
    assert idx is not None


@pytest.mark.asyncio
async def test_alerts_rls_enabled(pool):
    rls = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'alerts'"
    )
    assert rls is True


@pytest.mark.asyncio
async def test_alerts_rls_policies(pool):
    policies = await pool.fetch(
        "SELECT policyname FROM pg_policies WHERE tablename = 'alerts'"
    )
    names = {r["policyname"] for r in policies}
    assert "alerts_select" in names
    assert "alerts_insert" in names
    assert "alerts_update" in names
    assert "alerts_delete" in names


@pytest.mark.asyncio
async def test_alerts_insert_and_read(pool):
    """Basic insert/read round-trip."""
    await pool.execute(
        """
        INSERT INTO alerts (id, user_id, alert_type, title, trigger_at, status, channel, payload)
        VALUES ('a-1', NULL, 'custom', 'Test', now(), 'pending', 'log', '{}')
        """
    )
    row = await pool.fetchrow("SELECT * FROM alerts WHERE id = 'a-1'")
    assert row["title"] == "Test"
    assert row["status"] == "pending"
