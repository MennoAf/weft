"""Tests for alert store CRUD, polling, and scheduler logic."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from weft.alerts import (
    create_alert,
    dismiss_alert,
    get_alert,
    list_alerts,
    mark_alert_fired,
    poll_due_alerts,
)
from weft.models import AlertChannel, AlertCreate, AlertStatus, AlertType
from weft.scheduler import (
    dispatch_alert,
    dispatch_log,
    scheduler_loop,
)


# ── Store CRUD tests (real DB) ──────────────────────────────────────


def _make_create(
    *,
    title: str = "Test alert",
    alert_type: AlertType = AlertType.custom,
    trigger_at: datetime | None = None,
    channel: AlertChannel = AlertChannel.log,
    channel_target: str | None = None,
    payload: dict | None = None,
    project_id: str | None = None,
) -> AlertCreate:
    return AlertCreate(
        alert_type=alert_type,
        title=title,
        trigger_at=trigger_at or datetime.now(timezone.utc),
        channel=channel,
        channel_target=channel_target,
        payload=payload or {},
        project_id=project_id,
    )


class TestCreateAndGetAlert:
    @pytest.mark.asyncio
    async def test_create_and_get(self, pool):
        ac = _make_create(title="Hello alert")
        alert = await create_alert(pool, ac)
        assert alert.title == "Hello alert"
        assert alert.status == AlertStatus.pending
        assert alert.id.startswith("weft-")

        fetched = await get_alert(pool, alert.id)
        assert fetched is not None
        assert fetched.title == "Hello alert"
        assert fetched.alert_type == AlertType.custom

    @pytest.mark.asyncio
    async def test_create_with_payload(self, pool):
        ac = _make_create(
            title="With payload",
            payload={"task_id": "loom-123", "priority": "high"},
        )
        alert = await create_alert(pool, ac)
        assert alert.payload == {"task_id": "loom-123", "priority": "high"}

    @pytest.mark.asyncio
    async def test_create_with_slack_channel(self, pool):
        ac = _make_create(
            title="Slack alert",
            channel=AlertChannel.slack,
            channel_target="#alerts",
        )
        alert = await create_alert(pool, ac)
        assert alert.channel == AlertChannel.slack
        assert alert.channel_target == "#alerts"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, pool):
        result = await get_alert(pool, "weft-nonexistent")
        assert result is None


class TestListAlerts:
    @pytest.mark.asyncio
    async def test_list_all(self, pool):
        for i in range(3):
            await create_alert(pool, _make_create(title=f"Alert {i}"))
        alerts = await list_alerts(pool)
        assert len(alerts) == 3

    @pytest.mark.asyncio
    async def test_list_by_status(self, pool):
        a1 = await create_alert(pool, _make_create(title="Pending"))
        a2 = await create_alert(pool, _make_create(title="To dismiss"))
        await dismiss_alert(pool, a2.id)

        pending = await list_alerts(pool, status=AlertStatus.pending)
        assert len(pending) == 1
        assert pending[0].id == a1.id

    @pytest.mark.asyncio
    async def test_list_with_limit_offset(self, pool):
        for i in range(5):
            await create_alert(pool, _make_create(title=f"Alert {i}"))
        page = await list_alerts(pool, limit=2, offset=2)
        assert len(page) == 2


class TestDismissAlert:
    @pytest.mark.asyncio
    async def test_dismiss(self, pool):
        alert = await create_alert(pool, _make_create(title="Dismiss me"))
        result = await dismiss_alert(pool, alert.id)
        assert result is True

        fetched = await get_alert(pool, alert.id)
        assert fetched.status == AlertStatus.dismissed

    @pytest.mark.asyncio
    async def test_dismiss_already_dismissed(self, pool):
        alert = await create_alert(pool, _make_create(title="Dismiss twice"))
        await dismiss_alert(pool, alert.id)
        result = await dismiss_alert(pool, alert.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_dismiss_nonexistent(self, pool):
        result = await dismiss_alert(pool, "weft-nonexistent")
        assert result is False


# ── Poll and mark tests (real DB) ───────────────────────────────────


class TestPollDueAlerts:
    @pytest.mark.asyncio
    async def test_poll_returns_due(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        await create_alert(pool, _make_create(title="Due", trigger_at=past))
        alerts = await poll_due_alerts(pool)
        assert len(alerts) == 1
        assert alerts[0].title == "Due"

    @pytest.mark.asyncio
    async def test_poll_skips_future(self, pool):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        await create_alert(pool, _make_create(title="Future", trigger_at=future))
        alerts = await poll_due_alerts(pool)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_poll_skips_fired(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        alert = await create_alert(pool, _make_create(title="Fired", trigger_at=past))
        await mark_alert_fired(pool, alert.id)
        alerts = await poll_due_alerts(pool)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_poll_skip_locked(self, pool):
        """Two concurrent polls should not return the same alert."""
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        await create_alert(pool, _make_create(title="Locked test", trigger_at=past))

        results = await asyncio.gather(
            poll_due_alerts(pool, batch_size=10),
            poll_due_alerts(pool, batch_size=10),
        )
        # Combined, exactly 1 alert across both results
        total = sum(len(r) for r in results)
        assert total == 1

    @pytest.mark.asyncio
    async def test_poll_reclaims_stale_processing_reservation(self, pool):
        """A scheduler crash must not strand an alert in processing forever."""
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        alert = await create_alert(pool, _make_create(title="Recover me", trigger_at=past))
        await pool.execute(
            "UPDATE alerts SET status = 'processing', processing_at = now() - interval '10 minutes' WHERE id = $1",
            alert.id,
        )

        alerts = await poll_due_alerts(pool)

        assert [item.id for item in alerts] == [alert.id]
        assert await pool.fetchval("SELECT status FROM alerts WHERE id = $1", alert.id) == "processing"

    @pytest.mark.asyncio
    async def test_poll_respects_batch_size(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        for i in range(5):
            await create_alert(pool, _make_create(title=f"Batch {i}", trigger_at=past))
        alerts = await poll_due_alerts(pool, batch_size=3)
        assert len(alerts) == 3


class TestMarkAlertFired:
    @pytest.mark.asyncio
    async def test_mark_fired(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        alert = await create_alert(pool, _make_create(title="Fire me", trigger_at=past))
        result = await mark_alert_fired(pool, alert.id)
        assert result is True

        fetched = await get_alert(pool, alert.id)
        assert fetched.status == AlertStatus.fired
        assert fetched.fired_at is not None
        assert fetched.fired_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_mark_fired_idempotent(self, pool):
        alert = await create_alert(pool, _make_create(title="Fire twice"))
        await mark_alert_fired(pool, alert.id)
        result = await mark_alert_fired(pool, alert.id)
        assert result is False


# ── Dispatch tests ──────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_log_channel(self, caplog):
        from weft.models import Alert

        alert = Alert(
            alert_type=AlertType.follow_up,
            title="Log test",
            trigger_at=datetime.now(timezone.utc),
        )
        with caplog.at_level(logging.INFO, logger="weft.scheduler"):
            await dispatch_log(alert)
        assert "alert.fired" in caplog.text

    @pytest.mark.asyncio
    async def test_dispatch_unknown_channel(self, caplog):
        """If the registry doesn't have a handler, log a warning and skip."""
        from weft.models import Alert
        from weft.scheduler import _DISPATCH_REGISTRY

        alert = Alert(
            alert_type=AlertType.custom,
            title="Unknown channel",
            trigger_at=datetime.now(timezone.utc),
            channel=AlertChannel.log,
        )
        # Temporarily remove the log handler to simulate unknown channel
        saved = _DISPATCH_REGISTRY.pop("log")
        try:
            with caplog.at_level(logging.WARNING, logger="weft.scheduler"):
                await dispatch_alert(alert)
            assert "unknown_channel" in caplog.text
        finally:
            _DISPATCH_REGISTRY["log"] = saved


# ── Scheduler loop tests (mocked) ──────────────────────────────────


class TestSchedulerLoop:
    @pytest.mark.asyncio
    async def test_fires_and_marks(self):
        from weft.models import Alert

        alert = Alert(
            alert_type=AlertType.due_task,
            title="Fire this",
            trigger_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        with (
            patch("weft.scheduler.poll_due_alerts", new_callable=AsyncMock, return_value=[alert]) as mock_poll,
            patch("weft.scheduler.dispatch_alert", new_callable=AsyncMock) as mock_dispatch,
            patch("weft.scheduler.mark_alert_fired", new_callable=AsyncMock) as mock_mark,
            patch("weft.scheduler.release_alert", new_callable=AsyncMock) as mock_release,
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await scheduler_loop(AsyncMock(), interval=1)

            mock_poll.assert_called_once()
            mock_dispatch.assert_called_once_with(alert)
            mock_mark.assert_called_once_with(mock_poll.call_args[0][0], alert.id)

    @pytest.mark.asyncio
    async def test_isolates_dispatch_error(self):
        from weft.models import Alert

        alert = Alert(
            alert_type=AlertType.custom,
            title="Dispatch fails",
            trigger_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        with (
            patch("weft.scheduler.poll_due_alerts", new_callable=AsyncMock, return_value=[alert]) as mock_poll,
            patch("weft.scheduler.dispatch_alert", new_callable=AsyncMock, side_effect=RuntimeError("boom")) as mock_dispatch,
            patch("weft.scheduler.mark_alert_fired", new_callable=AsyncMock) as mock_mark,
            patch("weft.scheduler.release_alert", new_callable=AsyncMock) as mock_release,
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await scheduler_loop(AsyncMock(), interval=1)

            # Failed dispatch must release the durable reservation for retry.
            mock_mark.assert_not_called()
            mock_release.assert_awaited_once_with(mock_poll.call_args.args[0], alert.id)

    @pytest.mark.asyncio
    async def test_poll_error_does_not_crash(self):
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        with (
            patch("weft.scheduler.poll_due_alerts", new_callable=AsyncMock, side_effect=RuntimeError("db down")),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await scheduler_loop(AsyncMock(), interval=1)
            # Loop survived the poll error and reached sleep
