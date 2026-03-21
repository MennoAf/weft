"""Tests for the recurring Slack sync loop in weft/scheduler.py."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.scheduler import _MIN_SYNC_INTERVAL, slack_sync_loop


@dataclass
class _FakeSyncResult:
    channels_synced: int = 2
    messages_synced: int = 10
    memories_created: int = 8


@pytest.fixture
def mock_pool():
    return MagicMock()


def _cancel_after(n_sleeps: int):
    """Return an async side_effect for asyncio.sleep that cancels after n calls."""
    call_count = 0

    async def _side_effect(seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= n_sleeps:
            raise asyncio.CancelledError()

    return _side_effect


class TestSlackSyncLoop:
    @pytest.mark.asyncio
    async def test_happy_path(self, mock_pool):
        mock_sync = AsyncMock(return_value=_FakeSyncResult())

        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}),
            patch("weft.scheduler.sync_slack_sdk", mock_sync, create=True),
            patch("asyncio.sleep", new_callable=AsyncMock, side_effect=_cancel_after(2)),
        ):
            # Import inside patch context so the lazy import picks up the mock
            with patch("weft.slack.sync.sync_slack_sdk", mock_sync):
                with pytest.raises(asyncio.CancelledError):
                    await slack_sync_loop(mock_pool, interval=300)

        assert mock_sync.call_count == 2
        # Both calls should use the pool and token
        for call in mock_sync.call_args_list:
            assert call[0][0] is mock_pool
            assert call[0][1] == "xoxb-test"

    @pytest.mark.asyncio
    async def test_exception_resilience(self, mock_pool, caplog):
        """Loop survives sync errors and retries next cycle."""
        mock_sync = AsyncMock(
            side_effect=[Exception("boom"), _FakeSyncResult()]
        )

        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}),
            patch("weft.slack.sync.sync_slack_sdk", mock_sync),
            patch("asyncio.sleep", new_callable=AsyncMock, side_effect=_cancel_after(2)),
            caplog.at_level(logging.ERROR),
        ):
            with pytest.raises(asyncio.CancelledError):
                await slack_sync_loop(mock_pool, interval=300)

        assert mock_sync.call_count == 2
        assert any("slack_sync.error" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_token_exits_immediately(self, mock_pool, caplog):
        """Loop exits gracefully when no SLACK_BOT_TOKEN is set."""
        mock_sync = AsyncMock()

        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.environ.get", return_value=""),
            patch("weft.slack.sync.sync_slack_sdk", mock_sync),
            caplog.at_level(logging.WARNING),
        ):
            # Ensure SLACK_BOT_TOKEN is not set
            import os
            old = os.environ.pop("SLACK_BOT_TOKEN", None)
            try:
                await slack_sync_loop(mock_pool, interval=300)
            finally:
                if old is not None:
                    os.environ["SLACK_BOT_TOKEN"] = old

        mock_sync.assert_not_called()
        assert any("no_token" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_immediate_first_call(self, mock_pool):
        """sync_slack_sdk is called before the first sleep."""
        call_order = []

        mock_sync = AsyncMock(
            return_value=_FakeSyncResult(),
            side_effect=lambda *a, **kw: call_order.append("sync"),
        )

        async def mock_sleep(seconds):
            call_order.append("sleep")
            raise asyncio.CancelledError()

        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}),
            patch("weft.slack.sync.sync_slack_sdk", mock_sync),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await slack_sync_loop(mock_pool, interval=300)

        assert call_order == ["sync", "sleep"]

    @pytest.mark.asyncio
    async def test_minimum_interval_floor(self, mock_pool):
        """Interval is floored at _MIN_SYNC_INTERVAL to prevent API hammering."""
        sleep_intervals = []

        async def capture_sleep(seconds):
            sleep_intervals.append(seconds)
            raise asyncio.CancelledError()

        mock_sync = AsyncMock(return_value=_FakeSyncResult())

        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}),
            patch("weft.slack.sync.sync_slack_sdk", mock_sync),
            patch("asyncio.sleep", side_effect=capture_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await slack_sync_loop(mock_pool, interval=5)  # way too low

        assert sleep_intervals[0] >= _MIN_SYNC_INTERVAL

    @pytest.mark.asyncio
    @pytest.mark.parametrize("interval", [60, 300, 1800])
    async def test_sleep_uses_configured_interval(self, mock_pool, interval):
        sleep_intervals = []

        async def capture_sleep(seconds):
            sleep_intervals.append(seconds)
            raise asyncio.CancelledError()

        mock_sync = AsyncMock(return_value=_FakeSyncResult())

        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}),
            patch("weft.slack.sync.sync_slack_sdk", mock_sync),
            patch("asyncio.sleep", side_effect=capture_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await slack_sync_loop(mock_pool, interval=interval)

        assert sleep_intervals[0] == interval
