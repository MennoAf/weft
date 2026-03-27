"""Tests for check-in pattern alert evaluation (requires DB for alert CRUD)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.alerts import create_alert, list_alerts
from weft.check_in_patterns import (
    CheckInAlertConfig,
    evaluate_check_in_alerts,
)

# Use config defaults as test constants
_cfg = CheckInAlertConfig()
_ALERT_DEDUP_HOURS = _cfg.dedup_hours
_ALERT_LOW_MOOD_STREAK = _cfg.low_mood_streak
_ALERT_LOW_SLEEP_DAYS = _cfg.low_sleep_days
_ALERT_LOW_SLEEP_HOURS = _cfg.low_sleep_hours
from weft.models import AlertCreate, AlertStatus, AlertType, CheckIn


def _ci(
    *,
    mood: int | None = 3,
    energy: int | None = 3,
    sleep: float | None = 7.0,
    days_ago: int = 0,
) -> CheckIn:
    """Helper to build a CheckIn with sensible defaults."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return CheckIn(
        id=f"test-{days_ago}-{id(ts)}",
        user_id="test-user",
        mood=mood,
        sleep_hours=sleep,
        energy=energy,
        logged_at=ts,
        created_at=ts,
    )


class TestLowMoodAlert:
    @pytest.mark.asyncio
    async def test_fires_on_low_mood_streak(self, pool):
        """Alert fires when mood <= 2 for >= 3 consecutive days."""
        cis = [_ci(mood=2, days_ago=i) for i in range(_ALERT_LOW_MOOD_STREAK)]
        result = await evaluate_check_in_alerts(pool, cis)
        low_mood = [a for a in result if a["alert_type"] == AlertType.check_in_low_mood.value]
        assert len(low_mood) == 1
        assert str(_ALERT_LOW_MOOD_STREAK) in low_mood[0]["title"]

    @pytest.mark.asyncio
    async def test_no_alert_for_short_streak(self, pool):
        """No alert when streak is shorter than threshold."""
        cis = [_ci(mood=2, days_ago=i) for i in range(_ALERT_LOW_MOOD_STREAK - 1)]
        result = await evaluate_check_in_alerts(pool, cis)
        low_mood = [a for a in result if a["alert_type"] == AlertType.check_in_low_mood.value]
        assert len(low_mood) == 0

    @pytest.mark.asyncio
    async def test_no_alert_when_mood_ok(self, pool):
        """No alert when mood is above threshold."""
        cis = [_ci(mood=4, days_ago=i) for i in range(5)]
        result = await evaluate_check_in_alerts(pool, cis)
        low_mood = [a for a in result if a["alert_type"] == AlertType.check_in_low_mood.value]
        assert len(low_mood) == 0


class TestLowSleepAlert:
    @pytest.mark.asyncio
    async def test_fires_on_low_sleep(self, pool):
        """Alert fires when average sleep < threshold over N+ entries."""
        cis = [
            _ci(sleep=5.0, days_ago=i) for i in range(_ALERT_LOW_SLEEP_DAYS)
        ]
        result = await evaluate_check_in_alerts(pool, cis)
        low_sleep = [a for a in result if a["alert_type"] == AlertType.check_in_low_sleep.value]
        assert len(low_sleep) == 1
        assert "5.0h" in low_sleep[0]["title"]

    @pytest.mark.asyncio
    async def test_no_alert_when_sleep_ok(self, pool):
        """No alert when sleep is above threshold."""
        cis = [_ci(sleep=8.0, days_ago=i) for i in range(_ALERT_LOW_SLEEP_DAYS)]
        result = await evaluate_check_in_alerts(pool, cis)
        low_sleep = [a for a in result if a["alert_type"] == AlertType.check_in_low_sleep.value]
        assert len(low_sleep) == 0

    @pytest.mark.asyncio
    async def test_no_alert_insufficient_data(self, pool):
        """No alert when fewer entries than threshold."""
        cis = [_ci(sleep=4.0, days_ago=i) for i in range(_ALERT_LOW_SLEEP_DAYS - 1)]
        result = await evaluate_check_in_alerts(pool, cis)
        low_sleep = [a for a in result if a["alert_type"] == AlertType.check_in_low_sleep.value]
        assert len(low_sleep) == 0


class TestDecliningTrendAlert:
    @pytest.mark.asyncio
    async def test_fires_on_declining_mood(self, pool):
        """Alert fires when mood trend is 'down'."""
        # 10 days of declining mood: 5,5,4,4,3,3,2,2,1,1
        cis = [
            _ci(mood=max(1, 5 - i // 2), days_ago=9 - i) for i in range(10)
        ]
        result = await evaluate_check_in_alerts(pool, cis)
        declining = [a for a in result if a["alert_type"] == AlertType.check_in_declining_trend.value]
        assert len(declining) == 1

    @pytest.mark.asyncio
    async def test_no_alert_on_flat_mood(self, pool):
        """No alert when mood is flat."""
        cis = [_ci(mood=3, days_ago=i) for i in range(10)]
        result = await evaluate_check_in_alerts(pool, cis)
        declining = [a for a in result if a["alert_type"] == AlertType.check_in_declining_trend.value]
        assert len(declining) == 0


class TestDedup:
    @pytest.mark.asyncio
    async def test_dedup_suppresses_duplicate(self, pool):
        """Second evaluation within 24h should not create duplicate alerts."""
        cis = [_ci(mood=2, days_ago=i) for i in range(_ALERT_LOW_MOOD_STREAK)]

        # First evaluation: should create
        result1 = await evaluate_check_in_alerts(pool, cis)
        low_mood1 = [a for a in result1 if a["alert_type"] == AlertType.check_in_low_mood.value]
        assert len(low_mood1) == 1

        # Second evaluation: should be deduped
        result2 = await evaluate_check_in_alerts(pool, cis)
        low_mood2 = [a for a in result2 if a["alert_type"] == AlertType.check_in_low_mood.value]
        assert len(low_mood2) == 0


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_empty_check_ins(self, pool):
        result = await evaluate_check_in_alerts(pool, [])
        assert result == []
