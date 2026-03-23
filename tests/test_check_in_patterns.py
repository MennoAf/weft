"""Tests for check-in pattern detection (pure Python, no DB needed)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from weft.check_in_patterns import (
    _MIN_CORRELATION_POINTS,
    _MIN_STREAK_LENGTH,
    analyze_all,
    day_of_week_stats,
    detect_streaks,
    rolling_averages,
    sleep_energy_correlation,
    sleep_mood_correlation,
    trend_direction,
)
from weft.models import CheckIn


def _ci(
    *,
    mood: int | None = 3,
    energy: int | None = 3,
    sleep: float | None = 7.0,
    days_ago: int = 0,
    logged_at: datetime | None = None,
) -> CheckIn:
    """Helper to build a CheckIn with sensible defaults."""
    ts = logged_at or (datetime.now() - timedelta(days=days_ago))
    return CheckIn(
        id=f"test-{days_ago}-{id(ts)}",
        user_id="test-user",
        mood=mood,
        sleep_hours=sleep,
        energy=energy,
        logged_at=ts,
        created_at=ts,
    )


# --- day_of_week_stats ---


class TestDayOfWeekStats:
    def test_empty(self):
        result = day_of_week_stats([])
        assert result["best_day"] is None
        assert result["worst_day"] is None
        assert all(d["count"] == 0 for d in result["days"].values())

    def test_single_entry(self):
        ci = _ci(mood=4, energy=5, sleep=8.0, days_ago=0)
        result = day_of_week_stats([ci])
        day_name = ci.logged_at.strftime("%A")
        assert result["days"][day_name]["count"] == 1
        assert result["days"][day_name]["avg_mood"] == 4.0
        assert result["best_day"] == day_name
        assert result["worst_day"] == day_name

    def test_multiple_days(self):
        # Create entries on known days
        monday = datetime(2026, 3, 16, 12, 0)  # Monday
        tuesday = datetime(2026, 3, 17, 12, 0)  # Tuesday
        cis = [
            _ci(mood=5, energy=5, sleep=9.0, logged_at=monday),
            _ci(mood=2, energy=2, sleep=5.0, logged_at=tuesday),
        ]
        result = day_of_week_stats(cis)
        assert result["days"]["Monday"]["avg_mood"] == 5.0
        assert result["days"]["Tuesday"]["avg_mood"] == 2.0
        assert result["best_day"] == "Monday"
        assert result["worst_day"] == "Tuesday"

    def test_null_fields_excluded(self):
        ci = _ci(mood=None, energy=None, sleep=None, days_ago=0)
        result = day_of_week_stats([ci])
        day_name = ci.logged_at.strftime("%A")
        assert result["days"][day_name]["avg_mood"] is None
        assert result["days"][day_name]["count"] == 1


# --- sleep_energy_correlation ---


class TestSleepEnergyCorrelation:
    def test_insufficient_data(self):
        cis = [_ci(sleep=7.0, energy=3, days_ago=i) for i in range(_MIN_CORRELATION_POINTS - 1)]
        result = sleep_energy_correlation(cis)
        assert result["r"] is None
        assert "insufficient" in result["interpretation"]

    def test_empty(self):
        result = sleep_energy_correlation([])
        assert result["r"] is None
        assert result["n"] == 0

    def test_perfect_positive(self):
        # Energy tracks sleep perfectly: sleep 5,6,7,8,9 -> energy 1,2,3,4,5
        cis = [
            _ci(sleep=5.0 + i, energy=1 + i, days_ago=i) for i in range(5)
        ]
        result = sleep_energy_correlation(cis)
        assert result["r"] is not None
        assert result["r"] > 0.99
        assert "strong" in result["interpretation"]

    def test_negative_correlation(self):
        cis = [
            _ci(sleep=5.0 + i, energy=5 - i, days_ago=i) for i in range(5)
        ]
        result = sleep_energy_correlation(cis)
        assert result["r"] is not None
        assert result["r"] < -0.99
        assert "negative" in result["interpretation"]

    def test_no_variance(self):
        cis = [_ci(sleep=7.0, energy=3, days_ago=i) for i in range(5)]
        result = sleep_energy_correlation(cis)
        assert result["r"] is None
        assert "no variance" in result["interpretation"]

    def test_null_fields_excluded(self):
        cis = [
            _ci(sleep=7.0, energy=3, days_ago=0),
            _ci(sleep=None, energy=4, days_ago=1),
            _ci(sleep=8.0, energy=None, days_ago=2),
        ]
        result = sleep_energy_correlation(cis)
        assert result["n"] == 1  # Only one complete pair


# --- sleep_mood_correlation ---


class TestSleepMoodCorrelation:
    def test_insufficient_data(self):
        result = sleep_mood_correlation([_ci(days_ago=0)])
        assert result["r"] is None

    def test_positive_correlation(self):
        cis = [
            _ci(sleep=5.0 + i, mood=1 + i, days_ago=i) for i in range(5)
        ]
        result = sleep_mood_correlation(cis)
        assert result["r"] is not None
        assert result["r"] > 0.99


# --- detect_streaks ---


class TestDetectStreaks:
    def test_empty(self):
        result = detect_streaks([])
        assert result["logging_streak"] == 0
        assert result["good_mood_streaks"] == []
        assert result["low_mood_streaks"] == []

    def test_single_day(self):
        result = detect_streaks([_ci(days_ago=0)])
        assert result["logging_streak"] == 1

    def test_consecutive_days(self):
        cis = [_ci(days_ago=i) for i in range(5)]
        result = detect_streaks(cis)
        assert result["logging_streak"] == 5

    def test_gap_breaks_streak(self):
        # Today, yesterday, 3 days ago (gap on day 2)
        cis = [_ci(days_ago=0), _ci(days_ago=1), _ci(days_ago=3)]
        result = detect_streaks(cis)
        assert result["logging_streak"] == 2

    def test_good_mood_streak(self):
        cis = [_ci(mood=4, days_ago=i) for i in range(_MIN_STREAK_LENGTH)]
        result = detect_streaks(cis)
        assert len(result["good_mood_streaks"]) == 1
        assert result["good_mood_streaks"][0]["length"] == _MIN_STREAK_LENGTH

    def test_low_mood_streak(self):
        cis = [_ci(mood=2, days_ago=i) for i in range(_MIN_STREAK_LENGTH)]
        result = detect_streaks(cis)
        assert len(result["low_mood_streaks"]) == 1
        assert result["low_mood_streaks"][0]["length"] == _MIN_STREAK_LENGTH

    def test_two_day_streak_not_reported(self):
        """A 2-day streak should NOT appear since minimum is 3."""
        cis = [_ci(mood=4, days_ago=0), _ci(mood=4, days_ago=1)]
        result = detect_streaks(cis)
        assert result["good_mood_streaks"] == []

    def test_duplicate_same_day(self):
        """Multiple check-ins on the same day count as one day."""
        base = datetime(2026, 3, 20, 12, 0)
        cis = [
            _ci(mood=4, logged_at=base),
            _ci(mood=4, logged_at=base + timedelta(hours=3)),
            _ci(mood=4, logged_at=base - timedelta(days=1)),
            _ci(mood=4, logged_at=base - timedelta(days=2)),
        ]
        result = detect_streaks(cis)
        assert result["logging_streak"] == 3
        assert len(result["good_mood_streaks"]) == 1
        assert result["good_mood_streaks"][0]["length"] == 3


# --- rolling_averages ---


class TestRollingAverages:
    def test_empty(self):
        result = rolling_averages([])
        assert result["series"] == []

    def test_single_point(self):
        result = rolling_averages([_ci(mood=4, days_ago=0)])
        assert len(result["series"]) == 1
        assert result["series"][0]["avg_mood"] == 4.0

    def test_window_averaging(self):
        """7-day window should average across the window."""
        cis = [_ci(mood=2, days_ago=0), _ci(mood=4, days_ago=3)]
        result = rolling_averages(cis, window_days=7, days=30)
        # The most recent point's window includes both check-ins
        last = result["series"][-1]
        assert last["avg_mood"] == 3.0  # (2 + 4) / 2

    def test_respects_period(self):
        """Only check-ins within the period should be included."""
        cis = [_ci(mood=5, days_ago=0), _ci(mood=1, days_ago=60)]
        result = rolling_averages(cis, days=30)
        # The 60-day-old entry should be excluded
        moods = [s["avg_mood"] for s in result["series"] if s["avg_mood"] is not None]
        assert all(m == 5.0 for m in moods)


# --- trend_direction ---


class TestTrendDirection:
    def test_empty(self):
        result = trend_direction([])
        assert result["mood"] is None
        assert result["energy"] is None
        assert result["sleep"] is None

    def test_insufficient_data(self):
        cis = [_ci(days_ago=i) for i in range(3)]
        result = trend_direction(cis)
        assert result["mood"] is None  # < 5 points

    def test_flat_trend(self):
        cis = [_ci(mood=3, energy=3, sleep=7.0, days_ago=i) for i in range(10)]
        result = trend_direction(cis)
        assert result["mood"]["direction"] == "flat"
        assert result["energy"]["direction"] == "flat"

    def test_upward_trend(self):
        # Mood increasing over 10 days: 1,1,2,2,3,3,4,4,5,5
        cis = [
            _ci(mood=min(5, 1 + i // 2), days_ago=9 - i) for i in range(10)
        ]
        result = trend_direction(cis)
        assert result["mood"]["direction"] == "up"
        assert result["mood"]["slope"] > 0

    def test_downward_trend(self):
        cis = [
            _ci(mood=max(1, 5 - i // 2), days_ago=9 - i) for i in range(10)
        ]
        result = trend_direction(cis)
        assert result["mood"]["direction"] == "down"
        assert result["mood"]["slope"] < 0


# --- analyze_all ---


class TestAnalyzeAll:
    def test_empty(self):
        result = analyze_all([])
        assert result["total_check_ins"] == 0
        assert result["day_of_week"]["best_day"] is None
        assert result["streaks"]["logging_streak"] == 0

    def test_full_report_structure(self):
        cis = [_ci(mood=3 + (i % 3), energy=2 + (i % 3), sleep=6.0 + i * 0.5, days_ago=i) for i in range(10)]
        result = analyze_all(cis)
        assert "day_of_week" in result
        assert "sleep_energy_correlation" in result
        assert "sleep_mood_correlation" in result
        assert "streaks" in result
        assert "rolling_averages" in result
        assert "trends" in result
        assert result["total_check_ins"] == 10

    def test_custom_windows(self):
        cis = [_ci(days_ago=i) for i in range(10)]
        result = analyze_all(cis, trend_days=30, rolling_days=14, rolling_window=3)
        assert result["trends"]["period_days"] == 30
        assert result["rolling_averages"]["period_days"] == 14
        assert result["rolling_averages"]["window_days"] == 3
