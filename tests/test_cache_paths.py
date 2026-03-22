"""Tests for the date parser module (weft/date_parser.py).

Covers relative dates, absolute dates, timezone handling, and edge cases.
All tests use a fixed reference_time for determinism.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from weft.date_parser import parse_dates

# Monday 2025-01-13 12:00 UTC — a known weekday anchor for deterministic tests
REFERENCE_TIME = datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc)
ET = ZoneInfo("America/New_York")


class TestRelativeDates:
    def test_tomorrow(self):
        result = parse_dates("tomorrow", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].date().isoformat() == "2025-01-14"

    def test_today(self):
        result = parse_dates("today", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].date().isoformat() == "2025-01-13"

    def test_next_week(self):
        result = parse_dates("next week", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].date().isoformat() == "2025-01-20"

    @pytest.mark.parametrize("text,expected_weekday", [
        ("Saturday", 5),
        ("Friday", 4),
        ("Tuesday", 1),
    ])
    def test_bare_weekday(self, text, expected_weekday):
        """A bare weekday name returns the *next* occurrence."""
        result = parse_dates(text, reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].weekday() == expected_weekday
        # Must be in the future relative to reference
        assert result[0] > REFERENCE_TIME

    def test_saturday_on_saturday_returns_next_week(self):
        """When reference is Saturday, 'Saturday' should return NEXT Saturday."""
        saturday_ref = datetime(2025, 1, 18, 12, 0, 0, tzinfo=timezone.utc)
        assert saturday_ref.weekday() == 5  # confirm it's Saturday
        result = parse_dates("Saturday", reference_time=saturday_ref)
        assert len(result) == 1
        # Should be 7 days later, not today
        assert result[0].date().isoformat() == "2025-01-25"

    def test_next_tuesday_on_tuesday(self):
        """'next Tuesday' on a Tuesday should go forward a full week."""
        tuesday_ref = datetime(2025, 1, 14, 12, 0, 0, tzinfo=timezone.utc)
        assert tuesday_ref.weekday() == 1  # confirm Tuesday
        result = parse_dates("next Tuesday", reference_time=tuesday_ref)
        assert len(result) == 1
        assert result[0].date().isoformat() == "2025-01-21"

    @pytest.mark.parametrize("text,expected_date", [
        ("in 2 days", "2025-01-15"),
        ("in 1 week", "2025-01-20"),
        ("in 2 weeks", "2025-01-27"),
        ("in 3 months", "2025-04-13"),
    ])
    def test_in_delta(self, text, expected_date):
        result = parse_dates(text, reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].date().isoformat() == expected_date


class TestAbsoluteDates:
    def test_iso_format(self):
        result = parse_dates("2025-03-01", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].month == 3
        assert result[0].day == 1

    def test_month_day_format(self):
        result = parse_dates("Jan 15", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].month == 1
        assert result[0].day == 15

    def test_month_day_slash(self):
        result = parse_dates("3/15", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].month == 3
        assert result[0].day == 15


class TestTimezoneHandling:
    def test_results_are_timezone_aware(self):
        result = parse_dates("tomorrow", reference_time=REFERENCE_TIME)
        assert len(result) == 1
        assert result[0].tzinfo is not None

    def test_tz_name_parameter(self):
        result = parse_dates(
            "tomorrow",
            reference_time=REFERENCE_TIME,
            tz_name="America/New_York",
        )
        assert len(result) == 1
        assert result[0].tzinfo is not None

    def test_naive_reference_gets_tz(self):
        naive_ref = datetime(2025, 1, 13, 12, 0, 0)
        result = parse_dates("tomorrow", reference_time=naive_ref, tz_name="UTC")
        assert len(result) == 1
        assert result[0].tzinfo is not None

    def test_invalid_tz_falls_back_to_utc(self):
        """Invalid timezone should fall back to UTC, not crash."""
        result = parse_dates(
            "tomorrow",
            reference_time=REFERENCE_TIME,
            tz_name="Invalid/Zone",
        )
        assert len(result) == 1


class TestEdgeCases:
    def test_empty_string(self):
        assert parse_dates("", reference_time=REFERENCE_TIME) == []

    def test_whitespace_only(self):
        assert parse_dates("   ", reference_time=REFERENCE_TIME) == []

    def test_no_date_references(self):
        result = parse_dates("Buy some milk", reference_time=REFERENCE_TIME)
        assert result == []

    def test_none_input_returns_empty(self):
        assert parse_dates(None, reference_time=REFERENCE_TIME) == []

    def test_all_results_set_time_to_nine_am(self):
        """Relative dates default to 9:00 AM."""
        result = parse_dates("tomorrow", reference_time=REFERENCE_TIME)
        assert result[0].hour == 9
        assert result[0].minute == 0
