"""Tests for daily brief config, alert type, and is_daily_brief_due scheduling."""

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from weft.alerts import is_daily_brief_due
from weft.config import DailyBriefConfig, SlackSyncConfig, load_config
from weft.models import AlertCreate, AlertChannel, AlertType


# --- Config defaults and env var overrides ---


class TestSlackSyncConfig:
    def test_defaults(self):
        cfg = SlackSyncConfig()
        assert cfg.interval == 1800

    def test_env_override(self):
        with patch.dict("os.environ", {"WEFT_SLACK_SYNC_INTERVAL": "600"}, clear=False):
            config = load_config()
        assert config.slack_sync.interval == 600


class TestDailyBriefConfig:
    def test_defaults(self):
        cfg = DailyBriefConfig()
        assert cfg.time == "08:00"
        assert cfg.timezone == "America/New_York"
        assert cfg.channel == ""

    def test_env_override_time(self):
        with patch.dict("os.environ", {"WEFT_DAILY_BRIEF_TIME": "09:30"}, clear=False):
            config = load_config()
        assert config.daily_brief.time == "09:30"

    def test_env_override_tz(self):
        with patch.dict("os.environ", {"WEFT_DAILY_BRIEF_TZ": "US/Pacific"}, clear=False):
            config = load_config()
        assert config.daily_brief.timezone == "US/Pacific"

    def test_env_override_channel(self):
        with patch.dict("os.environ", {"WEFT_DAILY_BRIEF_CHANNEL": "C12345"}, clear=False):
            config = load_config()
        assert config.daily_brief.channel == "C12345"

    def test_strips_whitespace(self):
        with patch.dict(
            "os.environ",
            {
                "WEFT_DAILY_BRIEF_TIME": "  07:00  ",
                "WEFT_DAILY_BRIEF_TZ": "  UTC  ",
                "WEFT_DAILY_BRIEF_CHANNEL": "  C999  ",
            },
            clear=False,
        ):
            config = load_config()
        assert config.daily_brief.time == "07:00"
        assert config.daily_brief.timezone == "UTC"
        assert config.daily_brief.channel == "C999"


# --- AlertType model ---


class TestAlertTypeDailyBrief:
    def test_daily_brief_member_exists(self):
        assert AlertType.daily_brief.value == "daily_brief"

    def test_alert_create_accepts_daily_brief(self):
        alert = AlertCreate(
            alert_type=AlertType.daily_brief,
            title="Morning Brief",
            trigger_at=datetime.now(timezone.utc),
            channel=AlertChannel.slack,
            channel_target="C12345",
        )
        assert alert.alert_type == AlertType.daily_brief

    def test_invalid_alert_type_rejected(self):
        with pytest.raises(ValueError):
            AlertCreate(
                alert_type="foobar",  # type: ignore[arg-type]
                title="Bad",
                trigger_at=datetime.now(timezone.utc),
            )


# --- is_daily_brief_due ---


class TestIsDailyBriefDue:
    @pytest.mark.parametrize(
        "now_dt, brief_time, brief_tz, expected",
        [
            # Exact match
            (
                datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
                "08:00",
                "America/New_York",  # UTC-5 in January
                True,
            ),
            # One minute before
            (
                datetime(2024, 1, 15, 12, 59, 0, tzinfo=timezone.utc),
                "08:00",
                "America/New_York",
                False,
            ),
            # One minute after
            (
                datetime(2024, 1, 15, 13, 1, 0, tzinfo=timezone.utc),
                "08:00",
                "America/New_York",
                False,
            ),
            # Well after
            (
                datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc),
                "08:00",
                "America/New_York",
                False,
            ),
            # UTC timezone
            (
                datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
                "08:00",
                "UTC",
                True,
            ),
            # DST-affected zone (July = EDT = UTC-4)
            (
                datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                "08:00",
                "America/New_York",
                True,
            ),
            # Midnight brief
            (
                datetime(2024, 1, 15, 5, 0, 0, tzinfo=timezone.utc),
                "00:00",
                "America/New_York",
                True,
            ),
        ],
        ids=[
            "exact_match",
            "one_min_before",
            "one_min_after",
            "well_after",
            "utc_match",
            "dst_summer",
            "midnight",
        ],
    )
    def test_time_matching(self, now_dt, brief_time, brief_tz, expected):
        assert is_daily_brief_due(now_dt, brief_time=brief_time, brief_tz=brief_tz) is expected

    def test_naive_datetime_raises(self):
        with pytest.raises(TypeError, match="timezone-aware"):
            is_daily_brief_due(
                datetime(2024, 1, 15, 8, 0, 0),
                brief_time="08:00",
                brief_tz="UTC",
            )

    def test_invalid_timezone_raises(self):
        with pytest.raises(ValueError, match="Invalid timezone"):
            is_daily_brief_due(
                datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
                brief_time="08:00",
                brief_tz="Mars/Olympus",
            )

    def test_malformed_time_raises(self):
        now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Invalid brief time"):
            is_daily_brief_due(now, brief_time="8am", brief_tz="UTC")
        with pytest.raises(ValueError, match="Invalid brief time"):
            is_daily_brief_due(now, brief_time="25:00", brief_tz="UTC")

    def test_whitespace_in_time_stripped(self):
        now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert is_daily_brief_due(now, brief_time="  08:00  ", brief_tz="UTC") is True
