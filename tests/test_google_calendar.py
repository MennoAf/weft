"""Tests for weft.google_calendar credential management and daily brief calendar query."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from weft.config import DailyBriefConfig


class TestGetCredentials:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent" / "creds.json"
        with patch("weft.google_calendar.CREDENTIALS_PATH", missing):
            from weft.google_calendar import get_credentials

            with pytest.raises(RuntimeError, match="weft calendar-auth"):
                get_credentials()

    def test_valid_credentials_returned(self, tmp_path):
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False

        with patch(
            "weft.google_calendar.Credentials.from_authorized_user_file",
            return_value=mock_creds,
        ), patch("weft.google_calendar.CREDENTIALS_PATH", tmp_path / "creds.json"):
            # Create the file so the exists() check passes
            creds_path = tmp_path / "creds.json"
            creds_path.write_text("{}")

            from weft.google_calendar import get_credentials

            result = get_credentials()
            assert result is mock_creds
            mock_creds.refresh.assert_not_called()

    def test_expired_with_refresh_token(self, tmp_path):
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh-token"
        mock_creds.to_json.return_value = '{"token": "refreshed"}'

        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")

        with patch(
            "weft.google_calendar.Credentials.from_authorized_user_file",
            return_value=mock_creds,
        ), patch("weft.google_calendar.CREDENTIALS_PATH", creds_path):
            from weft.google_calendar import get_credentials

            result = get_credentials()
            assert result is mock_creds
            mock_creds.refresh.assert_called_once()
            assert creds_path.read_text() == '{"token": "refreshed"}'

    def test_expired_no_refresh_token_raises(self, tmp_path):
        mock_creds = MagicMock()
        mock_creds.expired = True
        mock_creds.refresh_token = None

        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")

        with patch(
            "weft.google_calendar.Credentials.from_authorized_user_file",
            return_value=mock_creds,
        ), patch("weft.google_calendar.CREDENTIALS_PATH", creds_path):
            from weft.google_calendar import get_credentials

            with pytest.raises(RuntimeError, match="re-authenticate"):
                get_credentials()


class TestBuildService:
    def test_builds_with_credentials(self):
        mock_creds = MagicMock()
        sentinel = object()

        with patch(
            "weft.google_calendar.get_credentials", return_value=mock_creds
        ), patch(
            "weft.google_calendar.build", return_value=sentinel
        ) as mock_build:
            from weft.google_calendar import build_service

            result = build_service()
            assert result is sentinel
            mock_build.assert_called_once_with(
                "calendar", "v3", credentials=mock_creds, cache_discovery=False
            )


class TestRunOAuthFlow:
    def test_saves_credentials(self, tmp_path):
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.to_json.return_value = '{"token": "new"}'

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        creds_path = tmp_path / ".weft" / "google_credentials.json"

        with patch(
            "weft.google_calendar.CREDENTIALS_PATH", creds_path
        ), patch(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            from weft.google_calendar import run_oauth_flow

            run_oauth_flow("/fake/secrets.json")

            assert creds_path.exists()
            assert creds_path.read_text() == '{"token": "new"}'
            # Check file permissions (owner read/write only)
            assert oct(creds_path.stat().st_mode)[-3:] == "600"

    def test_failed_flow_raises(self, tmp_path):
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = None

        with patch(
            "weft.google_calendar.CREDENTIALS_PATH", tmp_path / "creds.json"
        ), patch(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
            return_value=mock_flow,
        ):
            from weft.google_calendar import run_oauth_flow

            with pytest.raises(RuntimeError, match="did not complete"):
                run_oauth_flow("/fake/secrets.json")


class TestDailyBriefConfig:
    def test_default_calendar_id(self):
        config = DailyBriefConfig()
        assert config.calendar_id == "primary"

    def test_custom_calendar_id(self):
        config = DailyBriefConfig(calendar_id="work@group.calendar.google.com")
        assert config.calendar_id == "work@group.calendar.google.com"


class TestQueryCalendarEvents:
    @pytest.fixture
    def mock_events(self):
        return {
            "items": [
                {
                    "summary": "Team offsite",
                    "start": {"date": "2026-03-23"},
                    "status": "confirmed",
                },
                {
                    "summary": "Standup",
                    "start": {"dateTime": "2026-03-23T09:00:00-04:00"},
                    "status": "confirmed",
                },
                {
                    "summary": "Design review",
                    "start": {"dateTime": "2026-03-23T14:30:00-04:00"},
                    "status": "confirmed",
                },
            ]
        }

    async def test_mixed_events(self, mock_events):
        mock_service = MagicMock()
        mock_service.events().list().execute.return_value = mock_events

        with patch("weft.google_calendar.build_service", return_value=mock_service):
            from weft.daily_brief import _query_calendar_events

            tz = ZoneInfo("America/New_York")
            as_of = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
            result = await _query_calendar_events("primary", tz, as_of)

        assert result[0] == "All day: Team offsite"
        assert result[1] == "09:00 Standup"
        assert result[2] == "14:30 Design review"

    async def test_empty_calendar(self):
        mock_service = MagicMock()
        mock_service.events().list().execute.return_value = {"items": []}

        with patch("weft.google_calendar.build_service", return_value=mock_service):
            from weft.daily_brief import _query_calendar_events

            tz = ZoneInfo("America/New_York")
            as_of = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
            result = await _query_calendar_events("primary", tz, as_of)

        assert result == []

    async def test_no_title(self):
        mock_service = MagicMock()
        mock_service.events().list().execute.return_value = {
            "items": [{"start": {"date": "2026-03-23"}}]
        }

        with patch("weft.google_calendar.build_service", return_value=mock_service):
            from weft.daily_brief import _query_calendar_events

            tz = ZoneInfo("America/New_York")
            as_of = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
            result = await _query_calendar_events("primary", tz, as_of)

        assert result == ["All day: (No title)"]

    async def test_cancelled_events_skipped(self):
        mock_service = MagicMock()
        mock_service.events().list().execute.return_value = {
            "items": [
                {"summary": "Cancelled", "start": {"date": "2026-03-23"}, "status": "cancelled"},
                {"summary": "Active", "start": {"date": "2026-03-23"}, "status": "confirmed"},
            ]
        }

        with patch("weft.google_calendar.build_service", return_value=mock_service):
            from weft.daily_brief import _query_calendar_events

            tz = ZoneInfo("America/New_York")
            as_of = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
            result = await _query_calendar_events("primary", tz, as_of)

        assert result == ["All day: Active"]

    async def test_missing_credentials_raises(self):
        with patch(
            "weft.google_calendar.build_service",
            side_effect=RuntimeError("No creds"),
        ):
            from weft.daily_brief import _query_calendar_events

            tz = ZoneInfo("America/New_York")
            as_of = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
            with pytest.raises(RuntimeError):
                await _query_calendar_events("primary", tz, as_of)

    def test_calendar_in_section_meta(self):
        from weft.daily_brief import SECTION_META

        assert "calendar" in SECTION_META
        emoji, title = SECTION_META["calendar"]
        assert emoji == "📅"
        assert "Calendar" in title
