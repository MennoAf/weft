"""Tests for Slack alert dispatch."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.models import Alert, AlertChannel, AlertType
from weft.scheduler import dispatch_alert, dispatch_slack


def _make_alert(
    *,
    channel: AlertChannel = AlertChannel.slack,
    channel_target: str | None = "#alerts",
    title: str = "Test alert",
    body: str | None = "Alert body",
    alert_type: AlertType = AlertType.due_task,
    trigger_at: datetime | None = None,
) -> Alert:
    return Alert(
        alert_type=alert_type,
        title=title,
        body=body,
        trigger_at=trigger_at or datetime(2026, 3, 21, 10, 0, tzinfo=timezone.utc),
        channel=channel,
        channel_target=channel_target,
    )


class TestSlackDispatch:
    @pytest.mark.asyncio
    async def test_sends_message(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        mock_response = {"ok": True, "ts": "123456.789"}
        mock_client_instance = MagicMock()
        mock_client_instance.chat_postMessage = AsyncMock(return_value=mock_response)

        with patch(
            "weft.scheduler.AsyncWebClient", return_value=mock_client_instance
        ) as mock_cls:
            alert = _make_alert()
            await dispatch_slack(alert)

            mock_cls.assert_called_once_with(token="REDACTED")
            mock_client_instance.chat_postMessage.assert_called_once()
            call_kwargs = mock_client_instance.chat_postMessage.call_args[1]
            assert call_kwargs["channel"] == "#alerts"
            assert "*[due_task]*" in call_kwargs["text"]
            assert "Test alert" in call_kwargs["text"]
            assert "2026-03-21 10:00 UTC" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_includes_body(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        mock_client_instance = MagicMock()
        mock_client_instance.chat_postMessage = AsyncMock(return_value={"ok": True})

        with patch(
            "weft.scheduler.AsyncWebClient", return_value=mock_client_instance
        ):
            alert = _make_alert(body="Check the deadline")
            await dispatch_slack(alert)
            text = mock_client_instance.chat_postMessage.call_args[1]["text"]
            assert "Check the deadline" in text

    @pytest.mark.asyncio
    async def test_no_body(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        mock_client_instance = MagicMock()
        mock_client_instance.chat_postMessage = AsyncMock(return_value={"ok": True})

        with patch(
            "weft.scheduler.AsyncWebClient", return_value=mock_client_instance
        ):
            alert = _make_alert(body=None)
            await dispatch_slack(alert)
            mock_client_instance.chat_postMessage.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_token_logs_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        alert = _make_alert()
        with caplog.at_level(logging.WARNING, logger="weft.scheduler"):
            await dispatch_slack(alert)
        assert "no_token" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_token_no_sdk_call(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        with patch("weft.scheduler.AsyncWebClient") as mock_cls:
            await dispatch_slack(_make_alert())
            mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_channel_target_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        alert = _make_alert(channel_target=None)
        with caplog.at_level(logging.WARNING, logger="weft.scheduler"):
            await dispatch_slack(alert)
        assert "no_channel_target" in caplog.text

    @pytest.mark.asyncio
    async def test_api_error_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        mock_client_instance = MagicMock()
        mock_client_instance.chat_postMessage = AsyncMock(
            return_value={"ok": False, "error": "channel_not_found"}
        )

        with patch(
            "weft.scheduler.AsyncWebClient", return_value=mock_client_instance
        ):
            alert = _make_alert()
            with caplog.at_level(logging.WARNING, logger="weft.scheduler"):
                await dispatch_slack(alert)
            assert "api_error" in caplog.text

    @pytest.mark.asyncio
    async def test_http_error_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "REDACTED")

        mock_client_instance = MagicMock()
        mock_client_instance.chat_postMessage = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        with patch(
            "weft.scheduler.AsyncWebClient", return_value=mock_client_instance
        ):
            alert = _make_alert()
            await dispatch_slack(alert)  # Should not raise


class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_routes_to_slack(self):
        """dispatch_alert routes slack-channel alerts to dispatch_slack."""
        with patch("weft.scheduler.dispatch_slack", new_callable=AsyncMock) as mock:
            # Also patch the registry to use the mock
            from weft.scheduler import _DISPATCH_REGISTRY

            saved = _DISPATCH_REGISTRY["slack"]
            _DISPATCH_REGISTRY["slack"] = mock
            try:
                alert = _make_alert(channel=AlertChannel.slack)
                await dispatch_alert(alert)
                mock.assert_called_once_with(alert)
            finally:
                _DISPATCH_REGISTRY["slack"] = saved

    @pytest.mark.asyncio
    async def test_routes_to_log(self, caplog):
        alert = _make_alert(channel=AlertChannel.log)
        with caplog.at_level(logging.INFO, logger="weft.scheduler"):
            await dispatch_alert(alert)
        assert "alert.fired" in caplog.text
