"""Tests for Slack slash command check-in handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.mcp.slack_commands import handle_slash_checkin, parse_checkin_text


# ── Parser tests ────────────────────────────────────────────────────


class TestParseCheckinText:
    def test_named_fields(self):
        r = parse_checkin_text("mood 3 sleep 7 energy 4")
        assert r["mood"] == 3
        assert r["sleep_hours"] == 7.0
        assert r["energy"] == 4

    def test_named_with_colons(self):
        r = parse_checkin_text("mood:3 sleep:7.5 energy:4")
        assert r["mood"] == 3
        assert r["sleep_hours"] == 7.5
        assert r["energy"] == 4

    def test_short_names(self):
        r = parse_checkin_text("m3 s7 e4")
        assert r["mood"] == 3
        assert r["sleep_hours"] == 7.0
        assert r["energy"] == 4

    def test_positional(self):
        r = parse_checkin_text("3 7 4")
        assert r["mood"] == 3
        assert r["sleep_hours"] == 7.0
        assert r["energy"] == 4

    def test_partial(self):
        r = parse_checkin_text("mood 4")
        assert r["mood"] == 4
        assert "sleep_hours" not in r
        assert "energy" not in r

    def test_with_notes(self):
        r = parse_checkin_text("mood 3 energy 2 feeling tired today")
        assert r["mood"] == 3
        assert r["energy"] == 2
        assert "tired" in r["notes"]

    def test_notes_only(self):
        r = parse_checkin_text("feeling great after a long walk")
        assert "notes" in r

    def test_empty(self):
        r = parse_checkin_text("")
        assert r == {}

    def test_decimal_sleep(self):
        r = parse_checkin_text("sleep 6.5")
        assert r["sleep_hours"] == 6.5


# ── Handler tests (real DB) ─────────────────────────────────────────


def _make_request(text: str = "", user_id: str = "U123") -> MagicMock:
    """Build a mock Starlette Request with form data."""
    req = MagicMock()
    req.body = AsyncMock(return_value=b"")
    form_data = {"text": text, "user_id": user_id, "command": "/checkin"}
    req.form = AsyncMock(return_value=form_data)
    req.headers = {"X-Slack-Request-Timestamp": "", "X-Slack-Signature": ""}
    return req


class TestSlashCheckinHandler:
    @pytest.mark.asyncio
    async def test_success(self, pool):
        req = _make_request("mood 4 sleep 7 energy 3")
        resp = await handle_slash_checkin(req, pool)
        data = resp.body.decode()
        assert "Check-in logged" in data
        assert "Mood: 4/5" in data

    @pytest.mark.asyncio
    async def test_empty_text_returns_usage(self, pool):
        req = _make_request("")
        resp = await handle_slash_checkin(req, pool)
        data = resp.body.decode()
        assert "Usage" in data

    @pytest.mark.asyncio
    async def test_freetext_stored_as_notes(self, pool):
        req = _make_request("feeling great after a walk")
        resp = await handle_slash_checkin(req, pool)
        data = resp.body.decode()
        assert "Check-in logged" in data
        assert "great" in data

    @pytest.mark.asyncio
    async def test_mood_out_of_range(self, pool):
        req = _make_request("mood 7")
        resp = await handle_slash_checkin(req, pool)
        data = resp.body.decode()
        assert "1-5" in data

    @pytest.mark.asyncio
    async def test_notes_included(self, pool):
        req = _make_request("mood 3 feeling groggy")
        resp = await handle_slash_checkin(req, pool)
        data = resp.body.decode()
        assert "groggy" in data

    @pytest.mark.asyncio
    async def test_stores_in_db(self, pool):
        req = _make_request("mood 5 sleep 8 energy 5")
        await handle_slash_checkin(req, pool)

        row = await pool.fetchrow(
            "SELECT * FROM check_ins ORDER BY created_at DESC LIMIT 1"
        )
        assert row["mood"] == 5
        assert row["sleep_hours"] == 8.0
        assert row["energy"] == 5
