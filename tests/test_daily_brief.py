"""Tests for daily brief assembly, formatting, and scheduled delivery."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.daily_brief import (
    BRIEF_MAX_ITEMS_PER_SECTION,
    SECTION_META,
    BriefResult,
    _SLACK_TEXT_LIMIT,
    assemble_daily_brief,
    compute_trend,
    format_markdown,
    format_slack_blocks,
)


# --- Fixtures ---


@pytest.fixture
def populated_sections():
    return {
        "review_queue": ["[decision] Review pricing model… (review due 2024-01-15)"],
        "handoffs": ["Session completed Epic 5, deployed to Fly.io"],
        "checkin_trends": ["Mood: 3.5/5 (stable →)", "Sleep: 7.2h (improving ↑)", "Energy: 4.0/5 (stable →)"],
        "ready_tasks": ["[p0] Build daily brief assembly"],
        "alerts": ["[follow_up] Check deployment (due 09:00)"],
    }


@pytest.fixture
def empty_sections():
    return {key: [] for key in SECTION_META}


# --- compute_trend ---


class TestComputeTrend:
    def test_improving(self):
        assert compute_trend([4.0, 4.5, 5.0], [3.0, 3.2, 3.1]) == "improving ↑"

    def test_declining(self):
        assert compute_trend([2.0, 2.1, 2.0], [3.5, 3.8, 3.6]) == "declining ↓"

    def test_stable(self):
        assert compute_trend([3.5, 3.6, 3.4], [3.5, 3.4, 3.5]) == "stable →"

    def test_insufficient_recent(self):
        assert compute_trend([3.5], [3.5, 3.4, 3.5]) == "insufficient data"

    def test_insufficient_prior(self):
        assert compute_trend([3.5, 3.6, 3.4], [3.5]) == "insufficient data"

    def test_both_empty(self):
        assert compute_trend([], []) == "insufficient data"


# --- format_markdown ---


class TestFormatMarkdown:
    def test_populated(self, populated_sections):
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        md = format_markdown(populated_sections, now)
        assert "## 🌅 Daily Brief" in md
        assert "### 📋 Review Queue" in md
        assert "### 💤 Check-in Trends" in md
        assert "pricing model" in md
        assert "- [p0]" in md

    def test_empty_shows_all_clear(self, empty_sections):
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        md = format_markdown(empty_sections, now)
        assert "all clear" in md.lower()
        for key, (emoji, title) in SECTION_META.items():
            assert title in md

    def test_ends_with_rule(self, populated_sections):
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        md = format_markdown(populated_sections, now)
        assert md.strip().endswith("---")


# --- format_slack_blocks ---


class TestFormatSlackBlocks:
    def test_populated_structure(self, populated_sections):
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(populated_sections, now)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"

        # All blocks must have a type
        for block in blocks:
            assert "type" in block

        # Section blocks have mrkdwn text
        section_blocks = [b for b in blocks if b["type"] == "section"]
        for sb in section_blocks:
            assert sb["text"]["type"] == "mrkdwn"

    def test_empty_shows_all_clear(self, empty_sections):
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(empty_sections, now)
        texts = [b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"]
        assert any("All clear" in t for t in texts)

    def test_skips_empty_sections(self):
        sections = {
            "review_queue": ["item 1"],
            "handoffs": [],
            "checkin_trends": [],
            "ready_tasks": [],
            "alerts": [],
        }
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(sections, now)
        section_texts = [b["text"]["text"] for b in blocks if b["type"] == "section"]
        assert len(section_texts) == 1
        assert "Review Queue" in section_texts[0]

    def test_truncates_long_sections(self):
        items = [f"Item {i}: " + "x" * 50 for i in range(100)]
        sections = {"review_queue": items}
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(sections, now)
        section_blocks = [b for b in blocks if b["type"] == "section"]
        for sb in section_blocks:
            assert len(sb["text"]["text"]) <= _SLACK_TEXT_LIMIT + 100  # some headroom for truncation text

    def test_max_50_blocks(self):
        # Create many sections with items
        sections = {key: [f"item {i}" for i in range(20)] for key in SECTION_META}
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(sections, now)
        assert len(blocks) <= 50


# --- assemble_daily_brief ---


class TestAssembleDailyBrief:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        mock_pool = MagicMock()

        with (
            patch("weft.daily_brief._query_review_queue", new_callable=AsyncMock, return_value=["review item"]),
            patch("weft.daily_brief._query_handoffs", new_callable=AsyncMock, return_value=["handoff item"]),
            patch("weft.daily_brief._query_checkin_trends", new_callable=AsyncMock, return_value=["Mood: 3.5/5"]),
            patch("weft.daily_brief._query_loom_tasks", new_callable=AsyncMock, return_value=["[p0] task"]),
            patch("weft.daily_brief._query_alerts", new_callable=AsyncMock, return_value=["[custom] alert"]),
        ):
            result = await assemble_daily_brief(mock_pool)

        assert isinstance(result, BriefResult)
        assert "review item" in result.markdown
        assert "handoff item" in result.markdown
        assert len(result.slack_blocks) > 0
        assert isinstance(result.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_all_sources_empty(self):
        mock_pool = MagicMock()

        with (
            patch("weft.daily_brief._query_review_queue", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_handoffs", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_checkin_trends", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_loom_tasks", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_alerts", new_callable=AsyncMock, return_value=[]),
        ):
            result = await assemble_daily_brief(mock_pool)

        assert "all clear" in result.markdown.lower()

    @pytest.mark.asyncio
    async def test_partial_failure_still_assembles(self):
        mock_pool = MagicMock()

        with (
            patch("weft.daily_brief._query_review_queue", new_callable=AsyncMock, side_effect=Exception("db down")),
            patch("weft.daily_brief._query_handoffs", new_callable=AsyncMock, return_value=["handoff"]),
            patch("weft.daily_brief._query_checkin_trends", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_loom_tasks", new_callable=AsyncMock, side_effect=Exception("loom gone")),
            patch("weft.daily_brief._query_alerts", new_callable=AsyncMock, return_value=[]),
        ):
            result = await assemble_daily_brief(mock_pool)

        # Should still have the handoff data
        assert "handoff" in result.markdown

    @pytest.mark.asyncio
    async def test_custom_date(self):
        mock_pool = MagicMock()
        target = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        with (
            patch("weft.daily_brief._query_review_queue", new_callable=AsyncMock, return_value=[]) as mock_rq,
            patch("weft.daily_brief._query_handoffs", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_checkin_trends", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_loom_tasks", new_callable=AsyncMock, return_value=[]),
            patch("weft.daily_brief._query_alerts", new_callable=AsyncMock, return_value=[]),
        ):
            result = await assemble_daily_brief(mock_pool, target_date=target)

        # The review queue was called with our target date
        mock_rq.assert_called_once_with(mock_pool, target)
        assert "June" in result.markdown


# --- brief_state dedup ---


class TestBriefState:
    def test_roundtrip(self, tmp_path):
        state_file = tmp_path / "brief_state.json"
        with patch("weft.brief_state._STATE_PATH", state_file):
            from weft.brief_state import get_last_brief_date, set_last_brief_date

            assert get_last_brief_date() is None
            set_last_brief_date(date(2024, 1, 15))
            assert get_last_brief_date() == date(2024, 1, 15)

    def test_missing_file_returns_none(self, tmp_path):
        state_file = tmp_path / "nonexistent" / "brief_state.json"
        with patch("weft.brief_state._STATE_PATH", state_file):
            from weft.brief_state import get_last_brief_date

            assert get_last_brief_date() is None


# --- daily_brief_loop ---


class TestDailyBriefLoop:
    @pytest.mark.asyncio
    async def test_no_channel_exits(self, caplog):
        """Loop exits immediately when no channel is configured."""
        from weft.scheduler import daily_brief_loop

        import logging
        with caplog.at_level(logging.INFO):
            await daily_brief_loop(MagicMock(), brief_channel="")

        assert any("no_channel" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_delivers_when_due(self):
        """Loop assembles and posts brief when time matches and not yet delivered today."""
        from weft.scheduler import daily_brief_loop

        mock_pool = MagicMock()
        brief_result = BriefResult(
            markdown="# Test Brief",
            slack_blocks=[{"type": "header", "text": {"type": "plain_text", "text": "Test"}}],
        )

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        with (
            patch("weft.scheduler.is_daily_brief_due", return_value=True),
            patch("weft.brief_state.get_last_brief_date", return_value=None),
            patch("weft.brief_state.set_last_brief_date") as mock_set,
            patch("weft.daily_brief.assemble_daily_brief", new_callable=AsyncMock, return_value=brief_result),
            patch("weft.scheduler._post_brief_to_slack", new_callable=AsyncMock) as mock_post,
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await daily_brief_loop(mock_pool, brief_channel="C12345")

        mock_post.assert_called_once()
        mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_if_already_delivered(self):
        """Loop skips delivery if brief was already sent today."""
        from weft.scheduler import daily_brief_loop

        mock_pool = MagicMock()

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        today = date.today()

        with (
            patch("weft.scheduler.is_daily_brief_due", return_value=True),
            patch("weft.brief_state.get_last_brief_date", return_value=today),
            patch("weft.daily_brief.assemble_daily_brief", new_callable=AsyncMock) as mock_assemble,
            patch("weft.scheduler._post_brief_to_slack", new_callable=AsyncMock) as mock_post,
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await daily_brief_loop(mock_pool, brief_channel="C12345")

        mock_assemble.assert_not_called()
        mock_post.assert_not_called()
