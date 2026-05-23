"""Tests for daily brief assembly, formatting, and scheduled delivery."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.daily_brief import (
    BRIEF_MAX_ITEMS_PER_SECTION,
    SECTION_GROUPS,
    SECTION_META,
    BriefResult,
    _SLACK_TEXT_LIMIT,
    assemble_daily_brief,
    compute_trend,
    extract_next_step,
    format_markdown,
    format_slack_blocks,
)


# --- Fixtures ---


@pytest.fixture
def populated_sections():
    return {
        "calendar": ["All day: Team offsite", "09:00 Standup", "14:00 Design review"],
        "checkin_trends": ["Mood: 3.5/5 (stable →)", "Sleep: 7.2h (improving ↑)", "Energy: 4.0/5 (stable →)"],
        "birthdays": [],
        "active_projects": ["**weft** (5 commits, 1 handoff) — next: ship the brief"],
        "daily_spend": ["Today: $0.42", "7d total: $3.10 (daily avg $0.44) vs prior 7d $5.20 (improving ↓)"],
        "review_queue": ["[decision] Review pricing model… (review due 2024-01-15)"],
        "handoffs": ["Session completed Epic 5, deployed to Fly.io"],
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
        # Group headers
        assert "### 👤 Personal" in md
        assert "### 💻 Code" in md
        # Subsections under groups
        assert "#### 📋 Review Queue" in md
        assert "#### 💤 Check-in Trends" in md
        assert "#### 🔥 Top Active Projects" in md
        assert "#### 💰 Daily Spend" in md
        assert "pricing model" in md
        assert "weft" in md
        # Personal must appear before Code
        assert md.index("Personal") < md.index("Code")

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
            "active_projects": [],
            "daily_spend": [],
            "alerts": [],
        }
        now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        blocks = format_slack_blocks(sections, now)
        section_texts = [b["text"]["text"] for b in blocks if b["type"] == "section"]
        # One group header (Code) + one subsection (Review Queue) = 2
        assert len(section_texts) == 2
        assert any("Review Queue" in t for t in section_texts)
        assert any("💻 Code" in t for t in section_texts)

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


# --- extract_next_step ---


class TestExtractNextStep:
    def test_bold_inline(self):
        body = (
            "## Session Handoff\n\n"
            "**Summary:** shipped the thing.\n\n"
            "**Next Steps:** Pick up loom-c47075b3 (audit-log MCP tool, p2). "
            "Then move on to the Wick INTERCHANGE.\n\n"
            "**Blockers:** none."
        )
        out = extract_next_step(body)
        assert out is not None
        assert "loom-c47075b3" in out
        assert "Blockers" not in out

    def test_h2_section(self):
        body = "## Next Steps\n\nDo the next thing.\n\n## Other\n\nIgnored."
        out = extract_next_step(body)
        assert out is not None
        assert "Do the next thing" in out
        assert "Ignored" not in out

    def test_truncates_long_body(self):
        body = "**Next Steps:** " + ("very long step " * 100)
        out = extract_next_step(body, max_chars=50)
        assert out is not None
        assert len(out) <= 51  # +1 for ellipsis
        assert out.endswith("…")

    def test_no_section_returns_none(self):
        assert extract_next_step("just some prose, no section") is None

    def test_empty_input(self):
        assert extract_next_step("") is None
        assert extract_next_step(None) is None  # type: ignore[arg-type]


# --- assemble_daily_brief ---


def _patch_all_queries(**overrides):
    """Helper: patch every _query_* function used by assemble_daily_brief.

    Default returns are empty lists. Override any subset by name (no leading
    underscore), e.g. _patch_all_queries(handoffs=["x"]).
    """
    from contextlib import ExitStack

    # Map friendly name -> actual private function name in weft.daily_brief.
    # Calendar's underlying helper is _query_calendar_events.
    name_map = {
        "calendar": "_query_calendar_events",
        "review_queue": "_query_review_queue",
        "handoffs": "_query_handoffs",
        "checkin_trends": "_query_checkin_trends",
        "birthdays": "_query_birthdays",
        "active_projects": "_query_active_projects",
        "daily_spend": "_query_daily_spend",
        "alerts": "_query_alerts",
    }
    defaults = {key: [] for key in name_map}
    defaults.update(overrides)

    stack = ExitStack()
    patches = {}
    for name, value in defaults.items():
        target = f"weft.daily_brief.{name_map[name]}"
        if isinstance(value, Exception):
            p = patch(target, new_callable=AsyncMock, side_effect=value)
        else:
            p = patch(target, new_callable=AsyncMock, return_value=value)
        patches[name] = stack.enter_context(p)
    return stack, patches


class TestAssembleDailyBrief:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        mock_pool = MagicMock()
        stack, patches = _patch_all_queries(
            review_queue=["review item"],
            handoffs=["handoff item"],
            checkin_trends=["Mood: 3.5/5"],
            active_projects=["**weft** (3 commits, 1 handoff) — next: ship the brief"],
            daily_spend=["Today: $0.42"],
            alerts=["[custom] alert"],
        )
        with stack:
            result = await assemble_daily_brief(mock_pool)

        assert isinstance(result, BriefResult)
        assert "review item" in result.markdown
        assert "handoff item" in result.markdown
        assert "weft" in result.markdown
        assert "$0.42" in result.markdown
        assert len(result.slack_blocks) > 0
        assert isinstance(result.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_all_sources_empty(self):
        mock_pool = MagicMock()
        stack, _ = _patch_all_queries()
        with stack:
            result = await assemble_daily_brief(mock_pool)

        assert "all clear" in result.markdown.lower()

    @pytest.mark.asyncio
    async def test_partial_failure_still_assembles(self):
        mock_pool = MagicMock()
        stack, _ = _patch_all_queries(
            review_queue=Exception("db down"),
            handoffs=["handoff"],
            active_projects=Exception("git gone"),
        )
        with stack:
            result = await assemble_daily_brief(mock_pool)

        assert "handoff" in result.markdown

    @pytest.mark.asyncio
    async def test_custom_date(self):
        mock_pool = MagicMock()
        target = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        stack, patches = _patch_all_queries()
        with stack:
            result = await assemble_daily_brief(mock_pool, target_date=target)

        patches["review_queue"].assert_called_once_with(mock_pool, target)
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
            patch.dict("os.environ", {"WEFT_OUTBOUND_CONNECTOR": "slack"}),
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
            patch.dict("os.environ", {"WEFT_OUTBOUND_CONNECTOR": "slack"}),
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
