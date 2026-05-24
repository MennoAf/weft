"""Tests for scheduler loops — loom_awareness_loop, memory_hygiene_loop,
and the outbound event dispatch layer.

Tests verify that each loop:
1. Calls its evaluator and handles findings
2. Handles the evaluator returning empty results
3. Handles evaluator exceptions without crashing
"""

from __future__ import annotations

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_one_iteration(coro_fn, pool, **kwargs):
    """Run a scheduler loop for one iteration then cancel it."""
    task = asyncio.create_task(coro_fn(pool, interval=0, **kwargs))
    # Give the loop one iteration
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Outbound Connector Startup Validation
# ---------------------------------------------------------------------------


class TestOutboundConnectorValidation:
    """Tests for WEFT_OUTBOUND_CONNECTOR env var validation at startup."""

    @pytest.mark.parametrize("value", ["slack", "discord", "none", "", "SLACK", " slack "])
    def test_valid_connector_values_pass(self, monkeypatch, value):
        """Valid WEFT_OUTBOUND_CONNECTOR values do not raise at startup."""
        from weft.mcp.server import _validate_outbound_connector_env

        if value == "":
            monkeypatch.delenv("WEFT_OUTBOUND_CONNECTOR", raising=False)
        else:
            monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", value)

        # Must not raise
        _validate_outbound_connector_env()

    @pytest.mark.parametrize("invalid_value", ["badconnector", "slack2", "ftp", "http", "Discord!"])
    def test_invalid_connector_values_raise(self, monkeypatch, invalid_value):
        """Invalid WEFT_OUTBOUND_CONNECTOR values raise ValueError at startup."""
        from weft.mcp.server import _validate_outbound_connector_env

        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", invalid_value)

        with pytest.raises(ValueError, match="WEFT_OUTBOUND_CONNECTOR must be one of"):
            _validate_outbound_connector_env()


# ---------------------------------------------------------------------------
# Loom awareness loop
# ---------------------------------------------------------------------------


class TestLoomAwarenessLoop:
    @pytest.mark.asyncio
    async def test_calls_evaluator_with_findings(self):
        from weft.scheduler import loom_awareness_loop

        mock_alerts = [{"alert_type": "loom_stale_claim", "title": "test"}]
        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ) as mock_eval:
            await _run_one_iteration(loom_awareness_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_handles_empty_findings(self):
        from weft.scheduler import loom_awareness_loop

        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_eval:
            await _run_one_iteration(loom_awareness_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_survives_evaluator_exception(self):
        from weft.scheduler import loom_awareness_loop

        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            # Should not raise — the loop catches exceptions
            await _run_one_iteration(loom_awareness_loop, AsyncMock())


# ---------------------------------------------------------------------------
# Memory hygiene loop
# ---------------------------------------------------------------------------


class TestMemoryHygieneLoop:
    @pytest.mark.asyncio
    async def test_calls_evaluator_with_findings(self):
        from weft.scheduler import memory_hygiene_loop

        mock_alerts = [{"alert_type": "memory_consolidation_overdue", "title": "test"}]
        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ) as mock_eval:
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_handles_empty_findings(self):
        from weft.scheduler import memory_hygiene_loop

        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_eval:
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_survives_evaluator_exception(self):
        from weft.scheduler import memory_hygiene_loop

        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())


# ---------------------------------------------------------------------------
# Outbound event dispatch layer
# ---------------------------------------------------------------------------


class TestOutboundEventDispatch:
    """Tests for the event registry / emit_outbound_event layer."""

    @pytest.mark.asyncio
    async def test_emit_calls_active_connector(self, monkeypatch):
        """emit_outbound_event routes the event to the registered connector."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("test_event", "fake_connector", handler)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "fake_connector")

        await emit_outbound_event("test_event", channel="#test", brief_result=None)

        handler.assert_awaited_once_with(channel="#test", brief_result=None)

    @pytest.mark.asyncio
    async def test_slack_connector_registered(self):
        """The slack connector for daily_brief is registered at module load."""
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        assert "daily_brief" in _OUTBOUND_EVENT_REGISTRY
        assert "slack" in _OUTBOUND_EVENT_REGISTRY["daily_brief"]

    @pytest.mark.asyncio
    async def test_only_active_connector_receives_event(self, monkeypatch):
        """Only the env-selected connector receives the event; others are skipped."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler_a = AsyncMock()
        handler_b = AsyncMock()
        register_outbound_handler("singleton_event", "connector_a", handler_a)
        register_outbound_handler("singleton_event", "connector_b", handler_b)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "connector_a")

        await emit_outbound_event("singleton_event", x=1)

        handler_a.assert_awaited_once_with(x=1)
        handler_b.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_connector_unset(self, monkeypatch):
        """emit_outbound_event is a no-op when WEFT_OUTBOUND_CONNECTOR is unset."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("noop_event", "slack", handler)
        monkeypatch.delenv("WEFT_OUTBOUND_CONNECTOR", raising=False)

        # Must not raise
        await emit_outbound_event("noop_event", channel="#foo", brief_result=None)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_connector_is_none_string(self, monkeypatch):
        """emit_outbound_event is a no-op when WEFT_OUTBOUND_CONNECTOR='none'."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("none_event", "slack", handler)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "none")

        await emit_outbound_event("none_event", channel="#foo", brief_result=None)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_connector_value_warning(self, monkeypatch, caplog):
        """emit_outbound_event logs a warning when WEFT_OUTBOUND_CONNECTOR has an invalid value (invalid at dispatch time)."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("test_event", "slack", handler)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "invalid_connector")

        # Must not raise — invalid values at dispatch time are warnings
        await emit_outbound_event("test_event", channel="#foo", brief_result=None)

        # Should log a warning about missing handler
        assert "no_handler" in caplog.text

    @pytest.mark.asyncio
    async def test_daily_brief_loop_emits_event(self, monkeypatch):
        """daily_brief_loop emits a daily_brief event via emit_outbound_event."""
        from weft.scheduler import daily_brief_loop

        # Fake brief result with the fields _post_brief_to_slack expects
        fake_result = SimpleNamespace(
            markdown="Morning brief",
            slack_blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}],
        )

        # daily_brief_loop uses asyncio.sleep(_BRIEF_POLL_INTERVAL) at the end of
        # each cycle. Patch it to raise CancelledError so the loop exits cleanly
        # after exactly one iteration (the CancelledError propagates out of the
        # while-loop's outer try/except to the asyncio.CancelledError handler).
        async def sleep_then_cancel(_):
            raise asyncio.CancelledError("test cancel")

        mock_emit = AsyncMock()

        with (
            # is_daily_brief_due is imported into weft.scheduler at module load
            patch("weft.scheduler.is_daily_brief_due", return_value=True),
            # get/set_last_brief_date are imported inside the function body
            patch("weft.brief_state.get_last_brief_date", return_value=None),
            patch("weft.brief_state.set_last_brief_date"),
            # assemble_daily_brief is imported inside the function body
            patch(
                "weft.daily_brief.assemble_daily_brief",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
            patch("weft.scheduler.emit_outbound_event", mock_emit),
            patch("weft.scheduler.asyncio.sleep", side_effect=sleep_then_cancel),
        ):
            with pytest.raises(asyncio.CancelledError):
                await daily_brief_loop(
                    AsyncMock(),
                    brief_time="08:00",
                    brief_tz="UTC",
                    brief_channel="#test",
                )

        # The loop must have called emit_outbound_event with the daily_brief event
        assert mock_emit.call_count >= 1
        call_kwargs = mock_emit.call_args
        assert call_kwargs[0][0] == "daily_brief"
        assert call_kwargs[1]["channel"] == "#test"
        assert call_kwargs[1]["brief_result"] is fake_result

    @pytest.mark.asyncio
    async def test_slack_handler_calls_post_brief(self, monkeypatch):
        """The registered slack handler calls _post_brief_to_slack with correct args."""
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        fake_result = SimpleNamespace(
            markdown="x" * 400,
            slack_blocks=[],
        )

        slack_handler = _OUTBOUND_EVENT_REGISTRY["daily_brief"]["slack"]

        with patch(
            "weft.scheduler._post_brief_to_slack",
            new_callable=AsyncMock,
        ) as mock_post:
            await slack_handler(channel="#mychannel", brief_result=fake_result)

        mock_post.assert_awaited_once_with("#mychannel", fake_result)


# ---------------------------------------------------------------------------
# Discord connector + bot loop
# ---------------------------------------------------------------------------


class TestDiscordOutboundHandler:
    """Tests for the daily_brief discord handler and its bot-reference contract."""

    @pytest.mark.asyncio
    async def test_discord_connector_registered(self):
        """Importing weft.discord registers the discord handler for daily_brief."""
        import weft.discord  # noqa: F401  — import triggers registration
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        assert "daily_brief" in _OUTBOUND_EVENT_REGISTRY
        assert "discord" in _OUTBOUND_EVENT_REGISTRY["daily_brief"]

    @pytest.mark.asyncio
    async def test_handler_noops_when_bot_unset(self):
        """Handler logs and returns when no bot has been registered."""
        from weft.discord import connector
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        connector.clear_bot()
        handler = _OUTBOUND_EVENT_REGISTRY["daily_brief"]["discord"]
        # Must not raise
        await handler(channel="ignored", brief_result=SimpleNamespace(markdown="hi"))

    @pytest.mark.asyncio
    async def test_handler_noops_when_bot_not_ready(self):
        """Handler skips post when bot reports not-ready."""
        from weft.discord import connector
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        fake_bot = MagicMock()
        fake_bot.is_ready = False
        fake_bot.channel_id = 12345
        fake_bot.post = AsyncMock()
        connector.set_bot(fake_bot)
        try:
            handler = _OUTBOUND_EVENT_REGISTRY["daily_brief"]["discord"]
            await handler(channel="ignored", brief_result=SimpleNamespace(markdown="hi"))
            fake_bot.post.assert_not_called()
        finally:
            connector.clear_bot()

    @pytest.mark.asyncio
    async def test_handler_posts_markdown_when_ready(self):
        """Handler forwards brief_result.markdown to bot.post when ready."""
        from weft.discord import connector
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        fake_bot = MagicMock()
        fake_bot.is_ready = True
        fake_bot.channel_id = 12345
        fake_bot.post = AsyncMock(return_value=[111])
        connector.set_bot(fake_bot)
        try:
            handler = _OUTBOUND_EVENT_REGISTRY["daily_brief"]["discord"]
            await handler(
                channel="ignored-slack-channel",
                brief_result=SimpleNamespace(markdown="# Morning Brief\n\n..."),
            )
            fake_bot.post.assert_awaited_once_with("# Morning Brief\n\n...")
        finally:
            connector.clear_bot()


class TestDiscordBotLoop:
    """Tests for the discord_bot_loop env handling and lifecycle."""

    @pytest.mark.asyncio
    async def test_returns_early_without_token(self, monkeypatch):
        from weft.scheduler import discord_bot_loop

        monkeypatch.delenv("WEFT_DISCORD_BOT_TOKEN", raising=False)
        monkeypatch.setenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", "12345")

        # Returns without raising and without starting a bot
        await discord_bot_loop(AsyncMock())

    @pytest.mark.asyncio
    async def test_returns_early_without_channel_id(self, monkeypatch):
        from weft.scheduler import discord_bot_loop

        monkeypatch.setenv("WEFT_DISCORD_BOT_TOKEN", "fake.token")
        monkeypatch.delenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", raising=False)

        await discord_bot_loop(AsyncMock())

    @pytest.mark.asyncio
    async def test_returns_early_on_non_integer_channel_id(self, monkeypatch):
        from weft.scheduler import discord_bot_loop

        monkeypatch.setenv("WEFT_DISCORD_BOT_TOKEN", "fake.token")
        monkeypatch.setenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", "not-a-number")

        await discord_bot_loop(AsyncMock())

    @pytest.mark.asyncio
    async def test_registers_and_clears_bot(self, monkeypatch):
        """Loop calls set_bot after start and clear_bot on cancel."""
        from weft.discord import connector
        from weft.scheduler import discord_bot_loop

        monkeypatch.setenv("WEFT_DISCORD_BOT_TOKEN", "fake.token")
        monkeypatch.setenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", "12345")

        fake_bot = MagicMock()
        fake_bot.start = AsyncMock()
        fake_bot.close = AsyncMock()

        connector.clear_bot()
        with patch("weft.discord.bot.Bot", return_value=fake_bot):
            task = asyncio.create_task(discord_bot_loop(AsyncMock(), interval=0))
            # Give the loop time to start + set_bot
            await asyncio.sleep(0.05)
            assert connector.get_bot() is fake_bot
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # finally block must clear the reference and close the bot
        assert connector.get_bot() is None
        fake_bot.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_message_not_registered_without_idea_dump_channel(self, monkeypatch):
        """When WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID is unset, on_message is NOT registered."""
        monkeypatch.setenv("WEFT_DISCORD_BOT_TOKEN", "fake.token")
        monkeypatch.setenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", "12345")
        monkeypatch.delenv("WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID", raising=False)

        from weft.discord.bot import Bot

        # Capture the Bot constructor call to inspect the instance
        created_bots = []

        original_init = Bot.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            created_bots.append(self)

        with patch.object(Bot, "__init__", capturing_init):
            fake_bot = MagicMock()
            fake_bot.start = AsyncMock()
            fake_bot.close = AsyncMock()

            with patch("weft.discord.bot.Bot", return_value=fake_bot) as MockBot:
                task = asyncio.create_task(
                    __import__("weft.scheduler", fromlist=["discord_bot_loop"]).discord_bot_loop(
                        AsyncMock(), interval=0
                    )
                )
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                # Bot must have been called without idea_dump_channel_id
                call_kwargs = MockBot.call_args.kwargs
                assert call_kwargs.get("idea_dump_channel_id") is None

    @pytest.mark.asyncio
    async def test_on_message_registered_with_idea_dump_channel(self, monkeypatch):
        """When WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID is set, idea_dump_channel_id is passed to Bot."""
        monkeypatch.setenv("WEFT_DISCORD_BOT_TOKEN", "fake.token")
        monkeypatch.setenv("WEFT_DISCORD_BRIEF_CHANNEL_ID", "12345")
        monkeypatch.setenv("WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID", "99999")

        fake_bot = MagicMock()
        fake_bot.start = AsyncMock()
        fake_bot.close = AsyncMock()

        with patch("weft.discord.bot.Bot", return_value=fake_bot) as MockBot:
            task = asyncio.create_task(
                __import__("weft.scheduler", fromlist=["discord_bot_loop"]).discord_bot_loop(
                    AsyncMock(), interval=0
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            call_kwargs = MockBot.call_args.kwargs
            assert call_kwargs.get("idea_dump_channel_id") == 99999

    @pytest.mark.asyncio
    async def test_on_message_skips_wrong_channel(self):
        """Bot.on_message returns without ingesting when channel doesn't match."""
        import discord as discord_mod
        from weft.discord.bot import Bot

        # Build a real Bot instance with mocked discord internals
        fake_client = MagicMock()
        fake_client.user = MagicMock()
        fake_client.user.id = 1111

        bot = object.__new__(Bot)
        bot._pool = AsyncMock()
        bot._idea_dump_channel_id = 99999
        bot._client = fake_client

        # Message from a human in a DIFFERENT channel
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 5555
        msg.channel = MagicMock()
        msg.channel.id = 11111  # wrong channel

        with patch("weft.ingest_adapters.DiscordAdapter.ingest", new_callable=AsyncMock) as mock_ingest:
            await bot.on_message(msg)
            await asyncio.sleep(0)  # let any tasks run

        mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_skips_bot_author(self):
        """Bot.on_message skips messages where message.author.bot is True."""
        from weft.discord.bot import Bot

        fake_client = MagicMock()
        fake_client.user = MagicMock()
        fake_client.user.id = 1111

        bot = object.__new__(Bot)
        bot._pool = AsyncMock()
        bot._idea_dump_channel_id = 99999
        bot._client = fake_client

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = True  # bot message
        msg.author.id = 2222
        msg.channel = MagicMock()
        msg.channel.id = 99999  # correct channel

        with patch("weft.ingest_adapters.DiscordAdapter.ingest", new_callable=AsyncMock) as mock_ingest:
            await bot.on_message(msg)
            await asyncio.sleep(0)

        mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_skips_self_message(self):
        """Bot.on_message skips messages from the bot itself (self-ingest defense)."""
        from weft.discord.bot import Bot

        bot_user_id = 1111

        fake_client = MagicMock()
        fake_client.user = MagicMock()
        fake_client.user.id = bot_user_id

        bot = object.__new__(Bot)
        bot._pool = AsyncMock()
        bot._idea_dump_channel_id = 99999
        bot._client = fake_client

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False  # technically not .bot=True, but same user ID
        msg.author.id = bot_user_id  # same as bot's own ID
        msg.channel = MagicMock()
        msg.channel.id = 99999

        with patch("weft.ingest_adapters.DiscordAdapter.ingest", new_callable=AsyncMock) as mock_ingest:
            await bot.on_message(msg)
            await asyncio.sleep(0)

        mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_calls_discord_adapter_ingest_for_valid_message(self):
        """Bot.on_message calls DiscordAdapter.ingest for a non-bot message in the target channel."""
        from weft.discord.bot import Bot
        from weft.ingest_pipeline import IngestResult

        fake_client = MagicMock()
        fake_client.user = MagicMock()
        fake_client.user.id = 1111

        bot = object.__new__(Bot)
        bot._pool = AsyncMock()
        bot._idea_dump_channel_id = 99999
        bot._client = fake_client
        # __init__ sets _channel_map; bypassed via object.__new__, so seed it
        # here. None routes through DEFAULT_CHANNEL_MAP, but the adapter is
        # mocked below so the value doesn't reach resolve_channel_mapping.
        bot._channel_map = None

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 5555  # different from bot's own ID
        msg.author.name = "testuser"
        msg.id = 777
        msg.content = "This is a great idea for the dump channel"
        msg.channel = MagicMock()
        msg.channel.id = 99999  # correct channel
        msg.created_at = None

        fake_result = IngestResult(memories_created=1)

        # The bot now routes via DiscordChannelAdapter, not the generic
        # DiscordAdapter — patch the channel-aware adapter the WIP added.
        with patch(
            "weft.discord.adapter.DiscordChannelAdapter.ingest",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_ingest:
            await bot.on_message(msg)
            # Allow the create_task to run
            await asyncio.sleep(0.05)

        mock_ingest.assert_awaited_once()
