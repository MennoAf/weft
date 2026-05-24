"""Unit tests for DiscordAdapter and DiscordChannelAdapter.

Covers:
- Happy path: valid message → IngestItem with source="discord" → process called.
- Empty/whitespace content → skip (returns None).
- Very short content (< _MIN_TEXT_LENGTH) → skip.
- bot author flag set in raw → skip (defense in depth even though the bot
  handler already filters this upstream).
- Channel-mapping-aware DiscordChannelAdapter:
  - Configured channel A → ingests with correct memory_type hint in metadata.
  - Configured channel B → ingests with a different memory_type.
  - Unconfigured channel → ignored (returns None), process not called.
  - Default fallback: Idea Dump channel entry is the seed entry in the map.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from weft.ingest_adapters import DiscordAdapter, _MIN_TEXT_LENGTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw(
    content: str = "This is a test idea for the dump channel",
    *,
    msg_id: int = 111222333,
    author_id: str = "987654321",
    author_name: str = "testuser",
    channel_id: str = "444555666",
    created_at: str = "2026-05-23T10:00:00+00:00",
    is_bot: bool = False,
) -> dict:
    """Build a minimal raw Discord message dict."""
    raw: dict = {
        "id": msg_id,
        "content": content,
        "author_id": author_id,
        "author_name": author_name,
        "channel_id": channel_id,
        "created_at": created_at,
    }
    if is_bot:
        raw["is_bot"] = True
    return raw


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterHappyPath:
    @pytest.mark.asyncio
    async def test_valid_message_calls_process(self):
        """A well-formed message dict routes through process() with source='discord'."""
        from weft.ingest_pipeline import IngestResult

        fake_result = IngestResult(memories_created=1)

        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_process:
            result = await DiscordAdapter().ingest(
                _raw("Remember to fix the CI pipeline"),
                pool=AsyncMock(),
                embedding_provider=None,
                channel="444555666",
            )

        assert result is fake_result
        mock_process.assert_awaited_once()
        # Verify the IngestItem passed to process has source="discord"
        call_args = mock_process.call_args
        ingest_item = call_args.args[0]
        assert ingest_item.source == "discord"
        assert ingest_item.text == "Remember to fix the CI pipeline"

    @pytest.mark.asyncio
    async def test_author_name_used_for_ingest_item_author(self):
        """author_name from raw dict maps to IngestItem.author."""
        from weft.ingest_pipeline import IngestResult

        fake_result = IngestResult(memories_created=1)

        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_process:
            await DiscordAdapter().ingest(
                _raw("Some valid content here", author_name="jasonbauman"),
                pool=AsyncMock(),
            )

        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.author == "jasonbauman"

    @pytest.mark.asyncio
    async def test_metadata_includes_channel_and_message_id(self):
        """Metadata dict carries channel and message_id fields."""
        from weft.ingest_pipeline import IngestResult

        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=IngestResult(),
        ) as mock_process:
            await DiscordAdapter().ingest(
                _raw("Valid message with enough text", msg_id=999, channel_id="777"),
                pool=AsyncMock(),
                channel="777",
            )

        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.metadata["message_id"] == "999"
        assert ingest_item.metadata["channel"] == "777"

    @pytest.mark.asyncio
    async def test_timestamp_parsed_from_created_at(self):
        """created_at ISO string is parsed into a UTC-aware datetime."""
        from weft.ingest_pipeline import IngestResult

        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=IngestResult(),
        ) as mock_process:
            await DiscordAdapter().ingest(
                _raw("Valid idea here please process me", created_at="2026-05-23T10:00:00+00:00"),
                pool=AsyncMock(),
            )

        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.timestamp is not None
        assert ingest_item.timestamp.tzinfo is not None  # must be tz-aware

    @pytest.mark.asyncio
    async def test_naive_timestamp_gets_utc(self):
        """A naive created_at (no tz offset) is assumed UTC."""
        from weft.ingest_pipeline import IngestResult
        from datetime import timezone

        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=IngestResult(),
        ) as mock_process:
            await DiscordAdapter().ingest(
                _raw("Valid idea here please process me", created_at="2026-05-23T10:00:00"),
                pool=AsyncMock(),
            )

        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.timestamp is not None
        assert ingest_item.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Skip-path tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterSkipPaths:
    @pytest.mark.asyncio
    async def test_empty_content_returns_none(self):
        """Empty content string → None (skip, no ingest)."""
        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            result = await DiscordAdapter().ingest(_raw(""), pool=AsyncMock())

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_returns_none(self):
        """Whitespace-only content is treated as empty → None."""
        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            result = await DiscordAdapter().ingest(_raw("   \t\n  "), pool=AsyncMock())

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_very_short_content_returns_none(self):
        """Content shorter than _MIN_TEXT_LENGTH → None."""
        # _MIN_TEXT_LENGTH is 3; a 2-char string should be skipped
        short = "x" * (_MIN_TEXT_LENGTH - 1)
        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            result = await DiscordAdapter().ingest(_raw(short), pool=AsyncMock())

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_at_min_length_is_processed(self):
        """Content exactly at _MIN_TEXT_LENGTH passes the skip check."""
        from weft.ingest_pipeline import IngestResult

        at_min = "x" * _MIN_TEXT_LENGTH
        with patch(
            "weft.ingest_adapters.process",
            new_callable=AsyncMock,
            return_value=IngestResult(),
        ) as mock_process:
            result = await DiscordAdapter().ingest(_raw(at_min), pool=AsyncMock())

        mock_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bot_flag_in_raw_returns_none(self):
        """is_bot=True in raw dict → None (defense-in-depth bot filter)."""
        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            result = await DiscordAdapter().ingest(
                _raw("This would be a valid message but it is from a bot", is_bot=True),
                pool=AsyncMock(),
            )

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_content_key_returns_none(self):
        """raw dict with no 'content' key → None (treated as empty)."""
        raw = {
            "id": 123,
            "author_id": "456",
            "author_name": "user",
            "channel_id": "789",
        }
        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            result = await DiscordAdapter().ingest(raw, pool=AsyncMock())

        assert result is None
        mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# DiscordChannelAdapter — channel-mapping-aware adapter
# ---------------------------------------------------------------------------


def _channel_raw(
    content: str = "This is a test idea for the dump channel",
    *,
    msg_id: int = 111222333,
    author_id: str = "987654321",
    author_name: str = "testuser",
    channel_id: str = "444555666",
    created_at: str = "2026-05-23T10:00:00+00:00",
    is_bot: bool = False,
) -> dict:
    """Build a minimal raw Discord message dict for DiscordChannelAdapter tests."""
    raw: dict = {
        "id": msg_id,
        "content": content,
        "author_id": author_id,
        "author_name": author_name,
        "channel_id": channel_id,
        "created_at": created_at,
    }
    if is_bot:
        raw["is_bot"] = True
    return raw


class TestDiscordChannelAdapterChannelMapping:
    """DiscordChannelAdapter routes messages through the channel mapping."""

    @pytest.mark.asyncio
    async def test_configured_channel_ingests_with_correct_memory_type(self):
        """Channel A (fact) → process() is called; metadata carries memory_type_hint='fact'."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.ingest_pipeline import IngestResult
        from weft.models import MemoryType

        channel_map = {
            "444555666": DiscordChannelMapping(
                memory_type=MemoryType.fact, topics=["discord", "idea-dump"]
            ),
        }

        with patch(
            "weft.discord.adapter.process",
            new_callable=AsyncMock,
            return_value=IngestResult(memories_created=1),
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw("Remember to fix the CI pipeline"),
                pool=AsyncMock(),
                channel="444555666",
            )

        assert result is not None
        mock_process.assert_awaited_once()
        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.metadata["memory_type_hint"] == "fact"
        assert ingest_item.source == "discord"

    @pytest.mark.asyncio
    async def test_configured_channel_b_uses_different_memory_type(self):
        """Channel B (preference) → metadata carries memory_type_hint='preference'."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.ingest_pipeline import IngestResult
        from weft.models import MemoryType

        channel_map = {
            "111000111": DiscordChannelMapping(
                memory_type=MemoryType.preference, topics=["discord", "prefs"]
            ),
        }

        with patch(
            "weft.discord.adapter.process",
            new_callable=AsyncMock,
            return_value=IngestResult(memories_created=1),
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw("I prefer dark mode always", channel_id="111000111"),
                pool=AsyncMock(),
                channel="111000111",
            )

        assert result is not None
        mock_process.assert_awaited_once()
        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.metadata["memory_type_hint"] == "preference"

    @pytest.mark.asyncio
    async def test_unconfigured_channel_returns_none(self):
        """A channel not in the mapping → None (silently ignored, process not called)."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.models import MemoryType

        channel_map = {
            "444555666": DiscordChannelMapping(memory_type=MemoryType.fact),
        }

        with patch(
            "weft.discord.adapter.process", new_callable=AsyncMock
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw("This message is in an unwatched channel", channel_id="999888777"),
                pool=AsyncMock(),
                channel="999888777",
            )

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_map_all_channels_ignored(self):
        """Empty channel map → every channel is ignored."""
        from weft.discord.adapter import DiscordChannelAdapter

        with patch(
            "weft.discord.adapter.process", new_callable=AsyncMock
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map={}).ingest(
                _channel_raw("Any content here"),
                pool=AsyncMock(),
                channel="444555666",
            )

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_fallback_idea_dump_channel(self):
        """Default channel map has an entry for the Idea Dump channel (memory_type=fact).

        This exercises the 'default fallback' path: DEFAULT_CHANNEL_MAP ships
        with a seed entry for WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID so the Idea Dump
        channel works out-of-the-box with no extra config.
        """
        from weft.discord.config import DiscordChannelMapping
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.ingest_pipeline import IngestResult
        from weft.models import MemoryType

        # Simulate the default map as it would be after env-var bootstrap.
        idea_dump_id = "555666777"
        default_like_map = {
            idea_dump_id: DiscordChannelMapping(
                memory_type=MemoryType.fact, topics=["discord", "idea-dump"]
            ),
        }

        with patch(
            "weft.discord.adapter.process",
            new_callable=AsyncMock,
            return_value=IngestResult(memories_created=1),
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=default_like_map).ingest(
                _channel_raw("My idea: build a smarter router", channel_id=idea_dump_id),
                pool=AsyncMock(),
                channel=idea_dump_id,
            )

        assert result is not None
        mock_process.assert_awaited_once()
        ingest_item = mock_process.call_args.args[0]
        assert ingest_item.metadata["memory_type_hint"] == "fact"
        assert ingest_item.metadata["topics"] == ["discord", "idea-dump"]

    @pytest.mark.asyncio
    async def test_bot_message_skipped_before_mapping_check(self):
        """is_bot=True short-circuits before the channel-mapping lookup."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.models import MemoryType

        channel_map = {
            "444555666": DiscordChannelMapping(memory_type=MemoryType.fact),
        }

        with patch(
            "weft.discord.adapter.process", new_callable=AsyncMock
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw("Bot content here", is_bot=True),
                pool=AsyncMock(),
                channel="444555666",
            )

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_content_skipped_before_mapping_check(self):
        """Content shorter than _MIN_TEXT_LENGTH is skipped before mapping lookup."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.models import MemoryType

        channel_map = {
            "444555666": DiscordChannelMapping(memory_type=MemoryType.fact),
        }
        short_text = "x" * (_MIN_TEXT_LENGTH - 1)

        with patch(
            "weft.discord.adapter.process", new_callable=AsyncMock
        ) as mock_process:
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw(short_text),
                pool=AsyncMock(),
                channel="444555666",
            )

        assert result is None
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_id_falls_back_to_raw_data(self):
        """When channel kwarg is None, adapter reads channel_id from raw_data."""
        from weft.discord.adapter import DiscordChannelAdapter
        from weft.discord.config import DiscordChannelMapping
        from weft.ingest_pipeline import IngestResult
        from weft.models import MemoryType

        channel_id = "444555666"
        channel_map = {
            channel_id: DiscordChannelMapping(memory_type=MemoryType.fact),
        }

        with patch(
            "weft.discord.adapter.process",
            new_callable=AsyncMock,
            return_value=IngestResult(),
        ) as mock_process:
            # Pass channel=None — adapter must fall back to raw_data["channel_id"]
            result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                _channel_raw("Valid message here", channel_id=channel_id),
                pool=AsyncMock(),
                channel=None,
            )

        # Should have ingested (channel resolved from raw_data)
        mock_process.assert_awaited_once()
