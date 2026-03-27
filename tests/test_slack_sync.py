"""Tests for the Slack sync engine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.slack.hash_store import SlackSyncState
from weft.slack.sync import (
    ChannelInfo,
    SyncResult,
    _add_ingest_reaction,
    _store_message_memory,
    _sync_messages,
    sync_slack_messages,
)
from weft.slack.config import ChannelMapping
from weft.models import MemoryType


def _make_message(ts, text="hello", user="U123", **kwargs):
    """Helper to build a raw Slack message dict."""
    msg = {"text": text, "user": user, "ts": ts}
    msg.update(kwargs)
    return msg


def _make_thread_reply(ts, thread_ts, text="reply", user="U456"):
    return {"text": text, "user": user, "ts": ts, "thread_ts": thread_ts}


@pytest.fixture
def sync_state(tmp_path):
    return SlackSyncState(tmp_path / "state.json")


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    # Make store_memory return a mock with an id
    return pool


@pytest.fixture
def channel():
    return ChannelInfo(id="C123", name="general")


class TestSyncResult:
    def test_default_values(self):
        r = SyncResult()
        assert r.channels_synced == 0
        assert r.messages_found == 0
        assert r.messages_synced == 0
        assert r.memories_created == 0


class TestChannelInfo:
    def test_creation(self):
        ch = ChannelInfo(id="C123", name="general")
        assert ch.id == "C123"
        assert ch.name == "general"


class TestSyncMessages:
    @pytest.mark.asyncio
    async def test_skips_bot_messages(self, mock_pool, channel, sync_state):
        messages = [
            _make_message("100.0", bot_id="B123"),
            _make_message("200.0"),  # real message
        ]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert result.messages_found == 1  # bot message not counted
        assert result.messages_synced == 1

    @pytest.mark.asyncio
    async def test_skips_subtypes(self, mock_pool, channel, sync_state):
        messages = [
            _make_message("100.0", subtype="channel_join"),
            _make_message("200.0"),
        ]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert result.messages_found == 1
        assert result.messages_synced == 1

    @pytest.mark.asyncio
    async def test_skips_thread_replies(self, mock_pool, channel, sync_state):
        messages = [
            _make_message("100.0"),  # parent
            _make_thread_reply("200.0", "100.0"),  # reply — should be skipped
        ]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert result.messages_found == 1  # only parent counted

    @pytest.mark.asyncio
    async def test_attaches_thread_replies(self, mock_pool, channel, sync_state):
        parent_ts = "100.0"
        messages = [_make_message(parent_ts, reply_count=1)]
        threads = {parent_ts: [_make_thread_reply("200.0", parent_ts)]}
        result = SyncResult()

        stored_content = []

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            async def capture_store(pool, create, embedding=None):
                stored_content.append(create.content)
                return mock_mem

            mock_store.side_effect = capture_store

            await _sync_messages(
                channel, messages, threads, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert result.messages_synced == 1
        assert len(stored_content) == 1
        assert "Thread (1 replies)" in stored_content[0]

    @pytest.mark.asyncio
    async def test_skips_unchanged_messages(self, mock_pool, channel, sync_state):
        # Pre-populate sync state with a previously synced message
        sync_state.update_message("C123", "100.0", ["weft-old"], edited_ts=None)

        messages = [_make_message("100.0")]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        mock_store.assert_not_called()
        assert result.messages_skipped == 1

    @pytest.mark.asyncio
    async def test_updates_edited_messages(self, mock_pool, channel, sync_state):
        # Previously synced with no edit
        sync_state.update_message("C123", "100.0", ["weft-old"], edited_ts=None)

        # Now the message was edited
        messages = [_make_message("100.0", edited={"user": "U123", "ts": "150.0"})]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store, \
             patch("weft.slack.sync._archive_memory") as mock_archive:
            mock_mem = MagicMock()
            mock_mem.id = "weft-new"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert result.messages_updated == 1
        assert result.memories_archived == 1
        mock_archive.assert_called_once_with(mock_pool, "weft-old")

    @pytest.mark.asyncio
    async def test_updates_channel_cursor(self, mock_pool, channel, sync_state):
        messages = [
            _make_message("100.0"),
            _make_message("300.0"),
            _make_message("200.0"),
        ]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert sync_state.get_last_sync_ts("C123") == "300.0"

    @pytest.mark.asyncio
    async def test_uses_channel_mapping(self, mock_pool, channel, sync_state):
        messages = [_make_message("100.0")]
        result = SyncResult()
        stored_topics = []

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            async def capture_store(pool, create, embedding=None):
                stored_topics.append(create.topic)
                return mock_mem

            mock_store.side_effect = capture_store

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert "slack" in stored_topics[0]
        assert "general" in stored_topics[0]
        assert "channel:general" in stored_topics[0]

    @pytest.mark.asyncio
    async def test_reaction_topics(self, mock_pool, channel, sync_state):
        messages = [_make_message(
            "100.0",
            reactions=[{"name": "fire", "count": 2}],
        )]
        result = SyncResult()
        stored_topics = []

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            async def capture_store(pool, create, embedding=None):
                stored_topics.append(create.topic)
                return mock_mem

            mock_store.side_effect = capture_store

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
            )

        assert "reaction:fire" in stored_topics[0]


class TestSyncSlackMessages:
    @pytest.mark.asyncio
    async def test_multi_channel_sync(self, mock_pool, sync_state):
        channels = [
            ChannelInfo(id="C1", name="general"),
            ChannelInfo(id="C2", name="random"),
        ]
        messages = {
            "C1": [_make_message("100.0")],
            "C2": [_make_message("200.0")],
        }

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            result = await sync_slack_messages(
                mock_pool, channels, messages,
                sync_state=sync_state,
            )

        assert result.channels_synced == 2
        assert result.messages_synced == 2

    @pytest.mark.asyncio
    async def test_empty_channel(self, mock_pool, sync_state):
        channels = [ChannelInfo(id="C1", name="general")]
        messages = {"C1": []}

        result = await sync_slack_messages(
            mock_pool, channels, messages, sync_state=sync_state,
        )

        assert result.channels_synced == 1
        assert result.messages_synced == 0

    @pytest.mark.asyncio
    async def test_saves_state(self, mock_pool, tmp_path):
        state = SlackSyncState(tmp_path / "state.json")
        channels = [ChannelInfo(id="C1", name="general")]
        messages = {"C1": [_make_message("100.0")]}

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await sync_slack_messages(
                mock_pool, channels, messages, sync_state=state,
            )

        # Verify state was saved to disk
        assert (tmp_path / "state.json").exists()
        saved = json.loads((tmp_path / "state.json").read_text())
        assert "C1" in saved["channels"]


import json

from weft.slack.sync import sync_slack_sdk


class TestSyncSlackSdkScopeFallback:
    """Test that sync_slack_sdk falls back to public channels when groups:read is missing."""

    @pytest.mark.asyncio
    async def test_falls_back_to_public_on_missing_groups_read(self, mock_pool):
        """When conversations_list fails with missing_scope groups:read,
        retry with public_channel only."""
        from slack_sdk.errors import SlackApiError

        mock_client_cls = AsyncMock()
        mock_client = mock_client_cls.return_value

        # First call (private+public) raises missing_scope error
        scope_error = SlackApiError(
            message="missing_scope",
            response=MagicMock(data={"error": "missing_scope", "needed": "groups:read"}),
        )
        # Second call (public only) succeeds
        success_resp = {
            "channels": [{"id": "C1", "name": "general"}],
            "response_metadata": {"next_cursor": ""},
        }
        mock_client.conversations_list = AsyncMock(
            side_effect=[scope_error, success_resp]
        )
        mock_client.users_list = AsyncMock(
            return_value={"members": [], "response_metadata": {"next_cursor": ""}}
        )
        mock_client.conversations_history = AsyncMock(
            return_value={"messages": [], "response_metadata": {"next_cursor": ""}}
        )

        with patch("slack_sdk.web.async_client.AsyncWebClient", return_value=mock_client):
            result = await sync_slack_sdk(mock_pool, "REDACTED")

        assert result.channels_synced == 1
        # Verify it called conversations_list twice
        assert mock_client.conversations_list.call_count == 2
        # Second call should use public_channel only
        second_call = mock_client.conversations_list.call_args_list[1]
        assert second_call.kwargs.get("types") == "public_channel"

    @pytest.mark.asyncio
    async def test_raises_non_scope_errors(self, mock_pool):
        """Non-scope errors from conversations_list should propagate."""
        mock_client = AsyncMock()
        mock_client.conversations_list = AsyncMock(
            side_effect=RuntimeError("network failure")
        )

        with patch("slack_sdk.web.async_client.AsyncWebClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="network failure"):
                await sync_slack_sdk(mock_pool, "REDACTED")


class TestAddIngestReaction:
    """Tests for the _add_ingest_reaction helper."""

    @pytest.mark.asyncio
    async def test_adds_reaction_on_success(self):
        client = AsyncMock()
        client.reactions_add.return_value = {"ok": True}

        result = await _add_ingest_reaction(client, "C123", "100.0", "brain")

        assert result is True
        client.reactions_add.assert_called_once_with(
            channel="C123", name="brain", timestamp="100.0",
        )

    @pytest.mark.asyncio
    async def test_handles_already_reacted(self):
        """already_reacted is not an error — returns True."""
        client = AsyncMock()
        client.reactions_add.return_value = {"ok": False, "error": "already_reacted"}

        result = await _add_ingest_reaction(client, "C123", "100.0")

        assert result is True

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """API errors return False but don't raise."""
        client = AsyncMock()
        client.reactions_add.return_value = {"ok": False, "error": "no_permission"}

        result = await _add_ingest_reaction(client, "C123", "100.0")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Network errors return False but don't raise."""
        client = AsyncMock()
        client.reactions_add.side_effect = RuntimeError("network")

        result = await _add_ingest_reaction(client, "C123", "100.0")

        assert result is False


class TestIngestReactionIntegration:
    """Test that reactions are added during message sync."""

    @pytest.mark.asyncio
    async def test_reaction_added_on_successful_sync(
        self, mock_pool, channel, sync_state,
    ):
        """When slack_client is provided and react_on_ingest=True, reactions fire."""
        messages = [_make_message("100.0")]
        result = SyncResult()
        mock_client = AsyncMock()
        mock_client.reactions_add.return_value = {"ok": True}

        with patch("weft.slack.sync.store_memory") as mock_store, \
             patch("weft.config.load_config") as mock_config:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem
            mock_cfg = MagicMock()
            mock_cfg.slack_sync.react_on_ingest = True
            mock_cfg.slack_sync.ingest_reaction_emoji = "brain"
            mock_config.return_value = mock_cfg

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
                slack_client=mock_client,
            )

        assert result.messages_synced == 1
        mock_client.reactions_add.assert_called_once_with(
            channel="C123", name="brain", timestamp="100.0",
        )

    @pytest.mark.asyncio
    async def test_no_reaction_without_client(
        self, mock_pool, channel, sync_state,
    ):
        """When slack_client is None (MCP mode), no reactions are attempted."""
        messages = [_make_message("100.0")]
        result = SyncResult()

        with patch("weft.slack.sync.store_memory") as mock_store:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
                slack_client=None,
            )

        assert result.messages_synced == 1

    @pytest.mark.asyncio
    async def test_reaction_disabled_by_config(
        self, mock_pool, channel, sync_state,
    ):
        """When react_on_ingest=False, no reactions fire."""
        messages = [_make_message("100.0")]
        result = SyncResult()
        mock_client = AsyncMock()

        with patch("weft.slack.sync.store_memory") as mock_store, \
             patch("weft.config.load_config") as mock_config:
            mock_mem = MagicMock()
            mock_mem.id = "weft-test1"
            mock_store.return_value = mock_mem
            mock_cfg = MagicMock()
            mock_cfg.slack_sync.react_on_ingest = False
            mock_cfg.slack_sync.ingest_reaction_emoji = "brain"
            mock_config.return_value = mock_cfg

            await _sync_messages(
                channel, messages, {}, mock_pool, None,
                sync_state=sync_state, channel_map=None,
                user_names={}, result=result,
                slack_client=mock_client,
            )

        assert result.messages_synced == 1
        mock_client.reactions_add.assert_not_called()
