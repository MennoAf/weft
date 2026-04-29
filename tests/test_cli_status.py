"""Tests for ingest adapters (weft/ingest_adapters.py) and smart sync integration."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.ingest_adapters import ADAPTERS, SlackAdapter, IngestAdapter
from weft.ingest_pipeline import IngestResult


@pytest.fixture(autouse=True)
def _patch_acquire_for_mock_pools():
    """Phase-2.5 wraps DB writes in weft.db.connection.acquire() in
    both ingest_pipeline.route and slack.sync. The smart-sync
    integration tests below run against AsyncMock pools that don't
    model the acquire contract — patch acquire to a no-op in both
    consumers so these tests keep exercising the routing logic.
    Real-pool coverage lives in tests/test_ingest_pipeline_acquire.py
    and tests/test_slack_sync_acquire.py."""
    @asynccontextmanager
    async def _noop_acquire(_pool):
        yield None

    with (
        patch("weft.ingest_pipeline.acquire", _noop_acquire),
        patch("weft.slack.sync.acquire", _noop_acquire),
    ):
        yield


# --- SlackAdapter pre-filtering tests ---


class TestSlackAdapterSkips:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", [
        {"ts": "1.0", "bot_id": "B123", "text": "hello"},
        {"ts": "1.0", "subtype": "bot_message", "text": "hello"},
        {"ts": "1.0", "subtype": "channel_join", "text": "joined"},
        {"ts": "1.0", "subtype": "message_changed", "text": "edited"},
        {"ts": "1.0", "subtype": "message_deleted"},
        {"ts": "1.0", "subtype": "channel_leave"},
        {"ts": "1.0", "text": ""},
        {"ts": "1.0", "text": "  "},
        {"ts": "1.0", "text": "hi"},  # too short (< 3 chars)
        {"ts": "1.0"},  # no text at all
    ])
    async def test_skip_cases(self, raw):
        adapter = SlackAdapter()
        pool = AsyncMock()

        result = await adapter.ingest(raw, pool)

        assert result is None

    @pytest.mark.asyncio
    async def test_attachment_only_message_skipped(self):
        adapter = SlackAdapter()
        pool = AsyncMock()
        raw = {"ts": "1.0", "text": None, "attachments": [{"text": "file"}]}

        result = await adapter.ingest(raw, pool)

        assert result is None


class TestSlackAdapterIngest:
    @pytest.mark.asyncio
    async def test_valid_message_creates_ingest_item(self):
        adapter = SlackAdapter()
        pool = AsyncMock()
        expected_result = IngestResult(memories_created=1)
        raw = {"ts": "1000.0", "user": "U123", "text": "Bob is the CEO of TechCorp"}

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = expected_result

            result = await adapter.ingest(raw, pool, channel="general")

        assert result is expected_result
        # Verify IngestItem was constructed correctly
        call_args = mock_process.call_args
        item = call_args[0][0]
        assert item.text == "Bob is the CEO of TechCorp"
        assert item.source == "slack"
        assert item.author == "U123"
        assert item.metadata["channel"] == "general"

    @pytest.mark.asyncio
    async def test_html_entities_decoded(self):
        adapter = SlackAdapter()
        pool = AsyncMock()
        raw = {"ts": "1.0", "user": "U1", "text": "A &amp; B &lt; C"}

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = IngestResult(memories_created=1)

            await adapter.ingest(raw, pool)

        item = mock_process.call_args[0][0]
        assert item.text == "A & B < C"

    @pytest.mark.asyncio
    async def test_missing_user_defaults_to_unknown(self):
        adapter = SlackAdapter()
        pool = AsyncMock()
        raw = {"ts": "1.0", "text": "a valid message here"}

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = IngestResult(memories_created=1)

            await adapter.ingest(raw, pool)

        item = mock_process.call_args[0][0]
        assert item.author == "unknown"

    @pytest.mark.asyncio
    async def test_process_exception_propagates(self):
        adapter = SlackAdapter()
        pool = AsyncMock()
        raw = {"ts": "1.0", "user": "U1", "text": "a valid message here"}

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process:
            mock_process.side_effect = RuntimeError("LLM timeout")

            with pytest.raises(RuntimeError, match="LLM timeout"):
                await adapter.ingest(raw, pool)


class TestAdaptersRegistry:
    def test_slack_registered(self):
        assert "slack" in ADAPTERS
        assert ADAPTERS["slack"] is SlackAdapter

    def test_slack_adapter_implements_protocol(self):
        assert isinstance(SlackAdapter(), IngestAdapter)


class TestSmartSyncIntegration:
    """Tests for smart_ingest wiring in _sync_messages."""

    @pytest.mark.asyncio
    async def test_smart_ingest_skips_flat_storage_on_success(self, tmp_path):
        from weft.slack.hash_store import SlackSyncState
        from weft.slack.sync import ChannelInfo, _sync_messages, SyncResult

        pool = AsyncMock()
        state = SlackSyncState(tmp_path / "state.json")
        channel = ChannelInfo(id="C1", name="general")
        raw_messages = [{"ts": "100.0", "user": "U1", "text": "Bob is CEO of Acme"}]
        result = SyncResult()

        mock_ingest_result = IngestResult(memories_created=1)

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process, \
             patch("weft.slack.sync.store_memory", new_callable=AsyncMock) as mock_flat:
            mock_process.return_value = mock_ingest_result

            await _sync_messages(
                channel, raw_messages, {}, pool, None,
                sync_state=state, channel_map=None, user_names={},
                result=result, smart_ingest=True,
            )

        assert result.memories_created == 1
        assert result.messages_synced == 1
        # Flat storage should NOT have been called
        mock_flat.assert_not_called()

    @pytest.mark.asyncio
    async def test_smart_ingest_falls_back_on_failure(self, tmp_path):
        from weft.slack.hash_store import SlackSyncState
        from weft.slack.sync import ChannelInfo, _sync_messages, SyncResult

        pool = AsyncMock()
        state = SlackSyncState(tmp_path / "state.json")
        channel = ChannelInfo(id="C1", name="general")
        raw_messages = [{"ts": "100.0", "user": "U1", "text": "Bob is CEO of Acme"}]
        result = SyncResult()

        mock_mem = MagicMock()
        mock_mem.id = "weft-flat-1"

        with patch("weft.ingest_adapters.process", new_callable=AsyncMock) as mock_process, \
             patch("weft.slack.sync.store_memory", new_callable=AsyncMock) as mock_flat:
            mock_process.side_effect = RuntimeError("LLM timeout")
            mock_flat.return_value = mock_mem

            await _sync_messages(
                channel, raw_messages, {}, pool, None,
                sync_state=state, channel_map=None, user_names={},
                result=result, smart_ingest=True,
            )

        assert result.memories_created == 1
        assert result.messages_synced == 1
        # Flat storage should have been called as fallback
        mock_flat.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_smart_ingest_uses_flat_only(self, tmp_path):
        from weft.slack.hash_store import SlackSyncState
        from weft.slack.sync import ChannelInfo, _sync_messages, SyncResult

        pool = AsyncMock()
        state = SlackSyncState(tmp_path / "state.json")
        channel = ChannelInfo(id="C1", name="general")
        raw_messages = [{"ts": "100.0", "user": "U1", "text": "Hello world"}]
        result = SyncResult()

        mock_mem = MagicMock()
        mock_mem.id = "weft-flat-1"

        with patch("weft.slack.sync.store_memory", new_callable=AsyncMock) as mock_flat:
            mock_flat.return_value = mock_mem

            await _sync_messages(
                channel, raw_messages, {}, pool, None,
                sync_state=state, channel_map=None, user_names={},
                result=result, smart_ingest=False,
            )

        assert result.memories_created == 1
        mock_flat.assert_called_once()
