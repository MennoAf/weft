"""Tests for the Slack sync state tracker."""

import json

import pytest

from weft.slack.hash_store import SlackSyncState


@pytest.fixture
def state(tmp_path):
    return SlackSyncState(tmp_path / "slack_state.json")


class TestSlackSyncState:
    def test_initial_state(self, state):
        assert state.message_count == 0
        assert state.channel_count == 0

    def test_last_sync_ts(self, state):
        assert state.get_last_sync_ts("C123") is None
        state.set_last_sync_ts("C123", "1709740800.000000")
        assert state.get_last_sync_ts("C123") == "1709740800.000000"

    def test_message_tracking(self, state):
        assert state.get_memory_ids("C123", "1709740800.000000") == []
        state.update_message("C123", "1709740800.000000", ["weft-abc", "weft-def"])
        assert state.get_memory_ids("C123", "1709740800.000000") == ["weft-abc", "weft-def"]

    def test_edited_ts(self, state):
        assert state.get_edited_ts("C123", "1709740800.000000") is None
        state.update_message(
            "C123", "1709740800.000000", ["weft-abc"],
            edited_ts="1709740900.000000",
        )
        assert state.get_edited_ts("C123", "1709740800.000000") == "1709740900.000000"

    def test_remove_message(self, state):
        state.update_message("C123", "1709740800.000000", ["weft-abc"])
        assert state.message_count == 1
        state.remove_message("C123", "1709740800.000000")
        assert state.message_count == 0
        assert state.get_memory_ids("C123", "1709740800.000000") == []

    def test_persistence(self, tmp_path):
        path = tmp_path / "state.json"
        s1 = SlackSyncState(path)
        s1.set_last_sync_ts("C123", "1709740800.000000")
        s1.update_message("C123", "1709740800.000000", ["weft-abc"])
        s1.save()

        s2 = SlackSyncState(path)
        assert s2.get_last_sync_ts("C123") == "1709740800.000000"
        assert s2.get_memory_ids("C123", "1709740800.000000") == ["weft-abc"]

    def test_corrupted_file(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("not json", encoding="utf-8")
        state = SlackSyncState(path)
        assert state.message_count == 0

    def test_all_message_keys(self, state):
        state.update_message("C123", "100.0", ["weft-1"])
        state.update_message("C123", "200.0", ["weft-2"])
        state.update_message("C456", "300.0", ["weft-3"])

        all_keys = state.all_message_keys()
        assert len(all_keys) == 3

        c123_keys = state.all_message_keys("C123")
        assert len(c123_keys) == 2
        assert all(k.startswith("C123:") for k in c123_keys)

        c456_keys = state.all_message_keys("C456")
        assert len(c456_keys) == 1

    def test_update_overwrites(self, state):
        state.update_message("C123", "100.0", ["weft-old"])
        state.update_message("C123", "100.0", ["weft-new"])
        assert state.get_memory_ids("C123", "100.0") == ["weft-new"]
        assert state.message_count == 1
