"""Unit tests for weft.discord.config — DiscordChannelMapping + resolve_channel_mapping.

Tests cover:
- resolve_channel_mapping returns mapping for configured channel.
- resolve_channel_mapping returns None for unconfigured channel.
- Default-fallback: DEFAULT_CHANNEL_MAP populated when env var is set.
- Default-fallback: DEFAULT_CHANNEL_MAP empty when env var is not set.
- DiscordChannelMapping carries memory_type, topics, and confidence.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest

from weft.models import MemoryType


# ---------------------------------------------------------------------------
# DiscordChannelMapping shape tests
# ---------------------------------------------------------------------------


class TestDiscordChannelMappingShape:
    def test_memory_type_stored(self):
        from weft.discord.config import DiscordChannelMapping

        m = DiscordChannelMapping(memory_type=MemoryType.fact, topics=["discord", "general"])
        assert m.memory_type is MemoryType.fact

    def test_topics_stored(self):
        from weft.discord.config import DiscordChannelMapping

        m = DiscordChannelMapping(memory_type=MemoryType.preference, topics=["a", "b"])
        assert m.topics == ["a", "b"]

    def test_default_confidence(self):
        from weft.discord.config import DiscordChannelMapping

        m = DiscordChannelMapping(memory_type=MemoryType.fact)
        assert m.confidence == 0.7

    def test_custom_confidence(self):
        from weft.discord.config import DiscordChannelMapping

        m = DiscordChannelMapping(memory_type=MemoryType.decision, confidence=0.9)
        assert m.confidence == 0.9

    def test_default_topics_empty_list(self):
        from weft.discord.config import DiscordChannelMapping

        m = DiscordChannelMapping(memory_type=MemoryType.fact)
        assert m.topics == []


# ---------------------------------------------------------------------------
# resolve_channel_mapping tests
# ---------------------------------------------------------------------------


class TestResolveChannelMapping:
    def test_returns_mapping_for_configured_channel(self):
        """A channel present in the map returns its DiscordChannelMapping."""
        from weft.discord.config import DiscordChannelMapping, resolve_channel_mapping

        channel_map = {
            "111222333": DiscordChannelMapping(
                memory_type=MemoryType.fact, topics=["discord", "idea-dump"]
            ),
        }
        result = resolve_channel_mapping("111222333", channel_map)
        assert result is not None
        assert result.memory_type is MemoryType.fact
        assert "idea-dump" in result.topics

    def test_returns_none_for_unconfigured_channel(self):
        """A channel not in the map returns None — caller should skip it."""
        from weft.discord.config import DiscordChannelMapping, resolve_channel_mapping

        channel_map = {
            "111222333": DiscordChannelMapping(memory_type=MemoryType.fact),
        }
        result = resolve_channel_mapping("999999999", channel_map)
        assert result is None

    def test_empty_map_always_returns_none(self):
        """An empty channel map means all channels are ignored."""
        from weft.discord.config import resolve_channel_mapping

        result = resolve_channel_mapping("111222333", {})
        assert result is None

    def test_different_memory_types_per_channel(self):
        """Two channels can have different memory types."""
        from weft.discord.config import DiscordChannelMapping, resolve_channel_mapping

        channel_map = {
            "aaa": DiscordChannelMapping(memory_type=MemoryType.fact, topics=["ideas"]),
            "bbb": DiscordChannelMapping(memory_type=MemoryType.preference, topics=["prefs"]),
        }
        result_a = resolve_channel_mapping("aaa", channel_map)
        result_b = resolve_channel_mapping("bbb", channel_map)

        assert result_a is not None
        assert result_a.memory_type is MemoryType.fact
        assert result_b is not None
        assert result_b.memory_type is MemoryType.preference

    def test_uses_default_channel_map_when_none_supplied(self):
        """When channel_map=None, resolve uses DEFAULT_CHANNEL_MAP."""
        from weft.discord import config as discord_config

        # Temporarily replace DEFAULT_CHANNEL_MAP with a known fixture.
        from weft.discord.config import DiscordChannelMapping, resolve_channel_mapping

        test_map = {
            "fixture_channel": DiscordChannelMapping(memory_type=MemoryType.decision),
        }
        original = discord_config.DEFAULT_CHANNEL_MAP
        try:
            discord_config.DEFAULT_CHANNEL_MAP = test_map
            result = resolve_channel_mapping("fixture_channel", None)
        finally:
            discord_config.DEFAULT_CHANNEL_MAP = original

        assert result is not None
        assert result.memory_type is MemoryType.decision


# ---------------------------------------------------------------------------
# DEFAULT_CHANNEL_MAP env-var bootstrap
# ---------------------------------------------------------------------------


class TestDefaultChannelMapEnvVar:
    def test_default_map_populated_when_env_var_set(self):
        """When WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID is set, DEFAULT_CHANNEL_MAP
        contains an entry for that channel with memory_type=fact."""
        channel_id = "555666777"

        with patch.dict(os.environ, {"WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID": channel_id}):
            # Reload the module to re-run the module-level env-var logic.
            import weft.discord.config as m

            importlib.reload(m)
            try:
                assert channel_id in m.DEFAULT_CHANNEL_MAP
                entry = m.DEFAULT_CHANNEL_MAP[channel_id]
                assert entry.memory_type is MemoryType.fact
                assert "idea-dump" in entry.topics
            finally:
                importlib.reload(m)  # Restore module state

    def test_default_map_empty_without_env_var(self):
        """Without either *_CHANNEL_ID env var, DEFAULT_CHANNEL_MAP is empty."""
        env = {
            k: v for k, v in os.environ.items()
            if k not in {
                "WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID",
                "WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID",
            }
        }

        with patch.dict(os.environ, env, clear=True):
            import weft.discord.config as m

            importlib.reload(m)
            try:
                assert m.DEFAULT_CHANNEL_MAP == {}
            finally:
                importlib.reload(m)  # Restore module state

    def test_brain_dump_seeded_when_env_var_set(self):
        """WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID seeds an entry with
        memory_type=fact + topics=['discord', 'brain-dump']."""
        channel_id = "1507757010857496618"

        with patch.dict(os.environ, {"WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID": channel_id}):
            import weft.discord.config as m

            importlib.reload(m)
            try:
                assert channel_id in m.DEFAULT_CHANNEL_MAP
                entry = m.DEFAULT_CHANNEL_MAP[channel_id]
                assert entry.memory_type is MemoryType.fact
                assert entry.topics == ["discord", "brain-dump"]
            finally:
                importlib.reload(m)  # Restore module state

    def test_both_channels_seeded_independently(self):
        """Idea-dump and brain-dump entries coexist when both env vars set."""
        env = {
            "WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID": "111",
            "WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID": "222",
        }

        with patch.dict(os.environ, env):
            import weft.discord.config as m

            importlib.reload(m)
            try:
                assert "111" in m.DEFAULT_CHANNEL_MAP
                assert "222" in m.DEFAULT_CHANNEL_MAP
                assert m.DEFAULT_CHANNEL_MAP["111"].topics == ["discord", "idea-dump"]
                assert m.DEFAULT_CHANNEL_MAP["222"].topics == ["discord", "brain-dump"]
            finally:
                importlib.reload(m)  # Restore module state
