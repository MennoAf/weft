"""Tests for the Slack channel mapping config."""

from weft.models import MemoryType
from weft.slack.config import (
    DEFAULT_CHANNEL_MAP,
    DEFAULT_EXCLUDED_CHANNELS,
    ChannelMapping,
    resolve_channel_mapping,
)


class TestChannelMapping:
    def test_known_channel(self):
        mapping = resolve_channel_mapping("general")
        assert mapping.memory_type == MemoryType.fact
        assert "slack" in mapping.topics
        assert "general" in mapping.topics

    def test_client_channel(self):
        mapping = resolve_channel_mapping("ext-birdy-grey-seo")
        assert mapping.memory_type == MemoryType.fact
        assert "clients" in mapping.topics
        assert "birdy-grey" in mapping.topics

    def test_ai_channel(self):
        mapping = resolve_channel_mapping("claude")
        assert "ai" in mapping.topics
        assert "claude" in mapping.topics

    def test_automation_channel(self):
        mapping = resolve_channel_mapping("ai-automation-")
        assert "automation" in mapping.topics

    def test_unknown_channel_gets_default(self):
        mapping = resolve_channel_mapping("some-new-channel")
        assert mapping.memory_type == MemoryType.fact
        assert "slack" in mapping.topics
        assert "some-new-channel" in mapping.topics

    def test_custom_channel_map(self):
        custom = {"ops": ChannelMapping(MemoryType.fact, ["slack", "ops"], confidence=0.9)}
        mapping = resolve_channel_mapping("ops", custom)
        assert mapping.confidence == 0.9
        assert "ops" in mapping.topics

    def test_brain_dump_excluded(self):
        assert "brain-dump" in DEFAULT_EXCLUDED_CHANNELS

    def test_default_map_has_all_channels(self):
        expected = {"ext-birdy-grey-seo", "random", "claude", "general", "ai-automation-"}
        assert expected == set(DEFAULT_CHANNEL_MAP.keys())
