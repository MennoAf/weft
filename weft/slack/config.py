"""Channel-to-memory-type mapping and Slack sync configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from weft.models import MemoryType


@dataclass
class ChannelMapping:
    """Maps a Slack channel to a Weft memory type and topics."""

    memory_type: MemoryType
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.7


DEFAULT_CHANNEL_MAP: dict[str, ChannelMapping] = {
    "ext-birdy-grey-seo": ChannelMapping(
        MemoryType.fact, ["slack", "clients", "birdy-grey"]
    ),
    "random": ChannelMapping(MemoryType.fact, ["slack", "random"]),
    "claude": ChannelMapping(MemoryType.fact, ["slack", "ai", "claude"]),
    "general": ChannelMapping(MemoryType.fact, ["slack", "general"]),
    "ai-automation-": ChannelMapping(
        MemoryType.fact, ["slack", "ai", "automation"]
    ),
}

# Channels to skip during sync
DEFAULT_EXCLUDED_CHANNELS: set[str] = {"brain-dump"}


def resolve_channel_mapping(
    channel_name: str,
    channel_map: dict[str, ChannelMapping] | None = None,
) -> ChannelMapping:
    """Find the mapping for a channel name, falling back to a generic default."""
    if channel_map is None:
        channel_map = DEFAULT_CHANNEL_MAP

    if channel_name in channel_map:
        return channel_map[channel_name]

    return ChannelMapping(MemoryType.fact, ["slack", channel_name])
