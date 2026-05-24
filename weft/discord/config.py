"""Channel-to-memory-type mapping for Discord ingest.

Mirrors the shape of weft/slack/config.py ChannelMapping — a dataclass that
maps Discord channel ID (snowflake string) to a Weft memory type, with a
lookup helper and a default-fallback policy.

Design contract (mirrors Slack):
- DiscordChannelMapping: memory_type, topics, confidence
- DEFAULT_CHANNEL_MAP: seed mapping for the Idea Dump channel (loaded from
  WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID env var at module load time, if set)
- resolve_channel_mapping(channel_id, channel_map) → DiscordChannelMapping | None
  Returns None for channels not in the mapping (caller should skip/ignore).
  Unlike Slack, Discord ingest is opt-in: only explicitly configured channels
  are ingested; unconfigured channels are silently dropped.

Adding a new watched channel requires ONLY config changes:
  DEFAULT_CHANNEL_MAP["<channel_id>"] = DiscordChannelMapping(MemoryType.preference, [...])
  — no code changes needed anywhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from weft.models import MemoryType


@dataclass
class DiscordChannelMapping:
    """Maps a Discord channel to a Weft memory type and topics."""

    memory_type: MemoryType
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.7


# ---------------------------------------------------------------------------
# Default channel map — seed entries loaded from environment variables.
#
# Each watched channel has its own env var (channel snowflake IDs are
# user-secret so they don't belong in code). The map stays empty for any
# unset var, which is correct: Discord ingest is strict opt-in.
#
# Currently seeded:
#   WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID   → memory_type=fact, topics=["discord", "idea-dump"]
#   WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID  → memory_type=fact, topics=["discord", "brain-dump"]
#
# To add a new channel without changing code, extend this dict at startup
# (e.g. from a config file loader) or override it entirely in tests.
# ---------------------------------------------------------------------------

_idea_dump_channel_id: str | None = os.environ.get("WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID")
_brain_dump_channel_id: str | None = os.environ.get("WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID")

DEFAULT_CHANNEL_MAP: dict[str, DiscordChannelMapping] = {}

if _idea_dump_channel_id:
    DEFAULT_CHANNEL_MAP[_idea_dump_channel_id] = DiscordChannelMapping(
        memory_type=MemoryType.fact,
        topics=["discord", "idea-dump"],
        confidence=0.7,
    )

if _brain_dump_channel_id:
    DEFAULT_CHANNEL_MAP[_brain_dump_channel_id] = DiscordChannelMapping(
        memory_type=MemoryType.fact,
        topics=["discord", "brain-dump"],
        confidence=0.7,
    )


def resolve_channel_mapping(
    channel_id: str,
    channel_map: dict[str, DiscordChannelMapping] | None = None,
) -> DiscordChannelMapping | None:
    """Find the mapping for a Discord channel ID.

    Returns None when the channel has no entry in the map — the caller should
    treat this as "ignore this channel."  Unlike the Slack equivalent, there is
    no generic fallback: Discord ingest is strictly opt-in.

    Args:
        channel_id: Discord channel snowflake string.
        channel_map: Optional override for the default map (useful in tests).

    Returns:
        The DiscordChannelMapping for the channel, or None if not configured.
    """
    if channel_map is None:
        channel_map = DEFAULT_CHANNEL_MAP

    return channel_map.get(channel_id)
