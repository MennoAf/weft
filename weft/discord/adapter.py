"""Discord ingest adapter with channel-mapping support.

Thin wrapper around the generic DiscordAdapter from weft.ingest_adapters that
adds opt-in channel routing via weft.discord.config.DiscordChannelMapping.

Key behaviours:
- Configured channel  → ingests with the mapped memory_type stored in metadata.
- Unconfigured channel → returns None immediately (caller/bot skips silently).
- Default-fallback    → the Idea Dump channel has a seed entry in
  DEFAULT_CHANNEL_MAP (loaded from WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID).

Adding a new watched channel requires only a config change (extend
DEFAULT_CHANNEL_MAP or supply a custom channel_map at construction time);
no code changes are needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from weft.discord.config import (
    DEFAULT_CHANNEL_MAP,
    DiscordChannelMapping,
    resolve_channel_mapping,
)
from weft.ingest_adapters import _MIN_TEXT_LENGTH
from weft.ingest_pipeline import IngestItem, IngestResult, process

logger = logging.getLogger(__name__)


class DiscordChannelAdapter:
    """Discord ingest adapter that routes messages through the channel mapping.

    Instantiate once and reuse; the channel_map is injected at construction
    time, defaulting to DEFAULT_CHANNEL_MAP so production code needs no
    arguments while tests can supply any mapping they like.

    Usage::

        adapter = DiscordChannelAdapter()
        result = await adapter.ingest(raw, pool, embedding_provider,
                                      channel="123456789")
    """

    def __init__(
        self,
        channel_map: dict[str, DiscordChannelMapping] | None = None,
    ) -> None:
        self._channel_map = channel_map if channel_map is not None else DEFAULT_CHANNEL_MAP

    async def ingest(
        self,
        raw_data: dict,
        pool: Any,
        embedding_provider: Any = None,
        *,
        channel: str | None = None,
        project_id: str | None = None,
    ) -> IngestResult | None:
        """Ingest a raw Discord message dict.

        Args:
            raw_data: Raw message dict produced by Bot.on_message.
            pool: asyncpg connection pool.
            embedding_provider: Optional embedding provider (passed through).
            channel: Discord channel ID (snowflake string).  If not supplied,
                falls back to raw_data["channel_id"].
            project_id: Optional Weft project scope.

        Returns:
            IngestResult on success, None if the message should be skipped
            (bot author, short text, or unmapped channel).
        """
        msg_id = raw_data.get("id", "?")

        # Defense-in-depth: skip bot messages
        if raw_data.get("is_bot"):
            logger.debug("DiscordChannelAdapter.skip: bot author, id=%s", msg_id)
            return None

        # Extract and clean text
        text = (raw_data.get("content") or "").strip()

        if not text or len(text) < _MIN_TEXT_LENGTH:
            logger.debug("DiscordChannelAdapter.skip: short/empty text, id=%s", msg_id)
            return None

        # Resolve effective channel ID
        effective_channel = channel or raw_data.get("channel_id") or ""

        # Channel-mapping lookup — unconfigured channels are silently ignored.
        mapping = resolve_channel_mapping(str(effective_channel), self._channel_map)
        if mapping is None:
            logger.debug(
                "DiscordChannelAdapter.skip: unmapped channel=%s, id=%s",
                effective_channel, msg_id,
            )
            return None

        # Build IngestItem with memory_type_hint in metadata so downstream
        # tooling and tests can observe which memory type was selected.
        author = raw_data.get("author_name") or raw_data.get("author_id") or "unknown"
        timestamp: datetime | None = None
        created_at = raw_data.get("created_at")
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        item = IngestItem(
            text=text,
            source="discord",
            author=author,
            timestamp=timestamp,
            metadata={
                "channel": effective_channel,
                "message_id": str(msg_id),
                "author_id": raw_data.get("author_id"),
                # Carry the configured memory type through as a hint so callers
                # can inspect which type was chosen without re-querying config.
                "memory_type_hint": mapping.memory_type.value,
                "topics": mapping.topics,
            },
        )

        return await process(item, pool, embedding_provider, project_id=project_id)
