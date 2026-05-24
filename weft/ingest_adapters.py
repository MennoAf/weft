"""Ingest adapters — thin source-specific extractors for the smart pipeline.

Adapters transform raw data from a specific source (Slack, email, etc.) into
IngestItems and route them through the ingestion pipeline. They handle:
  - Pre-filtering (bots, empty, skip subtypes)
  - Field extraction (text, author, timestamp, metadata)
  - Calling process() with the constructed IngestItem

Adapters do NOT classify, score, or route — all intelligence lives in
ingest_pipeline.py. Adapters are intentionally thin.

Example — implementing a new adapter::

    class EmailAdapter:
        async def ingest(self, raw: dict, pool, embedding_provider=None,
                         *, project_id=None) -> IngestResult | None:
            text = raw.get("body", "").strip()
            if not text:
                return None
            item = IngestItem(text=text, source="email", author=raw.get("from"))
            return await process(item, pool, embedding_provider, project_id=project_id)
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from weft.ingest_pipeline import IngestItem, IngestResult, process

logger = logging.getLogger(__name__)

# Subtypes that should always be skipped
SKIP_SUBTYPES = frozenset({
    "bot_message", "message_changed", "message_deleted",
    "channel_join", "channel_leave", "channel_topic",
    "channel_purpose", "channel_name", "channel_archive",
    "channel_unarchive", "group_join", "group_leave",
    "group_topic", "group_purpose", "group_name",
    "group_archive", "group_unarchive",
})

_MIN_TEXT_LENGTH = 3


@runtime_checkable
class IngestAdapter(Protocol):
    """Protocol for source-specific ingest adapters."""

    async def ingest(
        self,
        raw_data: Any,
        pool: Any,
        embedding_provider: Any = None,
        *,
        channel: str | None = None,
        project_id: str | None = None,
    ) -> IngestResult | None: ...


class SlackAdapter:
    """Transforms raw Slack message dicts into IngestItems."""

    async def ingest(
        self,
        raw_data: dict,
        pool: Any,
        embedding_provider: Any = None,
        *,
        channel: str | None = None,
        project_id: str | None = None,
    ) -> IngestResult | None:
        ts = raw_data.get("ts", "?")

        # Skip bot messages
        if raw_data.get("bot_id") or raw_data.get("subtype") in SKIP_SUBTYPES:
            logger.debug("SlackAdapter.skip: bot/subtype, ts=%s", ts)
            return None

        # Skip any subtype (join, leave, etc.)
        if raw_data.get("subtype"):
            logger.debug("SlackAdapter.skip: subtype=%s, ts=%s", raw_data["subtype"], ts)
            return None

        # Extract and clean text
        text = raw_data.get("text") or ""
        # Decode Slack HTML entities
        text = html.unescape(text)
        text = text.strip()

        if not text or len(text) < _MIN_TEXT_LENGTH:
            logger.debug("SlackAdapter.skip: short/empty text, ts=%s", ts)
            return None

        # Build IngestItem
        author = raw_data.get("user") or raw_data.get("bot_id") or "unknown"
        timestamp = None
        try:
            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, TypeError):
            pass

        item = IngestItem(
            text=text,
            source="slack",
            author=author,
            timestamp=timestamp,
            metadata={
                "channel": channel,
                "ts": ts,
                "thread_ts": raw_data.get("thread_ts"),
            },
        )

        return await process(item, pool, embedding_provider, project_id=project_id)


class DiscordAdapter:
    """Transforms raw Discord message dicts into IngestItems.

    Raw dict shape (produced by the bot's on_message handler)::

        {
            "id": <int message snowflake>,
            "content": <str message text>,
            "author_id": <str>,
            "author_name": <str>,
            "channel_id": <str>,
            "created_at": <ISO-8601 str or None>,
        }

    Defense-in-depth bot filter: even if the bot's on_message handler already
    filtered self-messages, this adapter re-checks the ``is_bot`` flag so an
    inadvertent call path can't create an ingest loop.
    """

    async def ingest(
        self,
        raw_data: dict,
        pool: Any,
        embedding_provider: Any = None,
        *,
        channel: str | None = None,
        project_id: str | None = None,
    ) -> IngestResult | None:
        msg_id = raw_data.get("id", "?")

        # Defense-in-depth: skip if caller signals this is a bot message
        if raw_data.get("is_bot"):
            logger.debug("DiscordAdapter.skip: bot author, id=%s", msg_id)
            return None

        # Extract and clean text
        text = (raw_data.get("content") or "").strip()

        if not text or len(text) < _MIN_TEXT_LENGTH:
            logger.debug("DiscordAdapter.skip: short/empty text, id=%s", msg_id)
            return None

        # Build IngestItem
        author = raw_data.get("author_name") or raw_data.get("author_id") or "unknown"
        timestamp = None
        created_at = raw_data.get("created_at")
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        effective_channel = channel or raw_data.get("channel_id")

        item = IngestItem(
            text=text,
            source="discord",
            author=author,
            timestamp=timestamp,
            metadata={
                "channel": effective_channel,
                "message_id": str(msg_id),
                "author_id": raw_data.get("author_id"),
            },
        )

        return await process(item, pool, embedding_provider, project_id=project_id)


# Registry of adapters by source name
ADAPTERS: dict[str, type[IngestAdapter]] = {
    "slack": SlackAdapter,
    "discord": DiscordAdapter,
}

__all__ = ["IngestAdapter", "SlackAdapter", "DiscordAdapter", "ADAPTERS"]
