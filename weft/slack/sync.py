"""Slack sync — fetches channel history and stores messages as Weft memories."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import asyncpg

from weft.db.connection import acquire, get_db
from weft.embeddings.base import EmbeddingProvider
from weft.models import MemoryCreate, MemorySource
from weft.store import embed_text_for_memory, store_memory

from .config import (
    DEFAULT_EXCLUDED_CHANNELS,
    ChannelMapping,
    resolve_channel_mapping,
)
from .hash_store import SlackSyncState
from .parser import SlackMessage, build_memory_content, parse_message

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    channels_synced: int = 0
    messages_found: int = 0
    messages_synced: int = 0
    messages_skipped: int = 0
    messages_updated: int = 0
    messages_errored: int = 0
    memories_created: int = 0
    memories_archived: int = 0


@dataclass
class ChannelInfo:
    """Minimal channel info needed for sync."""

    id: str
    name: str


async def _add_ingest_reaction(
    client,
    channel_id: str,
    message_ts: str,
    emoji: str = "brain",
) -> bool:
    """Add a reaction emoji to a message after successful ingestion.

    Fire-and-forget — failures are logged but never block ingestion.
    Idempotent: adding the same reaction twice is safe (Slack returns already_reacted).
    """
    try:
        resp = await asyncio.wait_for(
            client.reactions_add(
                channel=channel_id,
                name=emoji,
                timestamp=message_ts,
            ),
            timeout=5.0,
        )
        if not resp.get("ok"):
            error = resp.get("error", "unknown")
            if error != "already_reacted":
                logger.warning(
                    "slack_sync.reaction_add_failed: %s (channel=%s, ts=%s)",
                    error, channel_id, message_ts,
                )
                return False
        return True
    except Exception:
        logger.warning(
            "slack_sync.reaction_add_error (channel=%s, ts=%s)",
            channel_id, message_ts, exc_info=True,
        )
        return False


async def sync_slack_sdk(
    pool: asyncpg.Pool,
    bot_token: str,
    embedding_provider: EmbeddingProvider | None = None,
    *,
    sync_state: SlackSyncState | None = None,
    channel_map: dict[str, ChannelMapping] | None = None,
    excluded_channels: set[str] | None = None,
    user_names: dict[str, str] | None = None,
    limit_per_channel: int = 200,
    smart_ingest: bool = False,
) -> SyncResult:
    """Sync Slack messages using the slack_sdk directly.

    Requires SLACK_BOT_TOKEN with channels:history, channels:read,
    users:read scopes.
    """
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=bot_token)

    if excluded_channels is None:
        excluded_channels = DEFAULT_EXCLUDED_CHANNELS

    # Discover channels — try private+public, fall back to public-only
    # if the bot token lacks groups:read scope.
    channels: list[ChannelInfo] = []
    channel_types = "public_channel,private_channel"
    try:
        channels = await _list_channels(client, channel_types, excluded_channels)
    except Exception as exc:
        if "missing_scope" in str(exc):
            logger.warning(
                "slack_sync.missing_scope — falling back to public channels "
                "only. Add groups:read to your Slack app's bot token scopes "
                "to sync private channels."
            )
            channels = await _list_channels(
                client, "public_channel", excluded_channels
            )
        else:
            raise

    # Resolve user names if not provided
    if user_names is None:
        user_names = await _fetch_user_names_sdk(client)

    if sync_state is None:
        sync_state = SlackSyncState()

    result = SyncResult()

    for channel in channels:
        try:
            await _sync_channel_sdk(
                client,
                channel,
                pool,
                embedding_provider,
                sync_state=sync_state,
                channel_map=channel_map,
                user_names=user_names,
                limit=limit_per_channel,
                result=result,
                smart_ingest=smart_ingest,
            )
            result.channels_synced += 1
        except Exception as exc:
            logger.warning("Error syncing channel %s: %s", channel.name, exc)

    sync_state.save()
    return result


async def _list_channels(
    client,
    channel_types: str,
    excluded_channels: set[str],
) -> list[ChannelInfo]:
    """Paginate conversations_list and return non-excluded channels."""
    channels: list[ChannelInfo] = []
    cursor = None
    while True:
        resp = await client.conversations_list(
            types=channel_types,
            cursor=cursor,
            limit=100,
        )
        for ch in resp["channels"]:
            if ch["name"] not in excluded_channels:
                channels.append(ChannelInfo(id=ch["id"], name=ch["name"]))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channels


async def sync_slack_messages(
    pool: asyncpg.Pool,
    channels: list[ChannelInfo],
    messages_by_channel: dict[str, list[dict]],
    embedding_provider: EmbeddingProvider | None = None,
    *,
    sync_state: SlackSyncState | None = None,
    channel_map: dict[str, ChannelMapping] | None = None,
    user_names: dict[str, str] | None = None,
    threads_by_channel: dict[str, dict[str, list[dict]]] | None = None,
) -> SyncResult:
    """Sync pre-fetched Slack messages into Weft memories.

    This is the backend for MCP tool mode — the caller fetches messages
    via Slack MCP tools and passes them here for processing.

    Args:
        channels: List of channels being synced.
        messages_by_channel: {channel_id: [raw_message_dicts]}
        threads_by_channel: {channel_id: {thread_ts: [reply_dicts]}}
    """
    if sync_state is None:
        sync_state = SlackSyncState()
    if user_names is None:
        user_names = {}
    if threads_by_channel is None:
        threads_by_channel = {}

    result = SyncResult()

    for channel in channels:
        raw_messages = messages_by_channel.get(channel.id, [])
        threads = threads_by_channel.get(channel.id, {})

        try:
            await _sync_messages(
                channel,
                raw_messages,
                threads,
                pool,
                embedding_provider,
                sync_state=sync_state,
                channel_map=channel_map,
                user_names=user_names,
                result=result,
            )
            result.channels_synced += 1
        except Exception as exc:
            logger.warning("Error syncing channel %s: %s", channel.name, exc)

    sync_state.save()
    return result


async def _sync_channel_sdk(
    client,
    channel: ChannelInfo,
    pool: asyncpg.Pool,
    embedding_provider: EmbeddingProvider | None,
    *,
    sync_state: SlackSyncState,
    channel_map: dict[str, ChannelMapping] | None,
    user_names: dict[str, str],
    limit: int,
    result: SyncResult,
    smart_ingest: bool = False,
):
    """Fetch and sync messages for one channel using slack_sdk."""
    oldest = sync_state.get_last_sync_ts(channel.id)

    kwargs: dict = {"channel": channel.id, "limit": min(limit, 100)}
    if oldest:
        kwargs["oldest"] = oldest

    raw_messages: list[dict] = []
    cursor = None
    fetched = 0
    while fetched < limit:
        if cursor:
            kwargs["cursor"] = cursor
        resp = await client.conversations_history(**kwargs)
        msgs = resp.get("messages", [])
        raw_messages.extend(msgs)
        fetched += len(msgs)
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor or not msgs:
            break

    # Fetch threads
    threads: dict[str, list[dict]] = {}
    for msg in raw_messages:
        if msg.get("reply_count", 0) > 0:
            thread_resp = await client.conversations_replies(
                channel=channel.id,
                ts=msg["ts"],
                limit=100,
            )
            # First message in replies is the parent — skip it
            replies = thread_resp.get("messages", [])[1:]
            if replies:
                threads[msg["ts"]] = replies

    await _sync_messages(
        channel,
        raw_messages,
        threads,
        pool,
        embedding_provider,
        sync_state=sync_state,
        channel_map=channel_map,
        user_names=user_names,
        result=result,
        smart_ingest=smart_ingest,
        slack_client=client,
    )


async def _sync_messages(
    channel: ChannelInfo,
    raw_messages: list[dict],
    threads: dict[str, list[dict]],
    pool: asyncpg.Pool,
    embedding_provider: EmbeddingProvider | None,
    *,
    sync_state: SlackSyncState,
    channel_map: dict[str, ChannelMapping] | None,
    user_names: dict[str, str],
    result: SyncResult,
    smart_ingest: bool = False,
    slack_client=None,
):
    """Process a batch of raw messages for a channel."""
    mapping = resolve_channel_mapping(channel.name, channel_map)
    max_ts: str | None = None

    # Load reaction config
    react_on_ingest = False
    ingest_emoji = "brain"
    if slack_client:
        from weft.config import load_config
        _sync_cfg = load_config().slack_sync
        react_on_ingest = _sync_cfg.react_on_ingest
        ingest_emoji = _sync_cfg.ingest_reaction_emoji

    # Load skip prefixes
    skip_prefixes: list[str] = []
    if slack_client:
        skip_prefixes = _sync_cfg.skip_prefixes
    else:
        from weft.config import load_config
        skip_prefixes = load_config().slack_sync.skip_prefixes

    # Lazy-init smart adapter
    adapter = None
    smart_ingest_remaining: int | None = None
    if smart_ingest:
        from weft.ingest_adapters import SlackAdapter

        from weft.config import load_config
        adapter = SlackAdapter()
        smart_ingest_remaining = load_config().slack_sync.smart_ingest_max_per_sync

    for raw in raw_messages:
        # Skip bot messages and subtypes (join/leave/etc)
        if raw.get("bot_id") or raw.get("subtype"):
            continue

        # Skip messages with prefixes already captured elsewhere (e.g. /checkin)
        msg_text = raw.get("text", "").strip()
        if skip_prefixes and any(msg_text.startswith(p) for p in skip_prefixes):
            continue

        # Skip thread replies — they'll be included with their parent
        if raw.get("thread_ts") and raw.get("thread_ts") != raw.get("ts"):
            continue

        result.messages_found += 1

        msg = parse_message(raw)

        # Check if already synced and unedited
        existing_ids = sync_state.get_memory_ids(channel.id, msg.ts)
        if existing_ids:
            stored_edited = sync_state.get_edited_ts(channel.id, msg.ts)
            if stored_edited == msg.edited_ts:
                # Already synced and no new edits
                result.messages_skipped += 1
                if max_ts is None or msg.ts > max_ts:
                    max_ts = msg.ts
                continue
            is_update = True
        else:
            is_update = False

        # Attach thread replies
        if msg.ts in threads:
            msg.replies = [parse_message(r) for r in threads[msg.ts]]

        # Archive old memories if this is an update
        if is_update:
            old_ids = sync_state.get_memory_ids(channel.id, msg.ts)
            for mid in old_ids:
                await _archive_memory(pool, mid)
                result.memories_archived += 1
            result.messages_updated += 1

        # --- Smart ingest path (try first, fall back to flat storage) ---
        if adapter and smart_ingest_remaining and smart_ingest_remaining > 0:
            try:
                ingest_result = await adapter.ingest(
                    raw, pool, embedding_provider, channel=channel.name,
                )
                if ingest_result and ingest_result.memories_created > 0:
                    smart_ingest_remaining -= 1
                    if react_on_ingest and slack_client:
                        await _add_ingest_reaction(
                            slack_client, channel.id, msg.ts, ingest_emoji,
                        )
                    # Smart ingest succeeded — update sync state and continue
                    sync_state.update_message(
                        channel.id, msg.ts, [], edited_ts=msg.edited_ts
                    )
                    result.messages_synced += 1
                    result.memories_created += ingest_result.memories_created
                    if max_ts is None or msg.ts > max_ts:
                        max_ts = msg.ts
                    continue
            except Exception:
                logger.warning(
                    "Smart ingest failed for %s in %s, falling back to flat storage",
                    msg.ts, channel.name, exc_info=True,
                )
        elif adapter and smart_ingest_remaining == 0:
            logger.info(
                "Smart ingest budget exhausted, falling back to flat storage "
                "for remaining messages in %s",
                channel.name,
            )
            smart_ingest_remaining = -1  # log once

        # --- Flat-memory storage (original path / fallback) ---
        try:
            memory_ids = await _store_message_memory(
                msg,
                channel,
                mapping,
                pool,
                embedding_provider,
                user_names=user_names,
            )
            if react_on_ingest and slack_client and memory_ids:
                await _add_ingest_reaction(
                    slack_client, channel.id, msg.ts, ingest_emoji,
                )
            sync_state.update_message(
                channel.id, msg.ts, memory_ids, edited_ts=msg.edited_ts
            )
            result.messages_synced += 1
            result.memories_created += len(memory_ids)
        except Exception as exc:
            logger.warning(
                "Error storing message %s in %s: %s", msg.ts, channel.name, exc
            )
            result.messages_errored += 1

        if max_ts is None or msg.ts > max_ts:
            max_ts = msg.ts

    # Update channel cursor
    if max_ts:
        sync_state.set_last_sync_ts(channel.id, max_ts)


async def _store_message_memory(
    message: SlackMessage,
    channel: ChannelInfo,
    mapping: ChannelMapping,
    pool: asyncpg.Pool,
    embedding_provider: EmbeddingProvider | None,
    *,
    user_names: dict[str, str],
) -> list[str]:
    """Create memory/memories from a Slack message. Returns memory IDs."""
    content = build_memory_content(message, channel.name, user_names)

    topics = list(mapping.topics)  # copy
    topics.append(f"channel:{channel.name}")
    topics.append(f"slack-ts:{message.ts}")

    # Add reaction-based topics
    for reaction in message.reactions:
        topics.append(f"reaction:{reaction.get('name', '')}")

    topics = list(dict.fromkeys(topics))  # dedupe preserving order

    embedding = None
    if embedding_provider:
        try:
            embedding = await embedding_provider.embed(
                embed_text_for_memory(content, topics)
            )
        except Exception as exc:
            logger.warning("Failed to embed message %s: %s", message.ts, exc)

    create = MemoryCreate(
        type=mapping.memory_type,
        content=content,
        topic=topics,
        source=MemorySource.ingest,
        confidence=mapping.confidence,
    )
    # acquire() so SET LOCAL app.user_id fires on the connection. Slack
    # sync runs from a background scheduler task, so the contextvar is
    # set at scheduler-loop entry (slack_sync_loop in weft.scheduler)
    # rather than by the HTTP middleware.
    async with acquire(pool):
        memory = await store_memory(pool, create, embedding=embedding)
    return [memory.id]


async def _fetch_user_names_sdk(client) -> dict[str, str]:
    """Fetch user ID → display name mapping via slack_sdk."""
    names: dict[str, str] = {}
    try:
        cursor = None
        while True:
            resp = await client.users_list(cursor=cursor, limit=200)
            for member in resp.get("members", []):
                uid = member["id"]
                profile = member.get("profile", {})
                name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or member.get("name", uid)
                )
                names[uid] = name
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except Exception as exc:
        logger.warning("Failed to fetch user list: %s", exc)
    return names


async def _archive_memory(pool: asyncpg.Pool, memory_id: str):
    """Archive a memory by ID."""
    try:
        # acquire() so SET LOCAL app.user_id fires — without it the RLS
        # WHERE predicate matches no rows under the non-superuser app
        # role and the UPDATE silently no-ops.
        async with acquire(pool):
            await get_db(pool).execute(
                "UPDATE memories SET status = 'archived', updated_at = NOW() WHERE id = $1",
                memory_id,
            )
    except Exception as exc:
        logger.warning("Failed to archive memory %s: %s", memory_id, exc)
