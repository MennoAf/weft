"""Outbound-only Discord bot wrapper.

Slimmed-down version of claude-discord-bridge/src/bridge/bot.py — drops the
message-receive path (and therefore the `message_content` privileged intent)
because Weft's current Discord use is one-way: post daily briefs into a
configured channel. If/when inbound replies are wanted, add the intent +
on_message hook back in.

Surface: start/close lifecycle, chunked text post, embed post, and an
is_ready probe that gates the connector handler.

Slash commands: the Bot accepts an optional `pool` parameter. When present, it
creates an `app_commands.CommandTree` on the underlying client, calls
`register_commands()` to attach slash command handlers, and syncs the tree
globally in `on_ready` after channel resolution but BEFORE setting _ready.
Tree sync failures are logged and swallowed — the outbound brief keeps working
regardless.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Awaitable, Callable, TypeVar

import aiohttp
import asyncpg
import discord
from discord import app_commands

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


# Discord 5xx during incidents — retry with backoff before propagating.
_RETRY_DELAYS_SECS = (0.5, 1.5, 4.0)

# Discord enforces 2000 chars per message; leave headroom for any
# attribution/header bytes we may prepend.
MAX_CHUNK = 1900


async def _with_retry(label: str, factory: Callable[[], Awaitable[_T]]) -> _T:
    last_exc: BaseException | None = None
    for attempt, delay in enumerate((0.0,) + _RETRY_DELAYS_SECS):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await factory()
        except (discord.DiscordServerError, aiohttp.ClientConnectionError) as e:
            last_exc = e
            logger.warning(
                "%s: transient discord error (attempt %d/%d): %s",
                label,
                attempt + 1,
                len(_RETRY_DELAYS_SECS) + 1,
                e,
            )
    assert last_exc is not None
    raise last_exc


def _chunk(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split text into <=limit-char chunks, preferring newline breaks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


class BotNotReady(RuntimeError):
    """Operation attempted before the gateway handshake completed."""


class _WeftClient(discord.Client):
    """Bare discord.Client subclass that carries Weft-specific state.

    Attributes stashed here are accessible inside slash-command handlers via
    ``interaction.client._weft_pool`` and ``interaction.client._weft_tree``.
    Using a subclass rather than monkeypatching the base client keeps the
    attribute access explicit and avoids type-checker complaints.
    """

    def __init__(self, *args, pool: asyncpg.Pool | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weft_pool: asyncpg.Pool | None = pool
        self._weft_tree: app_commands.CommandTree | None = None


class Bot:
    """Wraps a discord.py Client for outbound posting to one configured channel.

    Pass ``pool`` to enable slash-command registration. Without a pool the bot
    operates in outbound-only mode (same as before this change).

    Pass ``idea_dump_channel_id`` to enable the inbound idea-dump watcher. When
    set, the bot requests the ``message_content`` PRIVILEGED intent and registers
    an ``on_message`` handler that pipes matching messages through the ingest
    pipeline. Without this parameter, the bot stays on default intents — the
    same outbound-only behaviour as before.

    IMPORTANT: ``message_content`` is a Discord privileged intent. It requires
    manual enablement in the Discord Developer Portal under Bot → Privileged
    Gateway Intents → Message Content Intent. Discord will reject the gateway
    connection at runtime if the intent is enabled in code but not in the
    Developer Portal.
    """

    def __init__(
        self,
        token: str,
        channel_id: int,
        *,
        pool: asyncpg.Pool | None = None,
        idea_dump_channel_id: int | None = None,
        channel_map: "dict | None" = None,
    ) -> None:
        intents = discord.Intents.default()

        if idea_dump_channel_id is not None:
            # The message_content intent is PRIVILEGED — Discord requires manual
            # enablement in the Developer Portal (Bot → Privileged Gateway
            # Intents → Message Content Intent). Code can request it, but Discord
            # will reject the gateway connection at runtime if it's not toggled on
            # in the portal configuration.
            intents.message_content = True
            logger.warning(
                "discord_bot.idea_dump_enabled: WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID is set — "
                "requires message_content intent enabled in Discord Developer Portal"
            )

        self._client = _WeftClient(intents=intents, pool=pool)
        self._pool = pool
        self._token = token
        self._channel_id = channel_id
        self._idea_dump_channel_id = idea_dump_channel_id
        self._channel: discord.TextChannel | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

        # Channel-mapping-aware ingest adapter.  The custom channel_map is
        # injected here so the bot's message handler routes via the mapping
        # instead of using the hardcoded idea-dump-only logic.  None uses
        # DEFAULT_CHANNEL_MAP (env-var driven, populated at module import time).
        self._channel_map = channel_map

        # Build the command tree if a pool was supplied.
        if pool is not None:
            self._client._weft_tree = app_commands.CommandTree(self._client)
            self._register_commands()

        self._client.event(self.on_ready)

        # Register the inbound idea-dump handler only when the feature is enabled.
        # This keeps the outbound-only deployment path completely unaffected.
        if idea_dump_channel_id is not None:
            self._client.event(self.on_message)

    @property
    def channel_id(self) -> int:
        return self._channel_id

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._client.is_closed()

    def _register_commands(self) -> None:
        """Attach slash command handlers to the command tree.

        Imported lazily from weft.discord.commands to avoid a circular import
        at module load time (commands.py imports Bot).
        """
        from weft.discord.commands import register_checkin_command

        tree = self._client._weft_tree
        assert tree is not None  # only called when pool is set
        register_checkin_command(tree)

    async def on_ready(self) -> None:
        ch = self._client.get_channel(self._channel_id) or await self._client.fetch_channel(
            self._channel_id
        )
        if not isinstance(ch, discord.TextChannel):
            raise RuntimeError(
                f"WEFT_DISCORD_BRIEF_CHANNEL_ID={self._channel_id} "
                f"is not a TextChannel (got {type(ch).__name__})."
            )
        self._channel = ch

        # Sync the slash-command tree BEFORE marking ready so callers that
        # await wait_until_ready() can assume commands are registered.
        tree = self._client._weft_tree
        if tree is not None:
            try:
                await tree.sync()
                logger.info("discord_bot.commands_synced")
            except Exception:
                # Sync failure must NOT crash the bot — the outbound brief
                # should keep working even if command registration fails.
                logger.exception("discord_bot.commands_sync_failed — continuing")

        self._ready.set()
        logger.info("discord_bot.ready as %s, channel=#%s", self._client.user, ch.name)

    async def on_message(self, message: discord.Message) -> None:
        """Inbound message handler. Only active when idea_dump_channel_id is set.

        Routes messages through DiscordChannelAdapter, which consults the
        channel mapping to decide whether to ingest and which memory type to
        use.  Unconfigured channels are silently ignored by the adapter.

        Discord delivers gateway events serially per shard, so a slow ingest
        call would back up the entire event queue. We use asyncio.create_task to
        detach the ingest work from the gateway loop — the task runs concurrently
        and any failure is contained without blocking subsequent message events.
        """
        # Skip messages from any bot (including the bot itself) — defense layer 1.
        if message.author.bot:
            return

        # Skip messages from the bot itself — defense layer 2 (covers edge cases
        # where .bot might be False for the client user in unusual configurations).
        if self._client.user and message.author.id == self._client.user.id:
            return

        # Guard: only process messages from channels the bot is watching.
        # The idea_dump_channel_id param keeps backward compat — if it's set,
        # we only process that specific channel.  When channel mapping is used
        # directly (idea_dump_channel_id=None), the adapter's resolve step
        # handles per-channel filtering.
        channel_id_str = str(message.channel.id)
        if self._idea_dump_channel_id is not None:
            if message.channel.id != self._idea_dump_channel_id:
                return

        # Build the raw dict for DiscordChannelAdapter.ingest()
        raw: dict = {
            "id": message.id,
            "content": message.content,
            "author_id": str(message.author.id),
            "author_name": message.author.name,
            "channel_id": channel_id_str,
            "created_at": message.created_at.isoformat() if message.created_at else None,
        }

        pool = self._pool
        channel_map = self._channel_map

        async def _ingest() -> None:
            from weft.discord.adapter import DiscordChannelAdapter

            try:
                result = await DiscordChannelAdapter(channel_map=channel_map).ingest(
                    raw,
                    pool,
                    embedding_provider=None,
                    channel=channel_id_str,
                )
                logger.info(
                    "discord_bot.message_ingested: msg_id=%s memories=%s",
                    message.id,
                    result.memories_created if result else 0,
                )
            except Exception:
                # Must not raise — the gateway connection must stay alive.
                logger.exception("discord_bot.ingest_error: msg_id=%s", message.id)

        # Detach from the gateway event loop to avoid backing up the serial
        # event queue on slow ingest calls.
        asyncio.create_task(_ingest())

    async def start(self) -> None:
        """Schedule the gateway handshake in the background. Returns immediately."""
        self._task = asyncio.create_task(self._client.start(self._token))

    async def wait_until_ready(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)

    async def close(self) -> None:
        await self._client.close()
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task

    async def post(self, message: str) -> list[int]:
        """Post chunked text to the configured channel. Returns created message IDs.

        Partial-send semantics: if chunk N fails after chunks 1..N-1 landed,
        this raises but the earlier chunks remain visible. Callers should treat
        any exception from this method as "message may have been partially delivered".
        """
        if not self.is_ready or self._channel is None:
            raise BotNotReady("discord bot not connected")
        target = self._channel
        ids: list[int] = []
        for chunk in _chunk(message):
            msg = await _with_retry(
                "discord.send",
                lambda c=chunk: target.send(c),
            )
            ids.append(msg.id)
        return ids

    async def post_embed(self, embed: discord.Embed) -> int:
        """Send a single embed; return the message id."""
        if not self.is_ready or self._channel is None:
            raise BotNotReady("discord bot not connected")
        msg = await _with_retry(
            "discord.send-embed",
            lambda: self._channel.send(embed=embed),
        )
        return msg.id
