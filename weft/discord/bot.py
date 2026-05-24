"""Outbound-only Discord bot wrapper.

Slimmed-down version of claude-discord-bridge/src/bridge/bot.py — drops the
message-receive path (and therefore the `message_content` privileged intent)
because Weft's current Discord use is one-way: post daily briefs into a
configured channel. If/when inbound replies are wanted, add the intent +
on_message hook back in.

Surface: start/close lifecycle, chunked text post, embed post, and an
is_ready probe that gates the connector handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Awaitable, Callable, TypeVar

import aiohttp
import discord

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


class Bot:
    """Wraps a discord.py Client for outbound posting to one configured channel."""

    def __init__(self, token: str, channel_id: int) -> None:
        # Default intents only — no privileged `message_content` since we
        # don't read inbound messages here.
        intents = discord.Intents.default()
        self._client = discord.Client(intents=intents)
        self._token = token
        self._channel_id = channel_id
        self._channel: discord.TextChannel | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._client.event(self.on_ready)

    @property
    def channel_id(self) -> int:
        return self._channel_id

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._client.is_closed()

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
        self._ready.set()
        logger.info("discord_bot.ready as %s, channel=#%s", self._client.user, ch.name)

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
