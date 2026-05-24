"""DiscordConnector — registers an outbound handler for the daily_brief event.

The handler dispatches via a module-level Bot reference owned by
`weft.scheduler.discord_bot_loop`. The loop calls `set_bot(bot)` once the
gateway handshake completes; `_outbound_discord_brief` reads that reference
at dispatch time. If the bot isn't ready yet, the handler logs and returns
rather than raising — the daily_brief loop should never crash because the
gateway is reconnecting.
"""

from __future__ import annotations

import logging

from weft.discord.bot import Bot, BotNotReady
from weft.scheduler import register_outbound_handler

logger = logging.getLogger(__name__)


_BOT: Bot | None = None


def set_bot(bot: Bot) -> None:
    """Register the live Bot instance used by the outbound handler."""
    global _BOT
    _BOT = bot


def clear_bot() -> None:
    global _BOT
    _BOT = None


def get_bot() -> Bot | None:
    return _BOT


async def _outbound_discord_brief(channel: str, brief_result) -> None:
    """Outbound connector wrapper: post a daily_brief event to Discord.

    *channel* is accepted to match the event contract (Slack uses it as a
    channel ID) but ignored here — the Discord channel is fixed at bot
    startup via WEFT_DISCORD_BRIEF_CHANNEL_ID. We log it for traceability.
    """
    bot = _BOT
    if bot is None:
        logger.warning(
            "discord_brief.no_bot",
            extra={"requested_channel": channel},
        )
        return
    if not bot.is_ready:
        logger.warning(
            "discord_brief.bot_not_ready",
            extra={"requested_channel": channel, "discord_channel_id": bot.channel_id},
        )
        return
    try:
        await bot.post(brief_result.markdown)
    except BotNotReady:
        # Race: gateway disconnected between the is_ready check and the send.
        logger.warning(
            "discord_brief.bot_disconnected_mid_send",
            extra={"discord_channel_id": bot.channel_id},
        )
    except Exception:
        logger.exception(
            "discord_brief.post_error",
            extra={"discord_channel_id": bot.channel_id},
        )


register_outbound_handler("daily_brief", "discord", _outbound_discord_brief)
