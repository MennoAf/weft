"""Smoke test: post a fake daily_brief event through the Discord outbound path.

Reads WEFT_DISCORD_BOT_TOKEN and WEFT_DISCORD_BRIEF_CHANNEL_ID from env (or a
local .env via python-dotenv), spins up the bot loop in the background, waits
for the gateway handshake, then calls emit_outbound_event("daily_brief", ...)
with WEFT_OUTBOUND_CONNECTOR=discord set. If everything is wired correctly a
short brief lands in the configured Discord channel.

Usage:
    uv run python scripts/test_discord_brief.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from types import SimpleNamespace

from dotenv import load_dotenv

# Local .env (~/.weft/.env or project-level .env) — keep tokens out of argv.
load_dotenv()
load_dotenv(os.path.expanduser("~/.weft/.env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("test_discord_brief")


async def main() -> int:
    token = os.environ.get("WEFT_DISCORD_BOT_TOKEN", "").strip()
    channel = os.environ.get("WEFT_DISCORD_BRIEF_CHANNEL_ID", "").strip()
    if not token or not channel:
        logger.error(
            "Set WEFT_DISCORD_BOT_TOKEN and WEFT_DISCORD_BRIEF_CHANNEL_ID in env or ~/.weft/.env"
        )
        return 2

    # Force the active connector to discord for this smoke test, even if the
    # surrounding shell has WEFT_OUTBOUND_CONNECTOR set to slack/none.
    os.environ["WEFT_OUTBOUND_CONNECTOR"] = "discord"

    # Import after env is settled so the connector registers and the scheduler
    # picks up the right value at dispatch time.
    import weft.discord  # noqa: F401  — registers the discord handler
    from weft.discord.connector import get_bot
    from weft.scheduler import discord_bot_loop, emit_outbound_event

    pool_placeholder = None  # loop doesn't use the pool yet
    loop_task = asyncio.create_task(discord_bot_loop(pool_placeholder))

    # Wait for the bot to register itself and the gateway to come up.
    for _ in range(60):  # ~30s budget
        bot = get_bot()
        if bot is not None and bot.is_ready:
            break
        await asyncio.sleep(0.5)
    else:
        logger.error("discord bot never became ready within 30s")
        loop_task.cancel()
        return 1

    fake_brief = SimpleNamespace(
        markdown=(
            "**Weft daily brief — smoke test**\n\n"
            "If you can read this in Discord, the outbound event registry "
            "is correctly routing `daily_brief` through the Discord connector."
        ),
        slack_blocks=[],
    )
    await emit_outbound_event("daily_brief", channel="ignored", brief_result=fake_brief)
    logger.info("emit_outbound_event returned — check the configured Discord channel")

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
