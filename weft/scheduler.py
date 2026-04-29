"""Background schedulers — alert polling, Slack sync, and daily brief delivery.

Runs as asyncio tasks within the MCP server lifespan.

Design principles:
- Per-alert error isolation: one failed dispatch never crashes the loop
- Connection released before dispatch: no pool exhaustion risk
- Backpressure: polls again immediately when a full batch is returned
- Graceful shutdown: CancelledError is caught for clean logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

import asyncpg
from slack_sdk.web.async_client import AsyncWebClient

from weft.alerts import is_daily_brief_due, mark_alert_fired, poll_due_alerts
from weft.models import Alert, Trigger

logger = logging.getLogger(__name__)

_SLACK_TIMEOUT = 10  # seconds for Slack API calls

# Default settings — overridden by config in production
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_BATCH_SIZE = 50

# --- Dispatch registry ---

DispatchHandler = Callable[[Alert], Awaitable[None]]

_DISPATCH_REGISTRY: dict[str, DispatchHandler] = {}


def register_dispatch(channel: str, handler: DispatchHandler) -> None:
    """Register a dispatch handler for a channel."""
    _DISPATCH_REGISTRY[channel] = handler


async def dispatch_log(alert: Alert) -> None:
    """Dispatch an alert to Python logging."""
    logger.info(
        "alert.fired",
        extra={
            "alert_id": alert.id,
            "alert_type": alert.alert_type.value,
            "title": alert.title,
            "channel": alert.channel.value,
            "project_id": alert.project_id,
        },
    )


async def dispatch_slack(alert: Alert) -> None:
    """Dispatch an alert to Slack via the Web API.

    Uses SLACK_BOT_TOKEN from environment. Requires bot scopes:
    chat:write. Logs and returns gracefully on any failure.
    """
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.warning(
            "alert.slack.no_token",
            extra={"alert_id": alert.id},
        )
        return

    if not alert.channel_target:
        logger.warning(
            "alert.slack.no_channel_target",
            extra={"alert_id": alert.id},
        )
        return

    try:
        import ssl

        import certifi

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        client = AsyncWebClient(token=token, ssl=ssl_ctx)
        trigger_str = (
            alert.trigger_at.strftime("%Y-%m-%d %H:%M UTC")
            if alert.trigger_at
            else "now"
        )
        text = (
            f"*[{alert.alert_type.value}]* {alert.title}\n"
            f"{alert.body or ''}\n"
            f"_Due: {trigger_str}_"
        ).strip()

        response = await asyncio.wait_for(
            client.chat_postMessage(
                channel=alert.channel_target,
                text=text,
            ),
            timeout=_SLACK_TIMEOUT,
        )
        if not response.get("ok"):
            logger.warning(
                "alert.slack.api_error",
                extra={
                    "alert_id": alert.id,
                    "error": response.get("error", "unknown"),
                },
            )
    except Exception:
        logger.exception(
            "alert.slack.dispatch_error",
            extra={"alert_id": alert.id},
        )


# Register built-in handlers
register_dispatch("log", dispatch_log)
register_dispatch("slack", dispatch_slack)


async def dispatch_alert(alert: Alert) -> None:
    """Route an alert to the appropriate channel handler."""
    handler = _DISPATCH_REGISTRY.get(alert.channel.value)
    if handler is None:
        logger.warning(
            "alert.unknown_channel",
            extra={"alert_id": alert.id, "channel": alert.channel.value},
        )
        return
    await handler(alert)


async def scheduler_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = DEFAULT_POLL_INTERVAL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Main scheduler loop. Runs until cancelled.

    Polls for due alerts, dispatches each one, and marks fired on success.
    If dispatch fails, the alert stays pending and will be retried next cycle.
    If a full batch is returned, polls again immediately (backpressure).
    """
    logger.info("scheduler.started", extra={"interval": interval, "batch_size": batch_size})
    try:
        while True:
            try:
                alerts = await poll_due_alerts(pool, batch_size=batch_size)
            except Exception:
                logger.exception("scheduler.poll_error")
                await asyncio.sleep(interval)
                continue

            for alert in alerts:
                try:
                    await dispatch_alert(alert)
                    await mark_alert_fired(pool, alert.id)
                except Exception:
                    logger.exception(
                        "scheduler.dispatch_error",
                        extra={"alert_id": alert.id},
                    )
                    # Alert stays pending — will be retried next cycle

            # Backpressure: if we got a full batch, poll again immediately
            if len(alerts) >= batch_size:
                continue

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("scheduler.stopped")
        raise


# --- Slack sync loop ---

_MIN_SYNC_INTERVAL = 60  # floor to prevent API hammering


async def slack_sync_loop(
    pool: asyncpg.Pool,
    embedding_provider=None,
    *,
    interval: int = 1800,
    smart_ingest: bool = False,
) -> None:
    """Recurring Slack channel sync. Runs until cancelled.

    Calls sync_slack_sdk() on each cycle, auto-discovering all channels
    the bot is invited to. Syncs immediately on startup, then sleeps
    for *interval* seconds between cycles. Naturally serialized — a slow
    sync delays the next cycle rather than overlapping.
    """
    from weft.auth import current_user_id
    from weft.slack.sync import sync_slack_sdk

    interval = max(interval, _MIN_SYNC_INTERVAL)
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")

    if not bot_token:
        logger.warning("slack_sync.no_token — Slack sync loop disabled")
        return

    # Background scheduler tasks have no HTTP middleware setting the
    # request-scoped user identity. Slack-ingested memories belong to
    # the deployment owner — same user the legacy bootstrap row binds
    # to. Without this, every store_memory inside the sync would hit
    # the migration-34 NOT NULL on memories.user_id and silently fail
    # via per-message logger.warning.
    default_uid = os.environ.get("WEFT_DEFAULT_USER_ID")
    if not default_uid:
        logger.warning(
            "slack_sync.no_default_user — set WEFT_DEFAULT_USER_ID to "
            "the deployment owner's UUID; sync loop disabled"
        )
        return
    current_user_id.set(default_uid)

    logger.info(
        "slack_sync.started",
        extra={"interval": interval, "user_id": default_uid},
    )
    try:
        while True:
            t0 = time.monotonic()
            try:
                result = await sync_slack_sdk(
                    pool,
                    bot_token,
                    embedding_provider,
                    smart_ingest=smart_ingest,
                )
                elapsed = time.monotonic() - t0
                logger.info(
                    "slack_sync.complete",
                    extra={
                        "channels": result.channels_synced,
                        "messages_synced": result.messages_synced,
                        "memories_created": result.memories_created,
                        "elapsed_s": round(elapsed, 1),
                    },
                )
            except Exception:
                elapsed = time.monotonic() - t0
                logger.exception(
                    "slack_sync.error",
                    extra={"elapsed_s": round(elapsed, 1)},
                )
                # Continue — will retry next cycle

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("slack_sync.stopped")
        raise


# --- Daily brief delivery loop ---

_BRIEF_POLL_INTERVAL = 60  # check every minute whether brief is due


async def daily_brief_loop(
    pool: asyncpg.Pool,
    *,
    brief_time: str = "08:00",
    brief_tz: str = "America/New_York",
    brief_channel: str = "",
) -> None:
    """Scheduled daily brief delivery. Runs until cancelled.

    Polls every minute to check if the configured brief time has arrived.
    Assembles the brief and posts to Slack (if channel configured).
    Uses file-based dedup to prevent multiple sends per day.
    """
    from weft.brief_state import get_last_brief_date, set_last_brief_date
    from weft.config import DailyBriefConfig
    from weft.daily_brief import assemble_daily_brief

    if not brief_channel:
        logger.info("daily_brief.no_channel — daily brief delivery disabled")
        return

    logger.info(
        "daily_brief.started",
        extra={"time": brief_time, "tz": brief_tz, "channel": brief_channel},
    )

    try:
        while True:
            try:
                from datetime import datetime as dt_mod
                from datetime import timezone as tz_mod
                from zoneinfo import ZoneInfo

                now = dt_mod.now(tz_mod.utc)

                if is_daily_brief_due(now, brief_time=brief_time, brief_tz=brief_tz):
                    local_date = now.astimezone(ZoneInfo(brief_tz)).date()
                    last = get_last_brief_date()

                    if last != local_date:
                        logger.info("daily_brief.assembling")
                        brief_config = DailyBriefConfig(
                            time=brief_time, timezone=brief_tz, channel=brief_channel
                        )
                        result = await assemble_daily_brief(pool, brief_config, target_date=now)

                        # Post to Slack
                        await _post_brief_to_slack(brief_channel, result)
                        set_last_brief_date(local_date)
                        logger.info("daily_brief.delivered", extra={"date": str(local_date)})
            except Exception:
                logger.exception("daily_brief.loop_error")

            await asyncio.sleep(_BRIEF_POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("daily_brief.stopped")
        raise


# --- Loom awareness loop ---

_LOOM_CHECK_INTERVAL = 3600  # check every hour


async def loom_awareness_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = _LOOM_CHECK_INTERVAL,
) -> None:
    """Periodic Loom task state checks. Runs until cancelled.

    Checks for stale claims, epic completion readiness, and blocked pile-ups.
    Creates Weft alerts when thresholds are crossed (with 24h dedup).
    Gracefully skips if Loom tables don't exist.
    """
    from weft.loom_alerts import evaluate_loom_alerts

    logger.info("loom_awareness.started", extra={"interval": interval})
    try:
        while True:
            try:
                created = await evaluate_loom_alerts(pool)
                if created:
                    logger.info(
                        "loom_awareness.alerts_created",
                        extra={"count": len(created)},
                    )
            except Exception:
                logger.exception("loom_awareness.loop_error")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("loom_awareness.stopped")
        raise


# --- Memory hygiene loop ---

_HYGIENE_CHECK_INTERVAL = 3600  # check every hour


async def memory_hygiene_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = _HYGIENE_CHECK_INTERVAL,
) -> None:
    """Periodic memory health checks. Runs until cancelled.

    Checks for stale decisions, overdue consolidation, and memory count
    thresholds. Creates Weft alerts when issues are found (with 24h dedup).
    """
    from weft.memory_hygiene_alerts import evaluate_memory_hygiene_alerts

    logger.info("memory_hygiene.started", extra={"interval": interval})
    try:
        while True:
            try:
                created = await evaluate_memory_hygiene_alerts(pool)
                if created:
                    logger.info(
                        "memory_hygiene.alerts_created",
                        extra={"count": len(created)},
                    )
            except Exception:
                logger.exception("memory_hygiene.loop_error")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("memory_hygiene.stopped")
        raise


# --- Trigger evaluation loop ---

_TRIGGER_EVAL_INTERVAL = 300  # check every 5 minutes


async def trigger_evaluation_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = _TRIGGER_EVAL_INTERVAL,
) -> None:
    """Periodic trigger evaluation. Runs until cancelled.

    Checks for due triggers (cooldown elapsed, conditions met) and logs them.
    Actual firing is left to the agent or explicit weft_trigger_fire calls —
    this loop surfaces what's ready so the primer can include it.
    """
    from weft.triggers import get_triggers_due

    logger.info("trigger_eval.started", extra={"interval": interval})
    try:
        while True:
            try:
                due = await get_triggers_due(pool)
                if due:
                    logger.info(
                        "trigger_eval.due_triggers",
                        extra={
                            "count": len(due),
                            "trigger_ids": [t.id for t in due],
                        },
                    )
            except Exception:
                logger.exception("trigger_eval.loop_error")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("trigger_eval.stopped")
        raise


async def _post_brief_to_slack(channel: str, brief_result) -> None:
    """Post the assembled brief to Slack via Block Kit."""
    import ssl

    import certifi

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.warning("daily_brief.slack.no_token")
        return

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    client = AsyncWebClient(token=token, ssl=ssl_ctx)

    try:
        response = await asyncio.wait_for(
            client.chat_postMessage(
                channel=channel,
                text=brief_result.markdown[:300],  # fallback text
                blocks=brief_result.slack_blocks,
            ),
            timeout=_SLACK_TIMEOUT,
        )
        if not response.get("ok"):
            logger.warning(
                "daily_brief.slack.api_error",
                extra={"error": response.get("error", "unknown")},
            )
    except Exception:
        logger.exception("daily_brief.slack.post_error")
