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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

import asyncpg

from weft.alerts import is_daily_brief_due, mark_alert_fired, poll_due_alerts, release_alert
from weft.models import Alert

logger = logging.getLogger(__name__)

_SLACK_TIMEOUT = 10  # seconds for Slack API calls
# Kept as a patch seam for tests; the optional Slack SDK is loaded on first use.
AsyncWebClient = None

# Default settings — overridden by config in production
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_BATCH_SIZE = 50

# --- Alert dispatch registry ---

DispatchHandler = Callable[[Alert], Awaitable[None]]

_DISPATCH_REGISTRY: dict[str, DispatchHandler] = {}


def register_dispatch(channel: str, handler: DispatchHandler) -> None:
    """Register a dispatch handler for a channel."""
    _DISPATCH_REGISTRY[channel] = handler


# --- Outbound event registry ---
# Structurally separate from _DISPATCH_REGISTRY (which keys by alert.channel).
# This registry keys by *event name* and routes through a single active
# connector selected by the WEFT_OUTBOUND_CONNECTOR env var.
# Acceptable values: "slack", "discord", "none" (or unset → no-op).
# The discord handler self-registers from weft.discord.connector at import
# time; weft.scheduler.discord_bot_loop pulls that module in on startup.

OutboundEventHandler = Callable[..., Awaitable[None]]

_OUTBOUND_EVENT_REGISTRY: dict[str, dict[str, OutboundEventHandler]] = {}


def register_outbound_handler(event: str, connector: str, handler: OutboundEventHandler) -> None:
    """Register an outbound connector handler for an event.

    *event* — event name (e.g. "daily_brief")
    *connector* — connector name matching WEFT_OUTBOUND_CONNECTOR values (e.g. "slack")
    *handler* — async callable; receives keyword arguments specific to the event
    """
    _OUTBOUND_EVENT_REGISTRY.setdefault(event, {})[connector] = handler


async def emit_outbound_event(event: str, **kwargs) -> None:
    """Emit an outbound event through the active connector.

    Reads WEFT_OUTBOUND_CONNECTOR fresh at dispatch time so tests can
    monkeypatch the env var. When the connector is unset/none, logs at info level
    and returns silently without raising.
    """
    connector = os.environ.get("WEFT_OUTBOUND_CONNECTOR", "").strip().lower()
    if not connector or connector == "none":
        logger.info(
            "outbound_event.no_active_connector",
            extra={"event": event},
        )
        return
    handlers = _OUTBOUND_EVENT_REGISTRY.get(event, {})
    handler = handlers.get(connector)
    if handler is None:
        logger.warning(
            "outbound_event.no_handler",
            extra={"event": event, "connector": connector},
        )
        return
    await handler(**kwargs)


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

        global AsyncWebClient
        if AsyncWebClient is None:
            from slack_sdk.web.async_client import AsyncWebClient as slack_client

            AsyncWebClient = slack_client
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


# Register built-in alert dispatch handlers
register_dispatch("log", dispatch_log)
register_dispatch("slack", dispatch_slack)


async def _outbound_slack_brief(channel: str, brief_result) -> None:
    """Outbound connector wrapper: post a daily_brief event to Slack."""
    await _post_brief_to_slack(channel, brief_result)


# Register built-in outbound connector handlers
register_outbound_handler("daily_brief", "slack", _outbound_slack_brief)


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
                    # Release the durable reservation so the next cycle can retry.
                    try:
                        await release_alert(pool, alert.id)
                    except Exception:
                        logger.exception(
                            "scheduler.release_error",
                            extra={"alert_id": alert.id},
                        )

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


# --- Discord bot loop ---

# Floor mirrors slack_sync_loop's _MIN_SYNC_INTERVAL: anything tighter is
# just churn while a gateway reconnect is happening on its own clock.
_DISCORD_BOT_KEEPALIVE_INTERVAL = 60


async def discord_bot_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = _DISCORD_BOT_KEEPALIVE_INTERVAL,
) -> None:
    """Long-lived Discord gateway connection. Runs until cancelled.

    Reads WEFT_DISCORD_BOT_TOKEN and WEFT_DISCORD_BRIEF_CHANNEL_ID from env.
    On missing env, logs and returns (loop disabled). On startup, opens the
    gateway, registers the bot instance with weft.discord.connector so the
    outbound `daily_brief` handler can dispatch through it, then sleeps in
    *interval*-second ticks until cancelled. The discord.py client manages
    its own reconnect logic; this loop only owns the lifecycle.

    The pool parameter is unused for now — kept in the signature so the
    task can be started uniformly alongside the other scheduler loops.
    """
    from weft.auth import current_user_id

    token = os.environ.get("WEFT_DISCORD_BOT_TOKEN", "").strip()
    raw_channel = os.environ.get("WEFT_DISCORD_BRIEF_CHANNEL_ID", "").strip()
    if not token:
        logger.info("discord_bot.no_token — Discord bot loop disabled")
        return

    # Background scheduler tasks have no HTTP middleware setting the
    # request-scoped user identity. Discord-ingested memories belong to
    # the deployment owner — same pattern as slack_sync_loop. Without this,
    # every store_memory inside route() trips the migration-34 NOT NULL on
    # memories.user_id (asyncpg.exceptions.NotNullViolationError).
    # ContextVar set here propagates into discord.py's spawned tasks
    # (gateway loop, on_message dispatch, _ingest sub-task) via asyncio's
    # automatic context inheritance on create_task.
    default_uid = os.environ.get("WEFT_DEFAULT_USER_ID")
    if not default_uid:
        logger.warning(
            "discord_bot.no_default_user — set WEFT_DEFAULT_USER_ID to "
            "the deployment owner's UUID; inbound ingest would fail"
        )
    else:
        current_user_id.set(default_uid)
    if not raw_channel:
        logger.warning(
            "discord_bot.no_channel_id — set WEFT_DISCORD_BRIEF_CHANNEL_ID; loop disabled"
        )
        return
    try:
        channel_id = int(raw_channel)
    except ValueError:
        logger.error(
            "discord_bot.bad_channel_id — WEFT_DISCORD_BRIEF_CHANNEL_ID=%r is not an int",
            raw_channel,
        )
        return

    # Optional inbound watcher — enabled if ANY channel mapping is configured.
    # DEFAULT_CHANNEL_MAP is module-load populated from per-channel env vars
    # (WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID, WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID).
    # When the map is empty the watcher stays off and the bot keeps default
    # intents (no privileged message_content intent requested).
    #
    # idea_dump_channel_id is kept as the enable flag for backward compat;
    # any non-None value means "register on_message + request message_content
    # intent". The adapter's resolve_channel_mapping does the actual per-
    # channel routing — no in-bot filter on this value.
    from weft.discord.config import DEFAULT_CHANNEL_MAP

    idea_dump_channel_id: int | None = None
    raw_idea_dump = os.environ.get("WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID", "").strip()
    if raw_idea_dump:
        try:
            idea_dump_channel_id = int(raw_idea_dump)
        except ValueError:
            logger.error(
                "discord_bot.bad_idea_dump_channel_id — "
                "WEFT_DISCORD_IDEA_DUMP_CHANNEL_ID=%r is not an int",
                raw_idea_dump,
            )

    # If the idea-dump var isn't set but other channels are mapped (e.g.
    # WEFT_DISCORD_BRAIN_DUMP_CHANNEL_ID only), still enable inbound by
    # passing a sentinel non-None value so on_message gets registered.
    if idea_dump_channel_id is None and DEFAULT_CHANNEL_MAP:
        idea_dump_channel_id = 0  # sentinel: enable inbound, no channel-specific filter

    # Lazy import to keep discord.py off the import path for non-Discord
    # deployments and to avoid an import cycle (connector.py imports from
    # this module at load time).
    from weft.discord.bot import Bot
    from weft.discord.connector import clear_bot, set_bot

    bot = Bot(token, channel_id, pool=pool, idea_dump_channel_id=idea_dump_channel_id)
    logger.info("discord_bot.starting", extra={"channel_id": channel_id})
    try:
        await bot.start()
        set_bot(bot)
        while True:
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("discord_bot.stopped")
        raise
    finally:
        clear_bot()
        try:
            await bot.close()
        except Exception:
            logger.exception("discord_bot.close_error")


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

                        # Emit outbound event — routed by WEFT_OUTBOUND_CONNECTOR
                        await emit_outbound_event(
                            "daily_brief", channel=brief_channel, brief_result=result
                        )
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


# --- Quarantine LLM-review loop (Layer 3.5) ---

# Floor mirrors the existing Slack-sync minimum: a sub-minute interval
# would only burn API calls.
_MIN_QUARANTINE_REVIEW_INTERVAL = 60


async def quarantine_review_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = 21600,  # 6h default — agent writes don't accumulate fast
    limit: int = 100,
    concurrency: int = 4,
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    """Periodic Layer 3.5 LLM review of agent-provenance writes.

    Builds an ``AsyncAnthropic`` client once from the configured API key
    and reuses it across cycles. Skips quietly (no exception) if no API
    key is available — Layer 3 regex still applies. Per-cycle errors are
    logged and the loop continues.

    See ``weft.quarantine_review.llm_review_pending`` for the per-cycle
    semantics.
    """
    from anthropic import AsyncAnthropic

    from weft.config import load_config
    from weft.quarantine_review import llm_review_pending

    config = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY") or config.api_key
    if not api_key:
        logger.warning(
            "quarantine_review.no_api_key — Layer 3.5 LLM review disabled "
            "(Layer 3 regex still applies)"
        )
        return

    interval = max(interval, _MIN_QUARANTINE_REVIEW_INTERVAL)
    client = AsyncAnthropic(api_key=api_key)
    logger.info(
        "quarantine_review.started",
        extra={"interval": interval, "limit": limit, "concurrency": concurrency},
    )
    try:
        while True:
            t0 = time.monotonic()
            try:
                report = await llm_review_pending(
                    pool, client,
                    limit=limit,
                    concurrency=concurrency,
                    model=model,
                )
                elapsed = time.monotonic() - t0
                if report.checked or report.flagged or report.errors:
                    logger.info(
                        "quarantine_review.cycle",
                        extra={
                            "checked": report.checked,
                            "flagged": report.flagged,
                            "ambiguous": report.ambiguous,
                            "errors": len(report.errors),
                            "elapsed_s": round(elapsed, 1),
                        },
                    )
                if report.flagged:
                    logger.warning(
                        "quarantine_review.flagged_pending_review",
                        extra={
                            "flagged_count": report.flagged,
                            "flagged_ids": report.flagged_ids,
                        },
                    )
            except Exception:
                logger.exception("quarantine_review.cycle_error")
                # Loop continues — watermark is preserved on failure so the
                # next cycle picks up where this one left off.

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("quarantine_review.stopped")
        raise


# --- Re-ask feedback loop ---

_REASK_FEEDBACK_INTERVAL = 3600  # check every hour (matches memory_hygiene_loop cadence)
_REASK_WINDOW_MINUTES = 30  # same window as detect_reasked_queries default


async def _run_reask_feedback_pass(pool: asyncpg.Pool) -> int:
    """Execute one pass of the re-ask feedback loop. Returns pairs processed.

    Per-user fan-out (loom-fdd9282a). The pass enumerates the distinct users
    with unprocessed recall queries in the window, then processes each user
    in strict isolation via _run_reask_feedback_pass_for_user. No user's
    queries, memories, or boosts ever bleed into another user's pass — the
    enumerator is the ONLY cross-user read, and it is a system-context query
    that hands each user_id to a per-user scope.

    Designed to be callable in isolation (unit-testable without infinite loop
    machinery). Single-user deployments resolve to exactly one [None] user_id,
    so behavior there is unchanged.
    """
    from weft.store import get_distinct_reask_user_ids

    user_ids = await get_distinct_reask_user_ids(
        pool, window_minutes=_REASK_WINDOW_MINUTES
    )

    processed = 0
    for user_id in user_ids:
        processed += await _run_reask_feedback_pass_for_user(pool, user_id)
    return processed


async def _run_reask_feedback_pass_for_user(
    pool: asyncpg.Pool, user_id: str | None
) -> int:
    """Run one re-ask feedback pass scoped to a single user. Returns pairs processed.

    Steps:
      1. Fetch this user's recent unprocessed recall queries (explicit
         user scope — does NOT rely on RLS, which a system caller may bypass).
      2. Run pure detect_reasked_queries over the rows.
      3. For each (original, reask) pair, source the satisfying_memory_id from
         the user's OWN most-recently-accessed memory near the re-ask time.
      4. Call apply_reask_feedback — idempotent, so retries are safe.

    == Isolation (loom-fdd9282a) ==

    Both the query fetch and the satisfying-memory lookup carry an explicit
    ``user_id IS NOT DISTINCT FROM $user_id`` predicate, so attribution can
    never cross a tenant boundary even when the scheduler runs RLS-bypassing.
    The user identity is ALSO set on the contextvar for the duration of the
    pass, so the apply_reask_feedback write path runs under that user's RLS
    context as belt-and-suspenders.

    == Sourcing satisfying_memory_id ==

    weft_recall_queries records query text but NOT which memory IDs were
    returned per query. The current proxy picks the user's memory with the
    most recent accessed_at <= reask.created_at + 5s. It will be replaced when
    a per-query result-id column lands. If no recently-accessed memory is
    found, the pair is silently skipped — no boost, no stamp.
    """
    from weft.auth import current_user_id
    from weft.db.connection import get_db
    from weft.reask import detect_reasked_queries
    from weft.store import apply_reask_feedback, get_recent_recall_queries

    # Bind the user identity so nested write paths (apply_reask_feedback's
    # acquire()) run under this user's RLS context.
    token = current_user_id.set(user_id)
    try:
        rows = await get_recent_recall_queries(
            pool,
            window_minutes=_REASK_WINDOW_MINUTES,
            user_id=user_id,
            scope_to_user=True,
        )
        if not rows:
            return 0

        pairs = detect_reasked_queries(rows)
        if not pairs:
            return 0

        processed = 0
        for original, reask in pairs:
            try:
                # Source the satisfying_memory_id from THIS user's memories only:
                # the most recent accessed_at <= re-ask time (+5s grace for async
                # logging lag). The explicit user_id predicate is the tenant
                # boundary — never source another user's memory.
                row = await get_db(pool).fetchrow(
                    """
                    SELECT id FROM memories
                    WHERE accessed_at <= $1::timestamptz + interval '5 seconds'
                      AND user_id IS NOT DISTINCT FROM $2
                    ORDER BY accessed_at DESC
                    LIMIT 1
                    """,
                    reask.created_at,
                    user_id,
                )
                if row is None:
                    logger.debug(
                        "reask_feedback.no_satisfying_memory",
                        extra={
                            "original_query_id": original.query_id,
                            "reask_query_id": reask.query_id,
                        },
                    )
                    continue

                satisfying_memory_id = row["id"]
                result = await apply_reask_feedback(
                    pool, original.query_id, satisfying_memory_id
                )
                if result is not None:
                    # Fresh claim: EMA boost was applied.
                    logger.info(
                        "reask_feedback.boosted",
                        extra={
                            "original_query_id": original.query_id,
                            "satisfying_memory_id": satisfying_memory_id,
                            "new_usefulness_score": result.get("usefulness_score"),
                        },
                    )
                    processed += 1
                # result=None means already processed (idempotent no-op).
            except Exception:
                logger.exception(
                    "reask_feedback.pair_error",
                    extra={
                        "original_query_id": original.query_id,
                        "reask_query_id": reask.query_id,
                    },
                )
                # Per-pair isolation: one failure does not abort the rest.

        return processed
    finally:
        current_user_id.reset(token)


async def reask_feedback_loop(
    pool: asyncpg.Pool,
    *,
    interval: int = _REASK_FEEDBACK_INTERVAL,
) -> None:
    """Periodic re-ask detection and usefulness-score correction. Runs until cancelled.

    On each pass, fetches recent recall queries, detects near-duplicate re-asks
    (queries repeated because the first retrieval missed), and applies a usefulness
    boost (EMA) to the memory that satisfied the re-ask.

    This is the energizing loop for the compounding-recall improvement cycle.
    Each pass is idempotent: apply_reask_feedback atomically claims miss rows,
    so retries and scheduler restarts cannot double-boost any score.

    Interval defaults to 3600s (1 hour) — same cadence as memory_hygiene_loop
    and loom_awareness_loop — because re-ask correction is not latency-sensitive.
    """
    logger.info("reask_feedback.started", extra={"interval": interval})
    try:
        while True:
            try:
                processed = await _run_reask_feedback_pass(pool)
                if processed:
                    logger.info(
                        "reask_feedback.cycle_complete",
                        extra={"pairs_processed": processed},
                    )
            except Exception:
                logger.exception("reask_feedback.loop_error")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("reask_feedback.stopped")
        raise


# --- Recall canary audit loop ---

# The canary audit is a *daily* health check, but the loop polls more often so a
# server restart (Fly bluegreen deploy, machine auto-start) doesn't reset a
# single long sleep. Each poll asks whether an audit is actually due before
# running: a bare ``sleep(86400)`` would re-audit on every boot, inflating the
# per-probe ``audit_count`` / ``miss_count`` counters (which are not yet
# transactional — see the Phase 0 audit P2s).
_CANARY_AUDIT_POLL_INTERVAL = 3600   # check hourly whether a daily audit is due
_CANARY_AUDIT_MIN_AGE_HOURS = 23     # don't re-run within ~a day (restart-safe)
_CANARY_ERROR_MAX_CHARS = 1000


@dataclass
class CanaryAuditRuntimeState:
    """Observable state for the current process's canary scheduler.

    ``recall_canary.last_audit_at`` remains the durable, restart-safe marker of
    successful probe execution. This state explains what the live process has
    attempted since startup and why repeated hourly retries may be failing.
    """

    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int = 0
    last_exception: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "last_attempt_at": (
                self.last_attempt_at.isoformat() if self.last_attempt_at else None
            ),
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "last_exception": self.last_exception,
        }


async def _canary_audit_due(
    pool: asyncpg.Pool, user_id: str, *, min_age_hours: float = _CANARY_AUDIT_MIN_AGE_HOURS
) -> bool:
    """Return True when the daily canary audit should run for *user_id*.

    Uses ``max(last_audit_at)`` across the user's enabled probes as the
    "last run" marker so the cadence survives restarts (a sleep-only loop would
    re-audit on every boot). Returns True when no probe has ever been audited
    (all ``last_audit_at`` NULL) so the first run after enrollment proceeds, and
    fail-open True on query error — better to run than to silently never audit.

    The ``current_user_id`` contextvar is bound for the duration so the read
    runs under this user's RLS context (the scheduler is otherwise
    unauthenticated; ``get_db`` returns the raw pool with no ``app.user_id``).
    The explicit ``user_id = $1`` predicate is the belt-and-suspenders tenant
    boundary regardless.
    """
    from datetime import datetime, timezone

    from weft.auth import current_user_id
    from weft.db.connection import get_db

    token = current_user_id.set(user_id)
    try:
        last = await get_db(pool).fetchval(
            """
            SELECT max(last_audit_at) FROM recall_canary
            WHERE enabled = TRUE AND user_id = $1
            """,
            user_id,
        )
    except Exception:
        logger.exception("canary_audit.due_check_error")
        return True
    finally:
        current_user_id.reset(token)

    if last is None:
        return True
    age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return age_hours >= min_age_hours


async def canary_audit_loop(
    pool: asyncpg.Pool,
    embedder,
    *,
    interval: int = _CANARY_AUDIT_POLL_INTERVAL,
    min_age_hours: float = _CANARY_AUDIT_MIN_AGE_HOURS,
    runtime_state: CanaryAuditRuntimeState | None = None,
) -> None:
    """Periodic recall-canary audit — the reconciliation meter. Runs until cancelled.

    Wires ``weft.canary.run_canary_audit`` to a daily cadence. Only the
    high-confidence ``reaREDACTED`` probes run: ``active_probing_enabled``
    stays at its ``False`` default until the active-probe miss baseline is
    calibrated (RI-4).

    Background tasks use a service role that may bypass RLS. Each poll therefore
    enumerates canary owners once, checks cadence per owner, and runs every audit
    with explicit tenant predicates. Owners are processed sequentially to avoid
    an embedding burst as the deployment grows.
    """
    logger.info(
        "canary_audit.started",
        extra={
            "interval": interval,
            "min_age_hours": min_age_hours,
            "mode": "per_user",
        },
    )
    state = runtime_state or CanaryAuditRuntimeState()
    try:
        while True:
            state.last_attempt_at = datetime.now(timezone.utc)
            try:
                summary = await _run_canary_audit_pass(
                    pool, embedder, min_age_hours=min_age_hours
                )
                state.last_success_at = datetime.now(timezone.utc)
                state.consecutive_failures = 0
                state.last_exception = None
                logger.info("canary_audit.pass", extra=summary)
            except Exception as exc:
                state.consecutive_failures += 1
                detail = f"{type(exc).__name__}: {exc}"
                state.last_exception = detail[:_CANARY_ERROR_MAX_CHARS]
                logger.exception(
                    "canary_audit.loop_error",
                    extra={"consecutive_failures": state.consecutive_failures},
                )

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("canary_audit.stopped")
        raise


async def _run_canary_audit_pass(
    pool: asyncpg.Pool,
    embedder,
    *,
    min_age_hours: float = _CANARY_AUDIT_MIN_AGE_HOURS,
) -> dict[str, int]:
    """Run one sequential, RLS-scoped pass across canary owners.

    Owner discovery is the one intentional system-level query. Every due check,
    probe read, vector search, counter update, and audit-event insert then runs
    on a connection acquired with that owner's ``app.user_id``. Merely setting
    ``current_user_id`` is insufficient: without :func:`acquire`, ``get_db``
    returns the raw pool and PostgreSQL never receives the RLS GUC.
    """
    from weft.auth import current_user_id
    from weft.canary import list_canary_user_ids, run_canary_audit
    from weft.db.connection import acquire

    configured_owner = os.environ.get("WEFT_DEFAULT_USER_ID")
    if configured_owner:
        # Hosted Weft runs under a deliberately non-BYPASSRLS role. It cannot
        # enumerate arbitrary owners, nor should it. The deployment owner is an
        # explicit part of the scheduler contract and matches health/brief scope.
        user_ids = [configured_owner]
    else:
        # Local/test service-role contexts may support discovery. An empty list
        # is valid there: a fresh local database can genuinely have no probes.
        user_ids = await list_canary_user_ids(pool)
    summary = {
        "owners_considered": len(user_ids),
        "owners_audited": 0,
        "probes_checked": 0,
        "misses": 0,
    }
    for user_id in user_ids:
        token = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                if not await _canary_audit_due(
                    pool, user_id, min_age_hours=min_age_hours
                ):
                    continue
                result = await run_canary_audit(
                    pool,
                    embedder,
                    user_id=user_id,
                    active_probing_enabled=True,
                )
        finally:
            current_user_id.reset(token)

        if not result.get("audit_valid"):
            logger.info(
                "canary_audit.skipped",
                extra={
                    "user_id": user_id,
                    "status": result.get("status"),
                    "bootstrap_synced": result.get("bootstrap_synced"),
                },
            )
            continue
        summary["owners_audited"] += 1
        summary["probes_checked"] += int(result.get("probes_checked", 0))
        summary["misses"] += int(result.get("misses", 0))
        logger.info(
            "canary_audit.cycle",
            extra={
                "user_id": user_id,
                "probes_checked": result.get("probes_checked"),
                "misses": result.get("misses"),
                "miss_rate": result.get("miss_rate"),
                "bootstrap_synced": result.get("bootstrap_synced"),
            },
        )
    return summary


async def _post_brief_to_slack(channel: str, brief_result) -> None:
    """Post the assembled brief to Slack via Block Kit."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.warning("daily_brief.slack.no_token")
        return

    import ssl

    import certifi

    global AsyncWebClient
    if AsyncWebClient is None:
        from slack_sdk.web.async_client import AsyncWebClient as slack_client

        AsyncWebClient = slack_client

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
