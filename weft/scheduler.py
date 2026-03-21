"""Alert scheduler — background loop that polls and dispatches due alerts.

Runs as an asyncio task within the MCP server lifespan. Polls the alerts
table at a configurable interval, dispatches each due alert via channel-
specific handlers, and marks them fired on success.

Design principles:
- Per-alert error isolation: one failed dispatch never crashes the loop
- Connection released before dispatch: no pool exhaustion risk
- Backpressure: polls again immediately when a full batch is returned
- Graceful shutdown: CancelledError is caught for clean logging
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import asyncpg

from weft.alerts import mark_alert_fired, poll_due_alerts
from weft.models import Alert

logger = logging.getLogger(__name__)

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
    """Dispatch an alert to Slack. Not yet implemented — see Epic 5 Slack task."""
    raise NotImplementedError(
        "Slack dispatch not yet implemented — see loom-7b19085a"
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
