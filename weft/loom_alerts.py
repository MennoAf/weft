"""Loom task awareness alerts — proactive alerts from Loom task state.

Queries the Loom `tasks` table directly (same Postgres instance) to detect:
1. Stale claims — claimed for 48h+ with no heartbeat/update
2. Epic completion readiness — all children done, epic still pending
3. Blocked pile-ups — 5+ tasks blocked in a single project

Each check has 24h dedup to avoid alert spam.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.alerts import create_alert, list_alerts
from weft.loom_query import (
    LoomQueryError,
    get_blocked_pile_ups,
    get_completable_epics,
    get_stale_claimed_tasks,
    loom_tables_exist,
)
from weft.models import AlertCreate, AlertStatus, AlertType

logger = logging.getLogger(__name__)

# Thresholds
_STALE_CLAIM_HOURS = 48
_BLOCKED_PILE_UP_THRESHOLD = 5
_DEDUP_HOURS = 24


async def _recent_alert_types(pool: asyncpg.Pool) -> set[str]:
    """Collect alert types created in the last 24h for dedup."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_DEDUP_HOURS)
    pending = await list_alerts(pool, status=AlertStatus.pending, limit=200)
    fired = await list_alerts(pool, status=AlertStatus.fired, limit=200)
    return {
        a.alert_type.value
        for a in pending + fired
        if a.created_at >= cutoff
    }


async def check_stale_claims(pool: asyncpg.Pool) -> list[dict]:
    """Alert on tasks claimed for 48h+ without update.

    These are tasks where claimed_at is old and claim_expires_at has passed
    (or the task is still claimed but no heartbeat has refreshed the TTL).
    The Loom daemon handles retries, but this alerts the human.
    """
    now = datetime.now(timezone.utc)

    try:
        rows = await get_stale_claimed_tasks(pool, threshold_hours=_STALE_CLAIM_HOURS)
    except Exception:
        logger.warning("loom_alerts.stale_claims.query_error", exc_info=True)
        return []

    if not rows:
        return []

    recent_types = await _recent_alert_types(pool)
    if AlertType.loom_stale_claim.value in recent_types:
        return []

    # Build a single summary alert
    stale_list = []
    for r in rows:
        hours = (now - r["claimed_at"]).total_seconds() / 3600
        stale_list.append(
            f"- {r['title']} (claimed {hours:.0f}h ago by {r['assignee'] or 'unknown'}"
            f", project: {r['project_name'] or 'unknown'})"
        )

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_stale_claim,
            title=f"{len(rows)} stale claimed task(s) in Loom",
            body="Tasks claimed for 48h+ without update:\n" + "\n".join(stale_list),
            trigger_at=now,
        ),
    )
    return [alert.to_dict()]


async def check_epic_completion(pool: asyncpg.Pool) -> list[dict]:
    """Alert when all children of an epic are done but the epic is still open.

    Uses Loom's parent_id relationship — an epic is a task with children.
    """
    try:
        rows = await get_completable_epics(pool)
    except Exception:
        logger.warning("loom_alerts.epic_completion.query_error", exc_info=True)
        return []

    if not rows:
        return []

    recent_types = await _recent_alert_types(pool)
    if AlertType.loom_epic_ready.value in recent_types:
        return []

    now = datetime.now(timezone.utc)
    epic_list = [
        f"- {r['title']} ({r['child_count']} children done, project: {r['project_name'] or 'unknown'})"
        for r in rows
    ]

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_epic_ready,
            title=f"{len(rows)} epic(s) ready to close",
            body="All children are done/cancelled:\n" + "\n".join(epic_list),
            trigger_at=now,
        ),
    )
    return [alert.to_dict()]


async def check_blocked_pile_up(pool: asyncpg.Pool) -> list[dict]:
    """Alert when 5+ tasks are blocked in a single project."""
    try:
        rows = await get_blocked_pile_ups(pool, threshold=_BLOCKED_PILE_UP_THRESHOLD)
    except Exception:
        logger.warning("loom_alerts.blocked_pile_up.query_error", exc_info=True)
        return []

    if not rows:
        return []

    recent_types = await _recent_alert_types(pool)
    if AlertType.loom_blocked_pile_up.value in recent_types:
        return []

    now = datetime.now(timezone.utc)
    pile_list = [
        f"- {r['project_name']}: {r['blocked_count']} blocked tasks"
        for r in rows
    ]

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_blocked_pile_up,
            title=f"Blocked task pile-up in {len(rows)} project(s)",
            body="Projects with 5+ blocked tasks:\n" + "\n".join(pile_list),
            trigger_at=now,
        ),
    )
    return [alert.to_dict()]


async def evaluate_loom_alerts(pool: asyncpg.Pool) -> list[dict]:
    """Run all Loom awareness checks. Returns list of created alert dicts.

    Gracefully skips if Loom tables don't exist in this database.
    """
    try:
        if not await loom_tables_exist(pool):
            logger.debug("loom_alerts.skipped — Loom tables not found")
            return []
    except LoomQueryError:
        logger.debug("loom_alerts.skipped — Loom DB not reachable")
        return []

    created: list[dict] = []
    created.extend(await check_stale_claims(pool))
    created.extend(await check_epic_completion(pool))
    created.extend(await check_blocked_pile_up(pool))

    if created:
        logger.info("loom_alerts: created %d alerts", len(created))

    return created
