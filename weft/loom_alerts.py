"""Loom task awareness alerts — proactive alerts from Loom task state.

Queries the Loom `tasks` table directly (same Postgres instance) to detect:
1. Stale claims — claimed for 48h+ with no heartbeat/update
2. Epic completion readiness — all children done, epic still pending
3. Blocked pile-ups — 5+ tasks blocked in a single project

V2 dedup: each stale task / completable epic / blocked-project gets its
own dedup_key (``task:<id>`` / ``epic:<id>`` / ``project:<id>``), so
new staleness elsewhere can fire even if the same alert type fired for
a different target recently. Cooldowns come from
:class:`AlertCooldownConfig`. See :mod:`weft.alert_dedup`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg

from weft.alert_dedup import record_fire, should_fire
from weft.alerts import create_alert
from weft.config import AlertCooldownConfig
from weft.loom_query import (
    LoomQueryError,
    get_blocked_pile_ups,
    get_completable_epics,
    get_stale_claimed_tasks,
    loom_tables_exist,
)
from weft.models import AlertCreate, AlertType

logger = logging.getLogger(__name__)


@dataclass
class LoomAlertConfig:
    """Configuration for Loom task awareness alert thresholds."""

    stale_claim_hours: int = 48  # tasks claimed this long without heartbeat/update
    blocked_pile_up_threshold: int = 5  # blocked tasks per project before alerting


async def _filter_should_fire(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    keys: list[str],
    cooldown_minutes: float,
) -> list[str]:
    """Return the subset of *keys* that pass should_fire."""
    out: list[str] = []
    for k in keys:
        if await should_fire(
            pool, alert_type, k, cooldown_minutes=cooldown_minutes,
        ):
            out.append(k)
    return out


async def _record_fires(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    keys: list[str],
    alert_id: str,
) -> None:
    """Record fire for each dedup_key with a shared alert_id."""
    for k in keys:
        try:
            await record_fire(pool, alert_type, k, alert_id=alert_id)
        except Exception:
            logger.warning(
                "alert_dedup.record_fire_failed",
                extra={"alert_type": alert_type.value, "dedup_key": k},
                exc_info=True,
            )


async def check_stale_claims(
    pool: asyncpg.Pool,
    *,
    config: LoomAlertConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert on tasks claimed for 48h+ without update.

    Per-task dedup: a newly stale task fires even if another stale task
    already triggered an alert recently.
    """
    cfg = config or LoomAlertConfig()
    cd = cooldowns or AlertCooldownConfig()
    now = datetime.now(timezone.utc)

    try:
        rows = await get_stale_claimed_tasks(pool, threshold_hours=cfg.stale_claim_hours)
    except Exception:
        logger.warning("loom_alerts.stale_claims.query_error", exc_info=True)
        return []

    if not rows:
        return []

    cooldown_min = cd.minutes_for(AlertType.loom_stale_claim)
    keys = [f"task:{r['id']}" for r in rows]
    fireable = set(await _filter_should_fire(
        pool, AlertType.loom_stale_claim, keys, cooldown_min,
    ))

    fresh_rows = [r for r, k in zip(rows, keys) if k in fireable]
    if not fresh_rows:
        return []

    stale_list = []
    for r in fresh_rows:
        hours = (now - r["claimed_at"]).total_seconds() / 3600
        stale_list.append(
            f"- {r['title']} (claimed {hours:.0f}h ago by {r['assignee'] or 'unknown'}"
            f", project: {r['project_name'] or 'unknown'})"
        )

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_stale_claim,
            title=f"{len(fresh_rows)} stale claimed task(s) in Loom",
            body="Tasks claimed for 48h+ without update:\n" + "\n".join(stale_list),
            trigger_at=now,
        ),
    )
    await _record_fires(
        pool, AlertType.loom_stale_claim,
        [k for k in keys if k in fireable],
        alert.id,
    )
    return [alert.to_dict()]


async def check_epic_completion(
    pool: asyncpg.Pool,
    *,
    config: LoomAlertConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert when all children of an epic are done but the epic is still open.

    Per-epic dedup.
    """
    cfg = config or LoomAlertConfig()
    cd = cooldowns or AlertCooldownConfig()

    try:
        rows = await get_completable_epics(pool)
    except Exception:
        logger.warning("loom_alerts.epic_completion.query_error", exc_info=True)
        return []

    if not rows:
        return []

    cooldown_min = cd.minutes_for(AlertType.loom_epic_ready)
    keys = [f"epic:{r['id']}" for r in rows]
    fireable = set(await _filter_should_fire(
        pool, AlertType.loom_epic_ready, keys, cooldown_min,
    ))

    fresh_rows = [r for r, k in zip(rows, keys) if k in fireable]
    if not fresh_rows:
        return []

    now = datetime.now(timezone.utc)
    epic_list = [
        f"- {r['title']} ({r['child_count']} children done, project: {r['project_name'] or 'unknown'})"
        for r in fresh_rows
    ]

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_epic_ready,
            title=f"{len(fresh_rows)} epic(s) ready to close",
            body="All children are done/cancelled:\n" + "\n".join(epic_list),
            trigger_at=now,
        ),
    )
    await _record_fires(
        pool, AlertType.loom_epic_ready,
        [k for k in keys if k in fireable],
        alert.id,
    )
    return [alert.to_dict()]


async def check_blocked_pile_up(
    pool: asyncpg.Pool,
    *,
    config: LoomAlertConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert when blocked tasks exceed threshold in a single project.

    Per-project dedup.
    """
    cfg = config or LoomAlertConfig()
    cd = cooldowns or AlertCooldownConfig()

    try:
        rows = await get_blocked_pile_ups(pool, threshold=cfg.blocked_pile_up_threshold)
    except Exception:
        logger.warning("loom_alerts.blocked_pile_up.query_error", exc_info=True)
        return []

    if not rows:
        return []

    cooldown_min = cd.minutes_for(AlertType.loom_blocked_pile_up)
    keys = [f"project:{r['project_id']}" for r in rows]
    fireable = set(await _filter_should_fire(
        pool, AlertType.loom_blocked_pile_up, keys, cooldown_min,
    ))

    fresh_rows = [r for r, k in zip(rows, keys) if k in fireable]
    if not fresh_rows:
        return []

    now = datetime.now(timezone.utc)
    pile_list = [
        f"- {r['project_name']}: {r['blocked_count']} blocked tasks"
        for r in fresh_rows
    ]

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.loom_blocked_pile_up,
            title=f"Blocked task pile-up in {len(fresh_rows)} project(s)",
            body=f"Projects with {cfg.blocked_pile_up_threshold}+ blocked tasks:\n" + "\n".join(pile_list),
            trigger_at=now,
        ),
    )
    await _record_fires(
        pool, AlertType.loom_blocked_pile_up,
        [k for k in keys if k in fireable],
        alert.id,
    )
    return [alert.to_dict()]


async def evaluate_loom_alerts(
    pool: asyncpg.Pool,
    *,
    config: LoomAlertConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Run all Loom awareness checks. Returns list of created alert dicts.

    Gracefully skips if Loom tables don't exist in this database.
    """
    cfg = config or LoomAlertConfig()
    cd = cooldowns or AlertCooldownConfig()
    try:
        if not await loom_tables_exist(pool):
            logger.debug("loom_alerts.skipped — Loom tables not found")
            return []
    except LoomQueryError:
        logger.debug("loom_alerts.skipped — Loom DB not reachable")
        return []

    created: list[dict] = []
    created.extend(await check_stale_claims(pool, config=cfg, cooldowns=cd))
    created.extend(await check_epic_completion(pool, config=cfg, cooldowns=cd))
    created.extend(await check_blocked_pile_up(pool, config=cfg, cooldowns=cd))

    if created:
        logger.info("loom_alerts: created %d alerts", len(created))

    return created
