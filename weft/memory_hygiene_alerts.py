"""Memory hygiene alerts — proactive alerts for memory health.

Periodic checks for:
1. Stale decisions — decisions past review_after or old with low confidence
2. Consolidation overdue — weft_consolidate hasn't run in N days
3. Memory count threshold — active memory count crosses configurable limit
4. Contradiction auto-alerts — called from weft_remember when contradictions detected

V2 dedup: per-decision dedup for stale decisions and contradictions
(``memory:<id>``), singleton ``"global"`` dedup for the overdue and
count-threshold checks. Cooldowns from :class:`AlertCooldownConfig`.
See :mod:`weft.alert_dedup`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.alert_dedup import record_fire, should_fire
from weft.alerts import create_alert
from weft.config import AlertCooldownConfig
from weft.db.connection import get_db
from weft.models import AlertCreate, AlertType

logger = logging.getLogger(__name__)


@dataclass
class MemoryHygieneConfig:
    """Configuration for memory hygiene alert thresholds."""

    stale_decision_days: int = 90  # decisions older than this with low confidence
    stale_decision_confidence: float = 0.6  # below this = low confidence
    consolidation_overdue_hours: int = 72  # nudge if consolidation hasn't run in 3 days
    memory_count_threshold: int = 1000  # alert when active memories exceed this


async def check_stale_decisions(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert on decisions with overdue review_after or old + low confidence.

    Two triggers:
    1. review_after date has passed (explicit review schedule)
    2. Decision older than N days with confidence < threshold (implicit staleness)

    Per-decision dedup: each memory has its own cooldown so a newly-stale
    decision triggers an alert even if a different decision triggered one
    yesterday.
    """
    cfg = config or MemoryHygieneConfig()
    cd = cooldowns or AlertCooldownConfig()
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=cfg.stale_decision_days)

    try:
        overdue_rows = await get_db(pool).fetch(
            """
            SELECT id, content, review_after, confidence
            FROM memories
            WHERE status = 'active'
              AND type = 'decision'
              AND review_after IS NOT NULL
              AND review_after <= $1
            ORDER BY review_after ASC
            LIMIT 20
            """,
            now,
        )

        old_low_rows = await get_db(pool).fetch(
            """
            SELECT id, content, created_at, confidence
            FROM memories
            WHERE status = 'active'
              AND type = 'decision'
              AND review_after IS NULL
              AND created_at < $1
              AND confidence < $2
            ORDER BY confidence ASC
            LIMIT 20
            """,
            stale_cutoff,
            cfg.stale_decision_confidence,
        )
    except Exception:
        logger.warning("memory_hygiene.stale_decisions.query_error", exc_info=True)
        return []

    total = len(overdue_rows) + len(old_low_rows)
    if total == 0:
        return []

    cooldown_min = cd.minutes_for(AlertType.stale_decision)

    # Filter both row sets by per-decision should_fire.
    fireable_keys: set[str] = set()
    fresh_overdue = []
    for r in overdue_rows:
        key = f"memory:{r['id']}"
        if await should_fire(
            pool, AlertType.stale_decision, key, cooldown_minutes=cooldown_min,
        ):
            fresh_overdue.append(r)
            fireable_keys.add(key)
    fresh_old_low = []
    for r in old_low_rows:
        key = f"memory:{r['id']}"
        if await should_fire(
            pool, AlertType.stale_decision, key, cooldown_minutes=cooldown_min,
        ):
            fresh_old_low.append(r)
            fireable_keys.add(key)

    if not fresh_overdue and not fresh_old_low:
        return []

    items = []
    for r in fresh_overdue:
        preview = r["content"][:80].replace("\n", " ")
        items.append(f"- [overdue review] {preview}…")
    for r in fresh_old_low:
        preview = r["content"][:80].replace("\n", " ")
        items.append(f"- [low confidence: {r['confidence']:.1f}] {preview}…")

    fresh_total = len(fresh_overdue) + len(fresh_old_low)
    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.stale_decision,
            title=f"{fresh_total} stale decision(s) need review",
            body="Decisions that may be outdated:\n" + "\n".join(items[:10]),
            trigger_at=now,
        ),
    )
    for key in fireable_keys:
        try:
            await record_fire(pool, AlertType.stale_decision, key, alert_id=alert.id)
        except Exception:
            logger.warning(
                "alert_dedup.record_fire_failed",
                extra={"alert_type": "stale_decision", "dedup_key": key},
                exc_info=True,
            )
    return [alert.to_dict()]


async def check_consolidation_overdue(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert when weft_consolidate hasn't run in N+ hours.

    Singleton ``"global"`` dedup key — there's only one consolidation
    state, so no per-target distinction.
    """
    cfg = config or MemoryHygieneConfig()
    cd = cooldowns or AlertCooldownConfig()
    from weft.store import get_metadata

    now = datetime.now(timezone.utc)

    try:
        meta = await get_metadata(pool, "last_consolidation_run")
    except Exception:
        logger.warning("memory_hygiene.consolidation.query_error", exc_info=True)
        return []

    if meta is None:
        try:
            count = await get_db(pool).fetchval(
                "SELECT count(*) FROM memories WHERE status = 'active'"
            )
        except Exception:
            return []
        if count < 50:
            return []
        is_overdue = True
        hours_since = None
    else:
        ran_at = meta.get("ran_at")
        if not ran_at:
            return []
        last_run = datetime.fromisoformat(ran_at)
        hours_since = (now - last_run).total_seconds() / 3600
        is_overdue = hours_since >= cfg.consolidation_overdue_hours

    if not is_overdue:
        return []

    cooldown_min = cd.minutes_for(AlertType.memory_consolidation_overdue)
    if not await should_fire(
        pool, AlertType.memory_consolidation_overdue, "global",
        cooldown_minutes=cooldown_min,
    ):
        return []

    body = (
        f"Last consolidation was {hours_since:.0f}h ago."
        if hours_since
        else "Consolidation has never been run."
    )

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.memory_consolidation_overdue,
            title="Memory consolidation overdue",
            body=f"{body} Run weft_consolidate to decay stale memories and merge duplicates.",
            trigger_at=now,
        ),
    )
    await record_fire(
        pool, AlertType.memory_consolidation_overdue, "global", alert_id=alert.id,
    )
    return [alert.to_dict()]


async def check_memory_count(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Alert when active memory count exceeds threshold.

    Singleton ``"global"`` dedup key.
    """
    cfg = config or MemoryHygieneConfig()
    cd = cooldowns or AlertCooldownConfig()
    now = datetime.now(timezone.utc)

    try:
        count = await get_db(pool).fetchval(
            "SELECT count(*) FROM memories WHERE status = 'active'"
        )
    except Exception:
        logger.warning("memory_hygiene.count.query_error", exc_info=True)
        return []

    if count < cfg.memory_count_threshold:
        return []

    cooldown_min = cd.minutes_for(AlertType.memory_count_threshold)
    if not await should_fire(
        pool, AlertType.memory_count_threshold, "global",
        cooldown_minutes=cooldown_min,
    ):
        return []

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.memory_count_threshold,
            title=f"Memory count threshold reached: {count} active memories",
            body=(
                f"Active memory count ({count}) exceeds {cfg.memory_count_threshold}. "
                f"Consider running weft_consolidate to merge duplicates and archive stale memories."
            ),
            trigger_at=now,
        ),
    )
    await record_fire(
        pool, AlertType.memory_count_threshold, "global", alert_id=alert.id,
    )
    return [alert.to_dict()]


async def create_contradiction_alert(
    pool: asyncpg.Pool,
    *,
    new_memory_id: str,
    contradictions: list[dict],
    cooldowns: AlertCooldownConfig | None = None,
) -> dict | None:
    """Create an alert for detected contradictions during weft_remember.

    Per-new-memory dedup: ``memory:<new_memory_id>``. A contradiction
    alert for memory A doesn't suppress one for memory B.

    Called from weft_remember when check_contradictions finds conflicts.
    Returns the created alert dict, or None if deduped.
    """
    if not contradictions:
        return None

    cd = cooldowns or AlertCooldownConfig()
    now = datetime.now(timezone.utc)
    dedup_key = f"memory:{new_memory_id}"
    cooldown_min = cd.minutes_for(AlertType.memory_contradiction)

    if not await should_fire(
        pool, AlertType.memory_contradiction, dedup_key,
        cooldown_minutes=cooldown_min,
    ):
        return None

    items = []
    for c in contradictions[:5]:
        preview = c.get("content_preview", "")[:80]
        sim = c.get("similarity", 0)
        items.append(f"- {c.get('memory_id', '?')}: \"{preview}\" (sim: {sim:.2f})")

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.memory_contradiction,
            title=f"Memory contradiction detected ({len(contradictions)} conflict(s))",
            body=(
                f"New memory {new_memory_id} may contradict:\n"
                + "\n".join(items)
            ),
            trigger_at=now,
        ),
    )
    await record_fire(
        pool, AlertType.memory_contradiction, dedup_key, alert_id=alert.id,
    )
    return alert.to_dict()


async def evaluate_memory_hygiene_alerts(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
    cooldowns: AlertCooldownConfig | None = None,
) -> list[dict]:
    """Run all memory hygiene checks. Returns list of created alert dicts."""
    cfg = config or MemoryHygieneConfig()
    cd = cooldowns or AlertCooldownConfig()
    created: list[dict] = []
    created.extend(await check_stale_decisions(pool, config=cfg, cooldowns=cd))
    created.extend(await check_consolidation_overdue(pool, config=cfg, cooldowns=cd))
    created.extend(await check_memory_count(pool, config=cfg, cooldowns=cd))

    if created:
        logger.info("memory_hygiene: created %d alerts", len(created))

    return created
