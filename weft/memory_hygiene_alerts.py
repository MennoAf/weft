"""Memory hygiene alerts — proactive alerts for memory health.

Periodic checks for:
1. Stale decisions — decisions past review_after or old with low confidence
2. Consolidation overdue — weft_consolidate hasn't run in N days
3. Memory count threshold — active memory count crosses configurable limit
4. Contradiction auto-alerts — called from weft_remember when contradictions detected

Each periodic check has 24h dedup to avoid alert spam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.alerts import create_alert, list_alerts
from weft.db.connection import get_db
from weft.models import AlertCreate, AlertStatus, AlertType

logger = logging.getLogger(__name__)


@dataclass
class MemoryHygieneConfig:
    """Configuration for memory hygiene alert thresholds."""

    stale_decision_days: int = 90  # decisions older than this with low confidence
    stale_decision_confidence: float = 0.6  # below this = low confidence
    consolidation_overdue_hours: int = 72  # nudge if consolidation hasn't run in 3 days
    memory_count_threshold: int = 1000  # alert when active memories exceed this
    dedup_hours: int = 24  # suppress duplicate alerts within this window


async def _recent_alert_types(pool: asyncpg.Pool, *, dedup_hours: int = 24) -> set[str]:
    """Collect alert types created in the last N hours for dedup."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=dedup_hours)
    pending = await list_alerts(pool, status=AlertStatus.pending, limit=200)
    fired = await list_alerts(pool, status=AlertStatus.fired, limit=200)
    return {
        a.alert_type.value
        for a in pending + fired
        if a.created_at >= cutoff
    }


async def check_stale_decisions(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
) -> list[dict]:
    """Alert on decisions with overdue review_after or old + low confidence.

    Two triggers:
    1. review_after date has passed (explicit review schedule)
    2. Decision older than N days with confidence < threshold (implicit staleness)
    """
    cfg = config or MemoryHygieneConfig()
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=cfg.stale_decision_days)

    try:
        # Overdue review_after
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

        # Old + low confidence (no review_after set)
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

    recent_types = await _recent_alert_types(pool, dedup_hours=cfg.dedup_hours)
    if AlertType.stale_decision.value in recent_types:
        return []

    items = []
    for r in overdue_rows:
        preview = r["content"][:80].replace("\n", " ")
        items.append(f"- [overdue review] {preview}…")
    for r in old_low_rows:
        preview = r["content"][:80].replace("\n", " ")
        items.append(f"- [low confidence: {r['confidence']:.1f}] {preview}…")

    alert = await create_alert(
        pool,
        AlertCreate(
            alert_type=AlertType.stale_decision,
            title=f"{total} stale decision(s) need review",
            body="Decisions that may be outdated:\n" + "\n".join(items[:10]),
            trigger_at=now,
        ),
    )
    return [alert.to_dict()]


async def check_consolidation_overdue(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
) -> list[dict]:
    """Alert when weft_consolidate hasn't run in N+ hours."""
    cfg = config or MemoryHygieneConfig()
    from weft.store import get_metadata

    now = datetime.now(timezone.utc)

    try:
        meta = await get_metadata(pool, "last_consolidation_run")
    except Exception:
        logger.warning("memory_hygiene.consolidation.query_error", exc_info=True)
        return []

    if meta is None:
        # Never run — but don't alert on brand new installs.
        # Check if there are enough memories to warrant consolidation.
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

    recent_types = await _recent_alert_types(pool, dedup_hours=cfg.dedup_hours)
    if AlertType.memory_consolidation_overdue.value in recent_types:
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
    return [alert.to_dict()]


async def check_memory_count(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
) -> list[dict]:
    """Alert when active memory count exceeds threshold."""
    cfg = config or MemoryHygieneConfig()
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

    recent_types = await _recent_alert_types(pool, dedup_hours=cfg.dedup_hours)
    if AlertType.memory_count_threshold.value in recent_types:
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
    return [alert.to_dict()]


async def create_contradiction_alert(
    pool: asyncpg.Pool,
    *,
    new_memory_id: str,
    contradictions: list[dict],
) -> dict | None:
    """Create an alert for detected contradictions during weft_remember.

    Called from weft_remember when check_contradictions finds conflicts.
    Returns the created alert dict, or None if deduped.
    """
    if not contradictions:
        return None

    now = datetime.now(timezone.utc)

    # Dedup: don't fire if we already alerted on contradictions recently
    recent_types = await _recent_alert_types(pool)
    if AlertType.memory_contradiction.value in recent_types:
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
    return alert.to_dict()


async def evaluate_memory_hygiene_alerts(
    pool: asyncpg.Pool,
    *,
    config: MemoryHygieneConfig | None = None,
) -> list[dict]:
    """Run all memory hygiene checks. Returns list of created alert dicts."""
    cfg = config or MemoryHygieneConfig()
    created: list[dict] = []
    created.extend(await check_stale_decisions(pool, config=cfg))
    created.extend(await check_consolidation_overdue(pool, config=cfg))
    created.extend(await check_memory_count(pool, config=cfg))
    # Note: contradiction alerts are created inline from weft_remember,
    # not from the periodic scheduler.

    if created:
        logger.info("memory_hygiene: created %d alerts", len(created))

    return created
