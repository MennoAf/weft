"""Alert store — CRUD + polling for the proactive push system.

Follows the same patterns as modes.py: get_db() for RLS-aware queries,
nullif(current_setting('app.user_id', true), '') for user_id insertion.

The scheduler (weft/scheduler.py) calls poll_due_alerts() and mark_alert_fired()
to process alerts that have reached their trigger_at time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg

from weft.db.connection import get_db
from weft.models import Alert, AlertCreate, AlertStatus, _weft_id

logger = logging.getLogger(__name__)


async def create_alert(pool: asyncpg.Pool, create: AlertCreate) -> Alert:
    """Insert a new alert. Returns the created Alert."""
    alert_id = _weft_id()
    payload_json = json.dumps(create.payload)

    db = get_db(pool)
    row = await db.fetchrow(
        """
        INSERT INTO alerts (
            id, user_id, alert_type, title, body, trigger_at,
            status, channel, channel_target, payload, project_id, agent_id
        )
        VALUES (
            $1,
            nullif(current_setting('app.user_id', true), ''),
            $2, $3, $4, $5,
            'pending', $6, $7, $8::jsonb, $9, $10
        )
        RETURNING *
        """,
        alert_id,
        create.alert_type.value,
        create.title,
        create.body,
        create.trigger_at,
        create.channel.value,
        create.channel_target,
        payload_json,
        create.project_id,
        create.agent_id,
    )
    return _row_to_alert(row)


async def get_alert(pool: asyncpg.Pool, alert_id: str) -> Alert | None:
    """Fetch an alert by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM alerts WHERE id = $1",
        alert_id,
    )
    return _row_to_alert(row) if row else None


async def list_alerts(
    pool: asyncpg.Pool,
    *,
    status: AlertStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Alert]:
    """List alerts for the current user, newest first."""
    if status is not None:
        rows = await get_db(pool).fetch(
            """
            SELECT * FROM alerts
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            status.value,
            limit,
            offset,
        )
    else:
        rows = await get_db(pool).fetch(
            """
            SELECT * FROM alerts
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return [_row_to_alert(r) for r in rows]


async def dismiss_alert(pool: asyncpg.Pool, alert_id: str) -> bool:
    """Dismiss an alert (set status to 'dismissed'). Returns True if updated."""
    result = await get_db(pool).execute(
        """
        UPDATE alerts SET status = 'dismissed'
        WHERE id = $1 AND status = 'pending'
        """,
        alert_id,
    )
    return result.split()[-1] != "0"


_PROCESSING_LEASE = timedelta(minutes=5)


async def poll_due_alerts(
    pool: asyncpg.Pool,
    batch_size: int = 50,
) -> list[Alert]:
    """Poll for pending alerts whose trigger_at has passed.

    Uses SELECT FOR UPDATE SKIP LOCKED to reserve rows atomically while
    multiple scheduler instances are running. Reservation persists after the
    transaction ends because dispatch happens outside the transaction.

    Returns up to batch_size alerts as Python objects (connection-free).
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                WITH due AS (
                    SELECT id
                    FROM alerts
                    WHERE (
                        status = 'pending'
                        AND trigger_at <= now()
                    ) OR (
                        status = 'processing'
                        AND processing_at < now() - $2::interval
                    )
                    ORDER BY trigger_at ASC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE alerts AS a
                SET status = 'processing', processing_at = now()
                FROM due
                WHERE a.id = due.id
                RETURNING a.*
                """,
                batch_size,
                _PROCESSING_LEASE,
            )
            return [_row_to_alert(r) for r in rows]


async def mark_alert_fired(pool: asyncpg.Pool, alert_id: str) -> bool:
    """Mark a reserved alert as fired.

    Returns True if the alert was updated, False if already fired/dismissed.
    """
    result = await pool.execute(
        """
        UPDATE alerts
        SET status = 'fired', fired_at = now(), processing_at = NULL
        WHERE id = $1 AND status IN ('pending', 'processing')
        """,
        alert_id,
    )
    return result.split()[-1] != "0"


async def release_alert(pool: asyncpg.Pool, alert_id: str) -> bool:
    """Return a reserved alert to pending after dispatch failure."""
    result = await pool.execute(
        """
        UPDATE alerts
        SET status = 'pending', processing_at = NULL
        WHERE id = $1 AND status = 'processing'
        """,
        alert_id,
    )
    return result.split()[-1] != "0"


# --- Daily brief scheduling ---


def is_daily_brief_due(now: datetime, *, brief_time: str, brief_tz: str) -> bool:
    """Check if the daily brief should fire at the given moment.

    Compares the current wall-clock minute in the configured timezone against
    the configured brief time. Returns True when the current HH:MM matches
    exactly (1-minute window). The scheduler should call this each poll cycle.

    Args:
        now: timezone-aware datetime (raises TypeError if naive).
        brief_time: HH:MM string (e.g. "08:00").
        brief_tz: IANA timezone (e.g. "America/New_York").

    Raises:
        TypeError: if *now* is timezone-naive.
        ValueError: if *brief_time* is not HH:MM or *brief_tz* is invalid.
    """
    if now.tzinfo is None:
        raise TypeError("now must be timezone-aware")

    # Validate timezone
    try:
        tz = ZoneInfo(brief_tz)
    except (ZoneInfoNotFoundError, KeyError):
        raise ValueError(f"Invalid timezone: {brief_tz!r}")

    # Validate time format
    brief_time = brief_time.strip()
    try:
        parsed = datetime.strptime(brief_time, "%H:%M").time()
    except ValueError:
        raise ValueError(f"Invalid brief time (expected HH:MM): {brief_time!r}")

    local_now = now.astimezone(tz)
    return local_now.hour == parsed.hour and local_now.minute == parsed.minute


# --- Helpers ---


def _row_to_alert(row: asyncpg.Record) -> Alert:
    """Convert a database row to an Alert model."""
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    # ``processing`` is an internal durable reservation state; callers see
    # reserved alerts as still pending until dispatch marks them fired.
    status = row["status"]
    if status == "processing":
        status = AlertStatus.pending

    return Alert(
        id=row["id"],
        user_id=row["user_id"],
        alert_type=row["alert_type"],
        title=row["title"],
        body=row["body"],
        trigger_at=row["trigger_at"],
        status=status,
        channel=row["channel"],
        channel_target=row["channel_target"],
        payload=payload or {},
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        fired_at=row["fired_at"],
        created_at=row["created_at"],
    )
