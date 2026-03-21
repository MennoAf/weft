"""Alert store — CRUD + polling for the proactive push system.

Follows the same patterns as modes.py: get_db() for RLS-aware queries,
nullif(current_setting('app.user_id', true), '') for user_id insertion.

The scheduler (weft/scheduler.py) calls poll_due_alerts() and mark_alert_fired()
to process alerts that have reached their trigger_at time.
"""

from __future__ import annotations

import json
import logging

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


async def poll_due_alerts(
    pool: asyncpg.Pool,
    batch_size: int = 50,
) -> list[Alert]:
    """Poll for pending alerts whose trigger_at has passed.

    Uses SELECT FOR UPDATE SKIP LOCKED to prevent double-firing when
    multiple scheduler instances are running. The connection is released
    after fetching — dispatch happens outside the transaction.

    Returns up to batch_size alerts as Python objects (connection-free).
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT * FROM alerts
                WHERE status = 'pending'
                  AND trigger_at <= now()
                ORDER BY trigger_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
                """,
                batch_size,
            )
            return [_row_to_alert(r) for r in rows]


async def mark_alert_fired(pool: asyncpg.Pool, alert_id: str) -> bool:
    """Mark an alert as fired. Idempotent — only updates if still pending.

    Returns True if the alert was updated, False if already fired/dismissed.
    """
    result = await pool.execute(
        """
        UPDATE alerts
        SET status = 'fired', fired_at = now()
        WHERE id = $1 AND status = 'pending'
        """,
        alert_id,
    )
    return result.split()[-1] != "0"


# --- Helpers ---


def _row_to_alert(row: asyncpg.Record) -> Alert:
    """Convert a database row to an Alert model."""
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    return Alert(
        id=row["id"],
        user_id=row["user_id"],
        alert_type=row["alert_type"],
        title=row["title"],
        body=row["body"],
        trigger_at=row["trigger_at"],
        status=row["status"],
        channel=row["channel"],
        channel_target=row["channel_target"],
        payload=payload or {},
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        fired_at=row["fired_at"],
        created_at=row["created_at"],
    )
