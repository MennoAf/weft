"""Check-in store — CRUD for mood/sleep/energy tracking.

Follows the same patterns as modes.py and alerts.py: get_db() for
RLS-aware queries, nullif(current_setting('app.user_id', true), '')
for user_id insertion.
"""

from __future__ import annotations

import logging

import asyncpg

from weft.db.connection import get_db
from weft.models import CheckIn, CheckInCreate, _weft_id

logger = logging.getLogger(__name__)


async def create_check_in(pool: asyncpg.Pool, create: CheckInCreate) -> CheckIn:
    """Insert a new check-in. Returns the created CheckIn."""
    check_in_id = _weft_id()
    db = get_db(pool)

    if create.logged_at is not None:
        row = await db.fetchrow(
            """
            INSERT INTO check_ins (
                id, user_id, mood, sleep_hours, energy, notes, logged_at
            )
            VALUES (
                $1,
                nullif(current_setting('app.user_id', true), ''),
                $2, $3, $4, $5, $6
            )
            RETURNING *
            """,
            check_in_id,
            create.mood,
            create.sleep_hours,
            create.energy,
            create.notes,
            create.logged_at,
        )
    else:
        row = await db.fetchrow(
            """
            INSERT INTO check_ins (
                id, user_id, mood, sleep_hours, energy, notes
            )
            VALUES (
                $1,
                nullif(current_setting('app.user_id', true), ''),
                $2, $3, $4, $5
            )
            RETURNING *
            """,
            check_in_id,
            create.mood,
            create.sleep_hours,
            create.energy,
            create.notes,
        )
    return _row_to_check_in(row)


async def list_check_ins(
    pool: asyncpg.Pool,
    *,
    limit: int = 30,
    offset: int = 0,
) -> list[CheckIn]:
    """List check-ins for the current user, newest first."""
    rows = await get_db(pool).fetch(
        """
        SELECT * FROM check_ins
        ORDER BY logged_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset,
    )
    return [_row_to_check_in(r) for r in rows]


async def get_check_in_stats(
    pool: asyncpg.Pool,
    *,
    days: int = 30,
) -> dict:
    """Get aggregated stats over the last N days."""
    row = await get_db(pool).fetchrow(
        """
        SELECT
            count(*) AS total,
            round(avg(mood)::numeric, 1) AS avg_mood,
            round(avg(sleep_hours)::numeric, 1) AS avg_sleep,
            round(avg(energy)::numeric, 1) AS avg_energy,
            min(mood) AS min_mood,
            max(mood) AS max_mood,
            min(sleep_hours) AS min_sleep,
            max(sleep_hours) AS max_sleep,
            min(energy) AS min_energy,
            max(energy) AS max_energy
        FROM check_ins
        WHERE logged_at >= now() - make_interval(days => $1)
        """,
        days,
    )
    return {
        "days": days,
        "total": row["total"],
        "avg_mood": float(row["avg_mood"]) if row["avg_mood"] else None,
        "avg_sleep": float(row["avg_sleep"]) if row["avg_sleep"] else None,
        "avg_energy": float(row["avg_energy"]) if row["avg_energy"] else None,
        "min_mood": row["min_mood"],
        "max_mood": row["max_mood"],
        "min_sleep": float(row["min_sleep"]) if row["min_sleep"] else None,
        "max_sleep": float(row["max_sleep"]) if row["max_sleep"] else None,
        "min_energy": row["min_energy"],
        "max_energy": row["max_energy"],
    }


# --- Helpers ---


def _row_to_check_in(row: asyncpg.Record) -> CheckIn:
    """Convert a database row to a CheckIn model."""
    return CheckIn(
        id=row["id"],
        user_id=row["user_id"],
        mood=row["mood"],
        sleep_hours=float(row["sleep_hours"]) if row["sleep_hours"] is not None else None,
        energy=row["energy"],
        notes=row["notes"],
        logged_at=row["logged_at"],
        created_at=row["created_at"],
    )
