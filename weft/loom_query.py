"""Loom SQL abstraction layer — single source of truth for all Loom DB queries.

Weft observes Loom; it never mutates it. All functions in this module are
read-only SELECT queries against Loom's tasks and projects tables.

All functions are async and accept an asyncpg.Pool as their first argument.
They return plain dicts (not asyncpg Records) so callers are decoupled from
the database driver.

If the Loom DB pool is None or a query fails, LoomQueryError is raised so
callers can distinguish Loom DB problems from Weft logic errors.

Audit compliance: no raw Loom SQL should exist outside this module.
  grep -rn 'FROM tasks\\|FROM epics' weft/ | grep -v loom_query.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import asyncpg

logger = logging.getLogger(__name__)

# --- Defaults ---

STALE_CLAIMED_THRESHOLD_HOURS = 48
BLOCKED_PILE_UP_THRESHOLD = 5
READY_TASKS_LIMIT = 20


# --- Exception ---


class LoomQueryError(Exception):
    """Raised when a Loom DB query fails or the pool is unavailable."""


def _require_pool(pool: asyncpg.Pool | None) -> None:
    if pool is None:
        raise LoomQueryError("Loom DB pool is None — LOOM_DATABASE_URL may not be configured")


# --- Table existence ---


async def loom_tables_exist(pool: asyncpg.Pool | None) -> bool:
    """Check whether Loom's tasks table exists in the database.

    Used by loom_alerts and health_check to skip gracefully when
    Loom is not co-located in the same Postgres instance.
    """
    _require_pool(pool)
    try:
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tasks')"
        )
        return bool(exists)
    except asyncpg.PostgresError as e:
        logger.warning("loom_query.loom_tables_exist failed: %s", e)
        raise LoomQueryError(f"loom_tables_exist failed: {e}") from e


# --- Task Queries ---


async def get_stale_claimed_tasks(
    pool: asyncpg.Pool | None,
    threshold_hours: int = STALE_CLAIMED_THRESHOLD_HOURS,
) -> list[dict]:
    """Tasks claimed for longer than threshold_hours without update.

    Returns list of dicts with keys: id, title, assignee, claimed_at,
    claim_expires_at, project_name.

    Used by: loom_alerts.check_stale_claims, health_check._evaluate_loom
    """
    _require_pool(pool)
    stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=threshold_hours)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.id, t.title, t.assignee, t.claimed_at, t.claim_expires_at,
                       p.name AS project_name
                FROM tasks t
                LEFT JOIN projects p ON t.project_id = p.id
                WHERE t.status = 'claimed'
                  AND t.claimed_at < $1
                ORDER BY t.claimed_at ASC
                LIMIT 20
                """,
                stale_cutoff,
            )
        return [dict(r) for r in rows]
    except asyncpg.PostgresError as e:
        logger.warning("loom_query.get_stale_claimed_tasks failed: %s", e)
        raise LoomQueryError(f"get_stale_claimed_tasks failed: {e}") from e


# --- Epic Queries ---


async def get_completable_epics(pool: asyncpg.Pool | None) -> list[dict]:
    """Epics where all children are done/cancelled but the epic is still open.

    Returns list of dicts with keys: id, title, project_name, child_count.

    Used by: loom_alerts.check_epic_completion, health_check._evaluate_loom
    """
    _require_pool(pool)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.id, e.title, p.name AS project_name,
                       count(*) AS child_count
                FROM tasks e
                JOIN tasks c ON c.parent_id = e.id
                LEFT JOIN projects p ON e.project_id = p.id
                WHERE e.status NOT IN ('done', 'cancelled')
                  AND NOT EXISTS (
                      SELECT 1 FROM tasks child
                      WHERE child.parent_id = e.id
                        AND child.status NOT IN ('done', 'cancelled')
                  )
                GROUP BY e.id, e.title, p.name
                ORDER BY e.title
                LIMIT 20
                """,
            )
        return [dict(r) for r in rows]
    except asyncpg.PostgresError as e:
        logger.warning("loom_query.get_completable_epics failed: %s", e)
        raise LoomQueryError(f"get_completable_epics failed: {e}") from e


async def get_blocked_pile_ups(
    pool: asyncpg.Pool | None,
    threshold: int = BLOCKED_PILE_UP_THRESHOLD,
) -> list[dict]:
    """Projects with >= threshold blocked tasks.

    Returns list of dicts with keys: project_name, project_id, blocked_count.

    Used by: loom_alerts.check_blocked_pile_up, health_check._evaluate_loom
    """
    _require_pool(pool)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.name AS project_name, p.id AS project_id,
                       count(*) AS blocked_count
                FROM tasks t
                JOIN projects p ON t.project_id = p.id
                WHERE t.status = 'blocked'
                  AND p.status = 'active'
                GROUP BY p.id, p.name
                HAVING count(*) >= $1
                ORDER BY count(*) DESC
                LIMIT 10
                """,
                threshold,
            )
        return [dict(r) for r in rows]
    except asyncpg.PostgresError as e:
        logger.warning("loom_query.get_blocked_pile_ups failed: %s", e)
        raise LoomQueryError(f"get_blocked_pile_ups failed: {e}") from e


# --- Brief Queries ---


async def get_ready_tasks(
    pool: asyncpg.Pool | None,
    limit: int = READY_TASKS_LIMIT,
) -> list[dict]:
    """Pending tasks suitable for the daily brief.

    Returns list of dicts with keys: id, title, priority, project_name.
    The 'title' and 'priority' keys match the shape daily_brief.py expects
    from the old Loom CLI --json output.

    Replaces the broken subprocess call in daily_brief.py.

    Used by: daily_brief._query_loom_tasks
    """
    _require_pool(pool)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.id, t.title, t.priority, p.name AS project_name
                FROM tasks t
                LEFT JOIN projects p ON t.project_id = p.id
                WHERE t.status = 'pending'
                ORDER BY
                    CASE t.priority
                        WHEN 'p0' THEN 0
                        WHEN 'p1' THEN 1
                        WHEN 'p2' THEN 2
                        ELSE 3
                    END,
                    t.created_at ASC
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]
    except asyncpg.PostgresError as e:
        logger.warning("loom_query.get_ready_tasks failed: %s", e)
        raise LoomQueryError(f"get_ready_tasks failed: {e}") from e
