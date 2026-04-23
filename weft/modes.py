"""Modes store — CRUD for named retrieval personas with weight overrides.

Follows the same patterns as behaviors.py: get_db() for RLS-aware queries,
nullif(current_setting('app.user_id', true), '') for user_id insertion.
"""

from __future__ import annotations

import json
import logging

import asyncpg
from pydantic import ValidationError

from weft.db.connection import get_db
from weft.models import Mode, ModeCreate, ModeWeights, _weft_id

logger = logging.getLogger(__name__)


async def upsert_mode(pool: asyncpg.Pool, create: ModeCreate) -> Mode:
    """Create or update a mode by (user_id, name). Returns the Mode."""
    mode_id = _weft_id()
    weights_json = json.dumps(create.weights.model_dump())

    db = get_db(pool)
    user_id = await db.fetchval(
        "SELECT nullif(current_setting('app.user_id', true), '')"
    )

    # Try update first (handles both NULL and non-NULL user_id)
    if user_id is None:
        row = await db.fetchrow(
            """
            UPDATE modes SET
                description = $2,
                weights = $3::jsonb,
                project_id = $4,
                agent_id = $5,
                updated_at = now()
            WHERE user_id IS NULL AND name = $1
            RETURNING *
            """,
            create.name,
            create.description,
            weights_json,
            create.project_id,
            create.agent_id,
        )
    else:
        row = await db.fetchrow(
            """
            UPDATE modes SET
                description = $2,
                weights = $3::jsonb,
                project_id = $4,
                agent_id = $5,
                updated_at = now()
            WHERE user_id = $6 AND name = $1
            RETURNING *
            """,
            create.name,
            create.description,
            weights_json,
            create.project_id,
            create.agent_id,
            user_id,
        )

    if row is not None:
        return _row_to_mode(row)

    # Insert new
    row = await db.fetchrow(
        """
        INSERT INTO modes (id, user_id, name, description, weights, project_id, agent_id)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
        RETURNING *
        """,
        mode_id,
        user_id,
        create.name,
        create.description,
        weights_json,
        create.project_id,
        create.agent_id,
    )
    return _row_to_mode(row)


async def get_mode(pool: asyncpg.Pool, name: str) -> Mode | None:
    """Fetch a mode by name for the current user. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM modes WHERE name = $1",
        name,
    )
    return _row_to_mode(row) if row else None


async def list_modes(
    pool: asyncpg.Pool,
    *,
    user_id: str | None = None,
) -> list[Mode]:
    """List all modes, ordered by name.

    Args:
        pool: Database connection pool.
        user_id: If provided, filters to modes owned by this user OR globally-scoped (user_id IS NULL). If None, returns all modes.
    """
    conditions = []
    params: list = []
    idx = 1

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id IS NULL)")
        params.append(user_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM modes {where}
        ORDER BY name ASC
    """

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_mode(r) for r in rows]


async def delete_mode(pool: asyncpg.Pool, name: str) -> bool:
    """Delete a mode by name. Returns True if deleted, False if not found."""
    result = await get_db(pool).execute(
        "DELETE FROM modes WHERE name = $1",
        name,
    )
    return result.split()[-1] != "0"


async def get_active_weights(
    pool: asyncpg.Pool,
    mode_name: str | None,
) -> ModeWeights:
    """Resolve retrieval weights for a mode name. Never raises.

    Returns ModeWeights defaults when mode_name is None/empty or not found.
    Falls back to defaults on malformed DB data.
    """
    if not mode_name:
        return ModeWeights()

    mode = await get_mode(pool, mode_name)
    if mode is None:
        return ModeWeights()

    return mode.weights


# --- Helpers ---


def _row_to_mode(row: asyncpg.Record) -> Mode:
    """Convert a database row to a Mode model."""
    weights_data = row["weights"]
    try:
        if isinstance(weights_data, str):
            weights_data = json.loads(weights_data)
        weights = ModeWeights.model_validate(weights_data or {})
    except (ValidationError, json.JSONDecodeError, TypeError):
        logger.warning("malformed_mode_weights", extra={"mode_id": row["id"]})
        weights = ModeWeights()

    return Mode(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        description=row["description"],
        weights=weights,
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
