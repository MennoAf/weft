"""Episodes store — CRUD, memory linkage, and timeline queries.

Episodes group memories into time-bounded causal sequences,
enabling "what happened around that decision?" queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncpg

from weft.models import Episode, EpisodeCreate, EpisodeStatus, Memory, _weft_id
from weft.store import _row_to_memory
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)


async def create_episode(
    pool: asyncpg.Pool,
    create: EpisodeCreate,
) -> Episode:
    """Create a new open episode."""
    episode_id = _weft_id()
    now = datetime.now(timezone.utc)

    await pool.execute(
        """
        INSERT INTO episodes (
            id, title, summary, project_id, agent_id,
            started_at, status, token_count, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, 'open', 0, $6, $6)
        """,
        episode_id,
        create.title,
        create.summary,
        create.project_id,
        create.agent_id,
        now,
    )

    return Episode(
        id=episode_id,
        title=create.title,
        summary=create.summary,
        project_id=create.project_id,
        agent_id=create.agent_id,
        started_at=now,
        status=EpisodeStatus.open,
        created_at=now,
        updated_at=now,
    )


async def get_episode(pool: asyncpg.Pool, episode_id: str) -> Episode | None:
    """Fetch a single episode by ID."""
    row = await pool.fetchrow("SELECT * FROM episodes WHERE id = $1", episode_id)
    if not row:
        return None
    return _row_to_episode(row)


async def list_episodes(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    status: EpisodeStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Episode]:
    """List episodes with optional filters. Uses OR-NULL scoping on project_id/agent_id."""
    conditions = []
    params: list = []
    idx = 1

    if status is not None:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM episodes {where}
        ORDER BY started_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await pool.fetch(query, *params)
    return [_row_to_episode(r) for r in rows]


async def close_episode(
    pool: asyncpg.Pool,
    episode_id: str,
    *,
    summary: str | None = None,
) -> Episode | None:
    """Close an episode — sets ended_at and status='closed'."""
    now = datetime.now(timezone.utc)

    sets = ["ended_at = $1", "status = 'closed'", "updated_at = $1"]
    params: list = [now]
    idx = 2

    if summary is not None:
        sets.append(f"summary = ${idx}")
        params.append(summary)
        idx += 1

    set_clause = ", ".join(sets)
    params.append(episode_id)

    row = await pool.fetchrow(
        f"UPDATE episodes SET {set_clause} WHERE id = ${idx} RETURNING *",
        *params,
    )
    return _row_to_episode(row) if row else None


async def add_memory_to_episode(
    pool: asyncpg.Pool,
    episode_id: str,
    memory_id: str,
    position: int | None = None,
) -> bool:
    """Link a memory to an episode. Idempotent (ON CONFLICT DO NOTHING).

    If position is None, auto-assigns next position.
    Returns True if a new link was created, False if already existed.
    """
    if position is None:
        # Auto-assign next position
        max_pos = await pool.fetchval(
            "SELECT COALESCE(MAX(position), -1) FROM episode_memories WHERE episode_id = $1",
            episode_id,
        )
        position = max_pos + 1

    result = await pool.execute(
        """
        INSERT INTO episode_memories (episode_id, memory_id, position)
        VALUES ($1, $2, $3)
        ON CONFLICT (episode_id, memory_id) DO NOTHING
        """,
        episode_id,
        memory_id,
        position,
    )
    # "INSERT 0 1" = inserted, "INSERT 0 0" = conflict
    return result.split()[-1] != "0"


async def remove_memory_from_episode(
    pool: asyncpg.Pool,
    episode_id: str,
    memory_id: str,
) -> bool:
    """Unlink a memory from an episode. Returns True if removed."""
    result = await pool.execute(
        "DELETE FROM episode_memories WHERE episode_id = $1 AND memory_id = $2",
        episode_id,
        memory_id,
    )
    return result.split()[-1] != "0"


async def get_episode_memories(
    pool: asyncpg.Pool,
    episode_id: str,
    *,
    limit: int = 100,
) -> list[Memory]:
    """Get memories linked to an episode, ordered by position."""
    rows = await pool.fetch(
        """
        SELECT m.* FROM memories m
        JOIN episode_memories em ON m.id = em.memory_id
        WHERE em.episode_id = $1 AND m.status = 'active'
        ORDER BY em.position
        LIMIT $2
        """,
        episode_id,
        limit,
    )
    return [_row_to_memory(r) for r in rows]


async def get_episodes_for_memory(
    pool: asyncpg.Pool,
    memory_id: str,
) -> list[Episode]:
    """Reverse lookup — which episodes contain this memory?"""
    rows = await pool.fetch(
        """
        SELECT e.* FROM episodes e
        JOIN episode_memories em ON e.id = em.episode_id
        WHERE em.memory_id = $1
        ORDER BY e.started_at DESC
        """,
        memory_id,
    )
    return [_row_to_episode(r) for r in rows]


async def timeline_query(
    pool: asyncpg.Pool,
    start: datetime,
    end: datetime,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    status: EpisodeStatus | None = None,
    limit: int = 50,
) -> list[Episode]:
    """Find episodes overlapping a time range.

    Open episodes (ended_at IS NULL) are treated as ongoing and match
    any range that starts before or at their started_at.
    """
    conditions = [
        "started_at <= $1",
        "(ended_at IS NULL OR ended_at >= $2)",
    ]
    params: list = [end, start]
    idx = 3

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if status is not None:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT * FROM episodes {where}
        ORDER BY started_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(query, *params)
    return [_row_to_episode(r) for r in rows]


# --- Helpers ---


def _row_to_episode(row: asyncpg.Record) -> Episode:
    """Convert a database row to an Episode model."""
    return Episode(
        id=row["id"],
        title=row["title"],
        summary=row["summary"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        status=EpisodeStatus(row["status"]),
        token_count=row["token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
