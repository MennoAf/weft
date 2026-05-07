"""Episodes store — CRUD, memory linkage, and timeline queries.

Episodes group memories into time-bounded causal sequences,
enabling "what happened around that decision?" queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncpg

from datetime import timedelta

from weft.db.connection import get_db
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.models import (
    Episode,
    EpisodeCreate,
    EpisodeStatus,
    EpisodeWithMemories,
    Memory,
    MemoryCreate,
    MemorySource,
    MemoryType,
    _weft_id,
)
from weft.store import _row_to_memory, store_memory
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)


async def create_episode(
    pool: asyncpg.Pool,
    create: EpisodeCreate,
    embedding: list[float] | None = None,
) -> Episode:
    """Create a new open episode. If ttl_hours is set, computes expires_at.

    ``embedding`` is the precomputed vector for ``title || ' ' || COALESCE(summary, '')``.
    Mirrors :func:`weft.store.store_memory` — the embedder lives in the caller
    (typically MCP app context), the network call happens *outside* the SQL
    transaction, and a ``None`` value is written as NULL so the v47 startup
    backfill in :mod:`weft.db.reembed` can fill it in later. Pass ``None`` from
    callers that can't afford an embed call (background scripts, tests that
    don't care, transient open-then-close flows).
    """
    episode_id = _weft_id()
    now = datetime.now(timezone.utc)

    expires_at = None
    if create.ttl_hours is not None:
        expires_at = now + timedelta(hours=create.ttl_hours)

    await get_db(pool).execute(
        """
        INSERT INTO episodes (
            id, title, summary, project_id, agent_id,
            user_id, started_at, expires_at, status, token_count,
            embedding, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5,
                  nullif(current_setting('app.user_id', true), ''),
                  $6, $7, 'open', 0, $8::vector, $6, $6)
        """,
        episode_id,
        create.title,
        create.summary,
        create.project_id,
        create.agent_id,
        now,
        expires_at,
        embedding,
    )

    return Episode(
        id=episode_id,
        title=create.title,
        summary=create.summary,
        project_id=create.project_id,
        agent_id=create.agent_id,
        started_at=now,
        expires_at=expires_at,
        status=EpisodeStatus.open,
        created_at=now,
        updated_at=now,
    )


async def get_episode(pool: asyncpg.Pool, episode_id: str) -> Episode | None:
    """Fetch a single episode by ID."""
    row = await get_db(pool).fetchrow("SELECT * FROM episodes WHERE id = $1", episode_id)
    if not row:
        return None
    return _row_to_episode(row)


async def list_episodes(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    status: EpisodeStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Episode]:
    """List episodes with optional filters. Uses OR-NULL scoping on project_id/agent_id/user_id.

    Args:
        pool: asyncpg connection pool
        project_id: If provided, filters to episodes owned by this project OR globally-scoped (project_id IS NULL). If None, returns all episodes.
        agent_id: If provided, filters to episodes owned by this agent OR globally-scoped (agent_id IS NULL). If None, returns all episodes.
        user_id: If provided, filters to episodes owned by this user OR globally-scoped (user_id IS NULL). If None, returns all episodes.
        status: Filter to episodes with this status.
        limit: Maximum number of results.
        offset: Offset for pagination.
    """
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

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id = ${idx + 1})")
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM episodes {where}
        ORDER BY started_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_episode(r) for r in rows]


async def close_episode(
    pool: asyncpg.Pool,
    episode_id: str,
    *,
    summary: str | None = None,
    embedding: list[float] | None = None,
) -> Episode | None:
    """Close an episode — sets ended_at and status='closed'.

    ``embedding`` is the precomputed vector for the new
    ``title || ' ' || COALESCE(summary, '')``. Pass it ONLY when ``summary``
    changes — otherwise leave it as ``None`` and the existing embedding is
    untouched (no SET clause emitted). Callers that fail to embed should
    still be able to close; following :func:`weft.store.store_memory`'s
    semantics, an embed failure on the MCP side simply omits ``embedding``
    here and the v47 backfill catches it on the next startup pass.
    """
    now = datetime.now(timezone.utc)

    sets = ["ended_at = $1", "status = 'closed'", "updated_at = $1"]
    params: list = [now]
    idx = 2

    if summary is not None:
        sets.append(f"summary = ${idx}")
        params.append(summary)
        idx += 1

    if embedding is not None:
        sets.append(f"embedding = ${idx}::vector")
        params.append(embedding)
        idx += 1

    set_clause = ", ".join(sets)
    params.append(episode_id)

    row = await get_db(pool).fetchrow(
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
    db = get_db(pool)
    if position is None:
        # Auto-assign next position
        max_pos = await db.fetchval(
            "SELECT COALESCE(MAX(position), -1) FROM episode_memories WHERE episode_id = $1",
            episode_id,
        )
        position = max_pos + 1

    result = await db.execute(
        """
        INSERT INTO episode_memories (episode_id, memory_id, position, user_id)
        VALUES ($1, $2, $3, nullif(current_setting('app.user_id', true), ''))
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
    result = await get_db(pool).execute(
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
    rows = await get_db(pool).fetch(
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
    rows = await get_db(pool).fetch(
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
    user_id: str | None = None,
    status: EpisodeStatus | None = None,
    limit: int = 50,
) -> list[Episode]:
    """Find episodes overlapping a time range.

    Open episodes (ended_at IS NULL) are treated as ongoing and match
    any range that starts before or at their started_at.

    Args:
        pool: asyncpg connection pool
        start: Start of time range (exclusive).
        end: End of time range (exclusive).
        project_id: If provided, filters to episodes owned by this project OR globally-scoped (project_id IS NULL). If None, returns all episodes.
        agent_id: If provided, filters to episodes owned by this agent OR globally-scoped (agent_id IS NULL). If None, returns all episodes.
        user_id: If provided, filters to episodes owned by this user OR globally-scoped (user_id IS NULL). If None, returns all episodes.
        status: Filter to episodes with this status.
        limit: Maximum number of results.
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

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id = ${idx + 1})")
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

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

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_episode(r) for r in rows]


async def graduate_episode(
    pool: asyncpg.Pool,
    episode_id: str,
    *,
    memory_type: MemoryType = MemoryType.fact,
    content: str | None = None,
    topic: list[str] | None = None,
    confidence: float = 0.7,
    embedding: list[float] | None = None,
) -> tuple[Episode, Memory]:
    """Graduate an episode into a persistent memory.

    Creates a memory from the episode's content (or custom content),
    links the memory to the episode, sets graduated_memory_id, and
    transitions the episode to 'graduated' status.

    Returns the updated episode and the newly created memory.
    Raises ValueError if the episode doesn't exist or is already graduated.
    """
    ep = await get_episode(pool, episode_id)
    if ep is None:
        raise ValueError(f"Episode {episode_id} not found")
    if ep.status == EpisodeStatus.graduated:
        raise ValueError(f"Episode {episode_id} is already graduated")

    # Build memory content from episode if not provided
    if content is None:
        parts = [ep.title]
        if ep.summary:
            parts.append(ep.summary)
        content = "\n\n".join(parts)

    # Create the persistent memory
    memory = await store_memory(
        pool,
        MemoryCreate(
            type=memory_type,
            content=content,
            topic=topic or [],
            source=MemorySource.conversation,
            confidence=confidence,
            project_id=ep.project_id,
            agent_id=ep.agent_id,
        ),
        embedding=embedding,
    )

    # Link the memory to the episode
    await add_memory_to_episode(pool, episode_id, memory.id)

    # Update episode: set graduated status and graduated_memory_id.
    # NOTE: graduation never mutates `title` or `summary` (the only fields
    # the episode embedding is built from), so there's no embedding
    # recompute here. The graduated *memory* gets its own embedding via
    # the `embedding=` arg threaded into store_memory above; the episode's
    # vector stays valid because its source text is unchanged.
    now = datetime.now(timezone.utc)
    row = await get_db(pool).fetchrow(
        """
        UPDATE episodes
        SET status = 'graduated',
            graduated_memory_id = $1,
            ended_at = COALESCE(ended_at, $2),
            updated_at = $2
        WHERE id = $3
        RETURNING *
        """,
        memory.id,
        now,
        episode_id,
    )

    updated_ep = _row_to_episode(row)
    return updated_ep, memory


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
        expires_at=row.get("expires_at"),
        graduated_memory_id=row.get("graduated_memory_id"),
        status=EpisodeStatus(row["status"]),
        token_count=row["token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def get_working_memory(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
) -> list[EpisodeWithMemories]:
    """Get active working memory — open episodes that haven't expired.

    Returns episodes with their linked memories, ordered by most recent first.
    Excludes episodes where expires_at has passed.

    Args:
        pool: asyncpg connection pool
        project_id: If provided, filters to episodes owned by this project OR globally-scoped (project_id IS NULL). If None, returns all episodes.
        agent_id: If provided, filters to episodes owned by this agent OR globally-scoped (agent_id IS NULL). If None, returns all episodes.
        user_id: If provided, filters to episodes owned by this user OR globally-scoped (user_id IS NULL). If None, returns all episodes.
        limit: Maximum number of results.
    """
    conditions = [
        "status = 'open'",
        "(expires_at IS NULL OR expires_at > now())",
    ]
    params: list = []
    idx = 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id = ${idx + 1})")
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

    where = "WHERE " + " AND ".join(conditions)
    params.append(limit)

    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM episodes {where}
        ORDER BY started_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    result = []
    for row in rows:
        episode = _row_to_episode(row)
        memories = await get_episode_memories(pool, episode.id)
        result.append(EpisodeWithMemories(episode=episode, memories=memories))
    return result


async def expire_stale_episodes(pool: asyncpg.Pool) -> int:
    """Mark expired episodes as 'expired'. Returns count of episodes expired.

    Finds open episodes where expires_at <= now() and transitions them
    to 'expired' status with ended_at set to now().
    """
    now = datetime.now(timezone.utc)
    result = await get_db(pool).execute(
        """
        UPDATE episodes
        SET status = 'expired', ended_at = $1, updated_at = $1
        WHERE status = 'open'
          AND expires_at IS NOT NULL
          AND expires_at <= $1
        """,
        now,
    )
    count = int(result.split()[-1])
    if count > 0:
        logger.info("episodes.expired", extra={"count": count})
    return count
