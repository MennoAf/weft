"""Episode turns store — append-one writes, range queries, importance pruning.

Turn-tier dialogue trace lives under the existing episodes umbrella. Reads
and writes go through this module; the MCP `weft_turn_append` tool wraps
``append_turn`` and computes embeddings inline before insert (same pattern
as `weft_remember` → `store_memory`).

Design notes:

* ``append_turn`` is race-safe under concurrent inserts to the same
  ``episode_id``. We use a transaction-scoped Postgres advisory lock keyed
  to the episode_id to serialize concurrent appenders. The turn_index is
  computed inside the same transaction via
  ``(SELECT COALESCE(MAX(turn_index), -1) + 1 ...)`` so there's no read/write
  race. The advisory lock is per-transaction, so it auto-releases on commit
  or rollback. Cost is one cheap GUC call per append; no schema impact.
* All reads honor RLS — the caller's ``app.user_id`` GUC determines
  visibility. Service-side callers (graduation, benchmark adapters) set
  the GUC explicitly.

Spec: weft-d3a2ef78. Loom epic: loom-52ffc3a2.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncpg

from weft.db.connection import get_db
from weft.models import EpisodeTurn, EpisodeTurnCreate, TurnRole
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Namespace for pg_advisory_xact_lock(int, int) — keeps episode_turns locks
# disjoint from any other module's advisory lock space. Both args are
# signed int32, so this constant must fit in 31 bits.
_TURN_APPEND_LOCK_NAMESPACE = 0x4554_5552  # 'ETUR'


async def append_turn(
    pool: asyncpg.Pool,
    create: EpisodeTurnCreate,
    *,
    embedding: list[float] | None = None,
) -> EpisodeTurn:
    """Append a turn to an episode. Auto-assigns turn_index race-safely.

    Race-safety: we take a transaction-scoped Postgres advisory lock keyed
    to the episode_id. Concurrent appenders to the same episode serialize
    on the lock; appenders to different episodes proceed in parallel. The
    lock auto-releases at transaction end. Inside the lock, we compute
    ``turn_index`` from ``MAX(turn_index)+1`` and INSERT in the same
    transaction — no UNIQUE-violation retry path needed.
    """
    occurred_at = create.occurred_at or datetime.now(timezone.utc)
    token_count = estimate_tokens(create.content)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Scope the advisory lock by hashing episode_id into a 32-bit int.
            # Collisions across episodes are harmless (just rare extra serialization);
            # within an episode, concurrent appenders queue on the same lock.
            lock_key = _episode_lock_key(create.episode_id)
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1, $2)",
                _TURN_APPEND_LOCK_NAMESPACE,
                lock_key,
            )

            row = await conn.fetchrow(
                """
                INSERT INTO episode_turns (
                    id, episode_id, turn_index, role, content,
                    occurred_at, embedding, trace_id, token_count
                )
                SELECT
                    $1,
                    $2,
                    COALESCE(MAX(turn_index), -1) + 1,
                    $3,
                    $4,
                    $5,
                    $6::vector,
                    $7,
                    $8
                FROM episode_turns
                WHERE episode_id = $2
                RETURNING *
                """,
                f"et-{_short_id()}",
                create.episode_id,
                create.role.value,
                create.content,
                occurred_at,
                embedding,
                create.trace_id,
                token_count,
            )
            if row is None:
                raise RuntimeError(
                    f"append_turn returned no row for episode {create.episode_id}"
                )
            return _row_to_turn(row)


def _episode_lock_key(episode_id: str) -> int:
    """Stable 32-bit signed int derived from episode_id for advisory lock keying.

    Postgres pg_advisory_xact_lock(int, int) takes 32-bit signed ints, so we
    fold the hash to fit. Collisions between distinct episodes are harmless
    (concurrent appenders to different episodes that hash-collide will
    serialize unnecessarily, but correctness is preserved).
    """
    h = abs(hash(episode_id)) & 0x7FFFFFFF
    return h


async def get_turn(pool: asyncpg.Pool, turn_id: str) -> EpisodeTurn | None:
    """Fetch a single turn by ID."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM episode_turns WHERE id = $1", turn_id,
    )
    return _row_to_turn(row) if row else None


async def list_turns(
    pool: asyncpg.Pool,
    episode_id: str,
    *,
    limit: int = 200,
) -> list[EpisodeTurn]:
    """List all turns in an episode, ordered by turn_index ascending."""
    rows = await get_db(pool).fetch(
        """
        SELECT * FROM episode_turns
        WHERE episode_id = $1
        ORDER BY turn_index ASC
        LIMIT $2
        """,
        episode_id,
        limit,
    )
    return [_row_to_turn(r) for r in rows]


async def list_turns_in_range(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 200,
) -> list[EpisodeTurn]:
    """List turns within a time range, optionally scoped to a project.

    Joins through ``episodes`` for project scoping. The temporal range
    filter applies to ``occurred_at``, not ``created_at`` — turns may be
    backfilled with historical timestamps (e.g., Slack ingest, LongMemEval
    fixtures).
    """
    conditions: list[str] = []
    params: list = []
    idx = 1

    if since is not None:
        conditions.append(f"t.occurred_at >= ${idx}")
        params.append(since)
        idx += 1
    if until is not None:
        conditions.append(f"t.occurred_at <= ${idx}")
        params.append(until)
        idx += 1
    if project_id is not None:
        conditions.append(f"e.project_id = ${idx}")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    join = "JOIN episodes e ON t.episode_id = e.id" if project_id else ""
    params.append(limit)

    query = f"""
        SELECT t.* FROM episode_turns t
        {join}
        {where}
        ORDER BY t.occurred_at ASC
        LIMIT ${idx}
    """
    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_turn(r) for r in rows]


async def delete_turns_below_importance(
    pool: asyncpg.Pool,
    threshold: float,
    *,
    older_than_days: int = 30,
) -> int:
    """Prune low-importance turns whose owning episode graduated long enough ago.

    Used by graduation policy: high-importance turns persist indefinitely;
    mid/low-importance turns are dropped once their episode has been
    ``status = 'graduated'`` for more than ``older_than_days``. Turns whose
    score is NULL (Face offline) are NOT touched here — the age-only
    fallback policy lives in graduate_episode and runs as a separate
    statement.

    Returns the number of turns deleted.
    """
    result = await get_db(pool).execute(
        """
        DELETE FROM episode_turns
        WHERE importance_score IS NOT NULL
          AND importance_score < $1
          AND episode_id IN (
              SELECT id FROM episodes
              WHERE status = 'graduated'
                AND ended_at IS NOT NULL
                AND ended_at < now() - ($2 || ' days')::interval
          )
        """,
        threshold,
        str(older_than_days),
    )
    return int(result.split()[-1])


async def delete_turns_for_graduated_episode(
    pool: asyncpg.Pool,
    *,
    older_than_days: int,
) -> int:
    """Age-only fallback prune. Drops ALL turns of episodes graduated
    longer than ``older_than_days`` ago, regardless of importance_score.

    Used when Face is offline (score IS NULL across the board) and the
    only signal is age. Graduation policy invokes this with a longer TTL
    than the importance-aware path.
    """
    result = await get_db(pool).execute(
        """
        DELETE FROM episode_turns
        WHERE episode_id IN (
            SELECT id FROM episodes
            WHERE status = 'graduated'
              AND ended_at IS NOT NULL
              AND ended_at < now() - ($1 || ' days')::interval
        )
        """,
        str(older_than_days),
    )
    return int(result.split()[-1])


# --- Helpers ---


def _short_id() -> str:
    """Generate a short suffix for et-* IDs.

    Lives here (not in models.py) so the same generator can be used by
    callers that need to pre-allocate an ID before insert (rare, but useful
    for trace correlation).
    """
    import uuid
    return uuid.uuid4().hex[:10]


def _row_to_turn(row: asyncpg.Record) -> EpisodeTurn:
    return EpisodeTurn(
        id=row["id"],
        episode_id=row["episode_id"],
        turn_index=row["turn_index"],
        role=TurnRole(row["role"]),
        content=row["content"],
        occurred_at=row["occurred_at"],
        trace_id=row["trace_id"],
        importance_score=row["importance_score"],
        token_count=row["token_count"],
        user_id=row["user_id"],
        created_at=row["created_at"],
    )
