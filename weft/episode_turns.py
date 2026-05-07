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


# --- Recall ---


# RRF "k" constant — a value of 60 is the standard from Cormack et al. and
# is what weft.store.search_hybrid uses. Same value here to keep behavior
# consistent across tiers.
_RRF_K = 60


async def recall_turns(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    top_k: int = 20,
    embedding: list[float] | None = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> list[EpisodeTurn]:
    """Hybrid (vector + BM25) recall over episode_turns.

    Filters apply BEFORE scoring: ``project_id`` joins through ``episodes``,
    ``since`` / ``until`` constrain ``occurred_at``. Vector similarity uses
    cosine distance against the inline embedding column; keyword scoring
    uses Postgres FTS (``to_tsvector('english', content)``) — there is no
    persisted tsvector column on episode_turns yet, so this path is
    unindexed for now. Acceptable at Wick scale (one user's dialogue
    trace); add a stored search_tsv column + GIN index when a single
    installation crosses ~100k turns.

    RRF fusion mirrors ``weft.store.search_hybrid`` so callers can reason
    about belief-tier and turn-tier results in the same rank space.

    Args:
        embedding: precomputed query embedding. If None, the caller is
            responsible for skipping vector search (we don't reach into
            the embedding provider from this layer to keep the store
            module dependency-free).
    """
    candidate_limit = top_k * 3
    sql_filter, params = _build_turn_filters(
        project_id=project_id, since=since, until=until,
    )

    db = get_db(pool)

    # --- Vector half (skipped if embedding is None) ---
    vector_rows: list[asyncpg.Record] = []
    if embedding is not None:
        vector_sql = f"""
            SELECT t.*
              FROM episode_turns t
              {_join_episodes_if_needed(project_id)}
              WHERE t.embedding IS NOT NULL
                {sql_filter}
              ORDER BY t.embedding <=> $1::vector
              LIMIT ${len(params) + 2}
        """
        vector_rows = await db.fetch(vector_sql, embedding, *params, candidate_limit)

    # --- Keyword half ---
    keyword_sql = f"""
        SELECT t.*
          FROM episode_turns t
          {_join_episodes_if_needed(project_id)}
          WHERE to_tsvector('english', t.content)
                @@ websearch_to_tsquery('english', $1)
            {sql_filter}
          ORDER BY ts_rank(
              to_tsvector('english', t.content),
              websearch_to_tsquery('english', $1)
          ) DESC
          LIMIT ${len(params) + 2}
    """
    keyword_rows = await db.fetch(keyword_sql, query, *params, candidate_limit)

    # --- RRF fuse → rerank by usefulness × recency (P1.A3) ---
    fused = _rrf_fuse_turn_rows(
        vector_rows, keyword_rows,
        candidate_limit=candidate_limit,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    # Late import: relevance imports models, which we already loaded.
    # Done at call time to keep the store layer's import graph minimal.
    from weft.relevance import rank_turns
    ranked = rank_turns(fused)
    return [s.turn for s in ranked]


async def list_recent_turns(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
) -> list[EpisodeTurn]:
    """Time-ordered recent turns — used as a fallback when query has no
    keyword/vector signal (e.g., a temporal-anchor probe with anchor
    text that nothing matches semantically)."""
    sql_filter, params = _build_turn_filters(
        project_id=project_id, since=since, until=until,
    )
    db = get_db(pool)
    sql = f"""
        SELECT t.*
          FROM episode_turns t
          {_join_episodes_if_needed(project_id)}
          WHERE 1=1
            {sql_filter}
          ORDER BY t.occurred_at DESC
          LIMIT ${len(params) + 1}
    """
    rows = await db.fetch(sql, *params, limit)
    return [_row_to_turn(r) for r in rows]


# --- Helpers ---


def _join_episodes_if_needed(project_id: str | None) -> str:
    """Episodes JOIN only required for project scoping; cheaper to skip otherwise."""
    return "JOIN episodes e ON t.episode_id = e.id" if project_id is not None else ""


def _build_turn_filters(
    *,
    project_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[str, list]:
    """Compose AND-joined WHERE fragments; param numbering is offset by the
    caller's leading positional args (embedding or query string)."""
    fragments: list[str] = []
    params: list = []
    # Param numbering convention: caller's leading args are $1 (and $2 if
    # embedding+query both pre-bound). We emit fragments using $2, $3, ...
    # by counting from len(caller_leading_args) + 1 — the caller passes the
    # full param list to fetch().
    base = 2  # one leading arg (embedding OR query)
    if since is not None:
        fragments.append(f"AND t.occurred_at >= ${base + len(params)}")
        params.append(since)
    if until is not None:
        fragments.append(f"AND t.occurred_at <= ${base + len(params)}")
        params.append(until)
    if project_id is not None:
        fragments.append(f"AND e.project_id = ${base + len(params)}")
        params.append(project_id)
    return (" ".join(fragments), params)


def _rrf_fuse_turn_rows(
    vector_rows: list[asyncpg.Record],
    keyword_rows: list[asyncpg.Record],
    *,
    candidate_limit: int,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
) -> list[tuple[EpisodeTurn, float]]:
    """Reciprocal Rank Fusion over two sorted candidate lists.

    Mirrors weft.store.search_hybrid's behavior: missing-from-half rows
    get an absent-rank penalty equal to candidate_limit + 1 so a turn that
    appears in only one half can still surface if its rank is high.

    Returns ``(turn, rrf_score)`` pairs in RRF-descending order, capped at
    ``top_k``. Caller may rerank within the returned window using
    :func:`weft.relevance.rank_turns` (which is what ``recall_turns`` does
    today). Returning the score keeps the rerank step honest — without
    it, downstream layers would have to recompute RRF or rerank against
    rank-position, both of which lose information.
    """
    vector_ranks: dict[str, int] = {
        r["id"]: i + 1 for i, r in enumerate(vector_rows)
    }
    keyword_ranks: dict[str, int] = {
        r["id"]: i + 1 for i, r in enumerate(keyword_rows)
    }
    all_rows: dict[str, asyncpg.Record] = {}
    for r in vector_rows:
        all_rows[r["id"]] = r
    for r in keyword_rows:
        all_rows.setdefault(r["id"], r)

    absent = candidate_limit + 1
    scores: dict[str, float] = {}
    for tid in all_rows:
        v = vector_ranks.get(tid, absent)
        k = keyword_ranks.get(tid, absent)
        scores[tid] = (
            vector_weight / (_RRF_K + v) + keyword_weight / (_RRF_K + k)
        )
    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    return [(_row_to_turn(all_rows[tid]), scores[tid]) for tid in sorted_ids]


def _short_id() -> str:
    """Generate a short suffix for et-* IDs.

    Lives here (not in models.py) so the same generator can be used by
    callers that need to pre-allocate an ID before insert (rare, but useful
    for trace correlation).
    """
    import uuid
    return uuid.uuid4().hex[:10]


def _row_to_turn(row: asyncpg.Record) -> EpisodeTurn:
    # Boost-loop columns landed in v46 — read defensively so old fixtures
    # / pre-migration callers don't trip if a row predates the migration.
    def _opt(key: str, default):
        try:
            v = row[key]
        except (KeyError, IndexError):
            return default
        return default if v is None else v

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
        usefulness_score=float(_opt("usefulness_score", 0.7)),
        usefulness_count=int(_opt("usefulness_count", 0)),
        last_boosted_at=_opt("last_boosted_at", None),
    )
