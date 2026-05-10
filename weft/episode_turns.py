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
from typing import TYPE_CHECKING

import asyncpg

from weft.db.connection import get_db
from weft.models import EpisodeTurn, EpisodeTurnCreate, TurnRole
from weft.tokens import estimate_tokens

# Retention thresholds imported from episodes to keep a single source of truth.
# Import deferred to module body (not top-level) to avoid a circular import:
# episodes imports episode_turns (list_turns), episode_turns must not import
# episodes at module load time. The constants are read at call time inside the
# function bodies, which is safe.
def _graduation_constants() -> tuple[float, int, int]:
    """Lazy import guard — returns (HIGH_THRESHOLD, TTL_DAYS, TTL_DAYS_NO_SCORE)."""
    from weft.episodes import (  # noqa: PLC0415
        GRADUATION_HIGH_IMPORTANCE_THRESHOLD,
        GRADUATION_TURN_TTL_DAYS,
        GRADUATION_TURN_TTL_DAYS_NO_SCORE,
    )
    return (
        GRADUATION_HIGH_IMPORTANCE_THRESHOLD,
        GRADUATION_TURN_TTL_DAYS,
        GRADUATION_TURN_TTL_DAYS_NO_SCORE,
    )

if TYPE_CHECKING:
    from weft.embeddings.base import EmbeddingProvider

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


async def delete_turns_after_graduation(
    pool: asyncpg.Pool,
    *,
    high_threshold: float | None = None,
    ttl_days_scored: int | None = None,
    ttl_days_no_score: int | None = None,
) -> dict[str, int]:
    """Sweep turns whose episode has been graduated long enough.

    Two policies in one sweep:
    - Score-aware: turns whose episode graduated > ttl_days_scored ago AND
      importance_score < high_threshold are deleted.
    - Score-blind fallback (Face offline): turns whose episode graduated >
      ttl_days_no_score ago AND importance_score IS NULL are deleted.

    Turns whose importance_score >= high_threshold are RETAINED indefinitely
    regardless of age. Turns belonging to non-graduated episodes are ignored.

    Default values for threshold and TTLs come from the module-level constants
    in weft.episodes. Pass explicit values at the call site to override.

    Returns counts: {"scored_deleted": int, "no_score_deleted": int}.
    """
    _default_threshold, _default_ttl_scored, _default_ttl_no_score = _graduation_constants()
    if high_threshold is None:
        high_threshold = _default_threshold
    if ttl_days_scored is None:
        ttl_days_scored = _default_ttl_scored
    if ttl_days_no_score is None:
        ttl_days_no_score = _default_ttl_no_score

    db = get_db(pool)

    # Score-aware path: delete turns with a score below threshold whose
    # episode graduated long enough ago.
    scored_result = await db.execute(
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
        high_threshold,
        str(ttl_days_scored),
    )
    scored_deleted = int(scored_result.split()[-1])

    # Score-blind fallback: delete turns with NULL importance_score whose
    # episode graduated past the shorter no-score TTL.
    no_score_result = await db.execute(
        """
        DELETE FROM episode_turns
        WHERE importance_score IS NULL
          AND episode_id IN (
              SELECT id FROM episodes
              WHERE status = 'graduated'
                AND ended_at IS NOT NULL
                AND ended_at < now() - ($1 || ' days')::interval
          )
        """,
        str(ttl_days_no_score),
    )
    no_score_deleted = int(no_score_result.split()[-1])

    return {"scored_deleted": scored_deleted, "no_score_deleted": no_score_deleted}


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
    episode_ids: list[str] | None = None,
) -> list[EpisodeTurn]:
    """Hybrid (vector + BM25) recall over episode_turns.

    Filters apply BEFORE scoring: ``project_id`` joins through ``episodes``,
    ``since`` / ``until`` constrain ``occurred_at``, ``episode_ids`` (when
    supplied) restricts the candidate pool to turns whose ``episode_id``
    is in the list — used by the hierarchical-descent path in
    :func:`recall_turns_hierarchical` to fan out from a top-K episode set.
    Vector similarity uses cosine distance against the inline embedding
    column; keyword scoring uses Postgres FTS
    (``to_tsvector('english', content)``) — there is no persisted tsvector
    column on episode_turns yet, so this path is unindexed for now.
    Acceptable at Wick scale (one user's dialogue trace); add a stored
    search_tsv column + GIN index when a single installation crosses
    ~100k turns.

    RRF fusion mirrors ``weft.store.search_hybrid`` so callers can reason
    about belief-tier and turn-tier results in the same rank space.

    Args:
        embedding: precomputed query embedding. If None, the caller is
            responsible for skipping vector search (we don't reach into
            the embedding provider from this layer to keep the store
            module dependency-free).
        episode_ids: optional list of episode ids; when non-empty, both
            halves filter ``t.episode_id = ANY($N)``. ``None`` (default)
            preserves the legacy unscoped behavior. An empty list short-
            circuits to ``[]`` since no candidate episode could match.
    """
    if episode_ids is not None and len(episode_ids) == 0:
        # Empty filter would produce ``ANY('{}'::text[])`` which matches
        # nothing — shortcut so callers (the hierarchical path on an
        # empty episode result) don't pay the round-trip.
        return []
    candidate_limit = top_k * 3
    sql_filter, params = _build_turn_filters(
        project_id=project_id, since=since, until=until,
        episode_ids=episode_ids,
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
    # WEFT_TURN_RERANK_DISABLE=1 short-circuits back to RRF order.
    # Used by the P1.A5 warm-boost harness to A/B rerank-on vs rerank-off
    # on the same warmed dataset without needing a code revert.
    import os as _os
    if _os.environ.get("WEFT_TURN_RERANK_DISABLE") == "1":
        return [t for t, _ in fused]
    # Late import: relevance imports models, which we already loaded.
    # Done at call time to keep the store layer's import graph minimal.
    from weft.relevance import rank_turns
    ranked = rank_turns(fused)
    return [s.turn for s in ranked]


# --- Hierarchical descent: episodes → turns ---


async def recall_turns_hierarchical(
    pool: asyncpg.Pool,
    query: str,
    *,
    top_k_episodes: int = 10,
    top_k_turns: int = 20,
    embedder: "EmbeddingProvider | None" = None,
    embedding: list[float] | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[EpisodeTurn]:
    """Hierarchical retrieval: rank episodes first, then descend to turns.

    Two-step:

    1. :func:`weft.episodes.recall_episodes` ranks the top-K episodes that
       match the query (cosine + ts_rank + RRF + 30d recency). This is the
       coarse filter — the episode title/summary embedding tends to
       summarize the conversation arc, which gives a lower-noise signal
       than searching every turn flat.
    2. :func:`recall_turns` runs scoped to ``episode_id IN (top_k_ids)``,
       which means the vector + BM25 scoring in step 2 only considers the
       relevant slice. The hybrid score for the surviving turns is what
       the caller sees back; we don't blend the episode-level score in.

    Fallback: when step 1 returns no episodes (empty embedding column,
    too-aggressive temporal filter, etc.), fall through to a flat
    ``recall_turns(query, ...)`` so the caller always gets a populated
    list. This keeps the hierarchical path strictly additive — turning
    the flag on can only ever match or beat the flat baseline on recall,
    not regress it on empty pools.

    Embedder degradation: when no ``embedding`` is provided and no
    ``embedder`` is given, both halves run keyword-only via the
    :func:`recall_episodes` and :func:`recall_turns` keyword fallbacks.

    Scoping (matches the dispatch site convention from the belief tier):

    * ``project_id`` flows into BOTH halves so cross-project context
      cannot bleed in.
    * ``since`` / ``until`` flows into BOTH halves; for episodes it
      filters ``started_at``, for turns it filters ``occurred_at`` —
      both are "when did this happen" axes so the temporal intent is
      preserved through the descent.
    * ``agent_id`` / ``user_id`` flows into the episode half only
      (matching :func:`weft.episodes.recall_episodes`'s scoping shape;
      the turn-tier RLS path uses the session GUC, not a query param).
    """
    # Late imports keep the module's import graph minimal at startup
    # (recall_episodes pulls in weft.relevance and weft.store, which
    # import models and embeddings indirectly).
    from weft.episodes import recall_episodes

    # Resolve query embedding once and share with both halves so we don't
    # pay the embed cost twice. Mirrors recall_both's approach.
    if embedding is None and embedder is not None:
        try:
            embedding = await embedder.embed(query)
        except Exception as exc:
            logger.warning(
                "recall_turns_hierarchical embed failed, "
                "halves will run keyword-only: %s", exc,
            )
            embedding = None

    episodes = await recall_episodes(
        pool, query,
        project_id=project_id,
        agent_id=agent_id,
        user_id=user_id,
        since=since,
        until=until,
        top_k_episodes=top_k_episodes,
        embedding=embedding,
    )

    if not episodes:
        # Fallback path: flat recall_turns so the caller gets something.
        # Only project_id / since / until propagate — agent_id / user_id
        # don't apply at the turn tier (see docstring).
        return await recall_turns(
            pool, query,
            project_id=project_id,
            since=since,
            until=until,
            top_k=top_k_turns,
            embedding=embedding,
        )

    episode_id_set = [e.id for e in episodes]
    return await recall_turns(
        pool, query,
        project_id=project_id,
        since=since,
        until=until,
        top_k=top_k_turns,
        embedding=embedding,
        episode_ids=episode_id_set,
    )


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
    episode_ids: list[str] | None = None,
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
    if episode_ids is not None and len(episode_ids) > 0:
        # ANY($N::text[]) matches any element of the list. The cast is
        # explicit so asyncpg picks the right encoder when the list is
        # otherwise unbound (asyncpg has been known to misinfer when the
        # list mixes None / non-strings; episode IDs are always non-null
        # text so the cast is purely defensive).
        fragments.append(f"AND t.episode_id = ANY(${base + len(params)}::text[])")
        params.append(list(episode_ids))
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
