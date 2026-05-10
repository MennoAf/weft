"""Episodes store — CRUD, memory linkage, and timeline queries.

Episodes group memories into time-bounded causal sequences,
enabling "what happened around that decision?" queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import asyncpg

from datetime import timedelta

from weft.db.connection import get_db
from weft.relevance import recency_factor
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
from weft.store import _RRF_K, _row_to_memory, store_memory
from weft.tokens import estimate_tokens

if TYPE_CHECKING:
    from weft.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# --- Graduation retention constants ---

# Importance score above which a turn persists indefinitely after its episode
# graduates. Below this, retention falls back to age-based TTL.
GRADUATION_HIGH_IMPORTANCE_THRESHOLD = 0.7

# Default TTL for low-importance turns AFTER episode graduation. Used when
# importance_score is populated and below the high threshold.
GRADUATION_TURN_TTL_DAYS = 90

# Fallback TTL for turns with NULL importance_score (Face offline). Conservative
# (shorter than the scored fallback) because we cannot tell what to keep.
GRADUATION_TURN_TTL_DAYS_NO_SCORE = 30

# Maximum number of turns to include verbatim in the graduated memory's
# conversation section. Turns beyond this cap produce a truncation notice.
_GRADUATION_TURN_CAP = 60


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
        from weft.episode_turns import list_turns as _list_turns

        parts = [ep.title]
        if ep.summary:
            parts.append(ep.summary)

        # Pull turns and pre-existing linked memories to enrich the snapshot.
        turns = await _list_turns(pool, episode_id)
        prior_memories = await get_episode_memories(pool, episode_id)

        if turns:
            # Cap at _GRADUATION_TURN_CAP turns (windowed compression — no LLM).
            included = turns[:_GRADUATION_TURN_CAP]
            omitted = len(turns) - len(included)
            convo_lines = [f"{t.role.value}: {t.content}" for t in included]
            if omitted > 0:
                convo_lines.append(f"... [truncated, {omitted} turns omitted]")
            parts.append("Conversation:\n" + "\n".join(convo_lines))

        if prior_memories:
            mem_lines = [m.content for m in prior_memories]
            parts.append("Linked memories:\n" + "\n".join(mem_lines))

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


# --- Recall ---


# Recency half-life for the episode-tier recall score. 30 days mirrors the
# default `RelevanceWeights.recency_half_life_days` for the belief tier so
# behavior is consistent across tiers; episodes that never anchor in time
# don't lose more recency value than the memories that surround them.
_EPISODE_RECENCY_HALF_LIFE_DAYS = 30.0


async def recall_episodes(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    top_k_episodes: int = 10,
    embedding: list[float] | None = None,
    embedder: "EmbeddingProvider | None" = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    status: EpisodeStatus | None = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    now: datetime | None = None,
) -> list[Episode]:
    """Hybrid (vector + ts_rank) recall over episodes, ranked by RRF + recency.

    Mirrors :func:`weft.episode_turns.recall_turns` at the episode tier.
    Vector half scans :sql:`embedding <=> $1` (cosine distance) using the
    HNSW index added in v47; keyword half computes :sql:`ts_rank` over
    :sql:`title || ' ' || COALESCE(summary, '')`. The two ranked lists are
    fused via Reciprocal Rank Fusion (K=60, the same constant
    :mod:`weft.store.search_hybrid` and
    :mod:`weft.episode_turns.recall_turns` use), and the composite score
    is multiplied by an exponential recency factor over ``started_at`` so
    older episodes decay against newer ones at similar topical match.

    Args:
        embedding: precomputed query embedding. If None and ``embedder`` is
            provided, the embedder is invoked once to derive a vector from
            ``query``. If both are None, the vector half is skipped (BM25
            keyword half still runs).
        embedder: optional :class:`EmbeddingProvider` used to compute the
            query embedding when ``embedding`` is None. Mirrors how the
            MCP layer threads ``app.episode_embedding`` into call sites.
        project_id, agent_id, user_id: OR-NULL scoping — globally-scoped
            episodes (NULL on that column) always pass through, matching
            :func:`list_episodes` exactly.
        since/until: filter on ``started_at`` (matching ``list_episodes``'s
            ORDER BY axis); episodes that started outside the window are
            excluded before scoring.
        status: optional :class:`EpisodeStatus` filter.
    """
    if top_k_episodes <= 0:
        return []

    # Resolve query vector: prefer the explicit kwarg; otherwise call the
    # embedder if one is supplied; else skip vector half.
    if embedding is None and embedder is not None:
        try:
            embedding = await embedder.embed(query)
        except Exception as exc:
            logger.warning(
                "recall_episodes embed failed, vector half skipped: %s", exc,
            )
            embedding = None

    candidate_limit = top_k_episodes * 2
    sql_filter, filter_params = _build_episode_recall_filters(
        project_id=project_id,
        agent_id=agent_id,
        user_id=user_id,
        since=since,
        until=until,
        status=status,
        leading_args=1,  # $1 is reserved for embedding OR query
    )

    db = get_db(pool)

    # --- Vector half (skipped when embedding is None) ---
    vector_rows: list[asyncpg.Record] = []
    if embedding is not None:
        vector_sql = f"""
            SELECT *
              FROM episodes
              WHERE embedding IS NOT NULL
                {sql_filter}
              ORDER BY embedding <=> $1::vector
              LIMIT ${len(filter_params) + 2}
        """
        vector_rows = await db.fetch(
            vector_sql, embedding, *filter_params, candidate_limit,
        )

    # --- Keyword half (ts_rank over title + summary) ---
    keyword_sql = f"""
        SELECT *,
               ts_rank(
                   to_tsvector('english', title || ' ' || COALESCE(summary, '')),
                   plainto_tsquery('english', $1)
               ) AS rank
          FROM episodes
          WHERE to_tsvector('english', title || ' ' || COALESCE(summary, ''))
                @@ plainto_tsquery('english', $1)
            {sql_filter}
          ORDER BY rank DESC
          LIMIT ${len(filter_params) + 2}
    """
    keyword_rows = await db.fetch(
        keyword_sql, query, *filter_params, candidate_limit,
    )

    return _rrf_fuse_episode_rows(
        vector_rows,
        keyword_rows,
        candidate_limit=candidate_limit,
        top_k=top_k_episodes,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        now=now,
    )


def _build_episode_recall_filters(
    *,
    project_id: str | None,
    agent_id: str | None,
    user_id: str | None,
    since: datetime | None,
    until: datetime | None,
    status: EpisodeStatus | None,
    leading_args: int,
) -> tuple[str, list]:
    """AND-joined WHERE fragments for ``recall_episodes``.

    Mirrors :func:`list_episodes` scoping shape exactly: OR-NULL on
    project_id and agent_id, two-arg OR on user_id (caller value OR
    ``SYSTEM_GLOBAL_USER_ID`` sentinel). Param numbering is offset by
    ``leading_args`` so the caller's positional embedding/query argument
    keeps $1.
    """
    fragments: list[str] = []
    params: list = []
    base = leading_args + 1  # first param emitted is ${base}

    if status is not None:
        fragments.append(f"AND status = ${base + len(params)}")
        params.append(status.value)
    if project_id is not None:
        fragments.append(
            f"AND (project_id = ${base + len(params)} OR project_id IS NULL)"
        )
        params.append(project_id)
    if agent_id is not None:
        fragments.append(
            f"AND (agent_id = ${base + len(params)} OR agent_id IS NULL)"
        )
        params.append(agent_id)
    if user_id is not None:
        # Two params: caller value + global sentinel (matches list_episodes).
        fragments.append(
            f"AND (user_id = ${base + len(params)} "
            f"OR user_id = ${base + len(params) + 1})"
        )
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
    if since is not None:
        fragments.append(f"AND started_at >= ${base + len(params)}")
        params.append(since)
    if until is not None:
        fragments.append(f"AND started_at <= ${base + len(params)}")
        params.append(until)

    return (" ".join(fragments), params)


def _rrf_fuse_episode_rows(
    vector_rows: list[asyncpg.Record],
    keyword_rows: list[asyncpg.Record],
    *,
    candidate_limit: int,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    now: datetime | None = None,
) -> list[Episode]:
    """RRF-fuse the two halves, multiply each composite by a recency factor.

    RRF mirrors :func:`weft.episode_turns._rrf_fuse_turn_rows` — missing
    rows take an absent-rank penalty of ``candidate_limit + 1`` so an
    episode appearing in only one half can still surface if its rank is
    high. The recency multiplier comes from
    :func:`weft.relevance.recency_factor` evaluated against
    ``started_at`` (still meaningful for closed/graduated episodes — the
    episode's recall worth decays from when it began, not from its
    eventual end).
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
    for eid, row in all_rows.items():
        v = vector_ranks.get(eid, absent)
        k = keyword_ranks.get(eid, absent)
        rrf = (
            vector_weight / (_RRF_K + v)
            + keyword_weight / (_RRF_K + k)
        )
        rec = recency_factor(
            row["started_at"],
            now=now,
            half_life_days=_EPISODE_RECENCY_HALF_LIFE_DAYS,
        )
        scores[eid] = rrf * rec

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    return [_row_to_episode(all_rows[eid]) for eid in sorted_ids]


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
