"""Postgres store — ONLY writer to the database for memory data.

Handles CRUD operations, relationship management, and vector similarity search.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import asyncpg

from weft.db.connection import acquire, get_db
from weft.models import (
    Memory,
    MemoryCreate,
    MemoryRecall,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    RelationType,
    _weft_id,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None


async def store_memory(
    pool: asyncpg.Pool,
    create: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    """Store a new memory. Returns the created Memory."""
    memory_id = _weft_id()
    now = datetime.now(timezone.utc)
    token_count = estimate_tokens(create.content)

    db = get_db(pool)
    await db.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, agent_id, embedding, status, pinned,
            review_after, user_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $8, $8,
            0, $9, $10, $11::vector, 'active', $12,
            $13, nullif(current_setting('app.user_id', true), '')
        )
        """,
        memory_id,
        create.type.value,
        create.topic,
        create.content,
        create.source.value,
        create.confidence,
        token_count,
        now,
        create.project_id,
        create.agent_id,
        embedding,
        create.pinned,
        create.review_after,
    )

    return Memory(
        id=memory_id,
        type=create.type,
        topic=create.topic,
        content=create.content,
        source=create.source,
        confidence=create.confidence,
        token_count=token_count,
        created_at=now,
        updated_at=now,
        accessed_at=now,
        access_count=0,
        project_id=create.project_id,
        agent_id=create.agent_id,
        status=MemoryStatus.active,
        pinned=create.pinned,
        review_after=create.review_after,
    )


async def get_memory(pool: asyncpg.Pool, memory_id: str) -> Memory | None:
    """Fetch a single memory by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
    if not row:
        return None
    return _row_to_memory(row)


async def list_memories(
    pool: asyncpg.Pool,
    *,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    pinned: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Memory]:
    """List memories with optional filters.

    Scoping: pass project_id and/or agent_id to narrow results.
    Each axis uses OR-NULL logic (matches the value OR global memories).
    Omit both for brain-wide (unscoped) queries.
    """
    conditions = []
    params: list = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if pinned is not None:
        conditions.append(f"pinned = ${idx}")
        params.append(pinned)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM memories {where}
        ORDER BY updated_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_memory(r) for r in rows]


async def search_by_vector(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    limit: int = 10,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    exclude_ids: list[str] | None = None,
) -> list[MemoryRecall]:
    """Search memories by vector similarity (cosine distance).

    Scoping: pass project_id and/or agent_id to narrow results.
    Each axis uses OR-NULL logic (matches the value OR global memories).
    Omit both for brain-wide (unscoped) queries.

    exclude_ids: memory IDs to exclude from results (e.g., already surfaced
    by primer). Uses NOT id = ANY($N) for efficient filtering.
    """
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    # Similarity threshold in SQL so the DB handles filtering atomically
    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT *,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        {where}
        ORDER BY embedding <=> $1::vector
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(query, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        results.append(MemoryRecall(memory=memory, similarity=float(row["similarity"])))
    return results


async def search_by_keyword(
    pool: asyncpg.Pool,
    query: str,
    *,
    limit: int = 10,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    exclude_ids: list[str] | None = None,
) -> list[MemoryRecall]:
    """Search memories by full-text keyword match (BM25 ranking via ts_rank).

    Uses the search_tsv tsvector column with plainto_tsquery for robust
    keyword matching including stemming and stop-word removal.
    """
    conditions = ["search_tsv IS NOT NULL"]
    params: list = []
    idx = 1

    # $1 = tsquery
    conditions.append(f"search_tsv @@ plainto_tsquery('english', ${idx})")
    params.append(query)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT *,
               ts_rank(search_tsv, plainto_tsquery('english', $1)) AS rank
        FROM memories
        {where}
        ORDER BY rank DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(sql, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        # Normalize ts_rank (typically 0–1 but can exceed 1) into 0–1 range
        # for compatibility with MemoryRecall.similarity
        raw_rank = float(row["rank"])
        similarity = min(1.0, raw_rank)
        results.append(MemoryRecall(memory=memory, similarity=similarity))
    return results


# Reciprocal Rank Fusion constant (standard default from the literature)
_RRF_K = 60


async def search_hybrid(
    pool: asyncpg.Pool,
    query: str,
    embedding: list[float],
    *,
    limit: int = 10,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    exclude_ids: list[str] | None = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> list[MemoryRecall]:
    """Hybrid search combining vector similarity and BM25 keyword matching.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval
    methods. RRF is rank-based, so it handles the different score scales
    (cosine similarity vs ts_rank) naturally.

    vector_weight/keyword_weight control the relative importance of each
    signal in the RRF formula: score = w / (k + rank).
    """
    # Fetch broader candidate sets from both methods, then fuse
    candidate_limit = limit * 3  # over-fetch to ensure good fusion

    vector_results = await search_by_vector(
        pool,
        embedding,
        limit=candidate_limit,
        threshold=threshold,
        status=status,
        memory_type=memory_type,
        topic=topic,
        project_id=project_id,
        agent_id=agent_id,
        exclude_ids=exclude_ids,
    )

    keyword_results = await search_by_keyword(
        pool,
        query,
        limit=candidate_limit,
        status=status,
        memory_type=memory_type,
        topic=topic,
        project_id=project_id,
        agent_id=agent_id,
        exclude_ids=exclude_ids,
    )

    # Build rank maps (1-indexed)
    vector_ranks: dict[str, int] = {}
    for i, r in enumerate(vector_results):
        vector_ranks[r.memory.id] = i + 1

    keyword_ranks: dict[str, int] = {}
    for i, r in enumerate(keyword_results):
        keyword_ranks[r.memory.id] = i + 1

    # Collect all candidate memories
    all_memories: dict[str, MemoryRecall] = {}
    for r in vector_results:
        all_memories[r.memory.id] = r
    for r in keyword_results:
        if r.memory.id not in all_memories:
            all_memories[r.memory.id] = r

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    absent_rank = candidate_limit + 1  # penalty rank for missing results

    for mid in all_memories:
        v_rank = vector_ranks.get(mid, absent_rank)
        k_rank = keyword_ranks.get(mid, absent_rank)
        rrf_scores[mid] = (
            vector_weight / (_RRF_K + v_rank)
            + keyword_weight / (_RRF_K + k_rank)
        )

    # Sort by RRF score descending, take top `limit`
    sorted_ids = sorted(rrf_scores, key=lambda mid: rrf_scores[mid], reverse=True)[:limit]

    # Build results with RRF score as similarity (normalized to 0–1)
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    results = []
    for mid in sorted_ids:
        recall = all_memories[mid]
        normalized_score = rrf_scores[mid] / max_rrf if max_rrf > 0 else 0.0
        results.append(MemoryRecall(memory=recall.memory, similarity=normalized_score))

    return results


_CROSS_PROJECT_PENALTY = 0.8


async def search_cross_project(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    exclude_project_id: str | None,
    limit: int = 3,
    threshold: float = 0.5,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    exclude_ids: list[str] | None = None,
) -> list[MemoryRecall]:
    """Search memories from OTHER projects (cross-project insights).

    Returns memories NOT in exclude_project_id, with a 0.8x relevance penalty
    applied to similarity scores so cross-project hits never outrank
    same-project hits.

    If exclude_project_id is None (global context), returns only project-scoped
    memories (project_id IS NOT NULL) to avoid returning all globals.
    """
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    # Similarity threshold
    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    # Cross-project exclusion (NULL-safe)
    if exclude_project_id is not None:
        conditions.append(f"(project_id != ${idx} OR project_id IS NULL)")
        params.append(exclude_project_id)
        idx += 1
    else:
        # Global context: only show project-scoped memories
        conditions.append("project_id IS NOT NULL")

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT *,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        {where}
        ORDER BY embedding <=> $1::vector
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(query, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        # Apply 0.8x penalty for cross-project results
        raw_sim = float(row["similarity"])
        penalized_sim = raw_sim * _CROSS_PROJECT_PENALTY
        results.append(MemoryRecall(memory=memory, similarity=penalized_sim))
    return results


async def count_by_vector(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> int:
    """Count total memories matching a vector search (same filters as search_by_vector, no LIMIT)."""
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    query = f"SELECT COUNT(*) FROM memories {where}"

    return await get_db(pool).fetchval(query, *params)


async def get_recent_writes(
    pool: asyncpg.Pool,
    *,
    limit: int = 10,
    project_id: str | None = None,
) -> list[dict]:
    """Return the most recently created memories with provenance info."""
    db = get_db(pool)
    if project_id is not None:
        rows = await db.fetch(
            """SELECT id, type, content, source, agent_id, project_id, created_at
               FROM memories
               WHERE status = 'active' AND (project_id = $1 OR project_id IS NULL)
               ORDER BY created_at DESC LIMIT $2""",
            project_id, limit,
        )
    else:
        rows = await db.fetch(
            """SELECT id, type, content, source, agent_id, project_id, created_at
               FROM memories
               WHERE status = 'active'
               ORDER BY created_at DESC LIMIT $1""",
            limit,
        )
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "content": r["content"][:80] + ("..." if len(r["content"]) > 80 else ""),
            "source": r["source"],
            "agent_id": r["agent_id"],
            "project_id": r["project_id"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def update_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    content: str | None = None,
    confidence: float | None = None,
    status: MemoryStatus | None = None,
    memory_type: MemoryType | None = None,
    topic: list[str] | None = None,
    embedding: list[float] | None = None,
    pinned: bool | None = None,
    project_id: str | None = _UNSET,
    review_after: datetime | None = _UNSET,
) -> Memory | None:
    """Update mutable fields of a memory. Returns updated Memory or None."""
    sets = ["updated_at = now()"]
    params: list = []
    idx = 1

    if content is not None:
        sets.append(f"content = ${idx}")
        params.append(content)
        idx += 1
        sets.append(f"token_count = ${idx}")
        params.append(estimate_tokens(content))
        idx += 1

    if confidence is not None:
        sets.append(f"confidence = ${idx}")
        params.append(confidence)
        idx += 1

    if memory_type is not None:
        sets.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if topic is not None:
        sets.append(f"topic = ${idx}")
        params.append(topic)
        idx += 1

    if embedding is not None:
        sets.append(f"embedding = ${idx}::vector")
        params.append(embedding)
        idx += 1

    if pinned is not None:
        sets.append(f"pinned = ${idx}")
        params.append(pinned)
        idx += 1

    if project_id is not _UNSET:
        sets.append(f"project_id = ${idx}")
        params.append(project_id)
        idx += 1

    if review_after is not _UNSET:
        sets.append(f"review_after = ${idx}")
        params.append(review_after)
        idx += 1

    set_clause = ", ".join(sets)
    params.append(memory_id)

    row = await get_db(pool).fetchrow(
        f"UPDATE memories SET {set_clause} WHERE id = ${idx} RETURNING *",
        *params,
    )
    return _row_to_memory(row) if row else None


async def upsert_by_topic(
    pool: asyncpg.Pool,
    *,
    topic: list[str],
    project_id: str | None,
    content: str,
    memory_type: MemoryType,
    source: MemorySource,
    confidence: float,
    review_after: datetime | None = None,
    embedding: list[float] | None = None,
) -> Memory:
    """Find a memory by exact topic array + project_id match; update or create.

    Uses a transaction to ensure atomicity. If a matching active memory exists,
    updates its content, confidence, updated_at, and optionally embedding.
    Otherwise creates a new memory with all provided fields.

    Returns the updated or newly created Memory.
    """
    async with acquire(pool) as conn:
        async with conn.transaction():
            # Look for an existing active memory with exact topic match
            if project_id is not None:
                existing = await conn.fetchrow(
                    """
                    SELECT * FROM memories
                    WHERE topic @> $1 AND topic <@ $1
                      AND project_id = $2
                      AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    topic,
                    project_id,
                )
            else:
                existing = await conn.fetchrow(
                    """
                    SELECT * FROM memories
                    WHERE topic @> $1 AND topic <@ $1
                      AND project_id IS NULL
                      AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    topic,
                )

            if existing:
                # Update the existing memory
                sets = ["content = $1", "confidence = $2", "updated_at = now()"]
                params: list = [content, confidence]
                idx = 3

                sets.append(f"token_count = ${idx}")
                params.append(estimate_tokens(content))
                idx += 1

                if embedding is not None:
                    sets.append(f"embedding = ${idx}::vector")
                    params.append(embedding)
                    idx += 1

                set_clause = ", ".join(sets)
                params.append(existing["id"])

                row = await conn.fetchrow(
                    f"UPDATE memories SET {set_clause} WHERE id = ${idx} RETURNING *",
                    *params,
                )
                return _row_to_memory(row)

            # Create a new memory
            memory_id = _weft_id()
            now = datetime.now(timezone.utc)
            token_count = estimate_tokens(content)
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, project_id, embedding, status,
                    pinned, review_after, user_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $8, $8,
                    0, $9, $10::vector, 'active',
                    false, $11, nullif(current_setting('app.user_id', true), '')
                )
                """,
                memory_id,
                memory_type.value,
                topic,
                content,
                source.value,
                confidence,
                token_count,
                now,
                project_id,
                embedding,
                review_after,
            )

            return Memory(
                id=memory_id,
                type=memory_type,
                topic=topic,
                content=content,
                source=source,
                confidence=confidence,
                token_count=token_count,
                created_at=now,
                updated_at=now,
                accessed_at=now,
                access_count=0,
                project_id=project_id,
                status=MemoryStatus.active,
                pinned=False,
                review_after=review_after,
            )


async def delete_memory(pool: asyncpg.Pool, memory_id: str, *, hard: bool = False) -> bool:
    """Delete a memory. Soft-delete (archive) by default, hard-delete if specified."""
    db = get_db(pool)
    if hard:
        result = await db.execute("DELETE FROM memories WHERE id = $1", memory_id)
    else:
        result = await db.execute(
            "UPDATE memories SET status = 'archived', updated_at = now() WHERE id = $1",
            memory_id,
        )
    return result.split()[-1] != "0"


async def touch_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    implicit_alpha: float = 0.05,
) -> None:
    """Update accessed_at, increment access_count, and apply mild usefulness bump.

    The implicit bump treats retrieval as weak positive evidence of usefulness.
    Uses EMA with a small alpha (default 0.05) — much weaker than explicit
    feedback (alpha=0.3) so it takes many accesses to move the needle.

    Formula: new_score = (1 - alpha) * old_score + alpha * 1.0
    """
    await get_db(pool).execute(
        """
        UPDATE memories
        SET accessed_at = now(),
            access_count = access_count + 1,
            usefulness_score = LEAST(1.0, (1.0 - $2) * usefulness_score + $2)
        WHERE id = $1
        """,
        memory_id,
        implicit_alpha,
    )


async def record_feedback(
    pool: asyncpg.Pool,
    memory_id: str,
    helpful: bool,
    alpha: float = 0.3,
) -> dict:
    """Record usefulness feedback using exponential moving average.

    new_score = alpha * signal + (1 - alpha) * old_score
    where signal = 1.0 for helpful, 0.0 for not helpful.
    """
    async with acquire(pool) as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT usefulness_score, usefulness_count FROM memories WHERE id = $1 FOR UPDATE",
                memory_id,
            )
            if row is None:
                raise ValueError(f"Memory {memory_id} not found")

            old_score = float(row["usefulness_score"]) if row["usefulness_score"] is not None else 1.0
            signal = 1.0 if helpful else 0.0
            new_score = alpha * signal + (1 - alpha) * old_score
            new_score = max(0.0, min(1.0, new_score))
            new_count = (row["usefulness_count"] or 0) + 1

            await conn.execute(
                """
                UPDATE memories
                SET usefulness_score = $1, usefulness_count = $2, updated_at = now()
                WHERE id = $3
                """,
                new_score,
                new_count,
                memory_id,
            )

    return {
        "memory_id": memory_id,
        "usefulness_score": new_score,
        "usefulness_count": new_count,
    }


# --- Relationships ---


async def add_relationship(
    pool: asyncpg.Pool,
    source_id: str,
    target_id: str,
    relation: RelationType,
) -> MemoryRelationship:
    """Create a relationship between two memories."""
    now = datetime.now(timezone.utc)
    await get_db(pool).execute(
        """
        INSERT INTO memory_relationships (source_id, target_id, relation, created_at, user_id)
        VALUES ($1, $2, $3, $4, nullif(current_setting('app.user_id', true), ''))
        ON CONFLICT (source_id, target_id, relation) DO NOTHING
        """,
        source_id,
        target_id,
        relation.value,
        now,
    )
    return MemoryRelationship(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        created_at=now,
    )


async def get_relationships(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    relation: RelationType | None = None,
) -> list[MemoryRelationship]:
    """Get all relationships for a memory (as source or target)."""
    db = get_db(pool)
    if relation:
        rows = await db.fetch(
            """
            SELECT * FROM memory_relationships
            WHERE (source_id = $1 OR target_id = $1) AND relation = $2
            """,
            memory_id,
            relation.value,
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM memory_relationships WHERE source_id = $1 OR target_id = $1",
            memory_id,
        )
    return [
        MemoryRelationship(
            source_id=r["source_id"],
            target_id=r["target_id"],
            relation=RelationType(r["relation"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


async def remove_relationship(
    pool: asyncpg.Pool,
    source_id: str,
    target_id: str,
    relation: RelationType,
) -> bool:
    """Remove a specific relationship. Returns True if deleted."""
    result = await get_db(pool).execute(
        """
        DELETE FROM memory_relationships
        WHERE source_id = $1 AND target_id = $2 AND relation = $3
        """,
        source_id,
        target_id,
        relation.value,
    )
    return result.split()[-1] != "0"


# --- Stats ---


async def get_stats(pool: asyncpg.Pool) -> dict:
    """Get memory statistics."""
    db = get_db(pool)
    total = await db.fetchval("SELECT COUNT(*) FROM memories")
    by_status = await db.fetch(
        "SELECT status, COUNT(*) as count FROM memories GROUP BY status"
    )
    by_type = await db.fetch(
        "SELECT type, COUNT(*) as count FROM memories WHERE status = 'active' GROUP BY type"
    )
    by_topic = await db.fetch(
        """
        SELECT unnest(topic) as topic, COUNT(*) as count
        FROM memories WHERE status = 'active'
        GROUP BY topic ORDER BY count DESC LIMIT 20
        """
    )
    recent = await db.fetch(
        "SELECT id, content, accessed_at FROM memories WHERE status = 'active' ORDER BY accessed_at DESC LIMIT 5"
    )

    return {
        "total": total,
        "by_status": {r["status"]: r["count"] for r in by_status},
        "by_type": {r["type"]: r["count"] for r in by_type},
        "top_topics": {r["topic"]: r["count"] for r in by_topic},
        "recently_accessed": [
            {"id": r["id"], "content": r["content"][:80], "accessed_at": r["accessed_at"].isoformat()}
            for r in recent
        ],
    }


# --- Metadata (system-level key-value) ---


async def get_metadata(pool: asyncpg.Pool, key: str) -> dict | None:
    """Get a metadata value by key. Returns None if not found or table missing."""
    try:
        row = await get_db(pool).fetchrow(
            "SELECT value FROM weft_metadata WHERE key = $1", key,
        )
        if row is None:
            return None
        val = row["value"]
        # asyncpg returns JSONB as a string or dict depending on codec
        if isinstance(val, str):
            import json
            return json.loads(val)
        return dict(val)
    except Exception:
        # Table may not exist if migration hasn't run yet
        return None


async def set_metadata(pool: asyncpg.Pool, key: str, value: dict) -> None:
    """Upsert a metadata value (idempotent)."""
    import json
    await get_db(pool).execute(
        """INSERT INTO weft_metadata (key, value, updated_at)
           VALUES ($1, $2::jsonb, NOW())
           ON CONFLICT (key) DO UPDATE
           SET value = EXCLUDED.value, updated_at = NOW()""",
        key, json.dumps(value),
    )


# --- Changes since ---


async def get_last_handoff_timestamp(
    pool: asyncpg.Pool,
    project_id: str | None = None,
) -> datetime | None:
    """Return the created_at of the most recent handoff memory, or None.

    Scoping: project_id=None queries global handoffs (project_id IS NULL).
    A non-None project_id matches that exact project.
    """
    db = get_db(pool)
    if project_id is not None:
        row = await db.fetchrow(
            """
            SELECT MAX(created_at) AS ts
            FROM memories
            WHERE type = 'handoff' AND status = 'active'
              AND project_id = $1
            """,
            project_id,
        )
    else:
        row = await db.fetchrow(
            """
            SELECT MAX(created_at) AS ts
            FROM memories
            WHERE type = 'handoff' AND status = 'active'
              AND project_id IS NULL
            """,
        )
    return row["ts"] if row and row["ts"] else None


async def get_memory_changes_since(
    pool: asyncpg.Pool,
    *,
    since: datetime,
    project_id: str | None = None,
) -> dict:
    """Count memory mutations since a timestamp using conditional aggregation.

    Returns dict with keys: memories_created, memories_archived, memories_revised, since.
    - created: active memories with created_at > since
    - archived: archived memories with updated_at > since
    - revised: active memories updated since the timestamp but created before it
    """
    db = get_db(pool)
    if project_id is not None:
        row = await db.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'active' AND created_at > $1
                ) AS memories_created,
                COUNT(*) FILTER (
                    WHERE status = 'archived' AND updated_at > $1
                ) AS memories_archived,
                COUNT(*) FILTER (
                    WHERE status = 'active' AND updated_at > $1 AND created_at <= $1
                ) AS memories_revised
            FROM memories
            WHERE project_id = $2
            """,
            since, project_id,
        )
    else:
        row = await db.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'active' AND created_at > $1
                ) AS memories_created,
                COUNT(*) FILTER (
                    WHERE status = 'archived' AND updated_at > $1
                ) AS memories_archived,
                COUNT(*) FILTER (
                    WHERE status = 'active' AND updated_at > $1 AND created_at <= $1
                ) AS memories_revised
            FROM memories
            WHERE project_id IS NULL
            """,
            since,
        )
    return {
        "memories_created": row["memories_created"],
        "memories_archived": row["memories_archived"],
        "memories_revised": row["memories_revised"],
        "since": since.isoformat(),
    }


# --- Helpers ---


def _row_to_memory(row: asyncpg.Record) -> Memory:
    """Convert a database row to a Memory model."""
    return Memory(
        id=row["id"],
        type=MemoryType(row["type"]),
        topic=list(row["topic"]) if row["topic"] else [],
        content=row["content"],
        source=row["source"],
        confidence=row["confidence"],
        token_count=row["token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        accessed_at=row["accessed_at"],
        access_count=row["access_count"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        status=MemoryStatus(row["status"]),
        pinned=bool(row["pinned"]) if row.get("pinned") is not None else False,
        usefulness_score=float(row["usefulness_score"]) if row["usefulness_score"] is not None else 1.0,
        usefulness_count=row["usefulness_count"] if row["usefulness_count"] is not None else 0,
        last_boosted_at=row.get("last_boosted_at"),
        review_after=row["review_after"] if row.get("review_after") is not None else None,
    )
