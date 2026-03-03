"""Postgres store — ONLY writer to the database for memory data.

Handles CRUD operations, relationship management, and vector similarity search.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import asyncpg

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

    embedding_str = _vec_to_pgvector(embedding) if embedding else None

    await pool.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, agent_id, embedding, status, pinned,
            review_after
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $8, $8,
            0, $9, $10, $11, 'active', $12,
            $13
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
        embedding_str,
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
    row = await pool.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
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
    pinned: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Memory]:
    """List memories with optional filters."""
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

    rows = await pool.fetch(query, *params)
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
) -> list[MemoryRecall]:
    """Search memories by vector similarity (cosine distance)."""
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    embedding_str = _vec_to_pgvector(embedding)
    params.append(embedding_str)
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

    rows = await pool.fetch(query, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        results.append(MemoryRecall(memory=memory, similarity=float(row["similarity"])))
    return results


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
        sets.append(f"embedding = ${idx}")
        params.append(_vec_to_pgvector(embedding))
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

    row = await pool.fetchrow(
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
    async with pool.acquire() as conn:
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
                    sets.append(f"embedding = ${idx}")
                    params.append(_vec_to_pgvector(embedding))
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
            embedding_str = _vec_to_pgvector(embedding) if embedding else None

            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, project_id, embedding, status,
                    pinned, review_after
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $8, $8,
                    0, $9, $10, 'active',
                    false, $11
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
                embedding_str,
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
    if hard:
        result = await pool.execute("DELETE FROM memories WHERE id = $1", memory_id)
    else:
        result = await pool.execute(
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
    await pool.execute(
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
    async with pool.acquire() as conn:
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
    await pool.execute(
        """
        INSERT INTO memory_relationships (source_id, target_id, relation, created_at)
        VALUES ($1, $2, $3, $4)
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
    if relation:
        rows = await pool.fetch(
            """
            SELECT * FROM memory_relationships
            WHERE (source_id = $1 OR target_id = $1) AND relation = $2
            """,
            memory_id,
            relation.value,
        )
    else:
        rows = await pool.fetch(
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
    result = await pool.execute(
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
    total = await pool.fetchval("SELECT COUNT(*) FROM memories")
    by_status = await pool.fetch(
        "SELECT status, COUNT(*) as count FROM memories GROUP BY status"
    )
    by_type = await pool.fetch(
        "SELECT type, COUNT(*) as count FROM memories WHERE status = 'active' GROUP BY type"
    )
    by_topic = await pool.fetch(
        """
        SELECT unnest(topic) as topic, COUNT(*) as count
        FROM memories WHERE status = 'active'
        GROUP BY topic ORDER BY count DESC LIMIT 20
        """
    )
    recent = await pool.fetch(
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


# --- Helpers ---


def _vec_to_pgvector(vec: list[float]) -> str:
    """Convert a list of floats to pgvector string format."""
    return "[" + ",".join(str(v) for v in vec) + "]"


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
        review_after=row["review_after"] if row.get("review_after") is not None else None,
    )
