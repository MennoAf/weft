"""Entities store — CRUD, mention linkage, and vector search.

Entities are first-class people, projects, companies, tools, and concepts.
They link to memories via the entity_mentions join table, enabling
"what do I know about person X?" queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncpg

from weft.models import Entity, EntityCreate, EntityType, _weft_id
from weft.store import _row_to_memory

logger = logging.getLogger(__name__)


async def store_entity(
    pool: asyncpg.Pool,
    create: EntityCreate,
    embedding: list[float] | None = None,
) -> Entity:
    """Store a new entity. Returns the created Entity."""
    entity_id = _weft_id()
    now = datetime.now(timezone.utc)

    await pool.execute(
        """
        INSERT INTO entities (
            id, name, entity_type, aliases, description,
            project_id, agent_id, user_id, status, mention_count,
            created_at, updated_at, embedding
        ) VALUES ($1, $2, $3, $4, $5, $6, $7,
                  nullif(current_setting('app.user_id', true), ''),
                  'active', 0, $8, $8, $9::vector)
        """,
        entity_id,
        create.name,
        create.entity_type.value,
        create.aliases,
        create.description,
        create.project_id,
        create.agent_id,
        now,
        embedding,
    )

    return Entity(
        id=entity_id,
        name=create.name,
        entity_type=create.entity_type,
        aliases=create.aliases,
        description=create.description,
        project_id=create.project_id,
        agent_id=create.agent_id,
        status="active",
        mention_count=0,
        created_at=now,
        updated_at=now,
    )


async def get_entity(pool: asyncpg.Pool, entity_id: str) -> Entity | None:
    """Fetch a single entity by ID."""
    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", entity_id)
    if not row:
        return None
    return _row_to_entity(row)


async def list_entities(
    pool: asyncpg.Pool,
    *,
    entity_type: EntityType | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    status: str = "active",
    limit: int = 50,
    offset: int = 0,
) -> list[Entity]:
    """List entities with optional filters. Uses OR-NULL scoping on project_id/agent_id."""
    conditions = []
    params: list = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if entity_type is not None:
        conditions.append(f"entity_type = ${idx}")
        params.append(entity_type.value)
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
        SELECT * FROM entities {where}
        ORDER BY mention_count DESC, updated_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await pool.fetch(query, *params)
    return [_row_to_entity(r) for r in rows]


async def search_entities(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    entity_type: EntityType | None = None,
    project_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> list[tuple[Entity, float]]:
    """Search entities by vector similarity. Returns (entity, similarity) tuples."""
    conditions = ["status = 'active'", "embedding IS NOT NULL"]
    params: list = [embedding]
    idx = 2

    if entity_type is not None:
        conditions.append(f"entity_type = ${idx}")
        params.append(entity_type.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT *, 1 - (embedding <=> $1::vector) AS similarity
        FROM entities {where}
        ORDER BY embedding <=> $1::vector
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(query, *params)
    results = []
    for r in rows:
        sim = float(r["similarity"])
        if sim >= threshold:
            results.append((_row_to_entity(r), sim))
    return results


async def link_mention(
    pool: asyncpg.Pool,
    entity_id: str,
    memory_id: str,
) -> bool:
    """Link a memory to an entity. Idempotent (ON CONFLICT DO NOTHING).

    Also increments the entity's mention_count.
    Returns True if a new link was created, False if already existed.
    """
    result = await pool.execute(
        """
        INSERT INTO entity_mentions (entity_id, memory_id, user_id)
        VALUES ($1, $2, nullif(current_setting('app.user_id', true), ''))
        ON CONFLICT (entity_id, memory_id) DO NOTHING
        """,
        entity_id,
        memory_id,
    )
    created = result.split()[-1] != "0"
    if created:
        await pool.execute(
            "UPDATE entities SET mention_count = mention_count + 1, updated_at = $2 WHERE id = $1",
            entity_id,
            datetime.now(timezone.utc),
        )
    return created


async def unlink_mention(
    pool: asyncpg.Pool,
    entity_id: str,
    memory_id: str,
) -> bool:
    """Unlink a memory from an entity. Returns True if removed."""
    result = await pool.execute(
        "DELETE FROM entity_mentions WHERE entity_id = $1 AND memory_id = $2",
        entity_id,
        memory_id,
    )
    removed = result.split()[-1] != "0"
    if removed:
        await pool.execute(
            "UPDATE entities SET mention_count = GREATEST(mention_count - 1, 0), updated_at = $2 WHERE id = $1",
            entity_id,
            datetime.now(timezone.utc),
        )
    return removed


async def get_entity_memories(
    pool: asyncpg.Pool,
    entity_id: str,
    *,
    limit: int = 100,
):
    """Get memories linked to an entity, ordered by mention time (newest first)."""
    rows = await pool.fetch(
        """
        SELECT m.* FROM memories m
        JOIN entity_mentions em ON m.id = em.memory_id
        WHERE em.entity_id = $1 AND m.status = 'active'
        ORDER BY em.mentioned_at DESC
        LIMIT $2
        """,
        entity_id,
        limit,
    )
    return [_row_to_memory(r) for r in rows]


async def get_memory_entities(
    pool: asyncpg.Pool,
    memory_id: str,
) -> list[Entity]:
    """Reverse lookup — which entities are linked to this memory?"""
    rows = await pool.fetch(
        """
        SELECT e.* FROM entities e
        JOIN entity_mentions em ON e.id = em.entity_id
        WHERE em.memory_id = $1 AND e.status = 'active'
        ORDER BY e.name
        """,
        memory_id,
    )
    return [_row_to_entity(r) for r in rows]


# --- Helpers ---


def _row_to_entity(row: asyncpg.Record) -> Entity:
    """Convert a database row to an Entity model."""
    return Entity(
        id=row["id"],
        name=row["name"],
        entity_type=EntityType(row["entity_type"]),
        aliases=row["aliases"] or [],
        description=row["description"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        status=row["status"],
        mention_count=row["mention_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
