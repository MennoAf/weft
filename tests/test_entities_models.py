"""Tests for entity models and migration."""

from __future__ import annotations

import pytest

from weft.models import Entity, EntityCreate, EntityType, MemoryCreate, MemoryType
from weft.store import store_memory


# --- EntityType enum ---


def test_entity_type_values():
    assert EntityType.person.value == "person"
    assert EntityType.project.value == "project"
    assert EntityType.company.value == "company"
    assert EntityType.tool.value == "tool"
    assert EntityType.concept.value == "concept"


# --- EntityCreate validation ---


def test_entity_create_minimal():
    ec = EntityCreate(name="Casey Example")
    assert ec.name == "Casey Example"
    assert ec.entity_type == EntityType.concept
    assert ec.aliases == []
    assert ec.description is None
    assert ec.project_id is None
    assert ec.agent_id is None


def test_entity_create_full():
    ec = EntityCreate(
        name="Weft",
        entity_type=EntityType.project,
        aliases=["weft-memory", "weft-system"],
        description="Persistent agent memory system",
        project_id="weft",
        agent_id="warp",
    )
    assert ec.name == "Weft"
    assert ec.entity_type == EntityType.project
    assert ec.aliases == ["weft-memory", "weft-system"]
    assert ec.description == "Persistent agent memory system"


# --- Entity model ---


def test_entity_defaults():
    e = Entity(name="Test Entity")
    assert e.id.startswith("weft-")
    assert e.entity_type == EntityType.concept
    assert e.aliases == []
    assert e.status == "active"
    assert e.mention_count == 0
    assert e.description is None


def test_entity_to_dict():
    e = Entity(
        name="PostgreSQL",
        entity_type=EntityType.tool,
        description="Relational database",
        project_id="weft",
    )
    d = e.to_dict()
    assert d["name"] == "PostgreSQL"
    assert d["entity_type"] == "tool"
    assert d["project_id"] == "weft"
    assert "id" in d
    assert "created_at" in d


def test_entity_with_aliases():
    e = Entity(
        name="Casey Example",
        entity_type=EntityType.person,
        aliases=["Casey", "CE"],
    )
    d = e.to_dict()
    assert d["aliases"] == ["Casey", "CE"]


# --- Migration (entities + entity_mentions tables) ---


async def test_entities_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'entities'
        )
        """
    )
    assert exists is True


async def test_entity_mentions_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'entity_mentions'
        )
        """
    )
    assert exists is True


async def test_entities_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'entities'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "name" in columns
    assert "entity_type" in columns
    assert "aliases" in columns
    assert "description" in columns
    assert "project_id" in columns
    assert "agent_id" in columns
    assert "status" in columns
    assert "mention_count" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "embedding" in columns

    assert columns["name"]["is_nullable"] == "NO"
    assert columns["entity_type"]["is_nullable"] == "NO"
    assert columns["status"]["is_nullable"] == "NO"
    assert columns["description"]["is_nullable"] == "YES"
    assert columns["project_id"]["is_nullable"] == "YES"


async def test_entity_mentions_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'entity_mentions'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "entity_id" in columns
    assert "memory_id" in columns
    assert "mentioned_at" in columns

    assert columns["entity_id"]["is_nullable"] == "NO"
    assert columns["memory_id"]["is_nullable"] == "NO"


async def test_entities_table_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'entities'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "entities_pkey" in index_names
    assert "idx_entities_name" in index_names
    assert "idx_entities_type" in index_names
    assert "idx_entities_project" in index_names
    assert "idx_entities_status" in index_names
    assert "idx_entities_embedding_hnsw" in index_names


async def test_entity_mentions_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'entity_mentions'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "entity_mentions_pkey" in index_names
    assert "idx_entmem_memory" in index_names


async def test_entities_insert_and_read(pool):
    await pool.execute(
        """
        INSERT INTO entities (id, name, entity_type, description, project_id)
        VALUES ($1, $2, $3, $4, $5)
        """,
        "test-ent-1", "PostgreSQL", "tool", "A relational database", "weft",
    )

    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", "test-ent-1")
    assert row is not None
    assert row["name"] == "PostgreSQL"
    assert row["entity_type"] == "tool"
    assert row["status"] == "active"
    assert row["mention_count"] == 0


async def test_entities_defaults(pool):
    await pool.execute(
        "INSERT INTO entities (id, name) VALUES ($1, $2)",
        "test-ent-2", "Minimal",
    )

    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", "test-ent-2")
    assert row["entity_type"] == "concept"
    assert row["status"] == "active"
    assert row["mention_count"] == 0
    assert row["aliases"] == []
    assert row["project_id"] is None


async def test_entities_aliases_array(pool):
    await pool.execute(
        "INSERT INTO entities (id, name, aliases) VALUES ($1, $2, $3)",
        "test-ent-3", "Casey", ["Casey Example", "CE"],
    )

    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", "test-ent-3")
    assert row["aliases"] == ["Casey Example", "CE"]


async def test_entity_mentions_join(pool):
    """Can link a memory to an entity via join table."""
    await pool.execute(
        "INSERT INTO entities (id, name) VALUES ($1, $2)",
        "test-ent-4", "Join test entity",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="A fact about the entity",
    ))

    await pool.execute(
        """
        INSERT INTO entity_mentions (entity_id, memory_id)
        VALUES ($1, $2)
        """,
        "test-ent-4", mem.id,
    )

    rows = await pool.fetch(
        "SELECT * FROM entity_mentions WHERE entity_id = $1",
        "test-ent-4",
    )
    assert len(rows) == 1
    assert rows[0]["memory_id"] == mem.id


async def test_entity_mentions_cascade_on_entity_delete(pool):
    """Deleting an entity cascades to entity_mentions."""
    await pool.execute(
        "INSERT INTO entities (id, name) VALUES ($1, $2)",
        "test-ent-5", "Cascade test",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="linked memory",
    ))
    await pool.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES ($1, $2)",
        "test-ent-5", mem.id,
    )

    await pool.execute("DELETE FROM entities WHERE id = $1", "test-ent-5")

    links = await pool.fetch(
        "SELECT * FROM entity_mentions WHERE entity_id = $1", "test-ent-5",
    )
    assert len(links) == 0
    # Memory itself should still exist
    row = await pool.fetchrow("SELECT * FROM memories WHERE id = $1", mem.id)
    assert row is not None


async def test_entity_mentions_cascade_on_memory_delete(pool):
    """Hard-deleting a memory cascades to entity_mentions."""
    await pool.execute(
        "INSERT INTO entities (id, name) VALUES ($1, $2)",
        "test-ent-6", "Memory cascade",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="will be deleted",
    ))
    await pool.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES ($1, $2)",
        "test-ent-6", mem.id,
    )

    await pool.execute("DELETE FROM memories WHERE id = $1", mem.id)

    links = await pool.fetch(
        "SELECT * FROM entity_mentions WHERE memory_id = $1", mem.id,
    )
    assert len(links) == 0
    # Entity itself should still exist
    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", "test-ent-6")
    assert row is not None


async def test_entity_mentions_idempotent(pool):
    """Inserting the same memory into an entity twice uses ON CONFLICT."""
    await pool.execute(
        "INSERT INTO entities (id, name) VALUES ($1, $2)",
        "test-ent-7", "Idempotent test",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="unique link",
    ))

    for _ in range(2):
        await pool.execute(
            """
            INSERT INTO entity_mentions (entity_id, memory_id)
            VALUES ($1, $2)
            ON CONFLICT (entity_id, memory_id) DO NOTHING
            """,
            "test-ent-7", mem.id,
        )

    rows = await pool.fetch(
        "SELECT * FROM entity_mentions WHERE entity_id = $1", "test-ent-7",
    )
    assert len(rows) == 1
