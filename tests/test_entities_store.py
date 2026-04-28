"""Tests for entities store layer (CRUD + search + mention linking)."""

from __future__ import annotations

import pytest

from weft.entities import (
    get_entity,
    get_entity_memories,
    get_memory_entities,
    link_mention,
    list_entities,
    search_entities,
    store_entity,
    unlink_mention,
)
from weft.models import EntityCreate, EntityType, MemoryCreate, MemoryType
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.store import store_memory


# --- Helpers ---

async def _make_memory(pool, content="test memory"):
    return await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content=content,
    ))


async def _make_entity(pool, name="test entity", **kwargs):
    return await store_entity(pool, EntityCreate(name=name, **kwargs))


# --- store_entity ---


async def test_store_entity_minimal(pool):
    ent = await _make_entity(pool)
    assert ent.id.startswith("weft-")
    assert ent.name == "test entity"
    assert ent.entity_type == EntityType.concept
    assert ent.aliases == []
    assert ent.status == "active"
    assert ent.mention_count == 0
    assert ent.description is None
    assert ent.project_id is None


async def test_store_entity_full(pool):
    ent = await store_entity(pool, EntityCreate(
        name="Jason Bauman",
        entity_type=EntityType.person,
        aliases=["Jason", "JB"],
        description="Builder of Weft",
        project_id="weft",
        agent_id="warp",
    ))
    assert ent.name == "Jason Bauman"
    assert ent.entity_type == EntityType.person
    assert ent.aliases == ["Jason", "JB"]
    assert ent.description == "Builder of Weft"
    assert ent.project_id == "weft"
    assert ent.agent_id == "warp"


async def test_store_entity_with_embedding(pool):
    embedding = [0.1] * 768
    ent = await store_entity(
        pool,
        EntityCreate(name="Embedded entity"),
        embedding=embedding,
    )
    # Verify embedding was stored
    row = await pool.fetchrow(
        "SELECT embedding::text AS emb FROM entities WHERE id = $1", ent.id,
    )
    assert row["emb"] is not None
    assert row["emb"].startswith("[")


# --- get_entity ---


async def test_get_entity_found(pool):
    ent = await _make_entity(pool)
    found = await get_entity(pool, ent.id)
    assert found is not None
    assert found.id == ent.id
    assert found.name == ent.name


async def test_get_entity_not_found(pool):
    assert await get_entity(pool, "nonexistent") is None


# --- list_entities ---


async def test_list_entities_empty(pool):
    assert await list_entities(pool) == []


async def test_list_entities_filters_by_type(pool):
    await _make_entity(pool, "Alice", entity_type=EntityType.person)
    await _make_entity(pool, "PostgreSQL", entity_type=EntityType.tool)

    people = await list_entities(pool, entity_type=EntityType.person)
    assert len(people) == 1
    assert people[0].name == "Alice"

    tools = await list_entities(pool, entity_type=EntityType.tool)
    assert len(tools) == 1
    assert tools[0].name == "PostgreSQL"


async def test_list_entities_or_null_scoping(pool):
    await _make_entity(pool, "global")
    await _make_entity(pool, "proj-specific", project_id="proj-1")
    await _make_entity(pool, "other-proj", project_id="proj-2")

    results = await list_entities(pool, project_id="proj-1")
    names = {e.name for e in results}
    assert "global" in names
    assert "proj-specific" in names
    assert "other-proj" not in names


async def test_list_entities_user_id_none_returns_all(pool):
    """user_id=None should return all entities (no filter)."""
    await _make_entity(pool, "global", user_id=SYSTEM_GLOBAL_USER_ID)
    await _make_entity(pool, "user-a-entity", user_id="user-a")
    await _make_entity(pool, "user-b-entity", user_id="user-b")

    results = await list_entities(pool, user_id=None)
    names = {e.name for e in results}
    assert len(names) == 3
    assert "global" in names
    assert "user-a-entity" in names
    assert "user-b-entity" in names


async def test_list_entities_user_id_scoping(pool):
    """user_id='user-a' should return user-a rows + sentinel rows, exclude others."""
    await _make_entity(pool, "global", user_id=SYSTEM_GLOBAL_USER_ID)
    await _make_entity(pool, "user-a-entity", user_id="user-a")
    await _make_entity(pool, "user-b-entity", user_id="user-b")

    results = await list_entities(pool, user_id="user-a")
    names = {e.name for e in results}
    assert "global" in names
    assert "user-a-entity" in names
    assert "user-b-entity" not in names


async def test_list_entities_user_id_only_null_rows(pool):
    """user_id='user-a' should return sentinel rows even if no user-a rows exist."""
    await _make_entity(pool, "global", user_id=SYSTEM_GLOBAL_USER_ID)
    await _make_entity(pool, "user-b-entity", user_id="user-b")

    results = await list_entities(pool, user_id="user-a")
    names = {e.name for e in results}
    assert "global" in names
    assert "user-a-entity" not in names
    assert "user-b-entity" not in names


async def test_list_entities_ordered_by_mention_count(pool):
    e1 = await _make_entity(pool, "low")
    e2 = await _make_entity(pool, "high")

    # Give e2 more mentions
    m1 = await _make_memory(pool, "m1")
    m2 = await _make_memory(pool, "m2")
    await link_mention(pool, e2.id, m1.id)
    await link_mention(pool, e2.id, m2.id)

    results = await list_entities(pool)
    assert results[0].name == "high"
    assert results[-1].name == "low"


async def test_list_entities_respects_limit(pool):
    for i in range(5):
        await _make_entity(pool, f"ent-{i}")

    results = await list_entities(pool, limit=3)
    assert len(results) == 3


async def test_list_entities_excludes_archived(pool):
    ent = await _make_entity(pool, "archived")
    await pool.execute(
        "UPDATE entities SET status = 'archived' WHERE id = $1", ent.id,
    )

    results = await list_entities(pool)
    assert len(results) == 0


# --- search_entities ---


async def test_search_entities_by_vector(pool):
    embedding = [0.5] * 768
    ent = await store_entity(
        pool,
        EntityCreate(name="Searchable"),
        embedding=embedding,
    )

    # Search with similar vector
    results = await search_entities(pool, embedding, threshold=0.0)
    assert len(results) >= 1
    names = {e.name for e, _ in results}
    assert "Searchable" in names


async def test_search_entities_respects_threshold(pool):
    embedding = [0.5] * 768
    await store_entity(
        pool,
        EntityCreate(name="Far away"),
        embedding=embedding,
    )

    # Search with very different vector and high threshold
    opposite = [-0.5] * 768
    results = await search_entities(pool, opposite, threshold=0.99)
    assert len(results) == 0


async def test_search_entities_filters_by_type(pool):
    embedding = [0.5] * 768
    await store_entity(pool, EntityCreate(name="Alice", entity_type=EntityType.person), embedding=embedding)
    await store_entity(pool, EntityCreate(name="Redis", entity_type=EntityType.tool), embedding=embedding)

    results = await search_entities(
        pool, embedding, entity_type=EntityType.person, threshold=0.0,
    )
    names = {e.name for e, _ in results}
    assert "Alice" in names
    assert "Redis" not in names


async def test_search_entities_project_scoped(pool):
    embedding = [0.5] * 768
    await store_entity(pool, EntityCreate(name="global"), embedding=embedding)
    await store_entity(pool, EntityCreate(name="proj-1", project_id="proj-1"), embedding=embedding)
    await store_entity(pool, EntityCreate(name="proj-2", project_id="proj-2"), embedding=embedding)

    results = await search_entities(
        pool, embedding, project_id="proj-1", threshold=0.0,
    )
    names = {e.name for e, _ in results}
    assert "global" in names
    assert "proj-1" in names
    assert "proj-2" not in names


async def test_search_entities_user_id_none_returns_all(pool):
    """user_id=None should return all entities (no filter)."""
    embedding = [0.5] * 768
    await store_entity(pool, EntityCreate(name="global", user_id=SYSTEM_GLOBAL_USER_ID), embedding=embedding)
    await store_entity(pool, EntityCreate(name="user-a-entity", user_id="user-a"), embedding=embedding)
    await store_entity(pool, EntityCreate(name="user-b-entity", user_id="user-b"), embedding=embedding)

    results = await search_entities(pool, embedding, user_id=None, threshold=0.0)
    names = {e.name for e, _ in results}
    assert len(names) == 3
    assert "global" in names
    assert "user-a-entity" in names
    assert "user-b-entity" in names


async def test_search_entities_user_id_scoping(pool):
    """user_id='user-a' should return user-a rows + sentinel rows, exclude others."""
    embedding = [0.5] * 768
    await store_entity(pool, EntityCreate(name="global", user_id=SYSTEM_GLOBAL_USER_ID), embedding=embedding)
    await store_entity(pool, EntityCreate(name="user-a-entity", user_id="user-a"), embedding=embedding)
    await store_entity(pool, EntityCreate(name="user-b-entity", user_id="user-b"), embedding=embedding)

    results = await search_entities(pool, embedding, user_id="user-a", threshold=0.0)
    names = {e.name for e, _ in results}
    assert "global" in names
    assert "user-a-entity" in names
    assert "user-b-entity" not in names


# --- link_mention ---


async def test_link_mention(pool):
    ent = await _make_entity(pool)
    mem = await _make_memory(pool)

    created = await link_mention(pool, ent.id, mem.id)
    assert created is True

    # Check mention_count incremented
    found = await get_entity(pool, ent.id)
    assert found.mention_count == 1


async def test_link_mention_idempotent(pool):
    ent = await _make_entity(pool)
    mem = await _make_memory(pool)

    assert await link_mention(pool, ent.id, mem.id) is True
    assert await link_mention(pool, ent.id, mem.id) is False

    # mention_count should still be 1
    found = await get_entity(pool, ent.id)
    assert found.mention_count == 1


async def test_link_mention_multiple(pool):
    ent = await _make_entity(pool)
    m1 = await _make_memory(pool, "first")
    m2 = await _make_memory(pool, "second")
    m3 = await _make_memory(pool, "third")

    await link_mention(pool, ent.id, m1.id)
    await link_mention(pool, ent.id, m2.id)
    await link_mention(pool, ent.id, m3.id)

    found = await get_entity(pool, ent.id)
    assert found.mention_count == 3


# --- unlink_mention ---


async def test_unlink_mention(pool):
    ent = await _make_entity(pool)
    mem = await _make_memory(pool)
    await link_mention(pool, ent.id, mem.id)

    removed = await unlink_mention(pool, ent.id, mem.id)
    assert removed is True

    found = await get_entity(pool, ent.id)
    assert found.mention_count == 0


async def test_unlink_mention_not_linked(pool):
    ent = await _make_entity(pool)
    assert await unlink_mention(pool, ent.id, "nonexistent") is False


# --- get_entity_memories ---


async def test_get_entity_memories(pool):
    ent = await _make_entity(pool)
    m1 = await _make_memory(pool, "first")
    m2 = await _make_memory(pool, "second")

    await link_mention(pool, ent.id, m1.id)
    await link_mention(pool, ent.id, m2.id)

    memories = await get_entity_memories(pool, ent.id)
    assert len(memories) == 2


async def test_get_entity_memories_excludes_archived(pool):
    ent = await _make_entity(pool)
    mem = await _make_memory(pool)
    await link_mention(pool, ent.id, mem.id)

    await pool.execute(
        "UPDATE memories SET status = 'archived' WHERE id = $1", mem.id,
    )

    memories = await get_entity_memories(pool, ent.id)
    assert len(memories) == 0


async def test_get_entity_memories_respects_limit(pool):
    ent = await _make_entity(pool)
    for i in range(10):
        mem = await _make_memory(pool, f"mem-{i}")
        await link_mention(pool, ent.id, mem.id)

    memories = await get_entity_memories(pool, ent.id, limit=5)
    assert len(memories) == 5


# --- get_memory_entities ---


async def test_get_memory_entities(pool):
    e1 = await _make_entity(pool, "Alice")
    e2 = await _make_entity(pool, "Bob")
    mem = await _make_memory(pool)

    await link_mention(pool, e1.id, mem.id)
    await link_mention(pool, e2.id, mem.id)

    entities = await get_memory_entities(pool, mem.id)
    assert len(entities) == 2
    names = {e.name for e in entities}
    assert "Alice" in names
    assert "Bob" in names


async def test_get_memory_entities_none(pool):
    mem = await _make_memory(pool)
    assert await get_memory_entities(pool, mem.id) == []


async def test_get_memory_entities_excludes_archived(pool):
    ent = await _make_entity(pool)
    mem = await _make_memory(pool)
    await link_mention(pool, ent.id, mem.id)

    await pool.execute(
        "UPDATE entities SET status = 'archived' WHERE id = $1", ent.id,
    )

    entities = await get_memory_entities(pool, mem.id)
    assert len(entities) == 0
