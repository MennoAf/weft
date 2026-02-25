"""Tests for weft.store — Postgres CRUD and vector search."""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemoryStatus, MemoryType, MemorySource, RelationType
from weft.store import (
    add_relationship,
    delete_memory,
    get_memory,
    get_relationships,
    get_stats,
    list_memories,
    remove_relationship,
    search_by_vector,
    store_memory,
    touch_memory,
    update_memory,
)


async def test_store_and_get(pool):
    """Store a memory and retrieve it by ID."""
    create = MemoryCreate(
        type=MemoryType.fact,
        content="Weft uses pgvector for semantic search",
        topic=["weft", "architecture"],
        confidence=0.9,
    )
    mem = await store_memory(pool, create)
    assert mem.id.startswith("weft-")
    assert mem.type == MemoryType.fact
    assert mem.confidence == 0.9

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.content == create.content
    assert fetched.topic == ["weft", "architecture"]


async def test_get_nonexistent(pool):
    """Getting a non-existent memory returns None."""
    result = await get_memory(pool, "weft-00000000")
    assert result is None


async def test_list_memories(pool):
    """List memories with filters."""
    for i in range(5):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact if i < 3 else MemoryType.pattern,
            content=f"memory {i}",
            topic=["test"],
        ))

    all_mems = await list_memories(pool)
    assert len(all_mems) == 5

    facts = await list_memories(pool, memory_type=MemoryType.fact)
    assert len(facts) == 3

    patterns = await list_memories(pool, memory_type=MemoryType.pattern)
    assert len(patterns) == 2


async def test_list_by_topic(pool):
    """Filter memories by topic."""
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="a", topic=["alpha"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="b", topic=["beta"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="c", topic=["alpha", "beta"]))

    alpha = await list_memories(pool, topic="alpha")
    assert len(alpha) == 2

    beta = await list_memories(pool, topic="beta")
    assert len(beta) == 2


async def test_update_memory(pool):
    """Update mutable fields."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="original",
        confidence=0.5,
    ))

    updated = await update_memory(pool, mem.id, content="revised", confidence=0.9)
    assert updated is not None
    assert updated.content == "revised"
    assert updated.confidence == pytest.approx(0.9, abs=1e-6)
    assert updated.updated_at > mem.updated_at


async def test_soft_delete(pool):
    """Soft-delete archives the memory."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="to delete"))
    assert await delete_memory(pool, mem.id)

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.status == MemoryStatus.archived

    # Should not appear in active list
    active = await list_memories(pool, status=MemoryStatus.active)
    assert all(m.id != mem.id for m in active)


async def test_hard_delete(pool):
    """Hard-delete removes the memory entirely."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="to nuke"))
    assert await delete_memory(pool, mem.id, hard=True)
    assert await get_memory(pool, mem.id) is None


async def test_touch_memory(pool):
    """Touch updates accessed_at and increments access_count."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="touch me"))
    assert mem.access_count == 0

    await touch_memory(pool, mem.id)
    fetched = await get_memory(pool, mem.id)
    assert fetched.access_count == 1
    assert fetched.accessed_at >= mem.accessed_at


async def test_vector_search(pool):
    """Store memories with embeddings and search by similarity."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    texts = [
        "Python is a programming language",
        "PostgreSQL is a relational database",
        "Redis is an in-memory data store",
    ]

    for text in texts:
        emb = await provider.embed(text)
        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=text, topic=["tech"]),
            embedding=emb,
        )

    # Search for something related to databases
    query_emb = await provider.embed("database management system")
    results = await search_by_vector(pool, query_emb, limit=3)

    assert len(results) > 0
    # PostgreSQL should be most similar to "database management system"
    assert "PostgreSQL" in results[0].memory.content
    assert results[0].similarity > 0.5


async def test_relationships(pool):
    """Create and query relationships between memories."""
    m1 = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="old fact"))
    m2 = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="new fact"))

    rel = await add_relationship(pool, m2.id, m1.id, RelationType.supersedes)
    assert rel.source_id == m2.id
    assert rel.relation == RelationType.supersedes

    # Query relationships
    rels = await get_relationships(pool, m2.id)
    assert len(rels) == 1

    rels_typed = await get_relationships(pool, m2.id, relation=RelationType.supersedes)
    assert len(rels_typed) == 1

    # Remove
    assert await remove_relationship(pool, m2.id, m1.id, RelationType.supersedes)
    assert len(await get_relationships(pool, m2.id)) == 0


async def test_stats(pool):
    """Get memory statistics."""
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="a", topic=["x"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.pattern, content="b", topic=["x", "y"]))

    stats = await get_stats(pool)
    assert stats["total"] == 2
    assert stats["by_type"]["fact"] == 1
    assert stats["by_type"]["pattern"] == 1
    assert "x" in stats["top_topics"]
