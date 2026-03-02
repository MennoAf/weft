"""Tests for enhanced filtering: type, status, and combined filters."""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.store import search_by_vector, store_memory, update_memory


@pytest.fixture
async def mixed_memories(pool):
    """Seed pool with memories of different types and statuses."""
    provider = get_provider("fastembed")
    memories = []

    seed = [
        ("Always use fastembed for development", MemoryType.preference, "active"),
        ("store.py is the ONLY Postgres writer", MemoryType.architecture, "active"),
        ("pgvector supports cosine distance", MemoryType.fact, "active"),
        ("Testcontainers are great for testing", MemoryType.pattern, "active"),
        ("Old outdated architecture note", MemoryType.architecture, "archived"),
        ("Decayed fact about version 1.0", MemoryType.fact, "decayed"),
    ]

    for content, mem_type, status in seed:
        create = MemoryCreate(
            type=mem_type,
            content=content,
            source=MemorySource.code,
            confidence=0.9,
        )
        emb = await provider.embed(content)
        mem = await store_memory(pool, create, embedding=emb)
        if status != "active":
            await update_memory(pool, mem.id, status=MemoryStatus(status))
            mem.status = MemoryStatus(status)
        memories.append(mem)

    return pool, memories, provider


async def test_filter_by_type(mixed_memories):
    """Filter by memory_type should only return matching types."""
    pool, _, provider = mixed_memories
    emb = await provider.embed("architecture and design")

    results = await search_by_vector(
        pool, emb, limit=10, memory_type=MemoryType.architecture,
    )
    assert len(results) >= 1
    assert all(r.memory.type == MemoryType.architecture for r in results)


async def test_filter_by_status_active(mixed_memories):
    """Default status=active should exclude archived/decayed."""
    pool, _, provider = mixed_memories
    emb = await provider.embed("all memories")

    results = await search_by_vector(pool, emb, limit=10)
    assert all(r.memory.status == MemoryStatus.active for r in results)


async def test_filter_by_status_archived(mixed_memories):
    """Explicitly filtering by archived should return archived memories."""
    pool, _, provider = mixed_memories
    emb = await provider.embed("architecture")

    results = await search_by_vector(
        pool, emb, limit=10, status=MemoryStatus.archived,
    )
    assert len(results) >= 1
    assert all(r.memory.status == MemoryStatus.archived for r in results)


async def test_filter_by_status_none_returns_all(mixed_memories):
    """status=None should return memories of any status."""
    pool, memories, provider = mixed_memories
    emb = await provider.embed("all memories")

    results = await search_by_vector(pool, emb, limit=10, status=None, threshold=-1.0)
    assert len(results) == len(memories)


async def test_combined_type_and_status(mixed_memories):
    """Combining type + status should intersect filters."""
    pool, _, provider = mixed_memories
    emb = await provider.embed("fact")

    # Active facts only
    results = await search_by_vector(
        pool, emb, limit=10,
        memory_type=MemoryType.fact,
        status=MemoryStatus.active,
    )
    assert all(
        r.memory.type == MemoryType.fact and r.memory.status == MemoryStatus.active
        for r in results
    )

    # Decayed facts only
    results_decayed = await search_by_vector(
        pool, emb, limit=10,
        memory_type=MemoryType.fact,
        status=MemoryStatus.decayed,
    )
    assert all(r.memory.status == MemoryStatus.decayed for r in results_decayed)


async def test_type_filter_with_topic(mixed_memories):
    """Type filter should compose with existing topic filter."""
    pool, _, provider = mixed_memories

    # Store a memory with a specific topic + type
    create = MemoryCreate(
        type=MemoryType.solution,
        content="Fix Ryuk reaper in conftest",
        topic=["testing"],
        source=MemorySource.code,
    )
    emb = await provider.embed(create.content)
    await store_memory(pool, create, embedding=emb)

    # Search with both type and topic filter
    query_emb = await provider.embed("testing solutions")
    results = await search_by_vector(
        pool, query_emb, limit=10,
        memory_type=MemoryType.solution,
        topic="testing",
    )
    assert len(results) >= 1
    assert all(r.memory.type == MemoryType.solution for r in results)
    assert all("testing" in r.memory.topic for r in results)


async def test_no_results_for_nonexistent_type(mixed_memories):
    """Filtering by a type with no matching memories should return empty."""
    pool, _, provider = mixed_memories
    emb = await provider.embed("relationships")

    results = await search_by_vector(
        pool, emb, limit=10,
        memory_type=MemoryType.relationship,
    )
    assert len(results) == 0
