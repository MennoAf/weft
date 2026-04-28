"""End-to-end validation: 10 memories, semantic search, relevance verification.

This is the Phase 1 capstone test — validates the complete pipeline:
text → embedding → store → vector search → ranked results.
"""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType, MemorySource
from weft.store import get_stats, search_by_vector, store_memory, list_memories

# 10 diverse memories spanning different types and topics
SEED_MEMORIES = [
    {
        "type": MemoryType.preference,
        "content": "Always use fastembed as the default embedding provider for local development",
        "topic": ["embeddings", "development"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
    {
        "type": MemoryType.architecture,
        "content": "store.py is the ONLY module that writes to Postgres — all other modules go through it",
        "topic": ["weft", "architecture", "postgres"],
        "confidence": 0.95,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.fact,
        "content": "pgvector extension version 0.8.1 supports cosine distance operator <=>",
        "topic": ["postgres", "pgvector"],
        "confidence": 0.9,
        "source": MemorySource.documentation,
    },
    {
        "type": MemoryType.pattern,
        "content": "Using testcontainers with session-scoped fixtures gives the best balance of isolation and speed",
        "topic": ["testing", "patterns"],
        "confidence": 0.8,
        "source": MemorySource.inference,
    },
    {
        "type": MemoryType.solution,
        "content": "The Ryuk reaper container from testcontainers can become stale — clean it up in conftest.py",
        "topic": ["testing", "docker"],
        "confidence": 0.9,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.relationship,
        "content": "Jason Bauman owns Weft, Loom, and Muttr projects",
        "topic": ["people", "projects"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
    {
        "type": MemoryType.architecture,
        "content": "Weft MCP server uses FastMCP with stdio transport for agent communication",
        "topic": ["weft", "mcp", "architecture"],
        "confidence": 0.95,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.fact,
        "content": "Redis cache uses 1 hour TTL for memories and 24 hour TTL for embeddings",
        "topic": ["redis", "caching"],
        "confidence": 0.9,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.pattern,
        "content": "Three-layer config (global YAML, project YAML, env vars) provides good flexibility without complexity",
        "topic": ["configuration", "patterns"],
        "confidence": 0.85,
        "source": MemorySource.inference,
    },
    {
        "type": MemoryType.preference,
        "content": "Use dedicated infrastructure for each project — do not share Postgres between Loom and Weft",
        "topic": ["infrastructure", "docker"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
]


@pytest.fixture
async def seeded_pool(pool):
    """Pool with 10 memories pre-seeded with embeddings."""
    provider = get_provider("fastembed")
    for mem_data in SEED_MEMORIES:
        create = MemoryCreate(**mem_data)
        embedding = await provider.embed(create.content)
        await store_memory(pool, create, embedding=embedding)
    return pool


async def test_ten_memories_stored(seeded_pool):
    """Verify all 10 memories were stored."""
    memories = await list_memories(seeded_pool, limit=20)
    assert len(memories) == 10


async def test_stats_reflect_seed(seeded_pool):
    """Stats should reflect the 10 seeded memories."""
    stats = await get_stats(seeded_pool)
    assert stats["total"] == 10
    assert stats["by_status"]["active"] == 10
    # We have 2 preferences, 2 architectures, 2 facts, 2 patterns, 1 solution, 1 relationship
    assert stats["by_type"]["preference"] == 2
    assert stats["by_type"]["architecture"] == 2


async def test_recall_database_topics(seeded_pool):
    """Query about databases should surface pgvector and postgres memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("database and postgres configuration")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 2
    top_contents = [r.memory.content for r in results[:3]]
    # pgvector or postgres-related memories should rank high
    assert any("postgres" in c.lower() or "pgvector" in c.lower() for c in top_contents)


async def test_recall_testing_topics(seeded_pool):
    """Query about testing should surface testing-related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("how to set up tests with containers")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 2
    top_contents = [r.memory.content for r in results[:3]]
    assert any("testcontainer" in c.lower() or "testing" in c.lower() for c in top_contents)


async def test_recall_infrastructure_topics(seeded_pool):
    """Query about infrastructure should surface infra-related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("docker infrastructure setup for projects")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 1
    top_contents = [r.memory.content for r in results[:3]]
    assert any("infrastructure" in c.lower() or "docker" in c.lower() for c in top_contents)


async def test_recall_people_and_ownership(seeded_pool):
    """Query about project ownership should surface the relationship memory."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("who owns this project")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 1
    top_contents = [r.memory.content for r in results[:3]]
    assert any("jason" in c.lower() or "owns" in c.lower() for c in top_contents)


async def test_similarity_scores_reasonable(seeded_pool):
    """Similarity scores should be between 0 and 1 and decrease with rank."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("embedding provider configuration")
    results = await search_by_vector(seeded_pool, query_emb, limit=10)

    assert len(results) == 10
    similarities = [r.similarity for r in results]

    # All between 0 and 1
    assert all(0 <= s <= 1 for s in similarities)

    # Sorted descending (highest similarity first)
    assert similarities == sorted(similarities, reverse=True)

    # Top result should have reasonable similarity
    assert similarities[0] > 0.5


async def test_threshold_filters_irrelevant(seeded_pool):
    """A high threshold should filter out weakly related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("quantum physics and black holes")
    results = await search_by_vector(seeded_pool, query_emb, limit=10, threshold=0.7)

    # Nothing in our seed data is about quantum physics — should get very few or none
    assert len(results) <= 2


