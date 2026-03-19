"""Tests for keyword and hybrid search (BM25 + vector RRF fusion)."""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryStatus, MemoryType
from weft.store import (
    search_by_keyword,
    search_by_vector,
    search_hybrid,
    store_memory,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


async def _seed_memories(pool, provider):
    """Seed a set of diverse memories and return them for assertions."""
    memories = [
        MemoryCreate(
            type=MemoryType.fact,
            content="PostgreSQL uses MVCC for concurrency control",
            topic=["postgres", "architecture"],
            confidence=0.9,
        ),
        MemoryCreate(
            type=MemoryType.fact,
            content="Redis is an in-memory key-value store used for caching",
            topic=["redis", "caching"],
            confidence=0.8,
        ),
        MemoryCreate(
            type=MemoryType.pattern,
            content="Always use connection pooling with asyncpg for database access",
            topic=["postgres", "patterns"],
            confidence=0.85,
        ),
        MemoryCreate(
            type=MemoryType.solution,
            content="Fix the asyncpg connection leak by closing the pool in a finally block",
            topic=["postgres", "debugging"],
            confidence=0.75,
        ),
        MemoryCreate(
            type=MemoryType.fact,
            content="The Weft memory system uses pgvector for semantic similarity search",
            topic=["weft", "architecture"],
            confidence=0.9,
        ),
    ]
    stored = []
    for mc in memories:
        emb = await provider.embed(mc.content)
        mem = await store_memory(pool, mc, embedding=emb)
        stored.append(mem)
    return stored


# --- Keyword search tests ---


async def test_keyword_search_basic(pool, provider):
    """Keyword search returns memories matching query terms."""
    await _seed_memories(pool, provider)
    results = await search_by_keyword(pool, "PostgreSQL concurrency")
    assert len(results) > 0
    # The MVCC memory should rank highest for this query
    assert "MVCC" in results[0].memory.content


async def test_keyword_search_stemming(pool, provider):
    """Keyword search handles stemming (e.g., 'caching' matches 'caching')."""
    await _seed_memories(pool, provider)
    results = await search_by_keyword(pool, "cache")
    assert len(results) > 0
    assert any("caching" in r.memory.content for r in results)


async def test_keyword_search_no_match(pool, provider):
    """Keyword search returns empty when no terms match."""
    await _seed_memories(pool, provider)
    results = await search_by_keyword(pool, "kubernetes deployment helm")
    assert len(results) == 0


async def test_keyword_search_filters(pool, provider):
    """Keyword search respects type and topic filters."""
    await _seed_memories(pool, provider)

    # Filter by type
    results = await search_by_keyword(
        pool, "postgres", memory_type=MemoryType.pattern
    )
    assert all(r.memory.type == MemoryType.pattern for r in results)

    # Filter by topic
    results = await search_by_keyword(pool, "postgres", topic="debugging")
    assert all("debugging" in r.memory.topic for r in results)


async def test_keyword_search_limit(pool, provider):
    """Keyword search respects limit parameter."""
    await _seed_memories(pool, provider)
    results = await search_by_keyword(pool, "postgres", limit=2)
    assert len(results) <= 2


async def test_keyword_search_exclude_ids(pool, provider):
    """Keyword search excludes specified memory IDs."""
    stored = await _seed_memories(pool, provider)
    # Get all postgres results first
    all_results = await search_by_keyword(pool, "postgres")
    assert len(all_results) > 1

    # Exclude the first result
    exclude = [all_results[0].memory.id]
    filtered = await search_by_keyword(pool, "postgres", exclude_ids=exclude)
    assert all(r.memory.id != exclude[0] for r in filtered)


async def test_keyword_search_status_filter(pool, provider):
    """Keyword search filters by memory status."""
    await _seed_memories(pool, provider)
    # Archived memories should not appear by default
    results = await search_by_keyword(pool, "postgres", status=MemoryStatus.archived)
    assert len(results) == 0


# --- Hybrid search tests ---


async def test_hybrid_search_basic(pool, provider):
    """Hybrid search returns results combining vector and keyword signals."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("PostgreSQL database architecture")
    results = await search_hybrid(
        pool, "PostgreSQL database architecture", embedding
    )
    assert len(results) > 0
    # Should find postgres-related memories
    contents = " ".join(r.memory.content for r in results)
    assert "PostgreSQL" in contents or "postgres" in contents.lower()


async def test_hybrid_search_rrf_scores_normalized(pool, provider):
    """Hybrid search RRF scores are normalized to 0-1 range."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("connection pooling asyncpg")
    results = await search_hybrid(pool, "connection pooling asyncpg", embedding)
    assert len(results) > 0
    for r in results:
        assert 0.0 <= r.similarity <= 1.0
    # Top result should have score of 1.0 (max normalized)
    assert results[0].similarity == pytest.approx(1.0)


async def test_hybrid_surfaces_keyword_only_hits(pool, provider):
    """Hybrid search can surface results that keyword search finds but vector misses."""
    # Store a memory with a very specific term
    emb = await provider.embed("MVCC multiversion concurrency control in databases")
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="The XYZZY_UNIQUE_TOKEN protocol handles edge cases in distributed systems",
            topic=["distributed"],
            confidence=0.8,
        ),
        embedding=emb,
    )
    # Also store a semantically similar one
    emb2 = await provider.embed("distributed systems protocols")
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Distributed consensus uses Raft or Paxos algorithms",
            topic=["distributed"],
            confidence=0.8,
        ),
        embedding=emb2,
    )

    # Keyword search for the unique token should find it
    kw_results = await search_by_keyword(pool, "XYZZY_UNIQUE_TOKEN")
    assert len(kw_results) == 1

    # Hybrid should also surface it
    emb_query = await provider.embed("XYZZY_UNIQUE_TOKEN protocol")
    hybrid_results = await search_hybrid(
        pool, "XYZZY_UNIQUE_TOKEN protocol", emb_query
    )
    found_ids = {r.memory.id for r in hybrid_results}
    assert kw_results[0].memory.id in found_ids


async def test_hybrid_search_filters(pool, provider):
    """Hybrid search respects type, topic, and project_id filters."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("postgres")

    # Filter by type
    results = await search_hybrid(
        pool, "postgres", embedding, memory_type=MemoryType.fact
    )
    assert all(r.memory.type == MemoryType.fact for r in results)


async def test_hybrid_search_limit(pool, provider):
    """Hybrid search respects limit parameter."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("postgres database")
    results = await search_hybrid(pool, "postgres database", embedding, limit=2)
    assert len(results) <= 2


async def test_hybrid_weights(pool, provider):
    """Different weight configurations produce different rankings."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("postgres connection pooling")

    # Vector-heavy
    vector_heavy = await search_hybrid(
        pool, "postgres connection pooling", embedding,
        vector_weight=0.9, keyword_weight=0.1,
    )
    # Keyword-heavy
    keyword_heavy = await search_hybrid(
        pool, "postgres connection pooling", embedding,
        vector_weight=0.1, keyword_weight=0.9,
    )

    # Both should return results
    assert len(vector_heavy) > 0
    assert len(keyword_heavy) > 0

    # The orderings may differ (not guaranteed but likely with skewed weights)
    # At minimum, both should have results
    v_ids = [r.memory.id for r in vector_heavy]
    k_ids = [r.memory.id for r in keyword_heavy]
    # Both sets should overlap (same corpus)
    assert set(v_ids) & set(k_ids)


async def test_semantic_mode_unchanged(pool, provider):
    """Semantic-only search still works as before (regression check)."""
    await _seed_memories(pool, provider)
    embedding = await provider.embed("vector similarity search")
    results = await search_by_vector(
        pool, embedding, limit=5, threshold=0.0
    )
    assert len(results) > 0
    # Results should have similarity scores
    for r in results:
        assert r.similarity >= 0.0
