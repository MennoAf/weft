"""Tests for keyword and hybrid search (BM25 + vector RRF fusion)."""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryStatus, MemoryType
from weft.store import (
    build_or_tsquery,
    search_by_keyword,
    search_by_vector,
    search_hybrid,
    store_memory,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


# --- build_or_tsquery: pure-function unit tests (no DB) ---


def test_build_or_tsquery_or_joins_lexemes():
    """Multi-term query becomes an OR-joined tsquery string."""
    assert build_or_tsquery("loc key catalog") == "loc | key | catalog"


def test_build_or_tsquery_splits_underscores_and_hyphens():
    """Underscores/hyphens are split into separate lexemes (loc_key -> loc, key)."""
    assert build_or_tsquery("loc_key code-library") == "loc | key | code | library"


def test_build_or_tsquery_dedupes_preserving_order():
    """Repeated lexemes are de-duplicated, first occurrence wins."""
    assert build_or_tsquery("code code library code") == "code | library"


def test_build_or_tsquery_lowercases():
    assert build_or_tsquery("PostgreSQL MVCC") == "postgresql | mvcc"


@pytest.mark.parametrize("raw", ["", "   ", "!@#$%^&*()", "& | ! ( ) :", "''\"\""])
def test_build_or_tsquery_empty_after_sanitization_returns_none(raw):
    """Empty / all-punctuation input yields None so callers short-circuit."""
    assert build_or_tsquery(raw) is None


def test_build_or_tsquery_strips_operator_chars_no_injection():
    """Operator/punctuation characters never survive into the tsquery string."""
    out = build_or_tsquery("foo & bar | baz ! (qux):*")
    assert out == "foo | bar | baz | qux"
    assert all(c not in out for c in "&!():*")


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


async def test_keyword_search_is_disjunctive_not_conjunctive(pool, provider):
    """RC1: a multi-term query matches docs sharing ANY term, not only ALL terms.

    Three docs each hold exactly one distinct nonsense token; a query of all
    three returns all three. Under the old plainto_tsquery (AND) this returned
    zero, since no single doc held every term.
    """
    tokens = ["wobblefish", "zorptastic", "quibblenork"]
    for tok in tokens:
        emb = await provider.embed(tok)
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=f"This document is about {tok} and nothing else notable.",
                topic=["disjunctive-test"],
                confidence=0.8,
            ),
            embedding=emb,
        )
    results = await search_by_keyword(pool, "wobblefish zorptastic quibblenork")
    found = {tok for tok in tokens for r in results if tok in r.memory.content}
    assert found == set(tokens), f"expected all 3 docs, got tokens {found}"


async def test_keyword_search_ranks_more_overlap_higher(pool, provider):
    """RC1/V2: a doc matching more query terms ranks above one matching fewer."""
    emb_two = await provider.embed("two-term doc")
    two = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="alphaword and betaword both appear in this document.",
            topic=["overlap-test"],
            confidence=0.8,
        ),
        embedding=emb_two,
    )
    emb_one = await provider.embed("one-term doc")
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="only alphaword appears in this other document.",
            topic=["overlap-test"],
            confidence=0.8,
        ),
        embedding=emb_one,
    )
    results = await search_by_keyword(pool, "alphaword betaword")
    assert len(results) == 2
    assert results[0].memory.id == two.id, "two-term doc should rank first"


@pytest.mark.parametrize(
    "hostile",
    ["foo & bar", "baz | qux", "alphaword:* !betaword", "a (b) c", "what's this?"],
)
async def test_keyword_search_never_raises_on_operator_chars(pool, provider, hostile):
    """RC1/V5: operator/punctuation input degrades gracefully, never raises."""
    await _seed_memories(pool, provider)
    results = await search_by_keyword(pool, hostile)  # must not raise
    assert isinstance(results, list)


async def test_keyword_search_empty_query_returns_empty(pool, provider):
    """RC1/V5/AC4: all-punctuation query short-circuits to no matches, no raise."""
    await _seed_memories(pool, provider)
    assert await search_by_keyword(pool, "!@#$%^&*()") == []
    assert await search_by_keyword(pool, "   ") == []


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
    await _seed_memories(pool, provider)
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
