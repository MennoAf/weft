"""Tests for proactive contradiction detection on memory store."""

from __future__ import annotations

import pytest

from weft.consolidation import check_contradictions_on_store
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryType, RelationType
from weft.store import get_relationships, store_memory


@pytest.fixture
def provider():
    """Provide a FastEmbed embedding provider for tests."""
    return get_provider("fastembed")


async def test_store_detects_contradiction(pool, provider):
    """Storing a negated version of an existing memory should produce contradiction warnings."""
    content_a = "pgvector supports HNSW indexing"
    emb_a = await provider.embed(content_a)
    mem_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["pgvector"],
            source=MemorySource.documentation,
            confidence=0.9,
        ),
        embedding=emb_a,
    )

    content_b = "pgvector does not support HNSW indexing"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            topic=["pgvector"],
            source=MemorySource.conversation,
            confidence=0.7,
        ),
        embedding=emb_b,
    )

    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)

    assert len(warnings) >= 1
    # check_contradictions_on_store returns dicts with "memory_id" key
    warning_ids = [w["memory_id"] for w in warnings]
    assert mem_a.id in warning_ids
    # Verify warning structure
    w = next(w for w in warnings if w["memory_id"] == mem_a.id)
    assert "similarity" in w
    assert "content_preview" in w
    assert w["similarity"] > 0.5


async def test_store_no_contradiction_for_unrelated(pool, provider):
    """Storing unrelated memories should not produce contradiction warnings."""
    content_a = "Python is a popular programming language"
    emb_a = await provider.embed(content_a)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["python"],
        ),
        embedding=emb_a,
    )

    content_b = "Redis stores data in memory for fast access"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            topic=["redis"],
        ),
        embedding=emb_b,
    )

    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert warnings == []


async def test_contradiction_creates_relationship(pool, provider):
    """After contradiction detection, a 'contradicts' relationship should exist."""
    content_a = "pgvector supports HNSW indexing"
    emb_a = await provider.embed(content_a)
    mem_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["pgvector"],
        ),
        embedding=emb_a,
    )

    content_b = "pgvector does not support HNSW indexing"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            topic=["pgvector"],
        ),
        embedding=emb_b,
    )

    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert len(warnings) >= 1

    # Verify relationship was created
    rels = await get_relationships(pool, mem_b.id, relation=RelationType.contradicts)
    assert len(rels) >= 1
    rel_pairs = [(r.source_id, r.target_id) for r in rels]
    assert (mem_b.id, mem_a.id) in rel_pairs


async def test_check_contradictions_no_similar(pool, provider):
    """When no similar memories exist, no contradictions should be found."""
    # Store a memory that contradicts nothing
    content = "FastEmbed uses ONNX for local embeddings"
    emb = await provider.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content,
            topic=["fastembed"],
        ),
        embedding=emb,
    )

    # With no prior memories in the same semantic space, no contradictions
    warnings = await check_contradictions_on_store(pool, mem.id, emb)
    assert warnings == []

    # Also verify no contradicts relationships were created
    rels = await get_relationships(pool, mem.id, relation=RelationType.contradicts)
    assert len(rels) == 0
