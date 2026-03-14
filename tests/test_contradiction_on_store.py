"""Tests for proactive contradiction detection on memory store."""

from __future__ import annotations

import pytest

from weft.consolidation import check_contradictions_on_store
from weft.embeddings import get_provider
from weft.models import (
    ContradictionWarning,
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from weft.store import delete_memory, get_relationships, store_memory


@pytest.fixture
def provider():
    """Provide a FastEmbed embedding provider for tests."""
    return get_provider("fastembed")


async def test_store_detects_contradiction(pool, provider):
    """Storing a negated version of an existing memory should produce contradiction warnings."""
    content_a = "pgvector supports HNSW indexing for vector similarity search queries"
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

    content_b = "pgvector does not support HNSW indexing for vector similarity search queries"
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
    # check_contradictions_on_store returns dicts with ContradictionWarning fields
    warning_ids = [w["memory_id"] for w in warnings]
    assert mem_a.id in warning_ids
    # Verify warning structure
    w = next(w for w in warnings if w["memory_id"] == mem_a.id)
    assert "similarity" in w
    assert "content_preview" in w
    assert "type" in w
    assert w["type"] == "contradiction"
    assert w["similarity"] > 0.5


async def test_store_no_contradiction_for_unrelated(pool, provider):
    """Storing unrelated memories should not produce contradiction warnings."""
    content_a = "Python is a popular programming language used in data science and web development"
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

    content_b = "Redis stores data in memory for fast access and supports pub/sub messaging patterns"
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
    content_a = "pgvector supports HNSW indexing for vector similarity search queries"
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

    content_b = "pgvector does not support HNSW indexing for vector similarity search queries"
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
    content = "FastEmbed uses ONNX for local embeddings without requiring a remote API call"
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


async def test_different_types_no_contradiction(pool, provider):
    """Contradictory content with different memory types should not flag."""
    content_a = "pgvector supports HNSW indexing for vector similarity search queries"
    emb_a = await provider.embed(content_a)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["pgvector"],
        ),
        embedding=emb_a,
    )

    content_b = "pgvector does not support HNSW indexing for vector similarity search queries"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,  # different type
            content=content_b,
            topic=["pgvector"],
        ),
        embedding=emb_b,
    )

    # With type filter, should not match across types
    warnings = await check_contradictions_on_store(
        pool, mem_b.id, emb_b, memory_type=MemoryType.decision,
    )
    assert warnings == []


async def test_short_content_skips_check(pool, provider):
    """Content shorter than 50 chars should skip contradiction check entirely."""
    content_a = "Use PostgreSQL for this project's database backend and migrations"
    emb_a = await provider.embed(content_a)
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.decision, content=content_a),
        embedding=emb_a,
    )

    # Short content — should be skipped
    short = "Don't use PostgreSQL"
    emb_b = await provider.embed(short)
    mem_b = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.decision, content=short),
        embedding=emb_b,
    )

    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert warnings == []


async def test_archived_memory_not_flagged(pool, provider):
    """Archived memories should not trigger contradiction warnings."""
    content_a = "pgvector supports HNSW indexing for vector similarity search queries"
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

    # Archive the first memory
    await delete_memory(pool, mem_a.id, hard=False)

    content_b = "pgvector does not support HNSW indexing for vector similarity search queries"
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

    # Archived memory should not appear in contradiction search
    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert warnings == []


async def test_project_scoped_contradiction(pool, provider):
    """Contradiction check should scope to the same project when project_id is provided."""
    content_a = "Always use PostgreSQL for this project's database backend and data storage"
    emb_a = await provider.embed(content_a)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content=content_a,
            project_id="project-alpha",
        ),
        embedding=emb_a,
    )

    content_b = "Never use PostgreSQL for this project's database backend and data storage"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content=content_b,
            project_id="project-beta",  # different project
        ),
        embedding=emb_b,
    )

    # Scoped to project-beta — should not find project-alpha memories
    warnings = await check_contradictions_on_store(
        pool, mem_b.id, emb_b, project_id="project-beta",
    )
    assert warnings == []


async def test_complementary_not_contradictory(pool, provider):
    """Similar but complementary content should not be flagged as contradictory."""
    content_a = "Use Redis for caching frequently accessed data in the application layer"
    emb_a = await provider.embed(content_a)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content=content_a,
            topic=["redis"],
        ),
        embedding=emb_a,
    )

    content_b = "Redis cache TTL should be set to 300 seconds for optimal performance"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content=content_b,
            topic=["redis"],
        ),
        embedding=emb_b,
    )

    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert warnings == []


async def test_contradiction_warning_model():
    """ContradictionWarning model serializes correctly."""
    w = ContradictionWarning(
        memory_id="weft-abc123",
        content_preview="Some contradictory content...",
        similarity=0.85,
    )
    d = w.to_dict()
    assert d["type"] == "contradiction"
    assert d["memory_id"] == "weft-abc123"
    assert d["similarity"] == 0.85

    text = w.to_text()
    assert "weft-abc123" in text
    assert "Some contradictory content..." in text


async def test_first_write_no_warnings(pool, provider):
    """The very first memory of a type should produce no warnings."""
    content = "This project uses PostgreSQL 16 with pgvector extension for embeddings"
    emb = await provider.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content,
        ),
        embedding=emb,
    )

    warnings = await check_contradictions_on_store(
        pool, mem.id, emb, memory_type=MemoryType.fact,
    )
    assert warnings == []
