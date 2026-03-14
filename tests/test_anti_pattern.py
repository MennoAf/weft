"""Tests for anti_pattern memory type and prominent surfacing."""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.primer import build_primer
from weft.store import search_by_vector, store_memory


@pytest.fixture
def provider():
    return get_provider("fastembed")


def test_memory_type_has_anti_pattern():
    """MemoryType enum should have an anti_pattern member."""
    assert hasattr(MemoryType, "anti_pattern")
    assert MemoryType.anti_pattern.value == "anti_pattern"
    assert MemoryType("anti_pattern") == MemoryType.anti_pattern


async def test_store_anti_pattern_roundtrip(pool, provider):
    """anti_pattern memories can be stored and retrieved."""
    content = "CONTEXT: async database connections. PROBLEM: using sync psycopg2 in async code causes thread pool exhaustion. AVOID: importing psycopg2 directly."
    emb = await provider.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content=content,
            topic=["database", "async"],
            source=MemorySource.conversation,
            confidence=0.9,
        ),
        embedding=emb,
    )

    assert mem.type == MemoryType.anti_pattern
    assert mem.content == content

    # Verify it can be found via vector search
    results = await search_by_vector(
        pool, emb, limit=5, memory_type=MemoryType.anti_pattern,
    )
    assert any(r.memory.id == mem.id for r in results)


async def test_type_filter_excludes_anti_pattern(pool, provider):
    """Searching with type=decision should NOT return anti_pattern memories."""
    content = "CONTEXT: caching. PROBLEM: caching user sessions in Redis without TTL leads to memory leaks. AVOID: storing sessions without expiry."
    emb = await provider.embed(content)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content=content,
            topic=["caching"],
        ),
        embedding=emb,
    )

    results = await search_by_vector(
        pool, emb, limit=5, memory_type=MemoryType.decision,
    )
    assert all(r.memory.type != MemoryType.anti_pattern for r in results)


async def test_primer_surfaces_anti_patterns(pool, provider):
    """Anti-patterns should appear in the primer output."""
    content = "CONTEXT: database migrations. PROBLEM: running migrations without a backup causes irreversible data loss. AVOID: deploying migrations to production without a recent backup."
    emb = await provider.embed(content)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content=content,
            topic=["database"],
            project_id="test-project",
        ),
        embedding=emb,
    )

    result = await build_primer(pool, project_id="test-project")

    # Should have anti_patterns key in the primer output
    assert "anti_patterns" in result
    assert len(result["anti_patterns"]) >= 1
    assert result["anti_patterns"][0]["type"] == "anti_pattern"
    assert "migration" in result["anti_patterns"][0]["content"].lower()


async def test_primer_anti_patterns_before_decisions(pool, provider):
    """Anti-patterns section should be packed before decisions in the primer."""
    emb_ap = await provider.embed("CONTEXT: API design. PROBLEM: exposing internal IDs in URLs creates security vulnerabilities. AVOID: using sequential database IDs in public-facing endpoints.")
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content="CONTEXT: API design. PROBLEM: exposing internal IDs in URLs creates security vulnerabilities. AVOID: using sequential database IDs in public-facing endpoints.",
            topic=["api"],
            project_id="test-project",
        ),
        embedding=emb_ap,
    )

    emb_d = await provider.embed("Use UUIDs for all public-facing resource identifiers in the API layer")
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content="Use UUIDs for all public-facing resource identifiers in the API layer",
            topic=["api"],
            project_id="test-project",
        ),
        embedding=emb_d,
    )

    result = await build_primer(pool, project_id="test-project")

    assert "anti_patterns" in result
    assert "decisions" in result
    # Both should be present
    assert len(result["anti_patterns"]) >= 1
    assert len(result["decisions"]) >= 1


async def test_anti_pattern_unfiltered_recall(pool, provider):
    """Anti-patterns should appear in unfiltered recall alongside other types."""
    content_ap = "CONTEXT: error handling. PROBLEM: catching bare Exception hides bugs and makes debugging impossible. AVOID: using except Exception in production code."
    emb_ap = await provider.embed(content_ap)
    mem_ap = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content=content_ap,
            topic=["error-handling"],
        ),
        embedding=emb_ap,
    )

    content_fact = "Python exception hierarchy has BaseException at the root with Exception as a subclass for user-facing errors"
    emb_fact = await provider.embed(content_fact)
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_fact,
            topic=["error-handling"],
        ),
        embedding=emb_fact,
    )

    # Unfiltered search should return both
    emb_q = await provider.embed("exception handling python")
    results = await search_by_vector(pool, emb_q, limit=10)
    result_ids = {r.memory.id for r in results}
    assert mem_ap.id in result_ids


async def test_anti_pattern_to_dict(pool, provider):
    """anti_pattern memory serializes correctly."""
    content = "CONTEXT: testing. PROBLEM: mocking database in integration tests gives false confidence when schema changes. AVOID: using mocks for database calls in integration test suites."
    emb = await provider.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.anti_pattern,
            content=content,
        ),
        embedding=emb,
    )

    d = mem.to_dict()
    assert d["type"] == "anti_pattern"
