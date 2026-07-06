"""Tests for query-biased primer (context-aware weft_prime)."""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.primer import build_primer
from weft.store import list_memories, store_memory


@pytest.fixture
def provider():
    from weft.embeddings import get_provider

    return get_provider("fastembed")


# --- Helpers ---


async def _store(pool, provider, content: str, mem_type: MemoryType, **kwargs):
    """Store a memory with embedding."""
    create = MemoryCreate(
        type=mem_type,
        content=content,
        source=MemorySource.conversation,
        confidence=kwargs.pop("confidence", 0.8),
        topic=kwargs.pop("topic", []),
        pinned=kwargs.pop("pinned", False),
        project_id=kwargs.pop("project_id", None),
    )
    vec = await provider.embed(content)
    return await store_memory(pool, create, embedding=vec)


# --- Unbiased fallback ---


async def test_no_query_identical_to_existing(pool, provider):
    """build_primer without query_vec produces the same result as before."""
    await _store(pool, provider, "Use pytest for all tests", MemoryType.decision)
    await _store(pool, provider, "Redis caching is broken", MemoryType.issue)

    result_none = await build_primer(pool, query_vec=None, disclosure="full")
    result_omitted = await build_primer(pool, disclosure="full")

    assert result_none["decisions"] == result_omitted["decisions"]
    assert result_none["issues"] == result_omitted["issues"]


# --- Query-biased sections ---


async def test_query_biases_decisions(pool, provider):
    """When query_vec is provided, decisions are re-ranked by relevance."""
    # Store two decisions: one about testing, one about deployment
    await _store(pool, provider, "Always use pytest-asyncio for async tests", MemoryType.decision, topic=["testing"])
    await _store(pool, provider, "Deploy to production using Docker Compose", MemoryType.decision, topic=["deployment"])

    # Query about testing should rank the testing decision first
    query_vec = await provider.embed("testing patterns and pytest")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    decisions = result["decisions"]
    assert len(decisions) == 2
    # Testing decision should come first when querying about testing
    assert "pytest" in decisions[0]["content"]


async def test_query_biases_issues(pool, provider):
    """When query_vec is provided, issues are re-ranked by relevance."""
    await _store(pool, provider, "Database connection pool leaks under load", MemoryType.issue, topic=["database"])
    await _store(pool, provider, "CSS styling breaks on mobile viewport", MemoryType.issue, topic=["frontend"])

    query_vec = await provider.embed("database performance and connection problems")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    issues = result["issues"]["items"]
    assert len(issues) == 2
    assert "Database" in issues[0]["content"]


async def test_query_biases_milestones(pool, provider):
    """When query_vec is provided, milestones are re-ranked by relevance."""
    await _store(pool, provider, "Implemented Redis caching layer for API responses", MemoryType.milestone, topic=["redis"])
    await _store(pool, provider, "Migrated authentication to JWT tokens", MemoryType.milestone, topic=["auth"])

    query_vec = await provider.embed("caching and Redis performance")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    recent = result["recent_work"]
    assert len(recent) == 2
    assert "Redis" in recent[0]["summary"]


# --- Unbiased sections stay unbiased ---


async def test_rules_not_biased_by_query(pool, provider):
    """Pinned rules are always fetched by metadata, not biased by query."""
    await _store(pool, provider, "Always run linter before commit", MemoryType.preference, pinned=True)

    query_vec = await provider.embed("database migrations")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    assert len(result["rules"]) == 1
    assert "linter" in result["rules"][0]["content"]


async def test_handoff_not_biased_by_query(pool, provider):
    """Handoff is always the most recent, regardless of query."""
    await _store(
        pool,
        provider,
        "Worked on frontend refactoring today",
        MemoryType.handoff,
        project_id="test-proj",
    )

    query_vec = await provider.embed("backend API design")
    result = await build_primer(
        pool,
        project_id="test-proj",
        query_vec=query_vec,
        disclosure="full",
    )

    assert len(result["handoff"]) == 1
    assert "frontend" in result["handoff"][0]["content"]


# --- Edge cases ---


async def test_empty_store_with_query(pool, provider):
    """Query on empty store returns normal empty primer."""
    query_vec = await provider.embed("anything")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    assert result["decisions"] == []
    assert result["issues"]["items"] == []
    assert result["recent_work"] == []


async def test_query_with_no_matching_decisions(pool, provider):
    """Query with low similarity still returns decisions (threshold is permissive)."""
    await _store(pool, provider, "Use tabs not spaces for indentation", MemoryType.decision)

    # Completely unrelated query
    query_vec = await provider.embed("quantum physics and black holes")
    result = await build_primer(pool, query_vec=query_vec, disclosure="full")

    # Decision should still appear (threshold is 0.1, very permissive)
    assert len(result["decisions"]) >= 1


async def test_budget_still_enforced_with_query(pool, provider):
    """Token budget is still enforced when using query-biased fetching."""
    for i in range(10):
        await _store(pool, provider, f"Decision {i}: " + "x" * 200, MemoryType.decision)

    query_vec = await provider.embed("decisions")
    result = await build_primer(pool, query_vec=query_vec, budget_tokens=300, disclosure="full")

    # Should not exceed budget
    assert result["total_tokens"] <= 300
