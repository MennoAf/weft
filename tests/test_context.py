"""Tests for budget-aware context loading."""

from __future__ import annotations

import pytest

from weft.context import _deduplicate_by_topic, _pack_by_budget, build_context
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryRecall, MemorySource, MemoryType
from weft.relevance import ScoredMemory, score_memory
from weft.store import record_feedback, store_memory


# --- Unit tests for helpers ---


def _make_scored(
    content: str = "test",
    topic: list[str] | None = None,
    token_count: int = 10,
    score: float = 0.8,
) -> ScoredMemory:
    from weft.models import Memory, MemoryStatus

    mem = Memory(
        type=MemoryType.fact,
        content=content,
        topic=topic or [],
        token_count=token_count,
        status=MemoryStatus.active,
    )
    return ScoredMemory(
        memory=mem,
        similarity=score,
        confidence_factor=1.0,
        recency_factor=1.0,
        frequency_factor=1.0,
        usefulness_factor=1.0,
        type_boost_factor=1.0,
        score=score,
    )


def test_dedup_limits_per_topic():
    """Should limit to max_per_topic per topic."""
    scored = [
        _make_scored(content=f"pg fact {i}", topic=["postgres"], score=0.9 - i * 0.01)
        for i in range(5)
    ]
    result = _deduplicate_by_topic(scored, max_per_topic=2)
    assert len(result) == 2


def test_dedup_no_topic_passes_through():
    """Memories without topics should pass through uncapped."""
    scored = [
        _make_scored(content=f"no topic {i}", topic=[], score=0.9)
        for i in range(5)
    ]
    result = _deduplicate_by_topic(scored, max_per_topic=2)
    assert len(result) == 5


def test_dedup_mixed_topics():
    """Different topics should each get their own cap."""
    scored = [
        _make_scored(topic=["postgres"]),
        _make_scored(topic=["postgres"]),
        _make_scored(topic=["postgres"]),
        _make_scored(topic=["redis"]),
        _make_scored(topic=["redis"]),
    ]
    result = _deduplicate_by_topic(scored, max_per_topic=2)
    pg_count = sum(1 for s in result if "postgres" in s.memory.topic)
    redis_count = sum(1 for s in result if "redis" in s.memory.topic)
    assert pg_count == 2
    assert redis_count == 2


def test_pack_respects_budget():
    """Should stop packing when budget is exhausted."""
    scored = [
        _make_scored(token_count=100, score=0.9),
        _make_scored(token_count=100, score=0.8),
        _make_scored(token_count=100, score=0.7),
    ]
    result = _pack_by_budget(scored, budget_tokens=200)
    assert len(result) == 2


def test_pack_empty_list():
    assert _pack_by_budget([], budget_tokens=1000) == []


def test_pack_single_oversized():
    """A single memory larger than budget should be excluded."""
    scored = [_make_scored(token_count=5000)]
    result = _pack_by_budget(scored, budget_tokens=100)
    assert len(result) == 0


# --- Integration tests with DB ---


@pytest.fixture
async def context_pool(pool):
    """Pool seeded with diverse memories for context tests."""
    provider = get_provider("fastembed")

    seeds = [
        ("store.py is the ONLY Postgres writer", MemoryType.architecture, ["postgres", "weft"]),
        ("pgvector 0.8.1 supports cosine distance", MemoryType.fact, ["postgres", "pgvector"]),
        ("Redis cache uses 1hr TTL for memories", MemoryType.fact, ["redis", "caching"]),
        ("Use fastembed as default embedding provider", MemoryType.preference, ["embeddings"]),
        ("Testcontainers give best testing isolation", MemoryType.pattern, ["testing"]),
        ("Casey Example owns two related projects", MemoryType.relationship, ["people"]),
        ("Three-layer config for flexibility", MemoryType.pattern, ["configuration"]),
        ("Weft MCP server uses FastMCP stdio", MemoryType.architecture, ["weft", "mcp"]),
        ("Docker infra should be per-project", MemoryType.preference, ["docker", "infrastructure"]),
        ("Embedding dimension varies by provider", MemoryType.fact, ["embeddings", "pgvector"]),
    ]

    for content, mem_type, topics in seeds:
        create = MemoryCreate(
            type=mem_type,
            content=content,
            topic=topics,
            source=MemorySource.code,
            confidence=0.9,
        )
        emb = await provider.embed(content)
        await store_memory(pool, create, embedding=emb)

    return pool, provider


async def test_build_context_returns_within_budget(context_pool):
    """Result should respect token budget."""
    pool, provider = context_pool
    emb = await provider.embed("postgres database configuration")

    result = await build_context(pool, emb, budget_tokens=100)

    assert result["total_tokens"] <= 100
    assert result["remaining_budget"] >= 0
    assert result["total_tokens"] + result["remaining_budget"] == result["budget_tokens"]


async def test_build_context_large_budget_returns_all(context_pool):
    """With a large budget, should return all relevant memories."""
    pool, provider = context_pool
    emb = await provider.embed("everything about this project")

    result = await build_context(pool, emb, budget_tokens=100000)

    assert result["count"] == 10  # all seeded memories


async def test_build_context_topic_diversity(context_pool):
    """With max_per_topic=1, each topic should appear at most once."""
    pool, provider = context_pool
    emb = await provider.embed("postgres and embeddings")

    result = await build_context(
        pool, emb, budget_tokens=100000, max_per_topic=1,
    )

    # Check no topic appears more than once
    topic_counts: dict[str, int] = {}
    for mem in result["memories"]:
        for t in mem.get("topic", []):
            topic_counts[t] = topic_counts.get(t, 0) + 1

    for topic, count in topic_counts.items():
        assert count <= 1, f"Topic '{topic}' appears {count} times"


async def test_build_context_empty_query(context_pool):
    """Even an irrelevant query should return a valid structure."""
    pool, provider = context_pool
    emb = await provider.embed("quantum physics black holes")

    result = await build_context(pool, emb, budget_tokens=1000, threshold=0.9)

    assert "memories" in result
    assert "total_tokens" in result
    assert isinstance(result["count"], int)


async def test_build_context_with_type_filter(context_pool):
    """Type filter should pass through to search."""
    pool, provider = context_pool
    emb = await provider.embed("project architecture")

    result = await build_context(
        pool, emb, budget_tokens=100000, memory_type=MemoryType.architecture,
    )

    for mem in result["memories"]:
        assert mem["type"] == "architecture"


async def test_build_context_memories_sorted_by_relevance(context_pool):
    """Results should be sorted by relevance score descending."""
    pool, provider = context_pool
    emb = await provider.embed("database postgres configuration")

    result = await build_context(pool, emb, budget_tokens=100000)

    scores = [m["relevance_score"] for m in result["memories"]]
    assert scores == sorted(scores, reverse=True)


async def test_build_context_pinned_sorted_by_usefulness(pool):
    """Pinned memories should be sorted by usefulness_score descending before packing."""
    provider = get_provider("fastembed")

    # Create 3 pinned memories with large token counts so budget can only fit 2
    for content in ["pinned rule A", "pinned rule B", "pinned rule C"]:
        create = MemoryCreate(
            type=MemoryType.preference,
            content=content,
            topic=["rules"],
            source=MemorySource.conversation,
            confidence=0.9,
            pinned=True,
        )
        await store_memory(pool, create, embedding=await provider.embed(content))

    # Give different usefulness to each via feedback
    from weft.store import list_memories
    pinned = await list_memories(pool, pinned=True, limit=10)
    assert len(pinned) == 3

    # Make "B" the most useful, "C" middle, "A" least useful
    for mem in pinned:
        if "rule A" in mem.content:
            for _ in range(5):
                await record_feedback(pool, mem.id, helpful=False)
        elif "rule B" in mem.content:
            for _ in range(5):
                await record_feedback(pool, mem.id, helpful=True)
        # C stays at default (0.7)

    emb = await provider.embed("project rules")
    # Budget just big enough for 2 pinned memories
    token_cost = pinned[0].token_count or 1
    tight_budget = token_cost * 2 + 1

    result = await build_context(pool, emb, budget_tokens=tight_budget)
    pinned_contents = [m["content"] for m in result["memories"] if m.get("pinned")]

    # B (highest usefulness) should be included; A (lowest) should be dropped
    assert any("rule B" in c for c in pinned_contents), "High-usefulness pinned memory B should be included"
    assert not any("rule A" in c for c in pinned_contents), "Low-usefulness pinned memory A should be dropped"
