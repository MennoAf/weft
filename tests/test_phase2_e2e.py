"""End-to-end validation for Phase 2: Retrieval & Context.

Tests the complete Phase 2 pipeline with realistic data:
- Budget-aware context loading
- Version-aware revisions
- Enhanced filtering
- Relevance scoring (recency, confidence, frequency)
- Topic diversity
- MEMORY.md-style import scenario
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.context import build_context
from weft.embeddings import get_provider
from weft.models import (
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from weft.relevance import rank_memories
from weft.revise import revise_memory
from weft.store import (
    search_by_vector,
    store_memory,
    touch_memory,
    update_memory,
)

# Extended seed data — 15 memories spanning different types, topics,
# confidence levels, and simulated recency
PHASE2_SEEDS = [
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
    # 5 additional memories for Phase 2 testing
    {
        "type": MemoryType.fact,
        "content": "Loom uses loom_claim to atomically take ownership of a task with a TTL-based lease",
        "topic": ["loom", "orchestration"],
        "confidence": 0.9,
        "source": MemorySource.documentation,
    },
    {
        "type": MemoryType.architecture,
        "content": "Weft's embedding column uses variable-dimension vector type to support provider switching",
        "topic": ["weft", "embeddings", "postgres"],
        "confidence": 0.95,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.solution,
        "content": "Float precision issues with Postgres REAL columns are handled by pytest.approx in assertions",
        "topic": ["testing", "postgres"],
        "confidence": 0.85,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.pattern,
        "content": "MCP tools should be thin coordinators with 15 lines max — business logic goes in separate modules",
        "topic": ["mcp", "patterns", "architecture"],
        "confidence": 0.9,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.fact,
        "content": "The fastembed BAAI/bge-small-en-v1.5 model produces 384-dimensional embeddings using ONNX runtime",
        "topic": ["embeddings", "fastembed"],
        "confidence": 0.95,
        "source": MemorySource.documentation,
    },
]


@pytest.fixture
async def phase2_pool(pool):
    """Pool with 15 diverse memories seeded with embeddings."""
    provider = get_provider("fastembed")
    memories = []
    for mem_data in PHASE2_SEEDS:
        create = MemoryCreate(**mem_data)
        embedding = await provider.embed(create.content)
        mem = await store_memory(pool, create, embedding=embedding)
        memories.append(mem)
    return pool, memories, provider


# --- E2E Scenario 1: Budget-aware context loading ---


async def test_context_budget_fitting(phase2_pool):
    """weft_context should return a subset that fits within token budget."""
    pool, _, provider = phase2_pool
    emb = await provider.embed("tell me about the project architecture")

    # Small budget — shouldn't fit all 15 memories
    result = await build_context(pool, emb, budget_tokens=200)

    assert result["total_tokens"] <= 200
    assert result["count"] > 0
    assert result["count"] < 15  # shouldn't fit everything
    assert result["remaining_budget"] >= 0

    # All returned memories should have relevance scores
    for m in result["memories"]:
        assert "relevance_score" in m
        assert m["relevance_score"] > 0


# --- E2E Scenario 2: Revision chains ---


async def test_revision_chain_e2e(phase2_pool):
    """Create a memory, revise it twice, verify the chain."""
    pool, _, provider = phase2_pool

    # Create original
    create = MemoryCreate(
        type=MemoryType.fact,
        content="pgvector supports version 0.7.0",
        topic=["postgres"],
        source=MemorySource.documentation,
        confidence=0.8,
    )
    emb = await provider.embed(create.content)
    original = await store_memory(pool, create, embedding=emb)

    # Revise v1 → v2
    emb2 = await provider.embed("pgvector supports version 0.8.0")
    v2, old1 = await revise_memory(
        pool, original.id, "pgvector supports version 0.8.0",
        embedding=emb2, new_confidence=0.9,
    )
    assert old1.status == MemoryStatus.archived
    assert v2.confidence == 0.9

    # Revise v2 → v3
    emb3 = await provider.embed("pgvector supports version 0.8.1 with HNSW indexes")
    v3, old2 = await revise_memory(
        pool, v2.id, "pgvector supports version 0.8.1 with HNSW indexes",
        embedding=emb3, new_confidence=0.95,
    )
    assert old2.status == MemoryStatus.archived
    assert v3.status == MemoryStatus.active

    # Only v3 should appear in active search
    search_emb = await provider.embed("pgvector version")
    results = await search_by_vector(pool, search_emb, limit=20)
    active_versions = [
        r for r in results
        if r.memory.id in (original.id, v2.id, v3.id)
    ]
    assert len(active_versions) == 1
    assert active_versions[0].memory.id == v3.id


# --- E2E Scenario 3: Filtered recall ---


async def test_filtered_recall_by_type_and_topic(phase2_pool):
    """Combining type + topic filters should narrow results correctly."""
    pool, _, provider = phase2_pool
    emb = await provider.embed("testing best practices")

    # All testing-related
    all_results = await search_by_vector(pool, emb, limit=20, topic="testing")
    assert len(all_results) >= 2

    # Only patterns about testing
    pattern_results = await search_by_vector(
        pool, emb, limit=20, topic="testing", memory_type=MemoryType.pattern,
    )
    assert all(r.memory.type == MemoryType.pattern for r in pattern_results)
    assert len(pattern_results) <= len(all_results)


# --- E2E Scenario 4: Relevance scoring ranks correctly ---


async def test_relevance_prefers_recent_high_confidence(phase2_pool):
    """A recent high-confidence memory should rank above an old low-confidence one."""
    pool, memories, provider = phase2_pool

    # Touch one memory to make it "recently accessed"
    recent_mem = memories[1]  # architecture, 0.95 confidence
    for _ in range(5):
        await touch_memory(pool, recent_mem.id)

    # Make another memory "old" by setting accessed_at to 90 days ago
    old_mem = memories[3]  # pattern, 0.8 confidence
    await pool.execute(
        "UPDATE memories SET accessed_at = $1 WHERE id = $2",
        datetime.now(timezone.utc) - timedelta(days=90),
        old_mem.id,
    )

    emb = await provider.embed("project patterns and architecture")
    results = await search_by_vector(pool, emb, limit=20)
    ranked = rank_memories(results)

    # Find positions
    recent_pos = next(
        (i for i, s in enumerate(ranked) if s.memory.id == recent_mem.id), None
    )
    old_pos = next(
        (i for i, s in enumerate(ranked) if s.memory.id == old_mem.id), None
    )

    if recent_pos is not None and old_pos is not None:
        assert recent_pos < old_pos, "Recent high-confidence should rank higher"


# --- E2E Scenario 5: Topic diversity ---


async def test_topic_diversity_in_context(phase2_pool):
    """Context builder should diversify across topics."""
    pool, _, provider = phase2_pool
    emb = await provider.embed("everything about the project")

    # max_per_topic=1 forces diversity
    result = await build_context(
        pool, emb, budget_tokens=100000, max_per_topic=1,
    )

    # Verify no topic appears more than once
    topic_counts: dict[str, int] = {}
    for m in result["memories"]:
        for t in m.get("topic", []):
            topic_counts[t] = topic_counts.get(t, 0) + 1

    for topic, count in topic_counts.items():
        assert count <= 1, f"Topic '{topic}' appeared {count} times with max_per_topic=1"


# --- E2E Scenario 6: MEMORY.md-style import ---


SAMPLE_MEMORY_MD = """
## Preferences
- Always use bun instead of npm for package management
- Prefer sonnet for subagent tasks to minimize cost

## Architecture
- Loom uses a DAG-based task dependency system with topological ordering
- The MCP server runs on stdio transport and is registered in .mcp.json

## Facts
- Loom has 760 tests across unit, integration, and e2e suites
- The Loom project was started in January 2026
""".strip()


def _parse_memory_md(text: str) -> list[dict]:
    """Simple MEMORY.md parser — extracts bullets under sections."""
    memories = []
    current_type = MemoryType.fact
    type_map = {
        "preferences": MemoryType.preference,
        "architecture": MemoryType.architecture,
        "facts": MemoryType.fact,
        "patterns": MemoryType.pattern,
        "solutions": MemoryType.solution,
        "relationships": MemoryType.relationship,
    }

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip().lower()
            current_type = type_map.get(section, MemoryType.fact)
        elif line.startswith("- "):
            content = line[2:].strip()
            memories.append({
                "type": current_type,
                "content": content,
                "topic": [current_type.value],
                "source": MemorySource.documentation,
                "confidence": 0.8 if current_type == MemoryType.fact else 1.0,
            })

    return memories


async def test_memory_md_import_and_query(phase2_pool):
    """Import MEMORY.md-style content and verify it's queryable."""
    pool, _, provider = phase2_pool

    # Parse and import
    parsed = _parse_memory_md(SAMPLE_MEMORY_MD)
    assert len(parsed) == 6

    for mem_data in parsed:
        create = MemoryCreate(**mem_data)
        emb = await provider.embed(create.content)
        await store_memory(pool, create, embedding=emb)

    # Query for Loom task management
    query_emb = await provider.embed("how does Loom handle task dependencies?")
    result = await build_context(pool, query_emb, budget_tokens=2000)

    assert result["count"] > 0
    # At least one of the imported memories should surface
    contents = [m["content"] for m in result["memories"]]
    assert any(
        "loom" in c.lower() or "dag" in c.lower() or "task" in c.lower()
        for c in contents
    ), f"Expected Loom-related content in results: {contents[:3]}"


async def test_memory_md_import_respects_budget(phase2_pool):
    """Imported memories should respect budget constraints like any other memory."""
    pool, _, provider = phase2_pool

    for mem_data in _parse_memory_md(SAMPLE_MEMORY_MD):
        create = MemoryCreate(**mem_data)
        emb = await provider.embed(create.content)
        await store_memory(pool, create, embedding=emb)

    # Very tight budget
    query_emb = await provider.embed("project setup and tooling")
    result = await build_context(pool, query_emb, budget_tokens=50)

    assert result["total_tokens"] <= 50
    assert result["count"] >= 1  # should fit at least one short memory
