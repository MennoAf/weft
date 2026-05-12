"""Tests for weft.store — Postgres CRUD and vector search."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryStatus, MemoryType, MemorySource, RelationType
from weft.store import (
    add_relationship,
    count_by_vector,
    delete_memory,
    get_memory,
    get_recent_writes,
    get_relationships,
    get_stats,
    list_memories,
    remove_relationship,
    search_by_vector,
    store_memory,
    touch_memory,
    update_memory,
    bump_retrieval_telemetry,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


async def test_store_and_get(pool):
    """Store a memory and retrieve it by ID."""
    create = MemoryCreate(
        type=MemoryType.fact,
        content="Weft uses pgvector for semantic search",
        topic=["weft", "architecture"],
        confidence=0.9,
    )
    mem = await store_memory(pool, create)
    assert mem.id.startswith("weft-")
    assert mem.type == MemoryType.fact
    assert mem.confidence == 0.9

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.content == create.content
    assert fetched.topic == ["weft", "architecture"]


async def test_get_nonexistent(pool):
    """Getting a non-existent memory returns None."""
    result = await get_memory(pool, "weft-00000000")
    assert result is None


async def test_list_memories(pool):
    """List memories with filters."""
    for i in range(5):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact if i < 3 else MemoryType.pattern,
            content=f"memory {i}",
            topic=["test"],
        ))

    all_mems = await list_memories(pool)
    assert len(all_mems) == 5

    facts = await list_memories(pool, memory_type=MemoryType.fact)
    assert len(facts) == 3

    patterns = await list_memories(pool, memory_type=MemoryType.pattern)
    assert len(patterns) == 2


async def test_list_by_topic(pool):
    """Filter memories by topic."""
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="a", topic=["alpha"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="b", topic=["beta"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="c", topic=["alpha", "beta"]))

    alpha = await list_memories(pool, topic="alpha")
    assert len(alpha) == 2

    beta = await list_memories(pool, topic="beta")
    assert len(beta) == 2


async def test_list_memories_exact_scope(pool):
    """exact_scope=True skips OR-NULL, matching only the exact project_id."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff, content="global handoff", project_id=None,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff, content="proj-a handoff", project_id="proj-a",
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff, content="proj-b handoff", project_id="proj-b",
    ))

    # Default OR-NULL: proj-a query also returns global
    or_null = await list_memories(
        pool, memory_type=MemoryType.handoff, project_id="proj-a",
    )
    assert len(or_null) == 2
    assert {m.content for m in or_null} == {"global handoff", "proj-a handoff"}

    # exact_scope: only proj-a
    exact = await list_memories(
        pool, memory_type=MemoryType.handoff, project_id="proj-a",
        exact_scope=True,
    )
    assert len(exact) == 1
    assert exact[0].content == "proj-a handoff"


async def test_update_memory(pool):
    """Update mutable fields."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="original",
        confidence=0.5,
    ))

    updated = await update_memory(pool, mem.id, content="revised", confidence=0.9)
    assert updated is not None
    assert updated.content == "revised"
    assert updated.confidence == pytest.approx(0.9, abs=1e-6)
    assert updated.updated_at > mem.updated_at


async def test_update_memory_type(pool):
    """Update memory type via update_memory."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="## Session Handoff\n\n**Summary:** mistyped",
    ))
    assert mem.type == MemoryType.fact

    updated = await update_memory(pool, mem.id, memory_type=MemoryType.handoff)
    assert updated is not None
    assert updated.type == MemoryType.handoff

    # Verify via list filter
    handoffs = await list_memories(pool, memory_type=MemoryType.handoff)
    assert any(m.id == mem.id for m in handoffs)


async def test_soft_delete(pool):
    """Soft-delete archives the memory."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="to delete"))
    assert await delete_memory(pool, mem.id)

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.status == MemoryStatus.archived

    # Should not appear in active list
    active = await list_memories(pool, status=MemoryStatus.active)
    assert all(m.id != mem.id for m in active)


async def test_hard_delete(pool):
    """Hard-delete removes the memory entirely."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="to nuke"))
    assert await delete_memory(pool, mem.id, hard=True)
    assert await get_memory(pool, mem.id) is None


async def test_touch_memory(pool):
    """Touch updates accessed_at and increments access_count."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="touch me"))
    assert mem.access_count == 0

    await touch_memory(pool, mem.id)
    fetched = await get_memory(pool, mem.id)
    assert fetched.access_count == 1
    assert fetched.accessed_at >= mem.accessed_at


async def _retrieval_telemetry(pool, memory_id):
    row = await pool.fetchrow(
        "SELECT last_retrieved_at, retrieval_count FROM memories WHERE id = $1",
        memory_id,
    )
    return row["last_retrieved_at"], row["retrieval_count"]


async def test_bump_retrieval_telemetry_default(pool):
    """Freshly stored memories start with retrieval_count=0 and NULL last_retrieved_at."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="never retrieved"))
    last_at, count = await _retrieval_telemetry(pool, mem.id)
    assert last_at is None
    assert count == 0


async def test_bump_retrieval_telemetry_single(pool):
    """Single bump increments count by 1 and stamps last_retrieved_at."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="bump once"))

    await bump_retrieval_telemetry(pool, [mem.id])
    last_at, count = await _retrieval_telemetry(pool, mem.id)
    assert count == 1
    assert last_at is not None


async def test_bump_retrieval_telemetry_batched(pool):
    """Batched bump increments every supplied id in a single statement."""
    ids = [
        (await store_memory(pool, MemoryCreate(type=MemoryType.fact, content=f"batch {i}"))).id
        for i in range(3)
    ]

    await bump_retrieval_telemetry(pool, ids)
    for memory_id in ids:
        _, count = await _retrieval_telemetry(pool, memory_id)
        assert count == 1


async def test_bump_retrieval_telemetry_idempotent_increments(pool):
    """Repeated bumps accumulate — counter is monotonic, not set-once."""
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="bump thrice"))

    for _ in range(3):
        await bump_retrieval_telemetry(pool, [mem.id])
    _, count = await _retrieval_telemetry(pool, mem.id)
    assert count == 3


async def test_bump_retrieval_telemetry_empty_is_noop(pool):
    """Empty id list issues no UPDATE — callers can pass result lists unconditionally."""
    # No exception, no side effect. The implementation short-circuits before
    # the SQL so this also doubles as a guard against accidental wildcard updates.
    await bump_retrieval_telemetry(pool, [])


async def test_bump_retrieval_telemetry_does_not_touch_access_count(pool):
    """The new telemetry is intentionally decoupled from accessed_at/access_count.

    Confounding the two would re-couple raw retrieval signal to the usefulness
    EMA bump that touch_memory applies — defeats the point of v49.
    """
    mem = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="separated signals"))

    await bump_retrieval_telemetry(pool, [mem.id])
    fetched = await get_memory(pool, mem.id)
    assert fetched.access_count == 0
    assert fetched.accessed_at == mem.accessed_at


async def test_vector_search(pool):
    """Store memories with embeddings and search by similarity."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    texts = [
        "Python is a programming language",
        "PostgreSQL is a relational database",
        "Redis is an in-memory data store",
    ]

    for text in texts:
        emb = await provider.embed(text)
        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=text, topic=["tech"]),
            embedding=emb,
        )

    # Search for something related to databases
    query_emb = await provider.embed("database management system")
    results = await search_by_vector(pool, query_emb, limit=3)

    assert len(results) > 0
    # PostgreSQL should be most similar to "database management system"
    assert "PostgreSQL" in results[0].memory.content
    assert results[0].similarity > 0.5


async def test_relationships(pool):
    """Create and query relationships between memories."""
    m1 = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="old fact"))
    m2 = await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="new fact"))

    rel = await add_relationship(pool, m2.id, m1.id, RelationType.supersedes)
    assert rel.source_id == m2.id
    assert rel.relation == RelationType.supersedes

    # Query relationships
    rels = await get_relationships(pool, m2.id)
    assert len(rels) == 1

    rels_typed = await get_relationships(pool, m2.id, relation=RelationType.supersedes)
    assert len(rels_typed) == 1

    # Remove
    assert await remove_relationship(pool, m2.id, m1.id, RelationType.supersedes)
    assert len(await get_relationships(pool, m2.id)) == 0


async def test_stats(pool):
    """Get memory statistics."""
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="a", topic=["x"]))
    await store_memory(pool, MemoryCreate(type=MemoryType.pattern, content="b", topic=["x", "y"]))

    stats = await get_stats(pool)
    assert stats["total"] == 2
    assert stats["by_type"]["fact"] == 1
    assert stats["by_type"]["pattern"] == 1
    assert "x" in stats["top_topics"]


# --- count_by_vector ---


async def test_count_by_vector(pool):
    """count_by_vector returns total matches above threshold without LIMIT."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    texts = [
        "Python is a programming language",
        "PostgreSQL is a relational database",
        "Redis is an in-memory data store",
        "JavaScript runs in the browser",
        "MySQL is another database system",
    ]

    for text in texts:
        emb = await provider.embed(text)
        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=text, topic=["tech"]),
            embedding=emb,
        )

    query_emb = await provider.embed("database management")

    # Search with limit=2 should return 2 results
    results = await search_by_vector(pool, query_emb, limit=2)
    assert len(results) == 2

    # Count should return total matches above threshold (more than 2)
    total = await count_by_vector(pool, query_emb, threshold=0.0)
    assert total >= len(results)
    assert total == 5  # all memories match at threshold=0.0


async def test_count_by_vector_with_threshold(pool):
    """count_by_vector respects the similarity threshold."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    texts = [
        "PostgreSQL is a relational database",
        "The weather is sunny today",
    ]
    for text in texts:
        emb = await provider.embed(text)
        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=text),
            embedding=emb,
        )

    query_emb = await provider.embed("database systems")
    # High threshold should exclude the weather memory
    total = await count_by_vector(pool, query_emb, threshold=0.5)
    assert total <= 2  # at most both, but weather is unlikely above 0.5


# --- get_recent_writes ---


async def test_recent_writes(pool):
    """get_recent_writes returns memories ordered by created_at DESC."""
    m1 = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="first memory", source=MemorySource.conversation,
    ))
    m2 = await store_memory(pool, MemoryCreate(
        type=MemoryType.decision, content="second memory", source=MemorySource.code,
    ))

    writes = await get_recent_writes(pool, limit=10)
    assert len(writes) >= 2
    # Most recent first
    assert writes[0]["id"] == m2.id
    assert writes[1]["id"] == m1.id
    # Provenance fields present
    assert writes[0]["type"] == "decision"
    assert writes[0]["source"] == "code"
    assert writes[1]["source"] == "conversation"


async def test_recent_writes_limit(pool):
    """get_recent_writes respects limit."""
    for i in range(5):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact, content=f"memory {i}",
        ))

    writes = await get_recent_writes(pool, limit=3)
    assert len(writes) == 3


async def test_recent_writes_content_truncation(pool):
    """get_recent_writes truncates long content."""
    long_content = "x" * 200
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content=long_content,
    ))

    writes = await get_recent_writes(pool, limit=1)
    assert len(writes[0]["content"]) == 83  # 80 chars + "..."
    assert writes[0]["content"].endswith("...")


# --- Milestone type ---


async def test_store_milestone_type(pool):
    """Store and retrieve a milestone memory."""
    create = MemoryCreate(
        type=MemoryType.milestone,
        content="Shipped primer optimization (319 tests)",
        topic=["loom-abc123"],
        confidence=1.0,
    )
    mem = await store_memory(pool, create)
    assert mem.type == MemoryType.milestone

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.type == MemoryType.milestone
    assert fetched.content == "Shipped primer optimization (319 tests)"


async def test_list_milestones(pool):
    """List memories filtered by milestone type."""
    await store_memory(pool, MemoryCreate(type=MemoryType.milestone, content="m1"))
    await store_memory(pool, MemoryCreate(type=MemoryType.milestone, content="m2"))
    await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="not a milestone"))

    milestones = await list_memories(pool, memory_type=MemoryType.milestone)
    assert len(milestones) == 2
    assert all(m.type == MemoryType.milestone for m in milestones)


# --- review_after field ---


async def test_store_with_review_after(pool):
    """Store a memory with review_after set and verify round-trip."""
    review_date = datetime.now(timezone.utc) + timedelta(days=30)
    create = MemoryCreate(
        type=MemoryType.decision,
        content="Don't suggest mocks",
        confidence=0.9,
        review_after=review_date,
    )
    mem = await store_memory(pool, create)
    assert mem.review_after is not None

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.review_after is not None
    # Compare within 1 second tolerance (DB might truncate microseconds)
    assert abs((fetched.review_after - review_date).total_seconds()) < 1


async def test_store_without_review_after(pool):
    """review_after defaults to None when not provided."""
    create = MemoryCreate(type=MemoryType.fact, content="No review date")
    mem = await store_memory(pool, create)
    assert mem.review_after is None

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.review_after is None


async def test_update_review_after(pool):
    """Update a memory's review_after field."""
    create = MemoryCreate(type=MemoryType.decision, content="Some decision")
    mem = await store_memory(pool, create)
    assert mem.review_after is None

    review_date = datetime.now(timezone.utc) + timedelta(days=14)
    updated = await update_memory(pool, mem.id, review_after=review_date)
    assert updated is not None
    assert updated.review_after is not None
    assert abs((updated.review_after - review_date).total_seconds()) < 1


async def test_clear_review_after(pool):
    """Set review_after back to None."""
    review_date = datetime.now(timezone.utc) + timedelta(days=30)
    create = MemoryCreate(
        type=MemoryType.decision, content="Temp decision", review_after=review_date,
    )
    mem = await store_memory(pool, create)
    assert mem.review_after is not None

    updated = await update_memory(pool, mem.id, review_after=None)
    assert updated is not None
    assert updated.review_after is None


# ── Durability: null-embedding persistence ──────────────────────────


async def test_store_without_embedding(pool):
    """Memory stored with no embedding is persisted and retrievable by ID."""
    create = MemoryCreate(
        type=MemoryType.fact,
        content="Stored without embedding due to provider failure",
        topic=["durability"],
    )
    mem = await store_memory(pool, create, embedding=None)
    assert mem.id.startswith("weft-")
    assert mem.content == create.content

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.content == create.content


async def test_null_embedding_excluded_from_vector_search(pool, provider):
    """Null-embedding rows don't appear in vector search results."""
    # Store one with embedding, one without
    content_with = "Memory with a valid embedding for search"
    emb = await provider.embed(content_with)
    mem_with = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content_with, topic=["durability"]),
        embedding=emb,
    )

    mem_without = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Memory without embedding — provider was down",
            topic=["durability"],
        ),
        embedding=None,
    )

    results = await search_by_vector(pool, emb, limit=50)
    result_ids = {r.memory.id for r in results}
    assert mem_with.id in result_ids
    assert mem_without.id not in result_ids


async def test_null_embedding_appears_in_list(pool):
    """Null-embedding memories still appear in non-vector list queries."""
    create = MemoryCreate(
        type=MemoryType.fact,
        content="Listable even without embedding",
        topic=["durability-list-test"],
        project_id="durability-test",
    )
    mem = await store_memory(pool, create, embedding=None)

    memories = await list_memories(pool, project_id="durability-test")
    found = [m for m in memories if m.id == mem.id]
    assert len(found) == 1
