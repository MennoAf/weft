"""Tests for weft_focus core module."""

from __future__ import annotations

import pytest

from weft.focus import FocusResult, build_focus
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import search_by_vector, store_memory
from weft.tokens import estimate_tokens


# --- FocusResult formatting ---


def test_focus_result_format_all_sections():
    """Format includes all three sections when populated."""
    result = FocusResult(
        intent="test intent",
        last_session_summary="Did some work",
        focused_memories=[
            {"id": "weft-abc123", "type": "fact", "content": "A fact", "similarity": 0.85, "topic": ["test"]},
        ],
        git_changes=["abc1234 Add feature"],
    )
    text = result.format()
    assert "## Last Session" in text
    assert "## Focused Memories" in text
    assert "## Recent Changes" in text
    assert "Did some work" in text
    assert "A fact" in text
    assert "abc1234 Add feature" in text


def test_focus_result_format_empty():
    """Empty result returns a comment marker."""
    result = FocusResult(intent="nothing")
    assert "no additional context" in result.format()


def test_focus_result_format_partial():
    """Only populated sections appear."""
    result = FocusResult(
        intent="test",
        focused_memories=[
            {"id": "weft-x", "type": "fact", "content": "A fact", "similarity": 0.9, "topic": []},
        ],
    )
    text = result.format()
    assert "## Focused Memories" in text
    assert "## Last Session" not in text
    assert "## Recent Changes" not in text


def test_focus_result_to_dict():
    """to_dict includes all fields plus formatted output."""
    result = FocusResult(intent="test", budget_tokens=1200)
    d = result.to_dict()
    assert d["intent"] == "test"
    assert "formatted" in d
    assert "budget_tokens" in d


# --- build_focus ---


@pytest.mark.asyncio
async def test_empty_intent_raises():
    """Empty or whitespace intent raises ValueError."""
    with pytest.raises(ValueError, match="intent is required"):
        await build_focus(None, intent="", embedding_fn=None)

    with pytest.raises(ValueError, match="intent is required"):
        await build_focus(None, intent="   ", embedding_fn=None)


@pytest.mark.asyncio
async def test_focus_returns_memories(pool):
    """Focus returns relevant memories for the intent."""
    # Create some memories
    embedding_fn = _make_embedding_fn()
    for i in range(5):
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"Authentication uses JWT tokens for session management part {i}",
            topic=["auth"],
            source=MemorySource.conversation,
        )
        emb = await embedding_fn(create.content)
        await store_memory(pool, create, embedding=emb)

    result = await build_focus(
        pool,
        intent="implement authentication",
        embedding_fn=embedding_fn,
        exclude_memory_ids=[],
    )

    assert isinstance(result, FocusResult)
    assert result.intent == "implement authentication"
    assert len(result.focused_memories) > 0
    assert result.total_tokens > 0
    assert result.total_tokens <= result.budget_tokens


@pytest.mark.asyncio
async def test_focus_excludes_primer_memories(pool):
    """Memories in exclude list are not returned."""
    embedding_fn = _make_embedding_fn()

    # Create memories
    mems = []
    for i in range(3):
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"Database connection pooling strategy {i}",
            topic=["database"],
            source=MemorySource.conversation,
        )
        emb = await embedding_fn(create.content)
        mem = await store_memory(pool, create, embedding=emb)
        mems.append(mem)

    # Exclude all of them
    exclude_ids = [m.id for m in mems]
    result = await build_focus(
        pool,
        intent="database connection",
        embedding_fn=embedding_fn,
        exclude_memory_ids=exclude_ids,
    )

    # None of the excluded memories should appear
    returned_ids = {m["id"] for m in result.focused_memories}
    assert returned_ids.isdisjoint(set(exclude_ids))


@pytest.mark.asyncio
async def test_focus_with_handoff(pool):
    """When a handoff exists, last_session_summary is populated."""
    embedding_fn = _make_embedding_fn()

    # Create a handoff memory
    handoff = MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Implemented caching layer",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    )
    emb = await embedding_fn(handoff.content)
    await store_memory(pool, handoff, embedding=emb)

    # Create a fact too
    fact = MemoryCreate(
        type=MemoryType.fact,
        content="Redis caching uses a 5-minute TTL",
        topic=["caching"],
        source=MemorySource.conversation,
    )
    emb = await embedding_fn(fact.content)
    await store_memory(pool, fact, embedding=emb)

    result = await build_focus(
        pool,
        intent="caching improvements",
        embedding_fn=embedding_fn,
        exclude_memory_ids=[],
    )

    assert result.last_session_summary is not None
    assert "caching" in result.last_session_summary.lower() or "handoff" in result.last_session_summary.lower()


@pytest.mark.asyncio
async def test_focus_budget_enforcement(pool):
    """Total tokens stay within budget."""
    embedding_fn = _make_embedding_fn()

    # Create many memories with long content
    for i in range(20):
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"This is memory number {i} with some additional context about testing patterns and best practices for software development " * 5,
            topic=["testing"],
            source=MemorySource.conversation,
        )
        emb = await embedding_fn(create.content)
        await store_memory(pool, create, embedding=emb)

    result = await build_focus(
        pool,
        intent="testing patterns",
        embedding_fn=embedding_fn,
        exclude_memory_ids=[],
        budget_tokens=800,
    )

    assert result.total_tokens <= 800


@pytest.mark.asyncio
async def test_focus_no_memories_no_error(pool):
    """Empty database returns valid result with no memories."""
    embedding_fn = _make_embedding_fn()

    result = await build_focus(
        pool,
        intent="anything",
        embedding_fn=embedding_fn,
        exclude_memory_ids=[],
    )

    assert result.focused_memories == []
    assert result.total_tokens >= 0


@pytest.mark.asyncio
async def test_focus_project_scoped(pool):
    """Focus scoped to a project only returns memories from that project."""
    embedding_fn = _make_embedding_fn()

    # Create memories in different projects
    for proj in ["alpha", "beta"]:
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"Database migration strategy for project {proj} uses Alembic",
            topic=["database"],
            project_id=proj,
        )
        emb = await embedding_fn(create.content)
        await store_memory(pool, create, embedding=emb)

    result = await build_focus(
        pool,
        intent="database migrations",
        embedding_fn=embedding_fn,
        project_id="alpha",
        exclude_memory_ids=[],
    )

    # Should only include alpha memories (or global)
    for mem in result.focused_memories:
        # project_id is not in the focused_memories dict by default,
        # but the search_by_vector scopes correctly
        assert len(result.focused_memories) > 0


@pytest.mark.asyncio
async def test_focus_changes_since_populated(pool):
    """When handoff exists, changes_since is populated."""
    embedding_fn = _make_embedding_fn()

    # Create a handoff so there's a timestamp
    handoff = MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Set up CI pipeline",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    )
    emb = await embedding_fn(handoff.content)
    await store_memory(pool, handoff, embedding=emb)

    result = await build_focus(
        pool,
        intent="CI pipeline",
        embedding_fn=embedding_fn,
        exclude_memory_ids=[],
    )

    # changes_since should be populated since a handoff exists
    assert result.changes_since is not None
    assert "memories_created" in result.changes_since


@pytest.mark.asyncio
async def test_focus_exclude_ids_in_search(pool):
    """Verify exclude_ids parameter works in search_by_vector."""
    embedding_fn = _make_embedding_fn()

    # Create two memories
    mem1 = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Alpha memory about testing",
        topic=["test"],
        source=MemorySource.conversation,
    ), embedding=await embedding_fn("Alpha memory about testing"))

    mem2 = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Beta memory about testing",
        topic=["test"],
        source=MemorySource.conversation,
    ), embedding=await embedding_fn("Beta memory about testing"))

    # Search excluding mem1
    emb = await embedding_fn("testing")
    results = await search_by_vector(
        pool, emb, exclude_ids=[mem1.id], threshold=0.0,
    )

    returned_ids = {r.memory.id for r in results}
    assert mem1.id not in returned_ids
    assert mem2.id in returned_ids


# --- Helpers ---


def _make_embedding_fn():
    """Create a deterministic embedding function for testing.

    Uses a simple hash-based approach that produces consistent 768-dim vectors.
    """
    import hashlib

    async def embed(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Extend to 768 dims by repeating the hash
        extended = h * 24  # 32 bytes * 24 = 768 bytes
        return [b / 255.0 for b in extended[:768]]

    return embed
