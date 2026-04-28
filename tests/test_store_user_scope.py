"""Tests for memory user_id OR-NULL scoping in list and search functions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weft.db.connection import get_db
from weft.models import MemoryCreate, MemorySource, MemoryType, _weft_id
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.store import (
    list_memories,
    search_by_keyword,
    search_by_vector,
    search_hybrid,
)


def _normalized_embedding(seed: float = 0.1) -> list[float]:
    """Generate a normalized 768-dim embedding (unit vector)."""
    import math

    raw = [math.sin(seed * (i + 1)) for i in range(768)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


async def _insert_memory(
    pool,
    content: str,
    user_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    embedding: list[float] | None = None,
) -> str:
    """Helper to insert a memory with explicit user_id (bypasses current_setting).

    Caller-side ``user_id=None`` means "global row" — stored as the
    SYSTEM_GLOBAL_USER_ID sentinel under the post-mig-36 schema.
    """
    if user_id is None:
        user_id = SYSTEM_GLOBAL_USER_ID
    memory_id = _weft_id()
    now = datetime.now(timezone.utc)
    await get_db(pool).execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, agent_id, embedding, status, pinned,
            review_after, user_id, search_tsv
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $8, $8,
            0, $9, $10, $11::vector, 'active', $12,
            $13, $14,
            to_tsvector('english', $4)
        )
        """,
        memory_id,
        MemoryType.fact.value,
        ["test"],
        content,
        MemorySource.conversation.value,
        0.7,  # confidence
        len(content),  # token_count estimate
        now,
        project_id,
        agent_id,
        embedding,
        False,  # pinned
        None,  # review_after
        user_id,
    )
    return memory_id


# --- list_memories with user_id ---


async def test_list_memories_user_id_none_returns_all(pool):
    """user_id=None should return all rows (current behavior preserved)."""
    await _insert_memory(pool, "user-a owned", user_id="user-a")
    await _insert_memory(pool, "user-b owned", user_id="user-b")
    await _insert_memory(pool, "global")

    results = await list_memories(pool, user_id=None)
    contents = {m.content for m in results}
    assert len(results) == 3
    assert "user-a owned" in contents
    assert "user-b owned" in contents
    assert "global" in contents


async def test_list_memories_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows, excluding user-b."""
    await _insert_memory(pool, "user-a owned", user_id="user-a")
    await _insert_memory(pool, "user-b owned", user_id="user-b")
    await _insert_memory(pool, "global")

    results = await list_memories(pool, user_id="user-a")
    contents = {m.content for m in results}
    assert len(results) == 2
    assert "user-a owned" in contents
    assert "global" in contents
    assert "user-b owned" not in contents


async def test_list_memories_user_id_with_only_null(pool):
    """user_id='user-a' with only NULL rows should return those NULL rows."""
    await _insert_memory(pool, "global 1")
    await _insert_memory(pool, "global 2")

    results = await list_memories(pool, user_id="user-a")
    contents = {m.content for m in results}
    assert len(results) == 2
    assert "global 1" in contents
    assert "global 2" in contents


async def test_list_memories_user_id_with_mixed_filters(pool):
    """user_id filter should work alongside other filters (project_id, agent_id, etc)."""
    await _insert_memory(pool, "user-a proj-1", user_id="user-a", project_id="proj-1")
    await _insert_memory(pool, "user-a proj-2", user_id="user-a", project_id="proj-2")
    await _insert_memory(pool, "user-b proj-1", user_id="user-b", project_id="proj-1")
    await _insert_memory(pool, "global proj-1", project_id="proj-1")

    # user_id="user-a" AND project_id="proj-1"
    results = await list_memories(pool, user_id="user-a", project_id="proj-1")
    contents = {m.content for m in results}
    assert "user-a proj-1" in contents
    assert "global proj-1" in contents
    assert "user-a proj-2" not in contents
    assert "user-b proj-1" not in contents


# --- search_by_vector with user_id ---


async def test_search_by_vector_user_id_none_returns_all(pool):
    """user_id=None should return all rows from vector search."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "user-a result", user_id="user-a", embedding=emb)
    await _insert_memory(pool, "user-b result", user_id="user-b", embedding=emb)
    await _insert_memory(pool, "global result", embedding=emb)

    results = await search_by_vector(pool, emb, threshold=0.0, user_id=None)
    contents = {r.memory.content for r in results}
    assert len(results) == 3
    assert "user-a result" in contents
    assert "user-b result" in contents
    assert "global result" in contents


async def test_search_by_vector_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows from vector search."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "user-a result", user_id="user-a", embedding=emb)
    await _insert_memory(pool, "user-b result", user_id="user-b", embedding=emb)
    await _insert_memory(pool, "global result", embedding=emb)

    results = await search_by_vector(pool, emb, threshold=0.0, user_id="user-a")
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "user-a result" in contents
    assert "global result" in contents
    assert "user-b result" not in contents


async def test_search_by_vector_user_id_with_only_null(pool):
    """user_id='user-a' with only NULL rows should return those NULL rows."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "global result 1", embedding=emb)
    await _insert_memory(pool, "global result 2", embedding=emb)

    results = await search_by_vector(pool, emb, threshold=0.0, user_id="user-a")
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "global result 1" in contents
    assert "global result 2" in contents


# --- search_by_keyword with user_id ---


async def test_search_by_keyword_user_id_none_returns_all(pool):
    """user_id=None should return all rows from keyword search."""
    await _insert_memory(pool, "user-a test query", user_id="user-a")
    await _insert_memory(pool, "user-b test query", user_id="user-b")
    await _insert_memory(pool, "global test query")

    results = await search_by_keyword(pool, "test", user_id=None)
    contents = {r.memory.content for r in results}
    assert len(results) == 3
    assert "user-a test query" in contents
    assert "user-b test query" in contents
    assert "global test query" in contents


async def test_search_by_keyword_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows from keyword search."""
    await _insert_memory(pool, "user-a test query", user_id="user-a")
    await _insert_memory(pool, "user-b test query", user_id="user-b")
    await _insert_memory(pool, "global test query")

    results = await search_by_keyword(pool, "test", user_id="user-a")
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "user-a test query" in contents
    assert "global test query" in contents
    assert "user-b test query" not in contents


async def test_search_by_keyword_user_id_with_only_null(pool):
    """user_id='user-a' with only NULL rows should return those NULL rows."""
    await _insert_memory(pool, "global test query 1")
    await _insert_memory(pool, "global test query 2")

    results = await search_by_keyword(pool, "test", user_id="user-a")
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "global test query 1" in contents
    assert "global test query 2" in contents


# --- search_hybrid with user_id ---


async def test_search_hybrid_user_id_none_returns_all(pool):
    """user_id=None should return all rows from hybrid search."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "user-a test hybrid", user_id="user-a", embedding=emb)
    await _insert_memory(pool, "user-b test hybrid", user_id="user-b", embedding=emb)
    await _insert_memory(pool, "global test hybrid", embedding=emb)

    results = await search_hybrid(
        pool, "test", emb, threshold=0.0, user_id=None
    )
    contents = {r.memory.content for r in results}
    assert len(results) == 3
    assert "user-a test hybrid" in contents
    assert "user-b test hybrid" in contents
    assert "global test hybrid" in contents


async def test_search_hybrid_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows from hybrid search."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "user-a test hybrid", user_id="user-a", embedding=emb)
    await _insert_memory(pool, "user-b test hybrid", user_id="user-b", embedding=emb)
    await _insert_memory(pool, "global test hybrid", embedding=emb)

    results = await search_hybrid(
        pool, "test", emb, threshold=0.0, user_id="user-a"
    )
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "user-a test hybrid" in contents
    assert "global test hybrid" in contents
    assert "user-b test hybrid" not in contents


async def test_search_hybrid_user_id_with_only_null(pool):
    """user_id='user-a' with only NULL rows should return those NULL rows."""
    emb = _normalized_embedding(0.1)

    await _insert_memory(pool, "global test hybrid 1", embedding=emb)
    await _insert_memory(pool, "global test hybrid 2", embedding=emb)

    results = await search_hybrid(
        pool, "test", emb, threshold=0.0, user_id="user-a"
    )
    contents = {r.memory.content for r in results}
    assert len(results) == 2
    assert "global test hybrid 1" in contents
    assert "global test hybrid 2" in contents
