"""Tests for cross-project memory visibility.

When querying with a specific project_id, global memories (project_id IS NULL)
must also be returned alongside project-scoped memories.
"""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemoryType
from weft.store import list_memories, search_by_vector, store_memory


async def test_list_memories_includes_global_when_project_filter(pool):
    """list_memories(project_id='proj-a') returns both proj-a AND global memories."""
    # Create a global memory (no project_id)
    global_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.preference, content="I prefer dark mode", topic=["prefs"]),
    )
    assert global_mem.project_id is None

    # Create a project-scoped memory
    proj_mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Project A uses FastAPI",
            topic=["arch"],
            project_id="proj-a",
        ),
    )

    # Create a memory for a different project
    other_mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Project B uses Django",
            topic=["arch"],
            project_id="proj-b",
        ),
    )

    # Query for proj-a should include both global_mem and proj_mem but NOT other_mem
    results = await list_memories(pool, project_id="proj-a")
    result_ids = {m.id for m in results}
    assert global_mem.id in result_ids, "Global memory should be visible to project queries"
    assert proj_mem.id in result_ids, "Project-scoped memory should be visible"
    assert other_mem.id not in result_ids, "Other project memory should NOT be visible"


async def test_list_memories_no_project_filter_returns_all(pool):
    """list_memories() with no project_id filter returns ALL memories."""
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="global fact"),
    )
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="proj fact", project_id="proj-x"),
    )
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="another proj fact", project_id="proj-y"),
    )

    results = await list_memories(pool)
    assert len(results) == 3


async def test_vector_search_includes_global_when_project_filter(pool):
    """search_by_vector(project_id='proj-a') returns both proj-a AND global memories."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    # Global memory
    global_emb = await provider.embed("user prefers dark mode themes")
    global_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.preference, content="User prefers dark mode", topic=["prefs"]),
        embedding=global_emb,
    )

    # Project-scoped memory
    proj_emb = await provider.embed("project A uses PostgreSQL database")
    proj_mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Project A uses PostgreSQL",
            topic=["db"],
            project_id="proj-a",
        ),
        embedding=proj_emb,
    )

    # Other project memory
    other_emb = await provider.embed("project B uses MongoDB database")
    other_mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Project B uses MongoDB",
            topic=["db"],
            project_id="proj-b",
        ),
        embedding=other_emb,
    )

    # Search with project_id filter
    query_emb = await provider.embed("database preferences")
    results = await search_by_vector(pool, query_emb, limit=10, project_id="proj-a")
    result_ids = {r.memory.id for r in results}

    assert global_mem.id in result_ids, "Global memory should appear in project-filtered vector search"
    assert proj_mem.id in result_ids, "Project-scoped memory should appear"
    assert other_mem.id not in result_ids, "Other project memory should NOT appear"


async def test_vector_search_no_project_filter_returns_all(pool):
    """search_by_vector() with no project_id returns memories from all projects."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")

    emb1 = await provider.embed("PostgreSQL database")
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="global db fact"),
        embedding=emb1,
    )
    emb2 = await provider.embed("Redis cache")
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="proj db fact", project_id="proj-x"),
        embedding=emb2,
    )

    query_emb = await provider.embed("database systems")
    results = await search_by_vector(pool, query_emb, limit=10)
    assert len(results) == 2
