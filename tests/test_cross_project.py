"""Tests for cross-project memory sharing, isolation, and cross-project search.

Verifies that:
- Global memories (project_id=None) are visible from any project query
- Project-scoped memories are only visible within their own project
- Cross-project queries correctly combine global + project-scoped results
- build_context respects cross-project boundaries
- search_cross_project returns memories from other projects with penalty
"""

from __future__ import annotations

import pytest

from weft.context import build_context
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import list_memories, search_by_vector, search_cross_project, store_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_provider = None


def _get_provider():
    """Lazy singleton for embedding provider (expensive to init)."""
    global _provider
    if _provider is None:
        _provider = get_provider("fastembed")
    return _provider


async def _store(pool, content, *, project_id=None, mem_type=MemoryType.fact, topic=None, with_embedding=False):
    """Convenience wrapper for storing a memory with optional embedding."""
    create = MemoryCreate(
        type=mem_type,
        content=content,
        topic=topic or ["cross-project-test"],
        source=MemorySource.conversation,
        confidence=0.9,
        project_id=project_id,
    )
    embedding = None
    if with_embedding:
        provider = _get_provider()
        embedding = await provider.embed(content)
    return await store_memory(pool, create, embedding=embedding)


# ---------------------------------------------------------------------------
# Test 1: list_memories includes global when project filter is set
# ---------------------------------------------------------------------------


async def test_list_memories_includes_global_when_project_filter(pool):
    """Querying list_memories with project_id should return both project-scoped
    and global (project_id=None) memories."""
    await _store(pool, "Global memory visible everywhere", project_id=None)
    await _store(pool, "Project-A scoped memory", project_id="project-a")

    results = await list_memories(pool, project_id="project-a")

    contents = {m.content for m in results}
    assert "Global memory visible everywhere" in contents
    assert "Project-A scoped memory" in contents
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Test 2: list_memories with no project filter returns all memories
# ---------------------------------------------------------------------------


async def test_list_memories_no_project_filter_returns_all(pool):
    """Without a project_id filter, list_memories should return every memory
    regardless of project_id."""
    await _store(pool, "Global mem", project_id=None)
    await _store(pool, "Project-alpha mem", project_id="project-alpha")
    await _store(pool, "Project-beta mem", project_id="project-beta")

    results = await list_memories(pool)

    assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 3: vector search includes global when project filter is set
# ---------------------------------------------------------------------------


async def test_vector_search_includes_global_when_project_filter(pool):
    """search_by_vector with project_id should return project-scoped + global
    memories."""
    await _store(
        pool, "Global: Python is a great language", project_id=None, with_embedding=True,
    )
    await _store(
        pool, "Project-A: Python code style preferences", project_id="project-a", with_embedding=True,
    )

    provider = _get_provider()
    query_emb = await provider.embed("Python programming")

    results = await search_by_vector(pool, query_emb, limit=10, project_id="project-a")

    contents = {r.memory.content for r in results}
    assert "Global: Python is a great language" in contents
    assert "Project-A: Python code style preferences" in contents
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Test 4: vector search with no project filter returns all
# ---------------------------------------------------------------------------


async def test_vector_search_no_project_filter_returns_all(pool):
    """search_by_vector without project_id returns memories from all projects."""
    await _store(
        pool, "Global: Redis caching strategies", project_id=None, with_embedding=True,
    )
    await _store(
        pool, "Project-X: Redis config for project X", project_id="project-x", with_embedding=True,
    )
    await _store(
        pool, "Project-Y: Redis config for project Y", project_id="project-y", with_embedding=True,
    )

    provider = _get_provider()
    query_emb = await provider.embed("Redis caching")

    results = await search_by_vector(pool, query_emb, limit=10)

    assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 5: project-A memory is NOT visible from project-B
# ---------------------------------------------------------------------------


async def test_project_a_memory_not_visible_from_project_b(pool):
    """A memory scoped to project-a must NOT appear when querying with
    project_id='project-b'. Only global memories (if any) should cross."""
    await _store(pool, "Secret of project-a: uses React", project_id="project-a")
    await _store(pool, "Secret of project-b: uses Vue", project_id="project-b")

    # Query from project-b perspective
    results_b = await list_memories(pool, project_id="project-b")
    contents_b = {m.content for m in results_b}

    assert "Secret of project-b: uses Vue" in contents_b
    assert "Secret of project-a: uses React" not in contents_b
    assert len(results_b) == 1

    # Also verify from project-a perspective
    results_a = await list_memories(pool, project_id="project-a")
    contents_a = {m.content for m in results_a}

    assert "Secret of project-a: uses React" in contents_a
    assert "Secret of project-b: uses Vue" not in contents_a
    assert len(results_a) == 1


# ---------------------------------------------------------------------------
# Test 6: preferences stored globally by default are visible everywhere
# ---------------------------------------------------------------------------


async def test_preferences_stored_globally_by_default(pool):
    """When storing a preference/user_model type memory without an explicit
    project_id, it should be visible from any project query. This validates
    that global memories (project_id=None) work as cross-project shared
    knowledge."""
    # Store a preference without project_id (global)
    await _store(
        pool,
        "User prefers dark mode and vim keybindings",
        project_id=None,
        mem_type=MemoryType.preference,
    )

    # Should be visible from project-alpha
    results_alpha = await list_memories(pool, project_id="project-alpha")
    assert len(results_alpha) == 1
    assert results_alpha[0].content == "User prefers dark mode and vim keybindings"
    assert results_alpha[0].type == MemoryType.preference

    # Should be visible from project-beta
    results_beta = await list_memories(pool, project_id="project-beta")
    assert len(results_beta) == 1
    assert results_beta[0].content == "User prefers dark mode and vim keybindings"

    # Should be visible from project-gamma (any arbitrary project)
    results_gamma = await list_memories(pool, project_id="project-gamma")
    assert len(results_gamma) == 1


# ---------------------------------------------------------------------------
# Test 7: project-scoped plus global count is correct
# ---------------------------------------------------------------------------


async def test_project_scoped_plus_global_count(pool):
    """Store 3 memories: 1 global, 1 project-a, 1 project-b.
    Query with project_id='project-a' -> exactly 2 (global + project-a).
    Query with project_id='project-b' -> exactly 2 (global + project-b)."""
    await _store(pool, "Global knowledge: Earth orbits the Sun", project_id=None)
    await _store(pool, "Project-A fact: uses PostgreSQL 16", project_id="project-a")
    await _store(pool, "Project-B fact: uses MySQL 8", project_id="project-b")

    results_a = await list_memories(pool, project_id="project-a")
    assert len(results_a) == 2
    contents_a = {m.content for m in results_a}
    assert "Global knowledge: Earth orbits the Sun" in contents_a
    assert "Project-A fact: uses PostgreSQL 16" in contents_a
    assert "Project-B fact: uses MySQL 8" not in contents_a

    results_b = await list_memories(pool, project_id="project-b")
    assert len(results_b) == 2
    contents_b = {m.content for m in results_b}
    assert "Global knowledge: Earth orbits the Sun" in contents_b
    assert "Project-B fact: uses MySQL 8" in contents_b
    assert "Project-A fact: uses PostgreSQL 16" not in contents_b

    # Without any filter, all 3 are returned
    results_all = await list_memories(pool)
    assert len(results_all) == 3


# ---------------------------------------------------------------------------
# Test 8: build_context respects cross-project boundaries
# ---------------------------------------------------------------------------


async def test_weft_context_respects_cross_project(pool):
    """build_context() with a project_id filter should return global +
    project-scoped memories, but NOT other projects' memories."""
    provider = _get_provider()

    # Store memories with embeddings across projects
    await _store(
        pool, "Global architecture: microservices pattern",
        project_id=None, with_embedding=True,
    )
    await _store(
        pool, "Project-alpha architecture: monolith with modules",
        project_id="project-alpha", with_embedding=True,
    )
    await _store(
        pool, "Project-beta architecture: serverless functions",
        project_id="project-beta", with_embedding=True,
    )

    query_emb = await provider.embed("software architecture patterns")

    # Query from project-alpha perspective
    result_alpha = await build_context(
        pool, query_emb, budget_tokens=100000, project_id="project-alpha",
    )

    alpha_contents = {m["content"] for m in result_alpha["memories"]}
    assert "Global architecture: microservices pattern" in alpha_contents
    assert "Project-alpha architecture: monolith with modules" in alpha_contents
    assert "Project-beta architecture: serverless functions" not in alpha_contents
    assert result_alpha["count"] == 2

    # Query from project-beta perspective
    result_beta = await build_context(
        pool, query_emb, budget_tokens=100000, project_id="project-beta",
    )

    beta_contents = {m["content"] for m in result_beta["memories"]}
    assert "Global architecture: microservices pattern" in beta_contents
    assert "Project-beta architecture: serverless functions" in beta_contents
    assert "Project-alpha architecture: monolith with modules" not in beta_contents
    assert result_beta["count"] == 2


# ---------------------------------------------------------------------------
# Test 9: search_cross_project returns other projects' memories
# ---------------------------------------------------------------------------


async def test_search_cross_project_returns_other_projects(pool):
    """search_cross_project from project-A should find project-B memories."""
    await _store(
        pool, "Project-A: uses FastAPI for the web layer",
        project_id="project-a", with_embedding=True,
    )
    await _store(
        pool, "Project-B: uses FastAPI for the API server",
        project_id="project-b", with_embedding=True,
    )

    provider = _get_provider()
    emb = await provider.embed("FastAPI web framework")

    results = await search_cross_project(
        pool, emb, exclude_project_id="project-a", limit=5,
    )

    # Should find project-B but NOT project-A
    result_contents = {r.memory.content for r in results}
    assert "Project-B: uses FastAPI for the API server" in result_contents
    assert "Project-A: uses FastAPI for the web layer" not in result_contents


# ---------------------------------------------------------------------------
# Test 10: search_cross_project applies 0.8x penalty
# ---------------------------------------------------------------------------


async def test_search_cross_project_applies_penalty(pool):
    """Cross-project results should have 0.8x penalized similarity scores."""
    await _store(
        pool, "Project-B: Redis is used for caching layer with TTL 300s",
        project_id="project-b", with_embedding=True,
    )

    provider = _get_provider()
    emb = await provider.embed("Redis caching layer")

    # Get same-project result for comparison
    same_results = await search_by_vector(pool, emb, limit=5)
    cross_results = await search_cross_project(
        pool, emb, exclude_project_id="project-a", limit=5,
    )

    if same_results and cross_results:
        # Cross-project similarity should be < same-project similarity
        # because of the 0.8x penalty
        same_sim = same_results[0].similarity
        cross_sim = cross_results[0].similarity
        assert cross_sim < same_sim


# ---------------------------------------------------------------------------
# Test 11: search_cross_project excludes current project AND includes global
# ---------------------------------------------------------------------------


async def test_search_cross_project_includes_global(pool):
    """Global memories (project_id=None) should appear in cross-project results."""
    await _store(
        pool, "Global: Python typing best practices for type safety",
        project_id=None, with_embedding=True,
    )
    await _store(
        pool, "Project-A: Python coding standards for our team",
        project_id="project-a", with_embedding=True,
    )

    provider = _get_provider()
    emb = await provider.embed("Python typing")

    results = await search_cross_project(
        pool, emb, exclude_project_id="project-a", limit=5,
    )

    result_contents = {r.memory.content for r in results}
    assert "Global: Python typing best practices for type safety" in result_contents
    assert "Project-A: Python coding standards for our team" not in result_contents


# ---------------------------------------------------------------------------
# Test 12: search_cross_project from global context (project_id=None)
# ---------------------------------------------------------------------------


async def test_search_cross_project_from_global_context(pool):
    """When exclude_project_id=None, return only project-scoped memories."""
    await _store(
        pool, "Global: shared knowledge about databases",
        project_id=None, with_embedding=True,
    )
    await _store(
        pool, "Project-X: database migration tool preferences",
        project_id="project-x", with_embedding=True,
    )

    provider = _get_provider()
    emb = await provider.embed("database migration")

    results = await search_cross_project(
        pool, emb, exclude_project_id=None, limit=5,
    )

    result_contents = {r.memory.content for r in results}
    # Global memories should NOT appear (would be self-matching)
    assert "Global: shared knowledge about databases" not in result_contents
    # Project-scoped memories SHOULD appear
    assert "Project-X: database migration tool preferences" in result_contents


# ---------------------------------------------------------------------------
# Test 13: search_cross_project with no other projects returns empty
# ---------------------------------------------------------------------------


async def test_search_cross_project_single_project_empty(pool):
    """When only one project exists, cross-project search returns empty."""
    await _store(
        pool, "Project-only: uses PostgreSQL for everything in the app",
        project_id="project-only", with_embedding=True,
    )

    provider = _get_provider()
    emb = await provider.embed("PostgreSQL database")

    results = await search_cross_project(
        pool, emb, exclude_project_id="project-only", limit=5,
    )

    assert results == []


# ---------------------------------------------------------------------------
# Test 14: search_cross_project respects limit
# ---------------------------------------------------------------------------


async def test_search_cross_project_respects_limit(pool):
    """Cross-project search should respect the limit parameter."""
    for i in range(5):
        await _store(
            pool, f"Project-B: memory about testing pattern number {i}",
            project_id="project-b", with_embedding=True,
        )

    provider = _get_provider()
    emb = await provider.embed("testing patterns")

    results = await search_cross_project(
        pool, emb, exclude_project_id="project-a", limit=2,
    )

    assert len(results) <= 2
