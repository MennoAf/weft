"""Tests for L3 — Facet-boost recall: drop the project wall, rank by facet overlap.

Spec: loom-cbef102a  (parent epic: loom-1d6c8e5c)
Depends on: loom-ca3b3893 (project_facets column) + loom-a7664166 (write path)

Done-when assertions:
  1. A belief with project_facets {a, b} is returned by search_by_vector run
     under facet_boost_project_id=a AND under facet_boost_project_id=b.
  2. Two beliefs with equal cosine similarity — the one whose project_facets
     contains the current project outranks the one that does not.
  3. A global belief (project_facets='{}') still surfaces under any project.
  4. A recalled Memory/MemoryRecall carries its populated project_facets
     (proves the _row_to_memory mapping).
  5. No regression: existing recall semantics pass (project_id wall unchanged
     when facet_boost_project_id is NOT used).

Synthetic persona: Jim Boblaw (never real names from conversation).
"""

from __future__ import annotations

import math

import pytest

from weft.consolidation import init_project_facets
from weft.models import MemoryCreate, MemoryType
from weft.store import _FACET_BOOST, search_by_keyword, search_by_vector, store_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(n_dims: int = 768) -> list[float]:
    """Unit vector in dimension 0."""
    v = [0.0] * n_dims
    v[0] = 1.0
    return v


def _vec_at_similarity(base: list[float], target_sim: float) -> list[float]:
    """Unit vector with cosine similarity == target_sim to base.

    Uses Gram-Schmidt: w = target_sim*base + sqrt(1-s²)*e_perp.
    Works when base is a unit vector (which _unit_vec() always is).
    """
    n = len(base)
    w = [0.0] * n
    w[0] = target_sim
    if n > 1:
        w[1] = math.sqrt(max(0.0, 1.0 - target_sim ** 2))
    return w


async def _store(
    pool,
    content: str,
    *,
    project_id: str | None = None,
    embedding: list[float] | None = None,
    mem_type: MemoryType = MemoryType.preference,
) -> str:
    """Store a memory and return its id.  Does NOT init project_facets."""
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=mem_type,
            content=content,
            confidence=0.85,
            project_id=project_id,
        ),
        embedding=embedding,
    )
    return mem.id


async def _set_facets(pool, memory_id: str, facets: list[str]) -> None:
    """Overwrite project_facets to an explicit list (bypasses idempotency guard)."""
    await pool.execute(
        "UPDATE memories SET project_facets = $1::text[] WHERE id = $2",
        facets,
        memory_id,
    )


# ---------------------------------------------------------------------------
# Test 1: belief with project_facets {a, b} surfaces under both projects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facet_belief_surfaces_under_both_projects(pool):
    """A belief stored with project_facets={proj-a, proj-b} must be returned by
    search_by_vector when called with facet_boost_project_id=proj-a AND with
    facet_boost_project_id=proj-b.

    Jim Boblaw's belief about code review is relevant in both his weft and loom
    projects, so it should surface in either project context.
    """
    base = _unit_vec()
    content = (
        "Jim Boblaw always writes a summary comment at the top of every pull "
        "request explaining the motivation for the change."
    )
    mem_id = await _store(pool, content, project_id="proj-a", embedding=base)
    # Simulate the facet-merge write path: belief is now shared across two projects.
    await _set_facets(pool, mem_id, ["proj-a", "proj-b"])

    # Search under project a
    results_a = await search_by_vector(
        pool, base, limit=5, threshold=0.0,
        facet_boost_project_id="proj-a",
    )
    ids_a = [r.memory.id for r in results_a]
    assert mem_id in ids_a, "Faceted belief not returned for proj-a"

    # Search under project b
    results_b = await search_by_vector(
        pool, base, limit=5, threshold=0.0,
        facet_boost_project_id="proj-b",
    )
    ids_b = [r.memory.id for r in results_b]
    assert mem_id in ids_b, "Faceted belief not returned for proj-b"


# ---------------------------------------------------------------------------
# Test 2: equal-similarity beliefs — current-project facet outranks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facet_boost_outranks_equal_similarity(pool):
    """Given two beliefs at identical cosine similarity, the one whose
    project_facets contains the query project must be ranked higher.

    Jim Boblaw has two equally-relevant notes; the one tagged for the current
    project ('weft') should surface first.
    """
    base = _unit_vec()
    # Both beliefs at the same similarity to the query vector
    same_sim = 0.85
    query_emb = _vec_at_similarity(base, same_sim)

    content_in = (
        "Jim Boblaw prefers short, focused functions with a single responsibility — "
        "this is a project-specific coding style note for weft."
    )
    content_out = (
        "Jim Boblaw tracks time spent on tasks in a spreadsheet — "
        "this is a cross-project habit note."
    )

    id_in = await _store(pool, content_in, project_id="weft", embedding=base)
    id_out = await _store(pool, content_out, project_id="other", embedding=base)

    # id_in is in-project (facets = {weft}), id_out is cross-project (facets = {other})
    await _set_facets(pool, id_in, ["weft"])
    await _set_facets(pool, id_out, ["other"])

    results = await search_by_vector(
        pool, query_emb, limit=5, threshold=0.0,
        facet_boost_project_id="weft",
    )
    ids = [r.memory.id for r in results]
    assert id_in in ids, "In-project belief missing from results"
    assert id_out in ids, "Out-of-project belief missing from results"
    assert ids.index(id_in) < ids.index(id_out), (
        f"Expected in-project belief (idx {ids.index(id_in)}) to rank before "
        f"out-of-project belief (idx {ids.index(id_out)})"
    )


# ---------------------------------------------------------------------------
# Test 3: global belief (project_facets='{}') surfaces under any project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_belief_surfaces_under_any_project(pool):
    """A belief with empty project_facets (global) must still be returned when
    facet_boost_project_id is set — it just doesn't receive the boost.

    Jim Boblaw's universal preference for plain-text notes has no project tag
    but must still appear in any project's recall results.
    """
    base = _unit_vec()
    content = (
        "Jim Boblaw keeps all his personal notes in plain-text files rather "
        "than proprietary formats so they remain readable indefinitely."
    )
    mem_id = await _store(pool, content, project_id=None, embedding=base)
    # project_facets defaults to '{}' — no init_project_facets call needed.

    results = await search_by_vector(
        pool, base, limit=5, threshold=0.0,
        facet_boost_project_id="any-project",
    )
    ids = [r.memory.id for r in results]
    assert mem_id in ids, "Global belief (empty project_facets) not returned"


# ---------------------------------------------------------------------------
# Test 4: recalled MemoryRecall carries populated project_facets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recalled_memory_carries_project_facets(pool):
    """_row_to_memory must map project_facets from the DB row so that the
    returned Memory objects carry the correct list (not an always-empty default).

    This proves the ranking boost has real facet data to work with.
    """
    base = _unit_vec()
    content = (
        "Jim Boblaw documents every architectural decision in a dedicated ADR "
        "file so future team members can understand the reasoning."
    )
    mem_id = await _store(pool, content, project_id="weft", embedding=base)
    await init_project_facets(pool, mem_id, "weft")
    # Also add a second project via direct SQL
    await pool.execute(
        "UPDATE memories SET project_facets = ARRAY['weft', 'loom']::text[] WHERE id = $1",
        mem_id,
    )

    results = await search_by_vector(
        pool, base, limit=5, threshold=0.0,
        facet_boost_project_id="weft",
    )
    match = next((r for r in results if r.memory.id == mem_id), None)
    assert match is not None, "Memory not returned by search"
    assert set(match.memory.project_facets) == {"weft", "loom"}, (
        f"project_facets not populated: got {match.memory.project_facets!r}"
    )


# ---------------------------------------------------------------------------
# Test 5a: existing project_id wall still works (no facet boost param)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_wall_preserved_when_no_facet_boost(pool):
    """When facet_boost_project_id is NOT passed, search_by_vector uses the
    classic (project_id = $X OR project_id IS NULL) wall — memories from other
    projects are excluded.

    This is the regression guard: the catalog path must not be affected.
    """
    base = _unit_vec()
    content_a = "Jim Boblaw uses pytest for all Python unit tests in proj-alpha."
    content_b = "Jim Boblaw uses Jest for all JavaScript unit tests in proj-beta."

    id_a = await _store(pool, content_a, project_id="proj-alpha", embedding=base)
    id_b = await _store(pool, content_b, project_id="proj-beta", embedding=base)

    # Query scoped to proj-alpha — should see id_a, not id_b
    results = await search_by_vector(
        pool, base, limit=10, threshold=0.0,
        project_id="proj-alpha",
        # No facet_boost_project_id → hard wall is active
    )
    ids = [r.memory.id for r in results]
    assert id_a in ids, "proj-alpha memory missing from proj-alpha query"
    assert id_b not in ids, "proj-beta memory leaked into proj-alpha query"


# ---------------------------------------------------------------------------
# Test 5b: keyword search also applies facet boost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_facet_boost_outranks(pool):
    """search_by_keyword with facet_boost_project_id must boost in-project results.

    Uses two distinct content strings so BM25 gives each a nonzero rank,
    then checks that the in-project belief ranks higher.

    Jim Boblaw notes are used to ensure synthetic-persona discipline.
    """
    content_in = (
        "Jim Boblaw reviews all database migration scripts before merging "
        "them — this is a weft project convention."
    )
    content_out = (
        "Jim Boblaw reviews all database migration scripts before merging "
        "them — this is a general habit note."
    )

    id_in = await _store(pool, content_in, project_id="weft")
    id_out = await _store(pool, content_out, project_id="other")

    await _set_facets(pool, id_in, ["weft"])
    await _set_facets(pool, id_out, ["other"])

    results = await search_by_keyword(
        pool,
        "Jim Boblaw reviews database migration scripts",
        limit=10,
        facet_boost_project_id="weft",
    )
    ids = [r.memory.id for r in results]
    assert id_in in ids, "In-project belief missing from keyword results"
    assert id_out in ids, "Cross-project belief missing from keyword results"
    assert ids.index(id_in) < ids.index(id_out), (
        f"In-project belief (idx {ids.index(id_in)}) should rank above "
        f"cross-project (idx {ids.index(id_out)}) in keyword search"
    )
