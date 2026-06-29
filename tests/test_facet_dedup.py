"""Tests for L2 — cross-project two-tier belief-merge write path.

Spec: loom-a7664166  (parent epic: loom-1d6c8e5c)
Depends on: loom-ca3b3893 (project_facets column exists as of migration v64)

Done-when assertions:
  1. store belief X under project=weft then semantically-identical X under
     project=loom → exactly ONE active row, project_facets == {weft, loom}
  2. re-store X under project=weft again → still ONE row, project_facets still
     {weft} (idempotent, no duplicate facet entry)
  3. a CONTRADICTORY value for the same subject does NOT auto-merge
  4. a mid-similarity near-duplicate produces a CANDIDATE, not an auto-merge

Synthetic persona: Jim Boblaw (never real names from conversation).
"""

from __future__ import annotations

import math

import pytest

from weft.consolidation import (
    _FACET_AUTO_MERGE_THRESHOLD,
    _FACET_CANDIDATE_THRESHOLD,
    check_dedup_on_store,
    init_project_facets,
)
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.store import store_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_provider = None


async def _embed(text: str) -> list[float]:
    global _provider
    if _provider is None:
        _provider = get_provider("fastembed")
    return await _provider.embed(text)


async def _store(
    pool,
    content: str,
    *,
    project_id: str,
    mem_type: MemoryType = MemoryType.preference,
    confidence: float = 0.8,
    embedding: list[float] | None = None,
) -> tuple[str, list[float]]:
    """Store a memory and init its project_facets. Returns (id, embedding)."""
    emb = embedding if embedding is not None else await _embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=mem_type,
            content=content,
            confidence=confidence,
            project_id=project_id,
        ),
        embedding=emb,
    )
    await init_project_facets(pool, mem.id, project_id)
    return mem.id, emb


def _make_unit_embedding(n_dims: int = 768) -> list[float]:
    """Create a unit vector in dimension 0 (for synthetic similarity tests)."""
    v = [0.0] * n_dims
    v[0] = 1.0
    return v


def _make_embedding_at_similarity(base: list[float], target_sim: float) -> list[float]:
    """Create a unit vector with cosine similarity = target_sim to base.

    Uses Gram–Schmidt: w = target_sim * base + sqrt(1-target_sim²) * e_perp
    where e_perp is the unit vector in dim 1 (orthogonal to dim-0 base).
    Cosine(base, w) = target_sim exactly when both are unit vectors.
    """
    n = len(base)
    w = [0.0] * n
    w[0] = target_sim
    if n > 1:
        w[1] = math.sqrt(max(0.0, 1.0 - target_sim ** 2))
    return w


# ---------------------------------------------------------------------------
# Test 1: cross-project identical content → ONE row, project_facets={weft,loom}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_project_identical_collapses_to_one_row(pool):
    """Storing semantically-identical belief X under weft then loom yields
    exactly ONE active row whose project_facets contains both projects.

    Done-when assertion 1.
    """
    content = (
        "Jim Boblaw keeps detailed notes on every software project he works on, "
        "preferring Markdown for all his documentation"
    )

    # Store under weft
    mem_id, emb = await _store(pool, content, project_id="weft")

    # Store the same content under loom
    result = await check_dedup_on_store(
        pool,
        content,
        emb,
        new_confidence=0.8,
        memory_type=MemoryType.preference,
        project_id="loom",
    )

    # Must be an auto-merge (facet_appended), no new row inserted
    assert result.is_duplicate is True, (
        f"Expected is_duplicate=True for cross-project identical content, "
        f"got action={result.action!r}"
    )
    assert result.action == "facet_appended", (
        f"Expected action='facet_appended', got {result.action!r}"
    )
    assert result.existing_memory is not None
    assert result.existing_memory.id == mem_id

    # Verify DB: exactly ONE active row
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE status = 'active'"
    )
    assert count == 1, f"Expected exactly 1 active row, found {count}"

    # Verify DB: project_facets contains both projects
    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1", mem_id
    )
    facets = set(row["project_facets"])
    assert facets == {"weft", "loom"}, (
        f"Expected project_facets={{weft, loom}}, got {facets}"
    )


# ---------------------------------------------------------------------------
# Test 2: idempotent re-store under same project → ONE row, facets unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_project_restore_is_idempotent(pool):
    """Re-storing the same belief under the same project does not add a
    duplicate facet entry.

    Done-when assertion 2.
    """
    content = (
        "Jim Boblaw uses Git for version control on all his personal and "
        "professional software projects without exception"
    )

    # Store under weft once
    mem_id, emb = await _store(pool, content, project_id="weft")

    # Re-store the same content under weft — should be deduplicated (same scope)
    result = await check_dedup_on_store(
        pool,
        content,
        emb,
        new_confidence=0.8,
        project_id="weft",
    )

    # Same-scope dedup: is_duplicate=True, action in ("deduplicated", "revised")
    assert result.is_duplicate is True, (
        "Re-storing identical content under the same project must be deduplicated"
    )
    assert result.action in ("deduplicated", "revised"), (
        f"Expected deduplicated or revised for same-scope re-store, got {result.action!r}"
    )

    # Verify DB: still exactly ONE active row
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE status = 'active'"
    )
    assert count == 1, f"Re-store should not add a second row; found {count}"

    # Verify DB: project_facets still only contains {weft}
    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1", mem_id
    )
    facets = list(row["project_facets"])
    assert facets == ["weft"], (
        f"Re-store must not add a duplicate facet; expected ['weft'], got {facets}"
    )


# ---------------------------------------------------------------------------
# Test 3: contradictory cross-project content does NOT auto-merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contradictory_cross_project_blocked_by_contradiction_gate(pool):
    """A contradictory belief for the same subject must NOT be auto-merged even
    when cosine similarity exceeds the AUTO_MERGE threshold.

    Done-when assertion 3.
    """
    content_a = (
        "Jim Boblaw always uses spaces for indentation in all his Python source code"
    )
    content_b = (
        "Jim Boblaw does not use spaces for indentation in his Python source code"
    )

    emb_a = await _embed(content_a)
    emb_b = await _embed(content_b)

    # Store content_a under weft
    mem_id, _ = await _store(pool, content_a, project_id="weft", embedding=emb_a)

    # Try to cross-project merge contradictory content_b (from loom)
    result = await check_dedup_on_store(
        pool,
        content_b,
        emb_b,
        new_confidence=0.8,
        project_id="loom",
    )

    # Contradiction gate must block the auto-merge
    assert result.is_duplicate is False, (
        "Contradictory content must NOT auto-merge (is_duplicate must be False)"
    )

    # Verify DB: the original memory's project_facets was NOT modified
    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1", mem_id
    )
    facets = list(row["project_facets"])
    assert "loom" not in facets, (
        f"Contradictory merge must not append 'loom' to facets; got {facets}"
    )


# ---------------------------------------------------------------------------
# Test 4: mid-similarity cross-project match → CANDIDATE, not auto-merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_similarity_cross_project_produces_candidate(pool):
    """A cross-project match in the mid-tier (0.6 ≤ sim < 0.85) must produce
    a merge_candidate signal, not an auto-merge.

    Uses synthetic unit-vector embeddings to precisely control cosine similarity.
    Done-when assertion 4.
    """
    n_dims = 768  # matches FastEmbed BAAI/bge-small-en-v1.5 (padded)
    sim_target = 0.75  # between _FACET_CANDIDATE_THRESHOLD and _FACET_AUTO_MERGE_THRESHOLD

    assert _FACET_CANDIDATE_THRESHOLD <= sim_target < _FACET_AUTO_MERGE_THRESHOLD, (
        "Test invariant: target similarity must be in mid-tier range"
    )

    base_emb = _make_unit_embedding(n_dims)                          # [1, 0, 0, ...]
    query_emb = _make_embedding_at_similarity(base_emb, sim_target)  # cos_sim = 0.75

    content_existing = (
        "Jim Boblaw manages software development tasks using project management tools"
    )
    content_new = (
        "Jim Boblaw tracks work items and milestones for his development projects"
    )

    # Store existing memory under weft with the base embedding
    mem_id, _ = await _store(
        pool, content_existing, project_id="weft", embedding=base_emb,
    )

    # Check dedup from loom with the query embedding (sim=0.75 to base)
    result = await check_dedup_on_store(
        pool,
        content_new,
        query_emb,
        new_confidence=0.7,
        project_id="loom",
    )

    assert result.is_duplicate is False, (
        "Mid-tier cross-project match must NOT auto-merge (is_duplicate must be False)"
    )
    assert result.action == "merge_candidate", (
        f"Expected action='merge_candidate', got {result.action!r}. "
        f"Similarity was {result.similarity:.3f} (target {sim_target})"
    )
    assert result.existing_memory is not None, "merge_candidate should carry the existing memory"
    assert result.existing_memory.id == mem_id

    # Verify DB: project_facets was NOT modified (no auto-merge happened)
    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1", mem_id
    )
    facets = list(row["project_facets"])
    assert "loom" not in facets, (
        f"Mid-tier candidate must not append 'loom' to facets; got {facets}"
    )

    # Verify similarity is in the expected range
    assert _FACET_CANDIDATE_THRESHOLD <= result.similarity < _FACET_AUTO_MERGE_THRESHOLD, (
        f"Similarity {result.similarity:.3f} should be in mid-tier "
        f"[{_FACET_CANDIDATE_THRESHOLD}, {_FACET_AUTO_MERGE_THRESHOLD})"
    )
