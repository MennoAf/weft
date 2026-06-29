"""Cross-project merge-candidate review surface (loom-c82bd8d8).

L2 routes mid-similarity (0.6<=sim<0.85) cross-project near-duplicates to
review_status='pending_review' AND links them to the existing belief they
would merge into via a `merge_candidate` edge. This test pins that the
candidate is surfaced (not orphaned) and that the merge action appends the
candidate's facet to the target — the acceptance for loom-c82bd8d8.

Exercises the SAME helpers the weft_remember tool path calls
(quarantine.mark_merge_candidate / merge_pending), so the production linking
code is under test, not a re-implementation of it.

Synthetic persona: Jim Boblaw (never real names from conversation).
"""

from __future__ import annotations

import pytest

from weft.consolidation import init_project_facets
from weft.models import MemoryCreate, MemoryType
from weft.quarantine import (
    approve_pending,
    list_pending,
    mark_merge_candidate,
    merge_pending,
)
from weft.store import store_memory


async def _store(pool, content: str, *, project_id: str, confidence: float = 0.8) -> str:
    """Store a belief and seed its project_facets to [project_id]. Returns id."""
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.preference,
            content=content,
            confidence=confidence,
            project_id=project_id,
        ),
    )
    await init_project_facets(pool, mem.id, project_id)
    return mem.id


async def _make_candidate(pool) -> tuple[str, str]:
    """Set up a realistic merge-candidate pair the way the L2 tool path does.

    Returns (candidate_id, target_id). Target lives under 'weft', candidate
    under 'loom'; candidate is flagged pending_review + linked to target.
    """
    target_id = await _store(
        pool,
        "Jim Boblaw prefers Markdown for all project documentation",
        project_id="weft",
        confidence=0.7,
    )
    candidate_id = await _store(
        pool,
        "Jim Boblaw likes writing his project docs in Markdown format",
        project_id="loom",
        confidence=0.9,
    )
    await mark_merge_candidate(pool, candidate_id, target_id)
    return candidate_id, target_id


# ---------------------------------------------------------------------------
# A merge candidate is SURFACED for review, tagged with its target.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_candidate_listed_with_target(pool):
    candidate_id, target_id = await _make_candidate(pool)

    # A plain injection-quarantine pending row (no merge edge) for contrast.
    plain_id = await _store(
        pool, "Some unrelated flagged note", project_id="weft",
    )
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1",
        plain_id,
    )

    items = await list_pending(pool)
    by_id = {it["id"]: it for it in items}

    assert candidate_id in by_id, "merge candidate must appear in the review queue"
    assert by_id[candidate_id]["merge_target_id"] == target_id, (
        "merge candidate must be tagged with the belief it would merge into"
    )
    # Plain quarantine rows are surfaced too, but carry no merge target.
    assert plain_id in by_id
    assert by_id[plain_id]["merge_target_id"] is None


# ---------------------------------------------------------------------------
# Merge appends the candidate's facet to the target, archives the candidate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_appends_facet_and_archives_candidate(pool):
    candidate_id, target_id = await _make_candidate(pool)

    result = await merge_pending(pool, candidate_id)

    assert result is not None
    assert result["target_id"] == target_id
    assert set(result["target_project_facets"]) == {"weft", "loom"}

    # Target now carries both facets and the stronger confidence (0.9 > 0.7).
    target = await pool.fetchrow(
        "SELECT project_facets, confidence, status FROM memories WHERE id = $1",
        target_id,
    )
    assert set(target["project_facets"]) == {"weft", "loom"}
    assert target["confidence"] == pytest.approx(0.9)
    assert target["status"] == "active"

    # Candidate is archived and dropped from the review queue.
    cand = await pool.fetchrow(
        "SELECT status, review_status FROM memories WHERE id = $1", candidate_id
    )
    assert cand["status"] == "archived"
    assert cand["review_status"] != "pending_review"

    # Lineage edge target -> candidate recorded.
    superseded = await pool.fetchval(
        """
        SELECT count(*) FROM memory_relationships
        WHERE source_id = $1 AND target_id = $2 AND relation = 'supersedes'
        """,
        target_id, candidate_id,
    )
    assert superseded == 1

    # It no longer shows up for review.
    items = await list_pending(pool)
    assert candidate_id not in {it["id"] for it in items}


# ---------------------------------------------------------------------------
# Merge on a non-merge-candidate pending row is a no-op (returns None).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_on_plain_quarantine_row_returns_none(pool):
    plain_id = await _store(pool, "A plain flagged note", project_id="weft")
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1",
        plain_id,
    )

    assert await merge_pending(pool, plain_id) is None
    # Untouched — still pending, still active status.
    row = await pool.fetchrow(
        "SELECT status, review_status FROM memories WHERE id = $1", plain_id
    )
    assert row["status"] == "active"
    assert row["review_status"] == "pending_review"


# ---------------------------------------------------------------------------
# The keep-separate path: approve promotes the candidate to its own belief.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_keeps_candidate_as_separate_belief(pool):
    candidate_id, target_id = await _make_candidate(pool)

    ok = await approve_pending(pool, candidate_id)
    assert ok is True

    # Candidate is now its own active belief; target is unchanged (no facet added).
    cand = await pool.fetchrow(
        "SELECT status, review_status, project_facets FROM memories WHERE id = $1",
        candidate_id,
    )
    assert cand["status"] == "active"
    assert cand["review_status"] == "active"
    assert set(cand["project_facets"]) == {"loom"}

    target = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1", target_id
    )
    assert set(target["project_facets"]) == {"weft"}, (
        "keep-separate must NOT append the facet to the target"
    )
