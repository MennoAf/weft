#!/usr/bin/env python3
"""
test_smoke.py — Smoke test for enumeration_eval harness.

Validates that:
1. Fixtures seed correctly
2. Oracle path returns all members (recall == 1.0)
3. Candidate path produces stochastic runs with recall scores
"""

from __future__ import annotations

import pytest

from benchmarks.enumeration_eval.fixtures import (
    seed_fixtures,
    JIM_BOBLAW_USER_ID,
    JIM_BOBLAW_PROJECT_ID,
)
from benchmarks.enumeration_eval.harness import (
    enum_gather_oracle,
    enum_recall_candidate,
    run_enumeration_eval,
)


@pytest.mark.asyncio
async def test_fixtures_seed_correctly(pool) -> None:
    """Fixtures seed with correct cardinality."""
    fixtures = await seed_fixtures(
        pool,
        user_id=JIM_BOBLAW_USER_ID,
        project_id=JIM_BOBLAW_PROJECT_ID,
    )

    assert len(fixtures) == 3  # plants, medications, books
    assert fixtures["plants"].topic_tag == "plants"
    assert len(fixtures["plants"].members) == 12
    assert fixtures["medications"].topic_tag == "medications"
    assert len(fixtures["medications"].members) == 8
    assert fixtures["books"].topic_tag == "books"
    assert len(fixtures["books"].members) == 5

    # Verify memories were written
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE project_id = $1",
        JIM_BOBLAW_PROJECT_ID,
    )
    assert count == (12 + 8 + 5), f"Expected 25 memories, got {count}"


@pytest.mark.asyncio
async def test_oracle_path_returns_all_members(pool) -> None:
    """Oracle gather_topic_memories must return all members (recall == 1.0)."""
    fixtures = await seed_fixtures(
        pool,
        user_id=JIM_BOBLAW_USER_ID,
        project_id=JIM_BOBLAW_PROJECT_ID,
    )

    plants_collection = fixtures["plants"]
    returned, recall = await enum_gather_oracle(
        pool,
        topic_tag=plants_collection.topic_tag,
        user_id=JIM_BOBLAW_USER_ID,
        expected_members=plants_collection.members,
    )

    assert recall == 1.0, f"Oracle recall should be 1.0, got {recall}"
    assert len(returned) == len(plants_collection.members), (
        f"Oracle should return all {len(plants_collection.members)} members, "
        f"got {len(returned)}"
    )
    # Every expected member should be in the returned set
    returned_set = set(returned)
    expected_set = set(plants_collection.members)
    assert expected_set.issubset(returned_set), (
        f"Oracle returned set missing some expected members: "
        f"{expected_set - returned_set}"
    )


@pytest.mark.asyncio
async def test_candidate_path_produces_stochastic_runs(pool) -> None:
    """Candidate weft_recall produces k runs with recall scores."""
    fixtures = await seed_fixtures(
        pool,
        user_id=JIM_BOBLAW_USER_ID,
        project_id=JIM_BOBLAW_PROJECT_ID,
    )

    medications_collection = fixtures["medications"]
    recalls = await enum_recall_candidate(
        pool,
        topic_tag=medications_collection.topic_tag,
        user_id=JIM_BOBLAW_USER_ID,
        expected_members=medications_collection.members,
        k_runs=3,  # Smaller k for test speed
    )

    assert len(recalls) == 3, f"Expected 3 runs, got {len(recalls)}"
    assert all(0.0 <= r <= 1.0 for r in recalls), (
        f"Recalls must be in [0, 1], got {recalls}"
    )


@pytest.mark.asyncio
async def test_candidate_recall_is_non_degenerate(pool) -> None:
    """Non-degeneracy guard: seeded memories MUST carry real embeddings.

    If a future regression drops the embedding from seed_fixtures (the
    original Phase 0.4 defect), the vector/hybrid candidate path can never
    match a seeded row and recall collapses to 0.0 on every run — a
    structurally meaningless meter. This test fails LOUDLY in that case
    by asserting the candidate path produces NON-ZERO recall for a seeded
    collection.
    """
    fixtures = await seed_fixtures(
        pool,
        user_id=JIM_BOBLAW_USER_ID,
        project_id=JIM_BOBLAW_PROJECT_ID,
    )

    plants_collection = fixtures["plants"]
    recalls = await enum_recall_candidate(
        pool,
        topic_tag=plants_collection.topic_tag,
        user_id=JIM_BOBLAW_USER_ID,
        expected_members=plants_collection.members,
        k_runs=5,
    )

    assert max(recalls) > 0.0, (
        "Candidate path recall is 0.0 on every run — seeded memories likely "
        "have no embeddings (the Phase 0.4 degenerate-meter defect). "
        f"recalls={recalls}"
    )


@pytest.mark.asyncio
async def test_harness_end_to_end(pool) -> None:
    """End-to-end harness run produces RecallStats for all fixtures."""
    fixtures = await seed_fixtures(
        pool,
        user_id=JIM_BOBLAW_USER_ID,
        project_id=JIM_BOBLAW_PROJECT_ID,
    )

    results = await run_enumeration_eval(
        pool,
        fixtures,
        JIM_BOBLAW_USER_ID,
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Validate each result
    for result in results:
        # Oracle must have recall == 1.0
        assert result.oracle_recall_at_membership == 1.0, (
            f"Oracle recall for {result.collection_name} should be 1.0, "
            f"got {result.oracle_recall_at_membership}"
        )

        # Candidate runs must exist and have valid scores
        assert result.candidate_runs > 0, f"No candidate runs for {result.collection_name}"
        assert len(result.candidate_recalls_per_run) == result.candidate_runs
        assert all(0.0 <= r <= 1.0 for r in result.candidate_recalls_per_run)

        # Spread must be non-negative
        spread = result.candidate_max_recall - result.candidate_min_recall
        assert spread >= 0.0, f"Spread should be non-negative, got {spread}"

        # to_dict should serialize correctly
        d = result.to_dict()
        assert d["collection"] == result.collection_name
        assert d["oracle"]["recall_at_membership"] == 1.0
        assert "spread" in d["candidate"]
