#!/usr/bin/env python3
"""
test_paah_enumeration.py — Structural asserts for the PAAH enumeration shape.

These run against the project's testcontainers ``pool`` fixture (inherited via
conftest ``pytest_plugins``). They seed the manifest through the real
``weft_remember`` path and drive the real ``weft_recall`` path, then assert on
the agent-facing response — the coverage enumeration_eval's function-level
tests do not reach.

Post-V8 (consumption contract closed), the guarantees are:
  * Seed integrity — every intended member persisted as its own row.
  * The enumeration block is present and reports the exact manifest count.
  * recall@membership == 1.0 over ``enumeration.members`` (the complete list).
  * The MOST-OBVIOUS field ``response["count"]`` is corrected to the true count
    — an agent reading it can no longer undercount to len(results).
  * Legacy contrast held in place: the naive ``len(results)`` slice still does
    NOT equal the truth, so the test documents WHY the corrected fields matter.
    If a future change makes len(results) itself the complete answer, that
    contrast assert trips and forces a conscious update.
"""

from __future__ import annotations

import pytest

from benchmarks.personal_agent.harness import run_enumeration_paah
from benchmarks.personal_agent.manifest import get_collections
from benchmarks.personal_agent.seed import seed_corpus

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def seeded(pool):
    """Seed the manifest once and return (seed_results, stats)."""
    seed_results = await seed_corpus(pool)
    stats = await run_enumeration_paah(pool, seed_results, limit=10)
    return seed_results, stats


async def test_seed_integrity_matches_manifest(seeded):
    """Every intended member persisted as its own row (no dedup collapse)."""
    seed_results, _ = seeded
    by_name = {sr.collection.name: sr for sr in seed_results}
    for coll in get_collections():
        sr = by_name[coll.name]
        assert sr.stored == coll.cardinality, (
            f"{coll.name}: intended {coll.cardinality}, stored {sr.stored}; "
            f"dedup collisions={sr.dedup_collisions}"
        )
        assert sr.clean, f"{coll.name} seed not clean: {sr.dedup_collisions}"


async def test_enumeration_block_present_and_count_correct(seeded):
    """The enumeration answer is present with the exact membership count."""
    _, stats = seeded
    for s in stats:
        assert s.enum_present_rate == 1.0, (
            f"{s.name}: enumeration block missing on "
            f"{(1 - s.enum_present_rate) * 100:.0f}% of runs — router did not "
            f"fire (tier misroute or resolve_topic miss?)"
        )
        assert s.enum_count_correct_rate == 1.0, (
            f"{s.name}: enumeration.count wrong on "
            f"{(1 - s.enum_count_correct_rate) * 100:.0f}% of runs "
            f"(manifest={s.manifest_count})"
        )


async def test_obvious_count_field_is_corrected(seeded):
    """The most-obvious field response["count"] now equals the true membership.

    This is the consumption-contract closer: an agent reading the obvious count
    field can no longer undercount to len(results).
    """
    _, stats = seeded
    for s in stats:
        assert s.obvious_count_correct_rate == 1.0, (
            f"{s.name}: response['count'] wrong on "
            f"{(1 - s.obvious_count_correct_rate) * 100:.0f}% of runs — the "
            f"obvious field was not corrected to the true count"
        )


async def test_recall_at_membership_is_complete(seeded):
    """enumeration.members is the complete membership set — recall == 1.0."""
    _, stats = seeded
    for s in stats:
        assert s.recall_min == 1.0, (
            f"{s.name}: recall@membership over enumeration.members "
            f"min={s.recall_min:.3f} (<1.0) — the complete list is incomplete"
        )


async def test_naive_len_results_still_wrong_documents_why_fix_matters(seeded):
    """CONTRAST: the ranked len(results) slice still does NOT equal the truth.

    Kept as a guardrail: it proves the corrected fields (not len(results)) are
    what closed the contract. If a future change makes len(results) itself the
    complete answer, this trips and must be consciously updated.
    """
    _, stats = seeded
    for s in stats:
        assert s.naive_correct_rate == 0.0, (
            f"{s.name}: naive len(results) matched the truth on "
            f"{s.naive_correct_rate * 100:.0f}% of runs — the ranked slice now "
            f"equals membership; the contrast this test documents has changed"
        )
