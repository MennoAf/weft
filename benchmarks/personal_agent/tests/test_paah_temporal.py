#!/usr/bin/env python3
"""
test_paah_temporal.py — Structural asserts for the PAAH temporal/dialogue shape.

This is the Branch-A turn-tier probe (roadmap weft-c0a51a73) made into a
repeatable number: seed a dated dialogue trace through the real weft_turn_append
path, then drive the real weft_recall turn/both tiers and assert on the
agent-facing response.

Guarantees:
  * Seed integrity — every seeded turn persisted as its own row.
  * Routing — each probe's phrasings route to the expected tier (turns/both);
    a misroute means the turn path never fires.
  * Anchor recall — the one seeded turn that answers each probe is surfaced by
    id on every phrasing (never-miss floor == 1.0).

Because turn-tier recall was explicitly "untested at scale", the anchor-recall
assert is the load-bearing one: if it fails, that is the finding — the number
we came here to get — not a flaky test.
"""

from __future__ import annotations

import pytest

from benchmarks.personal_agent.seed import seed_turns
from benchmarks.personal_agent.temporal_harness import run_temporal_paah
from benchmarks.personal_agent.temporal_manifest import get_turn_specs

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def seeded_temporal(pool):
    seeded = await seed_turns(pool)
    stats = await run_temporal_paah(pool, seeded)
    return seeded, stats


async def test_turn_seed_integrity(seeded_temporal):
    seeded, _ = seeded_temporal
    assert seeded.stored == len(get_turn_specs()), (
        f"turn seed integrity: stored {seeded.stored} of "
        f"{len(get_turn_specs())} — a turn failed to persist"
    )
    assert seeded.clean


async def test_probes_route_to_expected_tier(seeded_temporal):
    """Temporal → 'turns', dialogue → 'both'. A misroute never fires the path."""
    _, stats = seeded_temporal
    for s in stats:
        assert s.routed_correct_rate == 1.0, (
            f"{s.key}: routed to expected tier {s.expected_tier!r} on only "
            f"{s.routed_correct_rate * 100:.0f}% of phrasings"
        )


async def test_anchor_turn_is_surfaced_every_run(seeded_temporal):
    """The seeded turn that answers each probe is surfaced by id, every phrasing.

    This is the never-miss floor for turn-tier recall (Branch-A). A failure here
    is the measurement, not a flake: it says the turn path dropped the answer.
    """
    _, stats = seeded_temporal
    for s in stats:
        assert s.anchor_present_min == 1.0, (
            f"{s.key}: anchor turn {s.anchor_key!r} missed on "
            f"{(1 - s.anchor_present_rate) * 100:.0f}% of phrasings "
            f"(present_rate={s.anchor_present_rate:.2f}) — turn-tier recall "
            f"dropped the answer under retrieval pressure (limit={s.limit})"
        )
