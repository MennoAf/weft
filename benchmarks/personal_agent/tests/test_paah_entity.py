#!/usr/bin/env python3
"""
test_paah_entity.py — Structural asserts for the PAAH entity-brief shape.

Seeds a person entity + its linked facts (+ a distractor person) through the
real graph write path, then measures how completely each read path answers
"what do I need to know about X":

  * ORACLE (weft_entity_context) MUST be complete — recall@links == 1.0. The
    graph edge walk is the guarantee; a regression here means links are lost.
  * CANDIDATE (weft_recall NL) is the MEASUREMENT. First run surfaced a real
    routing bug: brief phrasings containing a temporal word ("what do I need to
    know about X BEFORE our meeting") route to the turns tier and return NONE of
    the entity's belief-tier facts. So the two candidate asserts separate the
    two axes: (a) WHEN routed to belief, recall is complete; (b) the finding —
    temporal-worded briefs misroute away from belief and miss. If a router fix
    lands, (b) trips and must be consciously updated.
"""

from __future__ import annotations

import pytest

from benchmarks.personal_agent.entity_manifest import (
    get_brief_entity,
    get_distractor_entity,
)
from benchmarks.personal_agent.entity_harness import run_entity_brief_paah
from benchmarks.personal_agent.seed import seed_entity_brief

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def seeded_entity(pool):
    seeded = await seed_entity_brief(pool)
    stats = await run_entity_brief_paah(pool, seeded)
    return seeded, stats


async def test_entity_seed_integrity(seeded_entity):
    """Both entities' facts each persisted as their own row (no dedup collapse)."""
    seeded, _ = seeded_entity
    assert seeded.brief.stored == get_brief_entity().cardinality, (
        f"brief: stored {seeded.brief.stored} of {get_brief_entity().cardinality}; "
        f"collisions={seeded.brief.dedup_collisions}"
    )
    assert seeded.brief.clean
    assert seeded.distractor.stored == get_distractor_entity().cardinality


async def test_entity_context_returns_complete_brief(seeded_entity):
    """ORACLE: the entity graph walk returns every linked fact (recall@links==1)."""
    _, stats = seeded_entity
    assert stats.oracle_complete, (
        f"weft_entity_context returned only {stats.oracle_returned} of "
        f"{stats.fact_count} linked facts (recall@links={stats.oracle_recall:.3f}) "
        f"— the graph edge walk is lossy"
    )


async def test_brief_recall_is_complete_when_routed_to_belief(seeded_entity):
    """CANDIDATE (a): when a brief query routes to belief, recall is complete.

    Isolates recall quality from routing: every phrasing that actually reached
    the belief tier surfaced the full brief. So the brief IS recallable — the
    only failures are the ones the router sent elsewhere.
    """
    _, stats = seeded_entity
    belief_routed = stats.belief_routed
    assert belief_routed, "no brief phrasing routed to belief — check fixtures"
    for c in belief_routed:
        assert c.recall_at_links == 1.0, (
            f"belief-routed brief {c.query!r} surfaced only "
            f"{c.recall_at_links:.0%} of the linked facts"
        )


async def test_temporal_worded_briefs_misroute_and_miss(seeded_entity):
    """CANDIDATE (b) — FINDING: a brief phrasing with a temporal word ("before")
    routes off the belief tier and returns none of the entity's facts.

    This pins the routing bug so a future fix (router stops hijacking entity/
    belief briefs on bare temporal words) trips this test and forces an update.
    """
    _, stats = seeded_entity
    misrouted = stats.misrouted
    assert misrouted, (
        "expected at least one temporal-worded brief to misroute off belief; "
        "none did — the routing bug may be fixed, update this finding"
    )
    for c in misrouted:
        assert c.recall_at_links == 0.0, (
            f"misrouted brief {c.query!r} (tier={c.routed_tier}) unexpectedly "
            f"surfaced {c.recall_at_links:.0%} — routing behavior changed"
        )
