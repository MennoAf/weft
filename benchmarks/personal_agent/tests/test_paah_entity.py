#!/usr/bin/env python3
"""
test_paah_entity.py — Structural asserts for the PAAH entity-brief shape.

Seeds a person entity + its linked facts (+ a distractor person) through the
real graph write path, then measures how completely each read path answers
"what do I need to know about X":

  * ORACLE (weft_entity_context) MUST be complete — recall@links == 1.0. The
    graph edge walk is the guarantee; a regression here means links are lost.
  * CANDIDATE (weft_recall NL) is the MEASUREMENT. The first run surfaced a real
    routing bug: brief phrasings with a temporal word ("what do I need to know
    about X BEFORE our meeting") route to the turns tier and return NONE of the
    entity's belief-tier facts. The fix was NOT to tune the router to those
    phrasings but a general never-miss safety net: an empty turns result falls
    back to belief recall. So the asserts now prove the general fix: (a) no brief
    phrasing comes back empty; (b) the temporal-worded ones specifically recover
    VIA the fallback (marked in the response), across more than one marker.
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


async def test_no_brief_phrasing_comes_back_empty(seeded_entity):
    """CANDIDATE (a) — never-miss: every brief phrasing surfaces the full brief.

    This is the general fix in action: no phrasing (belief-routed or
    turns-misrouted-then-recovered) returns an empty hand. recall@links == 1.0
    across the board.
    """
    _, stats = seeded_entity
    assert stats.never_empty, (
        f"a brief phrasing returned an incomplete/empty result "
        f"(min recall@links={stats.candidate_min:.3f}) — never-miss violated: "
        + ", ".join(
            f"{c.query!r}={c.recall_at_links:.2f}"
            for c in stats.candidate if c.recall_at_links < 1.0
        )
    )


async def test_temporal_worded_briefs_recover_via_fallback(seeded_entity):
    """CANDIDATE (b): the temporal-worded briefs recover THROUGH the fallback.

    Proves the general fix (empty-turns → belief), not a router tweak: the
    phrasings that route to the turns tier come back marked tier_fallback and
    still surface the complete brief — and across more than one temporal marker
    ('before' and 'since'), so it isn't overfit to one word.
    """
    _, stats = seeded_entity
    fallbacks = stats.fallbacks
    assert fallbacks, (
        "expected temporal-worded briefs to route to turns and recover via the "
        "belief fallback; none did — routing changed, revisit this test"
    )
    for c in fallbacks:
        assert c.recall_at_links == 1.0, (
            f"fallback brief {c.query!r} recovered only {c.recall_at_links:.0%} "
            f"of the linked facts"
        )
    # Generality: the fallback fired for more than one distinct temporal marker.
    markers = {m for m in ("before", "since", "after", "until")
               for c in fallbacks if m in c.query.lower()}
    assert len(markers) >= 2, (
        f"fallback only exercised markers {markers} — add phrasing variety so "
        f"the fix isn't validated against a single word"
    )
