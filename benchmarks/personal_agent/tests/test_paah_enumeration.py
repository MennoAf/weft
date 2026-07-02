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

RENDER ALTITUDE (loop-closer, issue weft-09ae22a3). The asserts above check the
tool RESPONSE; an agent reads the RENDERED context, and the response being right
≠ the render showing it. So the render-altitude tests below mirror the agenda
shape's ORACLE + AGENT-FACING split via ``format_recall_context``:
  * FINDING — the legacy results-only render never equals the true count.
  * CLOSE  — the enum-aware render surfaces the complete membership AND makes
    the corrected count readable in its header.
  * PROSE GUARD — ``enumeration.summary`` carries the count as text.
  * SHAPE (item 2) — ``enumeration.members`` is the ratified light projection
    {id, type, content, topic, created_at}; nothing richer is needed to count.
"""

from __future__ import annotations

import pytest

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.harness import run_enumeration_paah
from benchmarks.personal_agent.manifest import PAAH_USER_ID, get_collections
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


# ── Render altitude: what the agent actually reads (loop-closer) ───────────


async def test_legacy_render_undercounts_documents_the_gap(seeded):
    """FINDING: the render an agent reads today (results-only) is NOT the answer.

    ``format_recall_context(use_enumeration=False)`` reproduces the legacy
    Reader render — a numbered dump of the limit-bounded ``results`` slice. Its
    row count never equals the true membership, so an agent reading it
    under/over-counts even though the response dict is correct. This is the same
    consumption-contract gap the agenda shape found in weft_daily_brief, one
    altitude down from the tool response.
    """
    _, stats = seeded
    for s in stats:
        assert s.render_legacy_wrong_rate == 1.0, (
            f"{s.name}: the legacy results-only render matched the true count on "
            f"{(1 - s.render_legacy_wrong_rate) * 100:.0f}% of runs — the render "
            f"gap this documents has changed; re-verify the finding"
        )


async def test_enum_aware_render_surfaces_complete_membership(seeded):
    """CLOSE: the enum-aware render carries the full membership + readable count.

    ``format_recall_context(use_enumeration=True)`` is the fix: it renders the
    complete member list (row count == manifest) and states the corrected count
    in its header where an agent can READ it, not just parse a JSON field.
    """
    _, stats = seeded
    for s in stats:
        assert s.render_close_rate == 1.0, (
            f"{s.name}: enum-aware render omitted members on "
            f"{(1 - s.render_close_rate) * 100:.0f}% of runs — rendered rows "
            f"!= manifest ({s.manifest_count}); the complete list did not "
            f"survive rendering"
        )
        assert s.render_count_shown_rate == 1.0, (
            f"{s.name}: corrected count missing from the rendered header on "
            f"{(1 - s.render_count_shown_rate) * 100:.0f}% of runs — the agent "
            f"cannot read the count off the render"
        )


async def test_summary_prose_carries_count(seeded):
    """PROSE GUARD: enumeration.summary states the count as text.

    Guards the natural-language sentence an agent skims against a regression to
    len(results): if the summary is ever rebuilt off the slice, this trips.
    """
    _, stats = seeded
    for s in stats:
        assert s.summary_shows_count_rate == 1.0, (
            f"{s.name}: enumeration.summary omitted the count on "
            f"{(1 - s.summary_shows_count_rate) * 100:.0f}% of runs — the prose "
            f"answer no longer carries the true count"
        )


async def test_members_projection_is_the_ratified_light_shape(pool):
    """SHAPE (item 2 decision): members is a LIGHT projection, ratified.

    ``enumeration.members`` carries exactly {id, type, content, topic,
    created_at} — enough to count and to list. The richer ``results[]`` shape
    (entities, similarity, relevance_score) is deliberately NOT duplicated here:
    answering "how many / list all" needs none of it. If the projection drifts,
    this trips and forces a conscious re-ratification rather than silent bloat.
    """
    await seed_corpus(pool)
    app = await build_app_context(pool)
    ctx = make_ctx(app)
    token = current_user_id.set(PAAH_USER_ID)
    try:
        from weft.mcp.tools import weft_recall

        coll = get_collections()[0]
        query = f"how many {coll.noun} do I have"
        resp = await weft_recall(ctx, query=query, limit=10, tier="auto")
    finally:
        current_user_id.reset(token)

    enum = resp.get("enumeration")
    assert enum is not None, f"enumeration block missing for {query!r}"
    assert enum.get("members"), "enumeration.members is empty — nothing to shape-check"
    expected_keys = {"id", "type", "content", "topic", "created_at"}
    for m in enum["members"]:
        assert set(m.keys()) == expected_keys, (
            f"members projection drifted: got {sorted(m.keys())}, "
            f"ratified {sorted(expected_keys)} — re-ratify the light shape "
            f"(item 2) before changing it"
        )
