#!/usr/bin/env python3
"""
test_paah_agenda.py — Structural asserts for the PAAH agenda shape (trackers).

Seeds 10 trackers with known due-fates through the real tracker lifecycle path
(create + snooze + close), then measures "what's on my plate" two ways:

  * ORACLE (weft_tracker_due) is the GUARANTEE. It must be complete (every open
    loop present), precise (no snoozed / future / terminal / no-nudge tracker
    leaks in — each excluded spec fails a distinct clause of the due predicate),
    and correctly ordered (the longest-overdue "keep pushing" loop first).
  * AGENT-FACING (weft_daily_brief) is the MEASUREMENT. The current finding: the
    morning digest an agent reads carries none of the open loops the oracle knows,
    because assemble_daily_brief has no trackers section — even though
    weft_tracker_due's docstring says it feeds a daily-brief "open loops" section.
    The last test is a change-detector for that gap: it passes while the gap
    exists and flips the day the brief is wired to open loops (update it then to
    assert full coverage).
"""

from __future__ import annotations

import pytest

from benchmarks.personal_agent.agenda_manifest import get_agenda_specs
from benchmarks.personal_agent.agenda_harness import run_agenda_paah
from benchmarks.personal_agent.seed import seed_agenda

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def seeded_agenda(pool):
    seeded = await seed_agenda(pool)
    stats = await run_agenda_paah(pool, seeded)
    return seeded, stats


async def test_agenda_seed_integrity(seeded_agenda):
    """Every spec created; every declared snooze/close applied (real lifecycle)."""
    seeded, _ = seeded_agenda
    assert seeded.intended == len(get_agenda_specs())
    assert seeded.clean, (
        "seed integrity failed — some tracker did not create, snooze, or close: "
        + ", ".join(
            f"{t.spec.key}(id={bool(t.tracker_id)},"
            f"close={t.close_applied},snooze={t.snooze_applied})"
            for t in seeded.trackers
        )
    )
    # The ground truth we assert against: 4 due open loops, 6 excluded.
    assert len(seeded.due_ids) == 4
    assert len(seeded.excluded_ids) == 6


async def test_tracker_due_is_complete(seeded_agenda):
    """ORACLE: the due query surfaces every seeded open loop (recall@due == 1)."""
    _, stats = seeded_agenda
    assert stats.oracle_complete, (
        f"weft_tracker_due surfaced only {stats.oracle_hits} of "
        f"{stats.due_expected} open loops (recall@due={stats.oracle_recall:.3f})"
    )


async def test_tracker_due_is_precise(seeded_agenda):
    """ORACLE: no snoozed / future / terminal / no-nudge tracker leaks into due.

    Each excluded spec trips a different clause of the due predicate, so a leak
    localizes exactly which clause regressed.
    """
    seeded, stats = seeded_agenda
    leaked_keys = [
        t.spec.key for t in seeded.trackers
        if t.tracker_id in set(stats.oracle_leaks)
    ]
    assert stats.oracle_precise, (
        f"weft_tracker_due leaked non-due trackers into the plate: {leaked_keys} "
        f"— a due-predicate clause (open-state / nudge-scheduled / past-due / "
        f"not-snoozed) is no longer honored"
    )


async def test_keep_pushing_surfaces_first(seeded_agenda):
    """ORACLE ordering: the longest-overdue loop (the one you keep pushing) leads.

    due_trackers orders by nudge_after ASC, so the oldest-due tracker must be the
    first thing an agent sees when asked "what am I forgetting / keep pushing."
    """
    seeded, stats = seeded_agenda
    assert stats.keep_pushing_first, (
        f"expected the longest-overdue loop {seeded.keep_pushing.spec.key!r} "
        f"(id={stats.keep_pushing_id}) to lead the due queue; got order "
        f"{stats.oracle_returned_ids}"
    )


async def test_daily_brief_surfaces_open_loops(seeded_agenda):
    """AGENT-FACING: the morning digest now carries every due open loop.

    The consumption-contract gap the agenda shape first measured (0/4) is CLOSED:
    assemble_daily_brief has an "Open Loops" section that renders due_trackers, so
    the brief an agent reads surfaces all of the open loops the oracle knows —
    coverage == 1.0.
    """
    _, stats = seeded_agenda
    assert stats.brief_available, "daily brief failed to assemble at all"
    assert stats.brief_surfaces_open_loops, (
        f"daily brief surfaced only {stats.brief_due_covered}/{stats.due_expected} "
        f"open loops (coverage={stats.brief_coverage:.2f}) — the Open Loops section "
        f"regressed. Sections present: {stats.brief_sections}"
    )
