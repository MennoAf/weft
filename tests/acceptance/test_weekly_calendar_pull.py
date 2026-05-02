"""Acceptance test — "What do I have this week?"

Personal agents need to answer "what's on my plate" by surfacing trackers
whose nudge window falls inside a date range. This test verifies the
filter primitive without depending on external calendar MCP integration:
trackers seeded with explicit ``nudge_after`` timestamps inside vs.
outside the next 7 days, and ``due_trackers`` should return only the
ones whose nudge has fired.

The richer "what do I have" path eventually combines this with
calendar-MCP events and ingested-Slack mentions; this test pins the
*tracker primitive* alone so regressions there are caught before the
combined query path is rebuilt.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import NudgeMode, TrackerCreate, TrackerKind
from weft.trackers import create_tracker, due_trackers, list_trackers

from tests.acceptance.conftest import cleanup_project, sandbox_project_id


CASE_ID = "weekly_calendar_pull"


@pytest.mark.asyncio
async def test_weekly_calendar_pull(pool) -> None:
    project_id = sandbox_project_id(CASE_ID)
    try:
        # --- Seed -------------------------------------------------------
        # Anchor "now" inside the test so the assertions don't drift with
        # wall-clock — same defensive pattern as the cost-tracking
        # spend-trend test.
        now = datetime.now(timezone.utc)
        seed = [
            ("yesterday-due",   now - timedelta(days=1),   True),   # already due
            ("today-due",       now - timedelta(hours=1),  True),   # due now
            ("tomorrow-due",    now + timedelta(days=1),   True),   # due this week
            ("late-week-due",   now + timedelta(days=5),   True),   # still this week
            ("next-week",       now + timedelta(days=10),  False),  # outside window
            ("month-out",       now + timedelta(days=30),  False),  # outside window
        ]
        created_ids: dict[str, str] = {}
        for label, nudge_after, _in_week in seed:
            t = await create_tracker(
                pool,
                TrackerCreate(
                    kind=TrackerKind.outreach,
                    title=f"weekly-test:{label}",
                    project_id=project_id,
                    nudge_mode=NudgeMode.once,
                    nudge_after=nudge_after,
                ),
            )
            created_ids[label] = t.id

        # --- Query ------------------------------------------------------
        # Trackers actively due "right now" — anchored at our test ``now``.
        due_now = await due_trackers(pool, now=now)
        # All open trackers in the project, irrespective of nudge timing.
        all_open = await list_trackers(
            pool, project_id=project_id, open_only=True,
        )

        # --- Assertions -------------------------------------------------
        due_ids = {t.id for t in due_now}
        # Already-due trackers fire.
        assert created_ids["yesterday-due"] in due_ids
        assert created_ids["today-due"] in due_ids
        # Future-dated trackers (even if "this week") do NOT fire on
        # ``due_trackers`` — nudge_after is strict.
        assert created_ids["tomorrow-due"] not in due_ids
        assert created_ids["late-week-due"] not in due_ids
        # Far-future trackers definitely don't fire.
        assert created_ids["next-week"] not in due_ids
        assert created_ids["month-out"] not in due_ids

        # The agent's "what do I have this week" path runs over the
        # project's open trackers and applies the 7-day filter itself.
        # Verify the underlying list is intact and filterable.
        seven_days_out = now + timedelta(days=7)
        this_week = [
            t for t in all_open
            if t.nudge_after is not None and t.nudge_after <= seven_days_out
        ]
        this_week_ids = {t.id for t in this_week}
        assert created_ids["yesterday-due"] in this_week_ids
        assert created_ids["today-due"] in this_week_ids
        assert created_ids["tomorrow-due"] in this_week_ids
        assert created_ids["late-week-due"] in this_week_ids
        assert created_ids["next-week"] not in this_week_ids
        assert created_ids["month-out"] not in this_week_ids

    finally:
        await cleanup_project(pool, project_id)
