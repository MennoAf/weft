"""Unit + integration tests for weft.board.

Tests cover:
- Item schema construction and serialization per PRD §Interfaces
- Urgency bucketing per PRD §Validation V2 (overdue/due_soon/pending/no_date)
- Ranking within a bucket (oldest-due-first, age_days desc, title)
- Boundary cases at exactly now and exactly horizon cutoff
- assemble_board() fan-out against a real testcontainers Postgres: multi-source
  assembly + bucketing (V2), per-source isolation on failure (V5), and
  per_source_cap truncation surfacing (V7)
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from weft.board import (
    Action,
    BOARD_TIMEZONE,
    Item,
    Urgency,
    _TRIGGER_HIDE_KINDS,
    alert_adapter,
    assemble_board,
    calculate_urgency,
    rank_items,
    review_adapter,
    task_adapter,
    task_due_at,
    tracker_adapter,
    trigger_adapter,
)
from weft.models import (
    Alert,
    AlertChannel,
    AlertCreate,
    AlertStatus,
    AlertType,
    Memory,
    MemoryCreate,
    MemoryType,
    NudgeMode,
    Tracker,
    TrackerCreate,
    TrackerKind,
    TrackerState,
    Trigger,
    TriggerConditionType,
    TriggerCreate,
    TriggerStatus,
)
from weft.skills import TaskEntry


# Fixtures

@pytest.fixture
def now():
    """Fixed 'current time' for deterministic tests."""
    return datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def default_horizon_days():
    """Default lookahead window per PRD."""
    return 7


# calculate_urgency tests

class TestCalculateUrgency:
    """Pure function tests for urgency bucketing."""

    def test_overdue_past_due_at(self, now):
        """past due_at → overdue (V2)."""
        past_due = now - timedelta(days=1)
        assert calculate_urgency(past_due, now) == "overdue"

    def test_overdue_far_past(self, now):
        """far-past due_at → overdue (robustness)."""
        far_past = now - timedelta(days=30)
        assert calculate_urgency(far_past, now) == "overdue"

    def test_due_soon_within_horizon(self, now):
        """due_at within horizon → due_soon (V2)."""
        due_in_3_days = now + timedelta(days=3)
        assert calculate_urgency(due_in_3_days, now, horizon_days=7) == "due_soon"

    def test_due_soon_at_boundary_now(self, now):
        """due_at exactly at now → due_soon (within horizon)."""
        assert calculate_urgency(now, now, horizon_days=7) == "due_soon"

    def test_due_soon_at_horizon_boundary(self, now):
        """due_at exactly at now + horizon → due_soon (inclusive boundary)."""
        at_horizon = now + timedelta(days=7)
        assert calculate_urgency(at_horizon, now, horizon_days=7) == "due_soon"

    def test_pending_beyond_horizon(self, now):
        """due_at beyond horizon → pending (V2)."""
        beyond_horizon = now + timedelta(days=8)
        assert calculate_urgency(beyond_horizon, now, horizon_days=7) == "pending"

    def test_pending_far_future(self, now):
        """far-future due_at → pending (robustness)."""
        far_future = now + timedelta(days=100)
        assert calculate_urgency(far_future, now, horizon_days=7) == "pending"

    def test_no_date_null_due_at(self, now):
        """null due_at → no_date (V2)."""
        assert calculate_urgency(None, now) == "no_date"

    def test_default_now(self):
        """Uses UTC now if not provided."""
        # Create a due_at that is definitely in the past
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        assert calculate_urgency(past) == "overdue"

    def test_default_horizon_days(self, now):
        """Default horizon is 7 days."""
        # On boundary: 7 days exactly should be due_soon
        at_boundary = now + timedelta(days=7)
        assert calculate_urgency(at_boundary, now) == "due_soon"

        # Just beyond: 8 days should be pending
        beyond = now + timedelta(days=8)
        assert calculate_urgency(beyond, now) == "pending"

    def test_custom_horizon(self, now):
        """Horizon parameter adjusts cutoff."""
        due_in_14_days = now + timedelta(days=14)

        # With 7-day horizon: pending
        assert calculate_urgency(due_in_14_days, now, horizon_days=7) == "pending"

        # With 14-day horizon: due_soon
        assert calculate_urgency(due_in_14_days, now, horizon_days=14) == "due_soon"

        # With 21-day horizon: due_soon
        assert calculate_urgency(due_in_14_days, now, horizon_days=21) == "due_soon"

    def test_timezone_naive_due_at_treated_as_utc(self, now):
        """Naive due_at is assumed to be UTC."""
        # Create naive datetime (no tzinfo)
        naive_past = datetime(2026, 7, 1, 12, 0, 0)  # yesterday
        result = calculate_urgency(naive_past, now)
        assert result == "overdue"

    def test_timezone_naive_now_treated_as_utc(self):
        """Naive now is assumed to be UTC."""
        naive_now = datetime(2026, 7, 2, 12, 0, 0)
        naive_past = datetime(2026, 7, 1, 12, 0, 0)
        result = calculate_urgency(naive_past, naive_now)
        assert result == "overdue"


# Item model tests

class TestItemConstruction:
    """Item dataclass construction and serialization."""

    def test_item_construction_minimal(self, now):
        """Construct Item with all fields."""
        item = Item(
            id="tracker-123",
            source="tracker",
            kind="project_task",
            title="Fix bug in parser",
            state="in_progress",
            due_at=now + timedelta(days=1),
            snoozed_until=None,
            age_days=3.5,
            urgency="due_soon",
        )

        assert item.id == "tracker-123"
        assert item.source == "tracker"
        assert item.kind == "project_task"
        assert item.title == "Fix bug in parser"
        assert item.state == "in_progress"
        assert item.urgency == "due_soon"
        assert item.project_id is None
        assert item.entity_id is None
        assert item.actions == []

    def test_item_construction_with_optional_fields(self, now):
        """Construct Item with all optional fields."""
        action = Action(verb="close", tool="weft_tracker_close", args={"id": "tracker-123"})
        item = Item(
            id="tracker-123",
            source="tracker",
            kind="project_task",
            title="Fix bug",
            state="in_progress",
            due_at=now,
            snoozed_until=now + timedelta(hours=1),
            age_days=2.0,
            urgency="due_soon",
            project_id="proj-1",
            entity_id="entity-1",
            actions=[action],
        )

        assert item.project_id == "proj-1"
        assert item.entity_id == "entity-1"
        assert len(item.actions) == 1
        assert item.actions[0].verb == "close"

    def test_item_to_dict_serializes_datetime(self, now):
        """to_dict converts datetime to ISO8601 string."""
        item = Item(
            id="alert-456",
            source="alert",
            kind="memory_review",
            title="Review memories",
            state="active",
            due_at=now,
            snoozed_until=now + timedelta(hours=2),
            age_days=1.0,
            urgency="due_soon",
        )

        d = item.to_dict()
        assert d["due_at"] == now.isoformat()
        assert d["snoozed_until"] == (now + timedelta(hours=2)).isoformat()

    def test_item_to_dict_null_datetimes(self):
        """to_dict handles null datetimes."""
        item = Item(
            id="task-789",
            source="task",
            kind="high",
            title="Read email",
            state=None,
            due_at=None,
            snoozed_until=None,
            age_days=0.0,
            urgency="no_date",
        )

        d = item.to_dict()
        assert d["due_at"] is None
        assert d["snoozed_until"] is None

    def test_item_to_dict_includes_actions(self, now):
        """to_dict serializes actions list."""
        actions = [
            Action(verb="close", tool="weft_tracker_close", args={"id": "t1"}),
            Action(verb="snooze", tool="weft_tracker_snooze", args={"id": "t1", "hours": 24}),
        ]
        item = Item(
            id="t1",
            source="tracker",
            kind="task",
            title="Test",
            state="open",
            due_at=now,
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
            actions=actions,
        )

        d = item.to_dict()
        assert len(d["actions"]) == 2
        assert d["actions"][0]["verb"] == "close"
        assert d["actions"][0]["tool"] == "weft_tracker_close"
        assert d["actions"][1]["verb"] == "snooze"

    def test_item_source_type_literal(self, now):
        """Item source is one of the five known values."""
        for source in ["tracker", "alert", "trigger", "task", "review"]:
            item = Item(
                id=f"{source}-1",
                source=source,  # type: ignore
                kind="test",
                title="Test",
                state="open",
                due_at=now,
                snoozed_until=None,
                age_days=1.0,
                urgency="due_soon",
            )
            assert item.source == source

    def test_item_urgency_type_literal(self, now):
        """Item urgency is one of the four buckets."""
        for urgency in ["overdue", "due_soon", "pending", "no_date"]:
            item = Item(
                id=f"urg-{urgency}",
                source="tracker",
                kind="test",
                title="Test",
                state="open",
                due_at=now,
                snoozed_until=None,
                age_days=1.0,
                urgency=urgency,  # type: ignore
            )
            assert item.urgency == urgency


# rank_items tests

class TestRankItems:
    """Ranking within a bucket (oldest-due-first, age_days desc, title)."""

    def test_rank_empty_list(self):
        """Empty list returns empty."""
        assert rank_items([]) == []

    def test_rank_single_item(self, now):
        """Single item is unchanged."""
        item = Item(
            id="single",
            source="tracker",
            kind="task",
            title="Only",
            state="open",
            due_at=now + timedelta(days=1),
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
        )
        result = rank_items([item])
        assert len(result) == 1
        assert result[0].id == "single"

    def test_rank_by_due_at_oldest_first(self, now):
        """Oldest due_at sorts first (ascending due_at)."""
        oldest = Item(
            id="1",
            source="tracker",
            kind="task",
            title="Oldest",
            state="open",
            due_at=now + timedelta(days=1),
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
        )
        middle = Item(
            id="2",
            source="tracker",
            kind="task",
            title="Middle",
            state="open",
            due_at=now + timedelta(days=3),
            snoozed_until=None,
            age_days=2.0,
            urgency="due_soon",
        )
        newest = Item(
            id="3",
            source="tracker",
            kind="task",
            title="Newest",
            state="open",
            due_at=now + timedelta(days=5),
            snoozed_until=None,
            age_days=3.0,
            urgency="due_soon",
        )

        result = rank_items([newest, oldest, middle])
        assert [r.id for r in result] == ["1", "2", "3"]

    def test_rank_by_age_days_desc_tiebreak(self, now):
        """Same due_at: older age_days (higher value) sorts first."""
        due_date = now + timedelta(days=2)

        very_old = Item(
            id="a",
            source="tracker",
            kind="task",
            title="Very old",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=10.0,
            urgency="due_soon",
        )
        moderately_old = Item(
            id="b",
            source="tracker",
            kind="task",
            title="Moderately old",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=5.0,
            urgency="due_soon",
        )
        young = Item(
            id="c",
            source="tracker",
            kind="task",
            title="Young",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
        )

        result = rank_items([young, very_old, moderately_old])
        assert [r.id for r in result] == ["a", "b", "c"]

    def test_rank_by_title_final_tiebreak(self, now):
        """Same due_at and age_days: title sorts alphabetically (case-insensitive)."""
        due_date = now + timedelta(days=1)
        age = 5.0

        apple = Item(
            id="1",
            source="tracker",
            kind="task",
            title="Apple task",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )
        zebra = Item(
            id="2",
            source="tracker",
            kind="task",
            title="Zebra task",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )
        banana = Item(
            id="3",
            source="tracker",
            kind="task",
            title="Banana task",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )

        result = rank_items([zebra, apple, banana])
        assert [r.id for r in result] == ["1", "3", "2"]  # Apple, Banana, Zebra

    def test_rank_title_tiebreak_case_insensitive(self, now):
        """Title sorting is case-insensitive."""
        due_date = now + timedelta(days=1)
        age = 5.0

        upper = Item(
            id="1",
            source="tracker",
            kind="task",
            title="APPLE",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )
        lower = Item(
            id="2",
            source="tracker",
            kind="task",
            title="apple",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )
        mixed = Item(
            id="3",
            source="tracker",
            kind="task",
            title="ApPlE",
            state="open",
            due_at=due_date,
            snoozed_until=None,
            age_days=age,
            urgency="due_soon",
        )

        result = rank_items([lower, upper, mixed])
        # All three are equal under case-insensitive comparison, order stable
        ids = [r.id for r in result]
        assert set(ids) == {"1", "2", "3"}

    def test_rank_null_due_at_sorts_last(self, now):
        """Items with null due_at sort last."""
        with_due = Item(
            id="1",
            source="tracker",
            kind="task",
            title="Has due",
            state="open",
            due_at=now + timedelta(days=1),
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
        )
        no_due = Item(
            id="2",
            source="tracker",
            kind="task",
            title="No due",
            state="open",
            due_at=None,
            snoozed_until=None,
            age_days=2.0,
            urgency="no_date",
        )

        result = rank_items([no_due, with_due])
        assert result[0].id == "1"
        assert result[1].id == "2"

    def test_rank_complex_scenario(self, now):
        """Mixed scenario: multiple due dates, ages, null due_at, titles."""
        items = [
            # Due tomorrow, age 5, "Zebra"
            Item(
                id="z1",
                source="tracker",
                kind="task",
                title="Zebra task",
                state="open",
                due_at=now + timedelta(days=1),
                snoozed_until=None,
                age_days=5.0,
                urgency="due_soon",
            ),
            # Due today, age 10, "Apple"
            Item(
                id="a1",
                source="tracker",
                kind="task",
                title="Apple task",
                state="open",
                due_at=now,
                snoozed_until=None,
                age_days=10.0,
                urgency="due_soon",
            ),
            # No due, age 3, "Banana"
            Item(
                id="b1",
                source="tracker",
                kind="task",
                title="Banana task",
                state="open",
                due_at=None,
                snoozed_until=None,
                age_days=3.0,
                urgency="no_date",
            ),
            # Due today, age 2, "Banana"
            Item(
                id="b2",
                source="alert",
                kind="alert",
                title="Banana alert",
                state="active",
                due_at=now,
                snoozed_until=None,
                age_days=2.0,
                urgency="due_soon",
            ),
            # No due, age 15, "Apple"
            Item(
                id="a2",
                source="task",
                kind="high",
                title="Apple review",
                state=None,
                due_at=None,
                snoozed_until=None,
                age_days=15.0,
                urgency="no_date",
            ),
        ]

        result = rank_items(items)
        ids = [r.id for r in result]

        # Expected order:
        # 1. a1 (due now, age 10) — earliest due
        # 2. b2 (due now, age 2) — same due, younger
        # 3. z1 (due tomorrow, age 5)
        # 4. a2 (no due, age 15) — null due_at, older age
        # 5. b1 (no due, age 3) — null due_at, younger age
        assert ids == ["a1", "b2", "z1", "a2", "b1"]

    def test_rank_does_not_mutate_input(self, now):
        """rank_items returns a new list, doesn't mutate input."""
        original = [
            Item(
                id="b",
                source="tracker",
                kind="task",
                title="B",
                state="open",
                due_at=now + timedelta(days=2),
                snoozed_until=None,
                age_days=1.0,
                urgency="due_soon",
            ),
            Item(
                id="a",
                source="tracker",
                kind="task",
                title="A",
                state="open",
                due_at=now + timedelta(days=1),
                snoozed_until=None,
                age_days=1.0,
                urgency="due_soon",
            ),
        ]
        original_ids_before = [i.id for i in original]

        result = rank_items(original)

        # Original should be unchanged
        original_ids_after = [i.id for i in original]
        assert original_ids_before == original_ids_after == ["b", "a"]

        # Result should be sorted
        assert [r.id for r in result] == ["a", "b"]


# Adapter tests

class TestTrackerAdapter:
    """Adapter for mapping Tracker objects to Items."""

    def test_tracker_adapter_empty_list(self, now):
        """Empty tracker list returns empty Items list."""
        result = tracker_adapter([], now)
        assert result == []

    def test_tracker_adapter_single_tracker(self, now):
        """Single tracker maps to single Item with all fields populated."""
        tracker = Tracker(
            id="tr-1234567890",
            user_id="user-1",
            project_id="proj-1",
            entity_id="entity-1",
            kind=TrackerKind.task,
            title="Fix parser bug",
            state=TrackerState.in_progress,
            nudge_mode=NudgeMode.once,
            nudge_after=now + timedelta(days=2),
            snooze_until=None,
            created_at=now - timedelta(days=5),
        )

        result = tracker_adapter([tracker], now)

        assert len(result) == 1
        item = result[0]
        assert item.id == "tr-1234567890"
        assert item.source == "tracker"
        assert item.kind == "task"
        assert item.title == "Fix parser bug"
        assert item.state == "in_progress"
        assert item.due_at == now + timedelta(days=2)
        assert item.snoozed_until is None
        assert item.age_days == 5.0
        assert item.urgency == "due_soon"
        assert item.project_id == "proj-1"
        assert item.entity_id == "entity-1"
        # Triage descriptors (PRD §Behavior / V1 actions[]): close/snooze/dismiss
        # pre-filled with the tracker id, each naming an existing write tool.
        verbs = {a.verb: a for a in item.actions}
        assert set(verbs) == {"close", "snooze", "dismiss"}
        assert verbs["close"].tool == "weft_tracker_close"
        assert verbs["close"].args["tracker_id"] == item.id
        assert verbs["snooze"].tool == "weft_tracker_snooze"
        assert "until" in verbs["snooze"].args
        assert verbs["dismiss"].tool == "weft_tracker_dismiss"

    def test_tracker_adapter_urgency_overdue(self, now):
        """Tracker with past nudge_after maps to overdue urgency."""
        tracker = Tracker(
            id="tr-overdue",
            kind=TrackerKind.task,
            title="Overdue task",
            state=TrackerState.in_progress,
            nudge_after=now - timedelta(days=1),
            created_at=now - timedelta(days=3),
        )

        result = tracker_adapter([tracker], now)
        assert result[0].urgency == "overdue"
        assert result[0].due_at == now - timedelta(days=1)

    def test_tracker_adapter_urgency_no_date(self, now):
        """Tracker with null nudge_after maps to no_date urgency."""
        tracker = Tracker(
            id="tr-no-date",
            kind=TrackerKind.task,
            title="No date task",
            state=TrackerState.in_progress,
            nudge_after=None,
            created_at=now,
        )

        result = tracker_adapter([tracker], now)
        assert result[0].urgency == "no_date"
        assert result[0].due_at is None

    def test_tracker_adapter_respects_snooze_until(self, now):
        """Tracker snooze_until is preserved in Item."""
        snooze_time = now + timedelta(hours=3)
        tracker = Tracker(
            id="tr-snoozed",
            kind=TrackerKind.task,
            title="Snoozed task",
            state=TrackerState.in_progress,
            nudge_after=now + timedelta(days=1),
            snooze_until=snooze_time,
            created_at=now - timedelta(days=1),
        )

        result = tracker_adapter([tracker], now)
        assert result[0].snoozed_until == snooze_time

    def test_tracker_adapter_age_days_calculation(self, now):
        """age_days is correctly calculated from created_at."""
        # Created 3.5 days ago
        created = now - timedelta(days=3, hours=12)
        tracker = Tracker(
            id="tr-age",
            kind=TrackerKind.task,
            title="Aged task",
            state=TrackerState.in_progress,
            nudge_after=now,
            created_at=created,
        )

        result = tracker_adapter([tracker], now)
        assert abs(result[0].age_days - 3.5) < 0.01

    def test_tracker_adapter_multiple_trackers(self, now):
        """Multiple trackers map to multiple Items."""
        trackers = [
            Tracker(
                id=f"tr-{i}",
                kind=TrackerKind.task,
                title=f"Task {i}",
                state=TrackerState.in_progress,
                nudge_after=now + timedelta(days=i),
                created_at=now - timedelta(days=i),
            )
            for i in range(1, 4)
        ]

        result = tracker_adapter(trackers, now)

        assert len(result) == 3
        for i, item in enumerate(result, start=1):
            assert item.id == f"tr-{i}"
            assert item.source == "tracker"
            assert item.title == f"Task {i}"

    def test_tracker_adapter_all_tracker_kinds(self, now):
        """All TrackerKind values map correctly."""
        for kind in [
            TrackerKind.outreach,
            TrackerKind.task,
            TrackerKind.follow_up,
            TrackerKind.meal_plan,
            TrackerKind.shopping_list,
            TrackerKind.pantry,
            TrackerKind.watch,
            TrackerKind.list,
        ]:
            tracker = Tracker(
                id=f"tr-{kind.value}",
                kind=kind,
                title=f"{kind.value} tracker",
                state=TrackerState.in_progress,
                nudge_after=now,
                created_at=now,
            )

            result = tracker_adapter([tracker], now)
            assert result[0].kind == kind.value

    def test_tracker_adapter_all_tracker_states(self, now):
        """All TrackerState values map correctly."""
        for state in [
            TrackerState.in_progress,
            TrackerState.awaiting_reply,
            TrackerState.blocked,
        ]:
            tracker = Tracker(
                id=f"tr-{state.value}",
                kind=TrackerKind.task,
                title=f"Task with state {state.value}",
                state=state,
                nudge_after=now,
                created_at=now,
            )

            result = tracker_adapter([tracker], now)
            assert result[0].state == state.value


class TestAlertAdapter:
    """Adapter for mapping Alert objects to Items."""

    def test_alert_adapter_empty_list(self, now):
        """Empty alert list returns empty Items list."""
        result = alert_adapter([], now)
        assert result == []

    def test_alert_adapter_single_alert(self, now):
        """Single alert maps to single Item with all fields populated."""
        alert = Alert(
            id="alert-abcd1234",
            user_id="user-1",
            alert_type=AlertType.due_task,
            title="Review memories",
            body="Time to review recent memories",
            trigger_at=now + timedelta(days=1),
            status=AlertStatus.pending,
            project_id="proj-1",
            created_at=now - timedelta(days=2),
        )

        result = alert_adapter([alert], now)

        assert len(result) == 1
        item = result[0]
        assert item.id == "alert-abcd1234"
        assert item.source == "alert"
        assert item.kind == "due_task"
        assert item.title == "Review memories"
        assert item.state == "pending"
        assert item.due_at == now + timedelta(days=1)
        assert item.snoozed_until is None
        assert item.age_days == 2.0
        assert item.urgency == "due_soon"
        assert item.project_id == "proj-1"
        assert item.entity_id is None
        assert len(item.actions) == 1
        assert item.actions[0].verb == "dismiss"
        assert item.actions[0].tool == "weft_alert_dismiss"
        assert item.actions[0].args["alert_id"] == item.id

    def test_alert_adapter_urgency_overdue(self, now):
        """Alert with past trigger_at maps to overdue urgency."""
        alert = Alert(
            id="alert-overdue",
            alert_type=AlertType.due_task,
            title="Overdue alert",
            trigger_at=now - timedelta(hours=2),
            status=AlertStatus.pending,
            created_at=now - timedelta(days=1),
        )

        result = alert_adapter([alert], now)
        assert result[0].urgency == "overdue"

    def test_alert_adapter_urgency_pending(self, now):
        """Alert with future trigger_at beyond horizon maps to pending."""
        alert = Alert(
            id="alert-pending",
            alert_type=AlertType.due_task,
            title="Pending alert",
            trigger_at=now + timedelta(days=15),
            status=AlertStatus.pending,
            created_at=now,
        )

        result = alert_adapter([alert], now)
        assert result[0].urgency == "pending"

    def test_alert_adapter_no_snooze_or_entity(self, now):
        """Alert never has snoozed_until or entity_id."""
        alert = Alert(
            id="alert-minimal",
            alert_type=AlertType.custom,
            title="Alert",
            trigger_at=now + timedelta(days=1),
            status=AlertStatus.pending,
            created_at=now,
        )

        result = alert_adapter([alert], now)
        assert result[0].snoozed_until is None
        assert result[0].entity_id is None

    def test_alert_adapter_age_days_calculation(self, now):
        """age_days is correctly calculated from created_at."""
        # Created 1.25 days ago
        created = now - timedelta(days=1, hours=6)
        alert = Alert(
            id="alert-age",
            alert_type=AlertType.custom,
            title="Alert",
            trigger_at=now + timedelta(days=1),
            status=AlertStatus.pending,
            created_at=created,
        )

        result = alert_adapter([alert], now)
        assert abs(result[0].age_days - 1.25) < 0.01

    def test_alert_adapter_multiple_alerts(self, now):
        """Multiple alerts map to multiple Items."""
        alerts = [
            Alert(
                id=f"alert-{i}",
                alert_type=AlertType.due_task,
                title=f"Alert {i}",
                trigger_at=now + timedelta(days=i),
                status=AlertStatus.pending,
                created_at=now - timedelta(days=i),
            )
            for i in range(1, 4)
        ]

        result = alert_adapter(alerts, now)

        assert len(result) == 3
        for i, item in enumerate(result, start=1):
            assert item.id == f"alert-{i}"
            assert item.source == "alert"
            assert item.title == f"Alert {i}"

    def test_alert_adapter_all_alert_types(self, now):
        """All AlertType values map correctly."""
        alert_types = [
            AlertType.due_task,
            AlertType.stale_decision,
            AlertType.follow_up,
            AlertType.custom,
            AlertType.daily_brief,
            AlertType.check_in_low_mood,
            AlertType.check_in_low_sleep,
            AlertType.check_in_declining_trend,
            AlertType.loom_stale_claim,
            AlertType.loom_epic_ready,
            AlertType.loom_blocked_pile_up,
            AlertType.memory_consolidation_overdue,
            AlertType.memory_count_threshold,
            AlertType.memory_contradiction,
            AlertType.board_feedback,
        ]

        for alert_type in alert_types:
            alert = Alert(
                id=f"alert-{alert_type.value}",
                alert_type=alert_type,
                title=f"Alert of type {alert_type.value}",
                trigger_at=now + timedelta(days=1),
                status=AlertStatus.pending,
                created_at=now,
            )

            result = alert_adapter([alert], now)
            assert result[0].kind == alert_type.value

    def test_alert_adapter_all_alert_statuses(self, now):
        """All AlertStatus values map correctly."""
        for status in [AlertStatus.pending, AlertStatus.fired, AlertStatus.dismissed]:
            alert = Alert(
                id=f"alert-{status.value}",
                alert_type=AlertType.custom,
                title=f"Alert with status {status.value}",
                trigger_at=now + timedelta(days=1),
                status=status,
                created_at=now,
            )

            result = alert_adapter([alert], now)
            assert result[0].state == status.value

    def test_adapter_schema_completeness_tracker(self, now):
        """Tracker adapter produces schema-complete Item per PRD §Interfaces."""
        tracker = Tracker(
            id="tr-complete",
            kind=TrackerKind.task,
            title="Complete tracker",
            state=TrackerState.in_progress,
            nudge_after=now + timedelta(days=1),
            snooze_until=None,
            created_at=now - timedelta(days=1),
            project_id="proj-x",
            entity_id="ent-x",
        )

        result = tracker_adapter([tracker], now)
        item = result[0]

        # All required fields present (can be None, but field exists)
        assert hasattr(item, "id")
        assert item.source == "tracker"
        assert hasattr(item, "kind")
        assert hasattr(item, "title")
        assert hasattr(item, "state")
        assert hasattr(item, "due_at")
        assert hasattr(item, "snoozed_until")  # Can be None but field exists
        assert hasattr(item, "age_days")
        assert hasattr(item, "urgency")
        assert hasattr(item, "project_id")
        assert hasattr(item, "entity_id")
        assert hasattr(item, "actions")

        # to_dict serialization works
        d = item.to_dict()
        assert d["id"] == "tr-complete"
        assert d["source"] == "tracker"

    def test_adapter_schema_completeness_alert(self, now):
        """Alert adapter produces schema-complete Item per PRD §Interfaces."""
        alert = Alert(
            id="alert-complete",
            alert_type=AlertType.due_task,
            title="Complete alert",
            trigger_at=now + timedelta(days=1),
            status=AlertStatus.pending,
            created_at=now - timedelta(days=1),
            project_id="proj-y",
        )

        result = alert_adapter([alert], now)
        item = result[0]

        # All required fields present (can be None, but field exists)
        assert hasattr(item, "id")
        assert item.source == "alert"
        assert hasattr(item, "kind")
        assert hasattr(item, "title")
        assert hasattr(item, "state")
        assert hasattr(item, "due_at")
        assert hasattr(item, "snoozed_until")  # None but field exists
        assert hasattr(item, "age_days")
        assert hasattr(item, "urgency")
        assert hasattr(item, "project_id")
        assert hasattr(item, "entity_id")  # None but field exists
        assert hasattr(item, "actions")

        # to_dict serialization works
        d = item.to_dict()
        assert d["id"] == "alert-complete"
        assert d["source"] == "alert"


class TestTriggerAdapter:
    """Adapter for mapping Trigger objects to Items."""

    def test_trigger_adapter_empty_list(self, now):
        """Empty trigger list returns empty Items list."""
        result = trigger_adapter([], now)
        assert result == []

    def test_trigger_adapter_time_condition_due_at_parsed(self, now):
        """time-condition trigger: due_at parsed from nested condition JSON."""
        trigger_at = now + timedelta(days=2)
        trigger = Trigger(
            id="trg-time",
            name="Renew library card",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": trigger_at.isoformat()},
            action="Remind to renew",
            status=TriggerStatus.enabled,
            created_at=now - timedelta(days=1),
        )

        result = trigger_adapter([trigger], now)

        assert len(result) == 1
        item = result[0]
        assert item.id == "trg-time"
        assert item.source == "trigger"
        assert item.kind == "time"
        assert item.title == "Renew library card"
        assert item.state == "enabled"
        assert item.due_at == trigger_at
        assert item.snoozed_until is None
        assert item.urgency == "due_soon"
        assert item.entity_id is None
        assert len(item.actions) == 1
        assert item.actions[0].verb == "delete"
        assert item.actions[0].tool == "weft_trigger_delete"
        assert item.actions[0].args["trigger_id"] == item.id

    def test_trigger_adapter_threshold_condition_no_date(self, now):
        """threshold-condition trigger: no due_at, urgency=no_date."""
        trigger = Trigger(
            id="trg-threshold",
            name="High memory count",
            condition_type=TriggerConditionType.threshold,
            condition={"metric": "memory_count", "threshold": 1000},
            action="Alert on threshold breach",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"
        assert result[0].kind == "threshold"

    def test_trigger_adapter_event_condition_no_date(self, now):
        """event-condition trigger: no due_at, urgency=no_date."""
        trigger = Trigger(
            id="trg-event",
            name="Deploy completed",
            condition_type=TriggerConditionType.event,
            condition={"event_name": "deploy_complete"},
            action="Notify",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_trigger_adapter_absence_condition_no_date(self, now):
        """absence-condition trigger: no due_at, urgency=no_date."""
        trigger = Trigger(
            id="trg-absence",
            name="No activity logged",
            condition_type=TriggerConditionType.absence,
            condition={"absence_hours": 48},
            action="Nudge",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_trigger_adapter_malformed_trigger_at_treated_as_no_date(self, now):
        """Malformed trigger_at in condition JSON doesn't crash; falls back to no_date."""
        trigger = Trigger(
            id="trg-malformed",
            name="Bad date",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": "not-a-valid-date"},
            action="Remind",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert len(result) == 1
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_trigger_adapter_missing_trigger_at_key_treated_as_no_date(self, now):
        """time-condition trigger with no trigger_at key in condition doesn't crash."""
        trigger = Trigger(
            id="trg-missing",
            name="Odd trigger",
            condition_type=TriggerConditionType.time,
            condition={},
            action="Remind",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_trigger_adapter_hides_canary_by_name(self, now):
        """Trigger whose name contains 'canary' is excluded from the board."""
        trigger = Trigger(
            id="trg-canary",
            name="recall_canary_health_check",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": now.isoformat()},
            action="Audit",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result == []

    def test_trigger_adapter_hides_check_in_variants_by_name(self, now):
        """Trigger names matching check_in / check-in variants are excluded."""
        triggers = [
            Trigger(
                id="trg-checkin-1",
                name="daily_check_in_reminder",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": now.isoformat()},
                action="Nudge",
                created_at=now,
            ),
            Trigger(
                id="trg-checkin-2",
                name="Weekly Check-In",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": now.isoformat()},
                action="Nudge",
                created_at=now,
            ),
        ]

        result = trigger_adapter(triggers, now)
        assert result == []

    def test_trigger_adapter_hide_list_case_insensitive(self, now):
        """Hide-list matching is case-insensitive."""
        trigger = Trigger(
            id="trg-canary-caps",
            name="CANARY Audit",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": now.isoformat()},
            action="Audit",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result == []

    def test_trigger_adapter_does_not_hide_non_matching_names(self, now):
        """A trigger whose name doesn't match the hide-list is kept."""
        trigger = Trigger(
            id="trg-keep",
            name="Follow up with vendor",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": now.isoformat()},
            action="Remind",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert len(result) == 1
        assert result[0].id == "trg-keep"

    def test_trigger_hide_kinds_is_a_module_level_config_list(self):
        """Hide-list is a config list, not buried in an if-branch."""
        assert isinstance(_TRIGGER_HIDE_KINDS, list)
        assert all(isinstance(k, str) for k in _TRIGGER_HIDE_KINDS)
        assert len(_TRIGGER_HIDE_KINDS) > 0

    def test_trigger_adapter_urgency_overdue(self, now):
        """time-condition trigger in the past maps to overdue."""
        trigger = Trigger(
            id="trg-overdue",
            name="Past due reminder",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": (now - timedelta(days=1)).isoformat()},
            action="Remind",
            created_at=now - timedelta(days=3),
        )

        result = trigger_adapter([trigger], now)
        assert result[0].urgency == "overdue"

    def test_trigger_adapter_urgency_pending(self, now):
        """time-condition trigger far in the future maps to pending."""
        trigger = Trigger(
            id="trg-pending",
            name="Future reminder",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": (now + timedelta(days=30)).isoformat()},
            action="Remind",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].urgency == "pending"

    def test_trigger_adapter_project_id_preserved(self, now):
        """Trigger's project_id carries through to the Item."""
        trigger = Trigger(
            id="trg-proj",
            name="Project reminder",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": now.isoformat()},
            action="Remind",
            project_id="proj-z",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        assert result[0].project_id == "proj-z"

    def test_trigger_adapter_multiple_triggers_mixed_visibility(self, now):
        """Mix of hidden and visible triggers: only visible ones are mapped."""
        triggers = [
            Trigger(
                id="trg-visible",
                name="Renew passport",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": now.isoformat()},
                action="Remind",
                created_at=now,
            ),
            Trigger(
                id="trg-hidden",
                name="canary_probe_check",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": now.isoformat()},
                action="Audit",
                created_at=now,
            ),
        ]

        result = trigger_adapter(triggers, now)
        assert len(result) == 1
        assert result[0].id == "trg-visible"

    def test_trigger_adapter_normalizes_naive_due_at_to_utc(self, now):
        """An offset-less trigger_at string yields a NAIVE datetime from
        datetime.fromisoformat — TriggerCreate._validate_condition
        (weft/models.py) does not require tz-aware, unlike
        AlertCreate.trigger_at. trigger_adapter must normalize the parsed
        due_at to UTC at adapter-construction time so downstream consumers
        (rank_items' `due_at.timestamp()`, which treats naive datetimes as
        LOCAL time, and Item.to_dict()'s isoformat()) never see a naive
        value.
        """
        naive_trigger_at_str = "2026-07-03T10:00:00"  # no offset -> naive
        trigger = Trigger(
            id="trg-naive",
            name="Naive due date",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": naive_trigger_at_str},
            action="Remind",
            created_at=now,
        )

        result = trigger_adapter([trigger], now)
        item = result[0]

        assert item.due_at is not None
        assert item.due_at.tzinfo is not None, (
            "due_at must be tz-aware — a naive datetime here means "
            "rank_items' due_at.timestamp() call would treat it as local "
            "time instead of UTC, mis-ordering it on non-UTC hosts"
        )
        assert item.due_at == datetime(2026, 7, 3, 10, 0, 0, tzinfo=timezone.utc)

        d = item.to_dict()
        assert d["due_at"] == "2026-07-03T10:00:00+00:00"

    def test_naive_trigger_due_at_ranks_correctly_against_tz_aware_items(self, now):
        """Mixing a naive-due_at trigger into a bucket with tz-aware items:
        rank_items must order by the true UTC instant. Once trigger_adapter
        normalizes the naive datetime to UTC (this fix), datetime.timestamp()
        on the resulting tz-aware value is host-timezone-independent, so
        ordering against other tz-aware items (alert, tracker, ...) is
        correct regardless of the machine's local timezone.
        """
        earlier_alert_item = Item(
            id="alert-earlier",
            source="alert",
            kind="due_task",
            title="Earlier alert",
            state="pending",
            due_at=now + timedelta(hours=1),
            snoozed_until=None,
            age_days=1.0,
            urgency="due_soon",
        )

        # Naive trigger_at string (no offset) representing a UTC instant
        # LATER than the alert above.
        later_naive_str = (now + timedelta(hours=2)).replace(tzinfo=None).isoformat()
        trigger = Trigger(
            id="trg-later",
            name="Later trigger",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": later_naive_str},
            action="Remind",
            created_at=now,
        )
        trigger_item = trigger_adapter([trigger], now)[0]

        bucket = rank_items([trigger_item, earlier_alert_item])

        assert [item.id for item in bucket] == ["alert-earlier", "trg-later"]

    def test_adapter_schema_completeness_trigger(self, now):
        """Trigger adapter produces schema-complete Item per PRD §Interfaces."""
        trigger = Trigger(
            id="trg-complete",
            name="Complete trigger",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": (now + timedelta(days=1)).isoformat()},
            action="Remind",
            created_at=now - timedelta(days=1),
            project_id="proj-x",
        )

        result = trigger_adapter([trigger], now)
        item = result[0]

        assert hasattr(item, "id")
        assert item.source == "trigger"
        assert hasattr(item, "kind")
        assert hasattr(item, "title")
        assert hasattr(item, "state")
        assert hasattr(item, "due_at")
        assert hasattr(item, "snoozed_until")
        assert hasattr(item, "age_days")
        assert hasattr(item, "urgency")
        assert hasattr(item, "project_id")
        assert hasattr(item, "entity_id")
        assert hasattr(item, "actions")

        d = item.to_dict()
        assert d["id"] == "trg-complete"
        assert d["source"] == "trigger"


class TestTaskAdapter:
    """Adapter for mapping task-memory TaskEntry objects to Items."""

    def test_task_adapter_empty_list(self, now):
        """Empty entry list returns empty Items list."""
        result = task_adapter([], now)
        assert result == []

    def test_task_adapter_single_entry_with_due_date(self, now):
        """Single entry with a due date maps to a fully populated Item."""
        due_str = (now + timedelta(days=2)).strftime("%Y-%m-%d")
        entry = TaskEntry(
            id="mem-1",
            content="Task: Buy groceries\nDue: " + due_str,
            topic=["tasks", f"due:{due_str}", "priority:high"],
            due_date=due_str,
            priority="high",
            created_at=now - timedelta(days=3),
        )

        result = task_adapter([entry], now)

        assert len(result) == 1
        item = result[0]
        assert item.id == "mem-1"
        assert item.source == "task"
        assert item.kind == "high"
        assert item.title == entry.content
        assert item.state is None
        # Ghost-F3 fix: a date-only due tag is end-of-day in BOARD_TIMEZONE,
        # not midnight UTC (see task_due_at).
        expected_due_date = datetime.strptime(due_str, "%Y-%m-%d").date()
        assert item.due_at == datetime.combine(
            expected_due_date, time(23, 59, 59), tzinfo=BOARD_TIMEZONE
        )
        assert item.snoozed_until is None
        assert abs(item.age_days - 3.0) < 0.01
        assert item.project_id is None
        assert item.entity_id is None
        assert item.actions == []

    def test_task_adapter_no_due_date_maps_to_no_date(self, now):
        """Entry without a due date maps to urgency=no_date."""
        entry = TaskEntry(
            id="mem-2",
            content="Task: No deadline",
            topic=["tasks"],
            due_date=None,
            priority=None,
            created_at=now,
        )

        result = task_adapter([entry], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_task_adapter_no_priority_defaults_kind_to_task(self, now):
        """Entry without a priority tag defaults kind to 'task'."""
        entry = TaskEntry(
            id="mem-3",
            content="Task: Unprioritized",
            topic=["tasks"],
            due_date=None,
            priority=None,
            created_at=now,
        )

        result = task_adapter([entry], now)
        assert result[0].kind == "task"

    def test_task_adapter_malformed_due_date_treated_as_no_date(self, now):
        """A malformed due-date string doesn't crash; falls back to no_date."""
        entry = TaskEntry(
            id="mem-4",
            content="Task: Weird date",
            topic=["tasks", "due:not-a-date"],
            due_date="not-a-date",
            priority=None,
            created_at=now,
        )

        result = task_adapter([entry], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_task_adapter_urgency_overdue(self, now):
        """Entry with past due date maps to overdue."""
        past_str = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        entry = TaskEntry(
            id="mem-5",
            content="Task: Overdue",
            topic=["tasks", f"due:{past_str}"],
            due_date=past_str,
            priority=None,
            created_at=now - timedelta(days=10),
        )

        result = task_adapter([entry], now)
        assert result[0].urgency == "overdue"

    def test_task_adapter_urgency_pending(self, now):
        """Entry with due date beyond horizon maps to pending."""
        future_str = (now + timedelta(days=30)).strftime("%Y-%m-%d")
        entry = TaskEntry(
            id="mem-6",
            content="Task: Far future",
            topic=["tasks", f"due:{future_str}"],
            due_date=future_str,
            priority=None,
            created_at=now,
        )

        result = task_adapter([entry], now)
        assert result[0].urgency == "pending"

    def test_task_adapter_multiple_entries(self, now):
        """Multiple entries map to multiple Items."""
        entries = [
            TaskEntry(
                id=f"mem-{i}",
                content=f"Task {i}",
                topic=["tasks"],
                due_date=None,
                priority=None,
                created_at=now - timedelta(days=i),
            )
            for i in range(1, 4)
        ]

        result = task_adapter(entries, now)
        assert len(result) == 3
        for i, item in enumerate(result, start=1):
            assert item.id == f"mem-{i}"
            assert item.source == "task"

    def test_adapter_schema_completeness_task(self, now):
        """Task adapter produces schema-complete Item per PRD §Interfaces."""
        entry = TaskEntry(
            id="mem-complete",
            content="Complete task",
            topic=["tasks", "priority:medium"],
            due_date=None,
            priority="medium",
            created_at=now - timedelta(days=1),
        )

        result = task_adapter([entry], now)
        item = result[0]

        assert hasattr(item, "id")
        assert item.source == "task"
        assert hasattr(item, "kind")
        assert hasattr(item, "title")
        assert hasattr(item, "state")
        assert hasattr(item, "due_at")
        assert hasattr(item, "snoozed_until")
        assert hasattr(item, "age_days")
        assert hasattr(item, "urgency")
        assert hasattr(item, "project_id")
        assert hasattr(item, "entity_id")
        assert hasattr(item, "actions")

        d = item.to_dict()
        assert d["id"] == "mem-complete"
        assert d["source"] == "task"


class TestTaskDueAtGhostF3:
    """Ghost-F3 fix: a date-only `due:YYYY-MM-DD` tag is end-of-day in
    BOARD_TIMEZONE (America/New_York), not midnight UTC — so a task due
    "today" stays due_soon for the whole ET calendar day instead of
    flipping to overdue the instant UTC crosses midnight (which is what
    happened before this fix, and disagreed with up_next's date-granularity
    rollover)."""

    def test_task_due_today_et_buckets_due_soon_not_overdue(self):
        """A task due:<today's ET calendar date> is due_soon (not overdue)
        while `now` is still within that same ET calendar day."""
        now = datetime(2026, 7, 2, 15, 0, 0, tzinfo=timezone.utc)  # mid-day UTC
        today_et = now.astimezone(BOARD_TIMEZONE).date()
        due_str = today_et.strftime("%Y-%m-%d")

        entry = TaskEntry(
            id="mem-today",
            content="Task: Due today",
            topic=["tasks", f"due:{due_str}"],
            due_date=due_str,
            priority=None,
            created_at=now - timedelta(days=1),
        )

        result = task_adapter([entry], now)
        assert result[0].urgency == "due_soon"

    def test_task_due_yesterday_et_flips_overdue_after_et_midnight(self):
        """The same due date rolls over to overdue once ET's midnight for
        that calendar day has passed — matches up_next's date-granularity
        rollover (a due date is only overdue once its whole day has
        elapsed, not at the UTC day boundary)."""
        due_str = "2026-07-02"
        entry = TaskEntry(
            id="mem-rollover",
            content="Task: Rolls over",
            topic=["tasks", f"due:{due_str}"],
            due_date=due_str,
            priority=None,
            created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )

        just_before_et_midnight = datetime(
            2026, 7, 2, 23, 59, 0, tzinfo=BOARD_TIMEZONE,
        )
        assert task_adapter([entry], just_before_et_midnight)[0].urgency == "due_soon"

        just_after_et_midnight = datetime(
            2026, 7, 3, 0, 0, 1, tzinfo=BOARD_TIMEZONE,
        )
        assert task_adapter([entry], just_after_et_midnight)[0].urgency == "overdue"

    def test_task_due_at_parses_to_end_of_day_board_timezone(self):
        """task_due_at parses a bare date into 23:59:59 BOARD_TIMEZONE."""
        due_at = task_due_at("2026-07-02")
        assert due_at == datetime(2026, 7, 2, 23, 59, 59, tzinfo=BOARD_TIMEZONE)

    def test_task_due_at_none_for_missing_or_malformed(self):
        assert task_due_at(None) is None
        assert task_due_at("") is None
        assert task_due_at("not-a-date") is None


class TestReviewAdapter:
    """Adapter for mapping Memory objects to review-queue Items."""

    def test_review_adapter_empty_list(self, now):
        """Empty memory list returns empty Items list."""
        result = review_adapter([], now)
        assert result == []

    def test_review_adapter_single_memory(self, now):
        """Single memory with review_after maps to a fully populated Item."""
        review_at = now + timedelta(days=1)
        memory = Memory(
            id="weft-review1",
            type=MemoryType.decision,
            content="We chose X over Y for reason Z",
            review_after=review_at,
            created_at=now - timedelta(days=10),
            project_id="proj-1",
        )

        result = review_adapter([memory], now)

        assert len(result) == 1
        item = result[0]
        assert item.id == "weft-review1"
        assert item.source == "review"
        assert item.kind == "review"
        assert item.title == "We chose X over Y for reason Z"
        assert item.state is None
        assert item.due_at == review_at
        assert item.snoozed_until is None
        assert abs(item.age_days - 10.0) < 0.01
        assert item.urgency == "due_soon"
        assert item.project_id == "proj-1"
        assert item.entity_id is None
        assert item.actions == []

    def test_review_adapter_null_review_after_maps_to_no_date(self, now):
        """Memory with no review_after maps to urgency=no_date."""
        memory = Memory(
            id="weft-review2",
            type=MemoryType.fact,
            content="Some fact",
            review_after=None,
            created_at=now,
        )

        result = review_adapter([memory], now)
        assert result[0].due_at is None
        assert result[0].urgency == "no_date"

    def test_review_adapter_urgency_overdue(self, now):
        """Memory with past review_after maps to overdue."""
        memory = Memory(
            id="weft-review3",
            type=MemoryType.decision,
            content="Old decision",
            review_after=now - timedelta(days=2),
            created_at=now - timedelta(days=30),
        )

        result = review_adapter([memory], now)
        assert result[0].urgency == "overdue"

    def test_review_adapter_urgency_pending(self, now):
        """Memory with far-future review_after maps to pending."""
        memory = Memory(
            id="weft-review4",
            type=MemoryType.fact,
            content="Far future review",
            review_after=now + timedelta(days=60),
            created_at=now,
        )

        result = review_adapter([memory], now)
        assert result[0].urgency == "pending"

    def test_review_adapter_kind_always_review(self, now):
        """kind is always the literal 'review' regardless of memory type."""
        for mtype in [MemoryType.fact, MemoryType.decision, MemoryType.solution]:
            memory = Memory(
                id=f"weft-{mtype.value}",
                type=mtype,
                content="Some content",
                review_after=now,
                created_at=now,
            )
            result = review_adapter([memory], now)
            assert result[0].kind == "review"

    def test_review_adapter_multiple_memories(self, now):
        """Multiple memories map to multiple Items."""
        memories = [
            Memory(
                id=f"weft-mem-{i}",
                type=MemoryType.fact,
                content=f"Memory {i}",
                review_after=now + timedelta(days=i),
                created_at=now - timedelta(days=i),
            )
            for i in range(1, 4)
        ]

        result = review_adapter(memories, now)
        assert len(result) == 3
        for i, item in enumerate(result, start=1):
            assert item.id == f"weft-mem-{i}"
            assert item.source == "review"

    def test_adapter_schema_completeness_review(self, now):
        """Review adapter produces schema-complete Item per PRD §Interfaces."""
        memory = Memory(
            id="weft-review-complete",
            type=MemoryType.decision,
            content="Complete memory",
            review_after=now + timedelta(days=1),
            created_at=now - timedelta(days=1),
            project_id="proj-y",
        )

        result = review_adapter([memory], now)
        item = result[0]

        assert hasattr(item, "id")
        assert item.source == "review"
        assert hasattr(item, "kind")
        assert hasattr(item, "title")
        assert hasattr(item, "state")
        assert hasattr(item, "due_at")
        assert hasattr(item, "snoozed_until")
        assert hasattr(item, "age_days")
        assert hasattr(item, "urgency")
        assert hasattr(item, "project_id")
        assert hasattr(item, "entity_id")
        assert hasattr(item, "actions")

        d = item.to_dict()
        assert d["id"] == "weft-review-complete"
        assert d["source"] == "review"


# assemble_board integration tests — real testcontainers Postgres

class TestAssembleBoardIntegration:
    """assemble_board() against a real Postgres pool (tests/conftest.py `pool`).

    Proves the behavior unit tests over pure functions can't: concurrent
    fan-out across the five real source-fetch functions, per-source
    isolation (V5), per_source_cap truncation surfacing (V7), multi-source
    bucketing (V2), and read-purity (V6).
    """

    async def test_assembles_and_buckets_items_from_all_five_sources(self, pool):
        """(a) items from multiple sources assemble + bucket (V2)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.alerts import create_alert
        from weft.store import store_memory
        from weft.trackers import create_tracker
        from weft.triggers import create_trigger

        now = datetime.now(timezone.utc)

        # Overdue tracker (nudge_after in the past)
        await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Overdue tracker item",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(days=2),
            ),
        )

        # Due-soon alert (trigger_at within default 7d horizon)
        await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Due soon alert",
                trigger_at=now + timedelta(days=1),
                channel=AlertChannel.log,
            ),
        )

        # Pending trigger (time-condition due_at beyond the default horizon)
        far_future = (now + timedelta(days=30)).isoformat()
        await create_trigger(
            pool,
            TriggerCreate(
                name="Renew library card",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": far_future},
                action="Remind to renew",
            ),
        )

        # No-date task-memory (no due: topic tag)
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Task: No deadline yet",
                topic=["tasks"],
            ),
        )

        # Overdue review-queue memory (review_after in the past)
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.decision,
                content="Old decision to review",
                review_after=now - timedelta(days=1),
            ),
        )

        result = await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)

        assert result["warnings"] == []
        assert result["horizon_days"] == 7
        assert set(result["buckets"].keys()) == {
            "overdue", "due_soon", "pending", "no_date",
        }

        sources_seen = {item["source"] for item in result["items"]}
        assert sources_seen == {"tracker", "alert", "trigger", "task", "review"}

        assert result["counts"]["total"] == 5
        assert result["counts"]["overdue"] == 2  # tracker + review
        assert result["counts"]["due_soon"] == 1  # alert
        assert result["counts"]["pending"] == 1  # trigger
        assert result["counts"]["no_date"] == 1  # task

        overdue_sources = {item["source"] for item in result["buckets"]["overdue"]}
        assert overdue_sources == {"tracker", "review"}
        assert result["buckets"]["due_soon"][0]["source"] == "alert"
        assert result["buckets"]["pending"][0]["source"] == "trigger"
        assert result["buckets"]["no_date"][0]["source"] == "task"

    async def test_one_source_failure_isolated_others_still_return(
        self, pool, monkeypatch,
    ):
        """(b) one source raising -> warnings entry + others still returned (V5)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.alerts import create_alert

        now = datetime.now(timezone.utc)

        await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Alert survives tracker failure",
                trigger_at=now + timedelta(hours=1),
                channel=AlertChannel.log,
            ),
        )

        async def _raise_due_trackers(*args, **kwargs):
            raise RuntimeError("simulated tracker source failure")

        monkeypatch.setattr("weft.trackers.due_trackers", _raise_due_trackers)

        result = await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)

        tracker_warnings = [w for w in result["warnings"] if w["source"] == "tracker"]
        assert len(tracker_warnings) == 1
        assert "error" in tracker_warnings[0]

        # The call itself did not raise, and the other source's item is intact.
        alert_items = [i for i in result["items"] if i["source"] == "alert"]
        assert len(alert_items) == 1
        assert alert_items[0]["title"] == "Alert survives tracker failure"

        tracker_items = [i for i in result["items"] if i["source"] == "tracker"]
        assert tracker_items == []

    async def test_per_source_cap_truncation_emits_warning(self, pool):
        """(c) per_source_cap truncation emits a warning (V7)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.trackers import create_tracker

        now = datetime.now(timezone.utc)
        for i in range(5):
            await create_tracker(
                pool,
                TrackerCreate(
                    kind=TrackerKind.task,
                    title=f"Tracker {i}",
                    nudge_mode=NudgeMode.once,
                    nudge_after=now - timedelta(hours=i),
                ),
            )

        result = await assemble_board(
            pool, now=now, per_source_cap=3, user_id=DEFAULT_TEST_USER_ID,
        )

        truncation_warnings = [
            w for w in result["warnings"]
            if w["source"] == "tracker" and w.get("truncated")
        ]
        assert len(truncation_warnings) == 1
        assert truncation_warnings[0]["cap"] == 3

        tracker_items = [i for i in result["items"] if i["source"] == "tracker"]
        assert len(tracker_items) == 3

    async def test_read_pure_no_source_row_mutation(self, pool):
        """assemble_board performs no writes — source rows unchanged (V6)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.alerts import create_alert
        from weft.store import store_memory
        from weft.trackers import create_tracker
        from weft.triggers import create_trigger

        now = datetime.now(timezone.utc)

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Read-purity tracker",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        alert = await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Read-purity alert",
                trigger_at=now + timedelta(hours=1),
                channel=AlertChannel.log,
            ),
        )
        trigger = await create_trigger(
            pool,
            TriggerCreate(
                name="Read-purity trigger",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": now.isoformat()},
                action="Remind",
            ),
        )
        memory = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.decision,
                content="Read-purity review memory",
                review_after=now - timedelta(hours=1),
            ),
        )

        counts_before = {
            "trackers": await pool.fetchval("SELECT count(*) FROM trackers"),
            "alerts": await pool.fetchval("SELECT count(*) FROM alerts"),
            "triggers": await pool.fetchval("SELECT count(*) FROM triggers"),
            "memories": await pool.fetchval("SELECT count(*) FROM memories"),
        }
        rows_before = {
            "tracker_state": await pool.fetchval(
                "SELECT state FROM trackers WHERE id = $1", tracker.id,
            ),
            "alert_status": await pool.fetchval(
                "SELECT status FROM alerts WHERE id = $1", alert.id,
            ),
            "trigger_status": await pool.fetchval(
                "SELECT status FROM triggers WHERE id = $1", trigger.id,
            ),
            "memory_status": await pool.fetchval(
                "SELECT status FROM memories WHERE id = $1", memory.id,
            ),
        }

        await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)

        counts_after = {
            "trackers": await pool.fetchval("SELECT count(*) FROM trackers"),
            "alerts": await pool.fetchval("SELECT count(*) FROM alerts"),
            "triggers": await pool.fetchval("SELECT count(*) FROM triggers"),
            "memories": await pool.fetchval("SELECT count(*) FROM memories"),
        }
        rows_after = {
            "tracker_state": await pool.fetchval(
                "SELECT state FROM trackers WHERE id = $1", tracker.id,
            ),
            "alert_status": await pool.fetchval(
                "SELECT status FROM alerts WHERE id = $1", alert.id,
            ),
            "trigger_status": await pool.fetchval(
                "SELECT status FROM triggers WHERE id = $1", trigger.id,
            ),
            "memory_status": await pool.fetchval(
                "SELECT status FROM memories WHERE id = $1", memory.id,
            ),
        }

        assert counts_before == counts_after
        assert rows_before == rows_after

    async def test_assemble_board_gets_fresh_connections_when_called_inside_acquire(
        self, pool,
    ):
        """assemble_board must not inherit the caller's already-bound
        connection via `weft.db.connection.acquire()`'s idempotency.

        Every MCP tool wraps its handler body in
        `async with acquire(app.pool):`, so once `weft_board` is wired that
        connection is already bound (via the `_current_conn` contextvar)
        when assemble_board's `asyncio.gather` fan-out starts. Each of the
        5 concurrent gather tasks gets a *copy* of that context; without a
        fresh-connection guarantee, all 5 would call `acquire()`'s
        idempotent branch and share ONE asyncpg connection, which cannot
        serve concurrent queries — asyncpg raises "another operation is in
        progress", `_safe` swallows it into `warnings`, and the board comes
        back with some/all sources silently missing.

        This test reproduces that exact calling convention directly (an
        outer `acquire(pool)` scope around the assemble_board call) and
        asserts no per-source failures occurred and items from multiple
        sources still made it into the result.
        """
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.alerts import create_alert
        from weft.db.connection import acquire
        from weft.trackers import create_tracker
        from weft.triggers import create_trigger

        now = datetime.now(timezone.utc)

        await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Tracker under outer acquire",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Alert under outer acquire",
                trigger_at=now + timedelta(hours=1),
                channel=AlertChannel.log,
            ),
        )
        await create_trigger(
            pool,
            TriggerCreate(
                name="Trigger under outer acquire",
                condition_type=TriggerConditionType.time,
                condition={"trigger_at": (now + timedelta(hours=2)).isoformat()},
                action="Remind",
            ),
        )

        # Reproduce the MCP-tool calling convention: assemble_board invoked
        # from INSIDE an already-open acquire() scope, so `_current_conn`
        # is already bound when the fan-out's asyncio.gather runs.
        async with acquire(pool):
            result = await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)

        assert result["warnings"] == [], (
            "fan-out sources failed under a shared inherited connection — "
            f"got warnings: {result['warnings']}"
        )

        sources_seen = {item["source"] for item in result["items"]}
        assert sources_seen == {"tracker", "alert", "trigger"}, (
            f"expected all 3 seeded sources present, got: {sources_seen}"
        )
        assert result["counts"]["total"] == 3

    async def test_sources_param_restricts_fan_out(self, pool):
        """`sources=` narrows the board to the named subset — the other
        sources are not fetched at all, not merely filtered post-hoc
        (Ratified decision weft-64c14697 / Epic Task 5)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.alerts import create_alert
        from weft.trackers import create_tracker

        now = datetime.now(timezone.utc)

        await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Tracker for sources filter",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.due_task,
                title="Alert for sources filter",
                trigger_at=now + timedelta(hours=1),
                channel=AlertChannel.log,
            ),
        )

        result = await assemble_board(
            pool, now=now, user_id=DEFAULT_TEST_USER_ID, sources=["tracker"],
        )

        sources_seen = {item["source"] for item in result["items"]}
        assert sources_seen == {"tracker"}
        assert result["counts"]["total"] == 1
        assert result["warnings"] == []

    async def test_include_snoozed_surfaces_snoozed_trackers(self, pool):
        """A tracker with a future `snooze_until` is hidden by default and
        surfaced only when `include_snoozed=True` (PRD §Validation V3;
        Ratified decision weft-64c14697 / Epic Task 5)."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.trackers import create_tracker, snooze_tracker

        now = datetime.now(timezone.utc)

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Snoozed tracker",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        await snooze_tracker(pool, tracker.id, now + timedelta(days=1))

        default_result = await assemble_board(
            pool, now=now, user_id=DEFAULT_TEST_USER_ID, sources=["tracker"],
        )
        assert default_result["counts"]["total"] == 0

        snoozed_result = await assemble_board(
            pool, now=now, user_id=DEFAULT_TEST_USER_ID, sources=["tracker"],
            include_snoozed=True,
        )
        tracker_ids = {item["id"] for item in snoozed_result["items"]}
        assert tracker.id in tracker_ids

    async def test_misconfigured_identity_emits_structured_warning(
        self, pool, monkeypatch,
    ):
        """No resolvable caller identity (no ambient caller, no
        WEFT_DEFAULT_USER_ID) surfaces a structured `warnings[]` entry, not
        just a log line — so 'misconfigured' is distinguishable from
        'genuinely nothing is due' (Ratified decision weft-64c14697 / Epic
        Task 5)."""
        monkeypatch.delenv("WEFT_DEFAULT_USER_ID", raising=False)

        now = datetime.now(timezone.utc)
        result = await assemble_board(pool, now=now)  # no user_id, no ambient caller

        identity_warnings = [
            w for w in result["warnings"] if w.get("source") == "identity"
        ]
        assert len(identity_warnings) == 1
        assert identity_warnings[0]["error"] == "no_default_user"

    async def test_every_board_action_tool_is_allowlisted(self, pool):
        """Security-boundary invariant: every action the board offers must name
        a tool in ACT_ALLOWLIST — otherwise POST /act would reject an action
        the UI legitimately rendered. Ties the descriptor producer (adapters)
        to the dispatch allowlist so they can't drift apart."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.board import ACT_ALLOWLIST
        from weft.alerts import create_alert
        from weft.trackers import create_tracker
        from weft.triggers import create_trigger

        now = datetime.now(timezone.utc)
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.task, title="T", nudge_mode=NudgeMode.once,
            nudge_after=now - timedelta(days=1)))
        await create_alert(pool, AlertCreate(
            alert_type=AlertType.due_task, title="A",
            trigger_at=now + timedelta(hours=1), channel=AlertChannel.log))
        await create_trigger(pool, TriggerCreate(
            name="Trg", condition_type=TriggerConditionType.time,
            condition={"trigger_at": now.isoformat()}, action="x"))

        board = await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)
        seen = 0
        for item in board["items"]:
            for action in item["actions"]:
                assert action["tool"] in ACT_ALLOWLIST, (
                    f"{item['source']} action {action['tool']} not in allowlist"
                )
                seen += 1
        assert seen >= 3, "expected tracker/alert/trigger items to carry actions"

    async def test_ingest_source_review_memories_excluded_from_board(self, pool):
        """loom-40c5fa78: codebase-ingest memories (source='ingest') carry a
        review_after but are system-generated noise, not owner triage items —
        they must NOT surface on the board's review bucket, while a genuine
        conversation-origin review memory still does."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.models import MemorySource
        from weft.store import store_memory

        now = datetime.now(timezone.utc)
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Summary: this file contains unit tests for the cache.",
                source=MemorySource.ingest,
                review_after=now - timedelta(days=30),
            ),
        )
        genuine = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.decision,
                content="Decided to go all-in on making Weft public",
                source=MemorySource.conversation,
                review_after=now - timedelta(days=1),
            ),
        )

        result = await assemble_board(pool, now=now, user_id=DEFAULT_TEST_USER_ID)
        review_ids = {
            i["id"] for i in result["items"] if i["source"] == "review"
        }
        assert genuine.id in review_ids, "genuine conversation review item must surface"
        review_contents = [
            i["title"] for i in result["items"] if i["source"] == "review"
        ]
        assert not any("unit tests for the cache" in c for c in review_contents), (
            "ingest-source review memories must be excluded from the board"
        )
