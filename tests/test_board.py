"""Unit tests for weft.board — Item model, urgency bucketing, and ranking.

Tests cover:
- Item schema construction and serialization per PRD §Interfaces
- Urgency bucketing per PRD §Validation V2 (overdue/due_soon/pending/no_date)
- Ranking within a bucket (oldest-due-first, age_days desc, title)
- Boundary cases at exactly now and exactly horizon cutoff
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.board import Action, Item, Urgency, calculate_urgency, rank_items


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
