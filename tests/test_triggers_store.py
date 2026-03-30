"""Tests for triggers store layer — CRUD, cooldown, and due-trigger queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import Trigger, TriggerConditionType, TriggerCreate, TriggerStatus
from weft.triggers import (
    create_trigger,
    delete_trigger,
    get_trigger,
    get_triggers_due,
    is_cooldown_elapsed,
    list_triggers,
    record_fire,
    update_trigger,
)


# --- Helpers ---


async def _make_trigger(pool, name="test trigger", **kwargs):
    defaults = {
        "condition_type": TriggerConditionType.event,
        "condition": {"event_name": "test_event"},
        "action": "notify user",
    }
    defaults.update(kwargs)
    return await create_trigger(pool, TriggerCreate(name=name, **defaults))


# --- create / get ---


async def test_create_trigger_minimal(pool):
    t = await _make_trigger(pool)
    assert t.id.startswith("weft-")
    assert t.name == "test trigger"
    assert t.condition_type == TriggerConditionType.event
    assert t.action == "notify user"
    assert t.status == TriggerStatus.enabled
    assert t.fire_count == 0
    assert t.last_fired_at is None
    assert t.cooldown_hours is None
    assert t.max_fires is None


async def test_create_trigger_with_condition(pool):
    t = await _make_trigger(
        pool,
        name="threshold trigger",
        condition_type=TriggerConditionType.threshold,
        condition={"metric": "memory_count", "threshold": 100},
        cooldown_hours=4.0,
        max_fires=3,
    )
    assert t.condition == {"metric": "memory_count", "threshold": 100}
    assert t.cooldown_hours == pytest.approx(4.0)
    assert t.max_fires == 3


async def test_get_trigger(pool):
    t = await _make_trigger(pool)
    fetched = await get_trigger(pool, t.id)
    assert fetched is not None
    assert fetched.id == t.id
    assert fetched.name == t.name


async def test_get_trigger_not_found(pool):
    result = await get_trigger(pool, "weft-nonexistent")
    assert result is None


# --- list ---


async def test_list_triggers_all(pool):
    await _make_trigger(pool, name="t1")
    await _make_trigger(pool, name="t2")
    await _make_trigger(pool, name="t3")
    result = await list_triggers(pool)
    assert len(result) == 3


async def test_list_triggers_by_condition_type(pool):
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    await _make_trigger(pool, name="time", condition_type=TriggerConditionType.time,
                        condition={"trigger_at": future})
    await _make_trigger(pool, name="event", condition_type=TriggerConditionType.event)
    await _make_trigger(pool, name="event2", condition_type=TriggerConditionType.event)

    result = await list_triggers(pool, condition_type=TriggerConditionType.event)
    assert len(result) == 2
    assert all(t.condition_type == TriggerConditionType.event for t in result)


async def test_list_triggers_by_status(pool):
    t = await _make_trigger(pool, name="will disable")
    await update_trigger(pool, t.id, status=TriggerStatus.disabled)
    await _make_trigger(pool, name="stays enabled")

    enabled = await list_triggers(pool, status=TriggerStatus.enabled)
    assert len(enabled) == 1
    assert enabled[0].name == "stays enabled"


# --- update ---


async def test_update_trigger_fields(pool):
    t = await _make_trigger(pool)
    updated = await update_trigger(
        pool, t.id,
        name="renamed",
        action="different action",
        cooldown_hours=2.0,
    )
    assert updated.name == "renamed"
    assert updated.action == "different action"
    assert updated.cooldown_hours == pytest.approx(2.0)
    assert updated.updated_at > t.updated_at


async def test_update_trigger_disable(pool):
    t = await _make_trigger(pool)
    updated = await update_trigger(pool, t.id, status=TriggerStatus.disabled)
    assert updated.status == TriggerStatus.disabled


async def test_update_trigger_not_found(pool):
    result = await update_trigger(pool, "weft-nonexistent", name="nope")
    assert result is None


# --- delete ---


async def test_delete_trigger(pool):
    t = await _make_trigger(pool)
    assert await delete_trigger(pool, t.id) is True
    assert await get_trigger(pool, t.id) is None


async def test_delete_trigger_not_found(pool):
    assert await delete_trigger(pool, "weft-nonexistent") is False


# --- record_fire ---


async def test_record_fire_increments(pool):
    t = await _make_trigger(pool)
    fired = await record_fire(pool, t.id)
    assert fired.fire_count == 1
    assert fired.last_fired_at is not None
    assert fired.status == TriggerStatus.enabled


async def test_record_fire_max_fires(pool):
    t = await _make_trigger(pool, max_fires=2)
    await record_fire(pool, t.id)
    fired = await record_fire(pool, t.id)
    assert fired.fire_count == 2
    assert fired.status == TriggerStatus.fired


async def test_record_fire_not_found(pool):
    with pytest.raises(LookupError, match="not found"):
        await record_fire(pool, "weft-nonexistent")


async def test_record_fire_disabled_trigger(pool):
    t = await _make_trigger(pool)
    await update_trigger(pool, t.id, status=TriggerStatus.disabled)
    with pytest.raises(ValueError, match="not enabled"):
        await record_fire(pool, t.id)


# --- is_cooldown_elapsed ---


def test_cooldown_no_cooldown():
    t = Trigger(
        name="no cd", condition_type=TriggerConditionType.event,
        action="x", cooldown_hours=None,
    )
    assert is_cooldown_elapsed(t) is True


def test_cooldown_never_fired():
    t = Trigger(
        name="never fired", condition_type=TriggerConditionType.event,
        action="x", cooldown_hours=1.0, last_fired_at=None,
    )
    assert is_cooldown_elapsed(t) is True


def test_cooldown_not_elapsed():
    now = datetime.now(timezone.utc)
    t = Trigger(
        name="recent", condition_type=TriggerConditionType.event,
        action="x", cooldown_hours=1.0,
        last_fired_at=now - timedelta(minutes=30),
    )
    assert is_cooldown_elapsed(t, now) is False


def test_cooldown_elapsed():
    now = datetime.now(timezone.utc)
    t = Trigger(
        name="old", condition_type=TriggerConditionType.event,
        action="x", cooldown_hours=1.0,
        last_fired_at=now - timedelta(hours=2),
    )
    assert is_cooldown_elapsed(t, now) is True


# --- get_triggers_due ---


async def test_get_triggers_due_basic(pool):
    """Enabled trigger with no cooldown is due."""
    await _make_trigger(pool, name="due")
    due = await get_triggers_due(pool)
    assert len(due) == 1
    assert due[0].name == "due"


async def test_get_triggers_due_respects_cooldown(pool):
    """Trigger within cooldown period is not due."""
    t = await _make_trigger(pool, name="cd", cooldown_hours=2.0)
    await record_fire(pool, t.id)

    due = await get_triggers_due(pool)
    assert len(due) == 0


async def test_get_triggers_due_cooldown_elapsed(pool):
    """Trigger past cooldown period is due again."""
    t = await _make_trigger(pool, name="cd", cooldown_hours=1.0)
    await record_fire(pool, t.id)

    future = datetime.now(timezone.utc) + timedelta(hours=2)
    due = await get_triggers_due(pool, now=future)
    assert len(due) == 1


async def test_get_triggers_due_disabled_excluded(pool):
    """Disabled triggers are not due."""
    t = await _make_trigger(pool, name="disabled")
    await update_trigger(pool, t.id, status=TriggerStatus.disabled)

    due = await get_triggers_due(pool)
    assert len(due) == 0


async def test_get_triggers_due_max_fires_reached(pool):
    """Trigger that hit max_fires is not due even if enabled."""
    t = await _make_trigger(pool, name="maxed", max_fires=1)
    await record_fire(pool, t.id)  # status -> fired

    due = await get_triggers_due(pool)
    assert len(due) == 0


async def test_get_triggers_due_time_condition(pool):
    """Time trigger is only due after trigger_at."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    await _make_trigger(
        pool, name="future",
        condition_type=TriggerConditionType.time,
        condition={"trigger_at": future},
    )
    await _make_trigger(
        pool, name="past",
        condition_type=TriggerConditionType.time,
        condition={"trigger_at": past},
    )

    due = await get_triggers_due(pool)
    assert len(due) == 1
    assert due[0].name == "past"


async def test_get_triggers_due_absence_condition(pool):
    """Absence trigger fires after absence_hours since creation."""
    now = datetime.now(timezone.utc)

    await _make_trigger(
        pool, name="absence",
        condition_type=TriggerConditionType.absence,
        condition={"absence_hours": 2},
    )

    # Not due yet (just created)
    due = await get_triggers_due(pool, now=now + timedelta(hours=1))
    assert len(due) == 0

    # Due after absence threshold
    due = await get_triggers_due(pool, now=now + timedelta(hours=3))
    assert len(due) == 1
    assert due[0].name == "absence"


async def test_get_triggers_due_filter_by_condition_type(pool):
    """Can filter due triggers by condition_type."""
    await _make_trigger(pool, name="event", condition_type=TriggerConditionType.event)
    await _make_trigger(pool, name="threshold", condition_type=TriggerConditionType.threshold,
                        condition={"metric": "memory_count", "threshold": 100})

    due = await get_triggers_due(pool, condition_type=TriggerConditionType.event)
    assert len(due) == 1
    assert due[0].name == "event"


async def test_get_triggers_due_filter_by_project(pool):
    """Can filter due triggers by project_id."""
    await _make_trigger(pool, name="proj-a", project_id="alpha")
    await _make_trigger(pool, name="proj-b", project_id="beta")

    due = await get_triggers_due(pool, project_id="alpha")
    # OR-NULL scoping means both project-specific and NULL project triggers match
    names = {t.name for t in due}
    assert "proj-a" in names


# --- to_dict ---


async def test_trigger_to_dict(pool):
    t = await _make_trigger(
        pool,
        condition={"event_name": "deploy", "extra": "metadata"},
        cooldown_hours=1.5,
    )
    d = t.to_dict()
    assert d["condition_type"] == "event"
    assert d["status"] == "enabled"
    assert d["condition"]["event_name"] == "deploy"
    assert d["cooldown_hours"] == pytest.approx(1.5)


# --- Condition validation ---


def test_validation_time_requires_trigger_at():
    with pytest.raises(ValueError, match="trigger_at"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.time,
            condition={}, action="x",
        )


def test_validation_time_rejects_bad_iso():
    with pytest.raises(ValueError, match="valid ISO datetime"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.time,
            condition={"trigger_at": "not-a-date"}, action="x",
        )


def test_validation_threshold_requires_metric_and_value():
    with pytest.raises(ValueError, match="metric"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.threshold,
            condition={"threshold": 10}, action="x",
        )
    with pytest.raises(ValueError, match="threshold"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.threshold,
            condition={"metric": "count"}, action="x",
        )


def test_validation_threshold_rejects_non_numeric():
    with pytest.raises(ValueError, match="number"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.threshold,
            condition={"metric": "count", "threshold": "high"}, action="x",
        )


def test_validation_event_requires_event_name():
    with pytest.raises(ValueError, match="event_name"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.event,
            condition={}, action="x",
        )


def test_validation_absence_requires_hours():
    with pytest.raises(ValueError, match="absence_hours"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.absence,
            condition={}, action="x",
        )


def test_validation_absence_rejects_negative():
    with pytest.raises(ValueError, match="positive"):
        TriggerCreate(
            name="bad", condition_type=TriggerConditionType.absence,
            condition={"absence_hours": -1}, action="x",
        )


def test_validation_allows_extra_keys():
    """Extra keys in condition are allowed — only required keys are enforced."""
    tc = TriggerCreate(
        name="ok", condition_type=TriggerConditionType.event,
        condition={"event_name": "deploy", "channel": "#ops"}, action="x",
    )
    assert tc.condition["channel"] == "#ops"
