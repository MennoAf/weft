"""Tests for proactive trigger MCP tools.

Exercises weft_trigger_create, weft_trigger_list, weft_trigger_due,
weft_trigger_fire, and weft_trigger_delete through the store layer
(tools wrap these with MCP context; store functions are the unit under test).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import TriggerConditionType, TriggerCreate, TriggerStatus
from weft.triggers import (
    create_trigger,
    delete_trigger,
    get_trigger,
    get_triggers_due,
    list_triggers,
    record_fire,
)


@pytest.fixture
async def _clean_triggers(pool):
    """Ensure triggers table is empty before each test."""
    await pool.execute("DELETE FROM triggers")
    yield
    await pool.execute("DELETE FROM triggers")


@pytest.mark.asyncio
async def test_create_and_get_trigger(pool, _clean_triggers):
    """Create a time-based trigger and retrieve it by ID."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    create = TriggerCreate(
        name="Morning standup reminder",
        condition_type=TriggerConditionType.time,
        condition={"trigger_at": future},
        action="Send standup reminder to Slack",
        cooldown_hours=24.0,
        max_fires=None,
        project_id="test-project",
    )
    trigger = await create_trigger(pool, create)

    assert trigger.id.startswith("weft-")
    assert trigger.name == "Morning standup reminder"
    assert trigger.condition_type == TriggerConditionType.time
    assert trigger.status == TriggerStatus.enabled
    assert trigger.fire_count == 0
    assert trigger.cooldown_hours == 24.0
    assert trigger.project_id == "test-project"

    fetched = await get_trigger(pool, trigger.id)
    assert fetched is not None
    assert fetched.id == trigger.id
    assert fetched.condition["trigger_at"] == future


@pytest.mark.asyncio
async def test_list_triggers_with_filters(pool, _clean_triggers):
    """List triggers filtered by condition_type and status."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    await create_trigger(
        pool,
        TriggerCreate(
            name="Time trigger",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": future},
            action="do something",
        ),
    )
    await create_trigger(
        pool,
        TriggerCreate(
            name="Absence trigger",
            condition_type=TriggerConditionType.absence,
            condition={"absence_hours": 48},
            action="check in",
        ),
    )

    # All triggers
    all_triggers = await list_triggers(pool)
    assert len(all_triggers) == 2

    # Filter by condition_type
    time_only = await list_triggers(pool, condition_type=TriggerConditionType.time)
    assert len(time_only) == 1
    assert time_only[0].name == "Time trigger"

    absence_only = await list_triggers(pool, condition_type=TriggerConditionType.absence)
    assert len(absence_only) == 1
    assert absence_only[0].name == "Absence trigger"


@pytest.mark.asyncio
async def test_trigger_due_respects_cooldown(pool, _clean_triggers):
    """Trigger that was recently fired should not appear in due list."""
    # Create a trigger that's been fired recently with 24h cooldown
    create = TriggerCreate(
        name="Event trigger",
        condition_type=TriggerConditionType.event,
        condition={"event_name": "deploy"},
        action="run smoke tests",
        cooldown_hours=24.0,
    )
    trigger = await create_trigger(pool, create)

    # Initially due (never fired)
    due = await get_triggers_due(pool)
    assert any(t.id == trigger.id for t in due)

    # Fire it
    await record_fire(pool, trigger.id)

    # No longer due (within cooldown)
    due_after = await get_triggers_due(pool)
    assert not any(t.id == trigger.id for t in due_after)

    # Simulate cooldown elapsed by querying with future time
    far_future = datetime.now(timezone.utc) + timedelta(hours=25)
    due_later = await get_triggers_due(pool, now=far_future)
    assert any(t.id == trigger.id for t in due_later)


@pytest.mark.asyncio
async def test_trigger_due_time_condition(pool, _clean_triggers):
    """Time triggers only appear due after their trigger_at time."""
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    future_trigger = await create_trigger(
        pool,
        TriggerCreate(
            name="Future trigger",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": future},
            action="do later",
        ),
    )
    past_trigger = await create_trigger(
        pool,
        TriggerCreate(
            name="Past trigger",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": past},
            action="do now",
        ),
    )

    due = await get_triggers_due(pool)
    due_ids = [t.id for t in due]
    assert past_trigger.id in due_ids
    assert future_trigger.id not in due_ids


@pytest.mark.asyncio
async def test_trigger_fire_increments_and_maxes_out(pool, _clean_triggers):
    """record_fire increments fire_count and transitions to 'fired' at max_fires."""
    create = TriggerCreate(
        name="One-shot",
        condition_type=TriggerConditionType.event,
        condition={"event_name": "test"},
        action="fire once",
        max_fires=2,
    )
    trigger = await create_trigger(pool, create)

    # First fire
    updated = await record_fire(pool, trigger.id)
    assert updated.fire_count == 1
    assert updated.status == TriggerStatus.enabled
    assert updated.last_fired_at is not None

    # Second fire — hits max_fires
    updated2 = await record_fire(pool, trigger.id)
    assert updated2.fire_count == 2
    assert updated2.status == TriggerStatus.fired

    # Cannot fire again — status is no longer enabled
    with pytest.raises(ValueError, match="not enabled"):
        await record_fire(pool, trigger.id)


@pytest.mark.asyncio
async def test_delete_trigger(pool, _clean_triggers):
    """Delete removes trigger from the database."""
    create = TriggerCreate(
        name="Temp trigger",
        condition_type=TriggerConditionType.event,
        condition={"event_name": "temp"},
        action="temporary",
    )
    trigger = await create_trigger(pool, create)

    deleted = await delete_trigger(pool, trigger.id)
    assert deleted is True

    fetched = await get_trigger(pool, trigger.id)
    assert fetched is None

    # Deleting again returns False
    deleted_again = await delete_trigger(pool, trigger.id)
    assert deleted_again is False


@pytest.mark.asyncio
async def test_trigger_due_absence_condition(pool, _clean_triggers):
    """Absence triggers fire after absence_hours since last fire or creation."""
    create = TriggerCreate(
        name="Check-in reminder",
        condition_type=TriggerConditionType.absence,
        condition={"absence_hours": 24},
        action="remind user to check in",
    )
    trigger = await create_trigger(pool, create)

    # Just created — absence period hasn't elapsed
    due_now = await get_triggers_due(pool)
    assert not any(t.id == trigger.id for t in due_now)

    # 25 hours later — absence period has elapsed
    far_future = datetime.now(timezone.utc) + timedelta(hours=25)
    due_later = await get_triggers_due(pool, now=far_future)
    assert any(t.id == trigger.id for t in due_later)


@pytest.mark.asyncio
async def test_trigger_to_dict_serialization(pool, _clean_triggers):
    """Trigger.to_dict() produces JSON-safe output."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    trigger = await create_trigger(
        pool,
        TriggerCreate(
            name="Serialization test",
            condition_type=TriggerConditionType.time,
            condition={"trigger_at": future},
            action="test",
        ),
    )
    d = trigger.to_dict()
    assert d["condition_type"] == "time"
    assert d["status"] == "enabled"
    assert isinstance(d["condition"], dict)
    assert "trigger_at" in d["condition"]
