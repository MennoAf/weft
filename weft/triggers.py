"""Triggers store — CRUD, cooldown logic, and due-trigger queries.

Proactive triggers are condition-driven rules that fire actions when
their conditions are met. Supports time, threshold, event, and absence
condition types with configurable cooldown and max-fire limits.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.db.connection import get_db
from weft.models import (
    Trigger,
    TriggerConditionType,
    TriggerCreate,
    TriggerStatus,
    _weft_id,
)

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None


# --- CRUD ---


async def create_trigger(
    pool: asyncpg.Pool,
    create: TriggerCreate,
) -> Trigger:
    """Create a new trigger. Returns the created Trigger."""
    trigger_id = _weft_id()
    now = datetime.now(timezone.utc)
    condition_json = json.dumps(create.condition)

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO triggers (
            id, name, condition_type, condition, action,
            status, cooldown_hours, max_fires,
            project_id, agent_id, user_id,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4::jsonb, $5,
            'enabled', $6, $7,
            $8, $9, nullif(current_setting('app.user_id', true), ''),
            $10, $10
        )
        RETURNING *
        """,
        trigger_id,
        create.name,
        create.condition_type.value,
        condition_json,
        create.action,
        create.cooldown_hours,
        create.max_fires,
        create.project_id,
        create.agent_id,
        now,
    )
    return _row_to_trigger(row)


async def get_trigger(pool: asyncpg.Pool, trigger_id: str) -> Trigger | None:
    """Fetch a trigger by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM triggers WHERE id = $1",
        trigger_id,
    )
    return _row_to_trigger(row) if row else None


async def list_triggers(
    pool: asyncpg.Pool,
    *,
    condition_type: TriggerConditionType | None = None,
    status: TriggerStatus | None = None,
    project_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Trigger]:
    """List triggers with optional filters."""
    conditions: list[str] = []
    params: list = []
    idx = 1

    if condition_type is not None:
        conditions.append(f"condition_type = ${idx}")
        params.append(condition_type.value)
        idx += 1

    if status is not None:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM triggers {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_trigger(r) for r in rows]


async def update_trigger(
    pool: asyncpg.Pool,
    trigger_id: str,
    *,
    name: str | None = None,
    action: str | None = None,
    condition: dict | None = None,
    cooldown_hours: float | None = _UNSET,
    max_fires: int | None = _UNSET,
    status: TriggerStatus | None = None,
) -> Trigger | None:
    """Update mutable fields on a trigger. Returns updated Trigger or None."""
    sets: list[str] = []
    params: list = []
    idx = 1

    if name is not None:
        sets.append(f"name = ${idx}")
        params.append(name)
        idx += 1

    if action is not None:
        sets.append(f"action = ${idx}")
        params.append(action)
        idx += 1

    if condition is not None:
        sets.append(f"condition = ${idx}::jsonb")
        params.append(json.dumps(condition))
        idx += 1

    if cooldown_hours is not _UNSET:
        sets.append(f"cooldown_hours = ${idx}")
        params.append(cooldown_hours)
        idx += 1

    if max_fires is not _UNSET:
        sets.append(f"max_fires = ${idx}")
        params.append(max_fires)
        idx += 1

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if not sets:
        return await get_trigger(pool, trigger_id)

    sets.append(f"updated_at = ${idx}")
    params.append(datetime.now(timezone.utc))
    idx += 1

    params.append(trigger_id)
    set_clause = ", ".join(sets)

    row = await get_db(pool).fetchrow(
        f"UPDATE triggers SET {set_clause} WHERE id = ${idx} RETURNING *",
        *params,
    )
    return _row_to_trigger(row) if row else None


async def delete_trigger(pool: asyncpg.Pool, trigger_id: str) -> bool:
    """Delete a trigger. Returns True if deleted."""
    result = await get_db(pool).execute(
        "DELETE FROM triggers WHERE id = $1",
        trigger_id,
    )
    return result.split()[-1] != "0"


# --- Firing & Cooldown ---


async def record_fire(
    pool: asyncpg.Pool,
    trigger_id: str,
) -> Trigger:
    """Record that a trigger has fired. Increments fire_count, sets last_fired_at.

    If max_fires is set and fire_count reaches it, transitions status to 'fired'.
    Raises LookupError if trigger not found.
    """
    db = get_db(pool)

    row = await db.fetchrow("SELECT * FROM triggers WHERE id = $1", trigger_id)
    if row is None:
        raise LookupError(f"Trigger {trigger_id} not found")

    trigger = _row_to_trigger(row)
    if trigger.status != TriggerStatus.enabled:
        raise ValueError(f"Trigger {trigger_id} is not enabled (status={trigger.status.value})")

    new_count = trigger.fire_count + 1
    new_status = TriggerStatus.enabled
    if trigger.max_fires is not None and new_count >= trigger.max_fires:
        new_status = TriggerStatus.fired

    now = datetime.now(timezone.utc)
    updated = await db.fetchrow(
        """
        UPDATE triggers
        SET fire_count = $1, last_fired_at = $2,
            status = $3, updated_at = $2
        WHERE id = $4
        RETURNING *
        """,
        new_count,
        now,
        new_status.value,
        trigger_id,
    )
    return _row_to_trigger(updated)


def is_cooldown_elapsed(trigger: Trigger, now: datetime | None = None) -> bool:
    """Check if a trigger's cooldown period has elapsed."""
    if trigger.cooldown_hours is None:
        return True
    if trigger.last_fired_at is None:
        return True
    now = now or datetime.now(timezone.utc)
    cooldown_delta = timedelta(hours=trigger.cooldown_hours)
    return now >= trigger.last_fired_at + cooldown_delta


async def get_triggers_due(
    pool: asyncpg.Pool,
    *,
    condition_type: TriggerConditionType | None = None,
    project_id: str | None = None,
    now: datetime | None = None,
) -> list[Trigger]:
    """Get enabled triggers whose cooldown has elapsed.

    For time-based triggers, also checks that the trigger_at time in the
    condition has passed. For absence triggers, checks that the
    absence_hours threshold has been exceeded since last_fired_at or created_at.

    Returns triggers sorted by creation time (oldest first).
    """
    now = now or datetime.now(timezone.utc)

    conditions = ["status = 'enabled'"]
    params: list = []
    idx = 1

    if condition_type is not None:
        conditions.append(f"condition_type = ${idx}")
        params.append(condition_type.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    rows = await get_db(pool).fetch(
        f"SELECT * FROM triggers {where} ORDER BY created_at ASC",
        *params,
    )

    due: list[Trigger] = []
    for row in rows:
        trigger = _row_to_trigger(row)

        # Check max_fires limit
        if trigger.max_fires is not None and trigger.fire_count >= trigger.max_fires:
            continue

        # Check cooldown
        if not is_cooldown_elapsed(trigger, now):
            continue

        # Condition-type-specific checks
        if trigger.condition_type == TriggerConditionType.time:
            trigger_at = trigger.condition.get("trigger_at")
            if trigger_at is not None:
                try:
                    target = datetime.fromisoformat(trigger_at)
                    if now < target:
                        continue
                except (ValueError, TypeError):
                    pass  # Malformed trigger_at — treat as due

        elif trigger.condition_type == TriggerConditionType.absence:
            absence_hours = trigger.condition.get("absence_hours")
            if absence_hours is not None:
                reference = trigger.last_fired_at or trigger.created_at
                if now < reference + timedelta(hours=float(absence_hours)):
                    continue

        due.append(trigger)

    return due


# --- Helpers ---

def _row_to_trigger(row: asyncpg.Record) -> Trigger:
    """Convert a database row to a Trigger model."""
    condition = row["condition"]
    if isinstance(condition, str):
        condition = json.loads(condition)

    return Trigger(
        id=row["id"],
        name=row["name"],
        condition_type=TriggerConditionType(row["condition_type"]),
        condition=condition or {},
        action=row["action"],
        status=TriggerStatus(row["status"]),
        cooldown_hours=row["cooldown_hours"],
        max_fires=row["max_fires"],
        fire_count=row["fire_count"],
        last_fired_at=row["last_fired_at"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
