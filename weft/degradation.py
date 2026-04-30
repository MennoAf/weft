"""Degradation policy store — CRUD, cooldown logic, and policy evaluation.

Degradation policies are condition-driven rules that fire response actions
when system health degrades. Supports low confidence, API errors, context
decay, and budget breach conditions with configurable cooldown and max-fire
limits.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.db.connection import get_db
from weft.models import (
    DegradationAction,
    DegradationPolicy,
    DegradationPolicyCreate,
    DegradationPolicyStatus,
    DegradationTriggerType,
    _weft_id,
)

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None


# --- Row mapping ---


def _row_to_policy(row: asyncpg.Record) -> DegradationPolicy:
    """Convert a database row to a DegradationPolicy model."""
    condition = row["condition"]
    if isinstance(condition, str):
        condition = json.loads(condition)

    return DegradationPolicy(
        id=row["id"],
        name=row["name"],
        trigger_type=DegradationTriggerType(row["trigger_type"]),
        condition=condition or {},
        action=DegradationAction(row["action"]),
        description=row["description"],
        status=DegradationPolicyStatus(row["status"]),
        cooldown_minutes=row["cooldown_minutes"],
        max_fires=row["max_fires"],
        fire_count=row["fire_count"],
        last_fired_at=row["last_fired_at"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# --- CRUD ---


async def create_policy(
    pool: asyncpg.Pool,
    create: DegradationPolicyCreate,
) -> DegradationPolicy:
    """Create a new degradation policy. Returns the created policy."""
    policy_id = _weft_id()
    now = datetime.now(timezone.utc)
    condition_json = json.dumps(create.condition)

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO degradation_policies (
            id, name, trigger_type, condition, action, description,
            status, cooldown_minutes, max_fires,
            project_id, agent_id, user_id,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4::jsonb, $5, $6,
            'active', $7, $8,
            $9, $10, nullif(current_setting('app.user_id', true), ''),
            $11, $11
        )
        RETURNING *
        """,
        policy_id,
        create.name,
        create.trigger_type.value,
        condition_json,
        create.action.value,
        create.description,
        create.cooldown_minutes,
        create.max_fires,
        create.project_id,
        create.agent_id,
        now,
    )
    return _row_to_policy(row)


async def get_policy(pool: asyncpg.Pool, policy_id: str) -> DegradationPolicy | None:
    """Fetch a degradation policy by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM degradation_policies WHERE id = $1",
        policy_id,
    )
    return _row_to_policy(row) if row else None


async def list_policies(
    pool: asyncpg.Pool,
    *,
    trigger_type: DegradationTriggerType | None = None,
    action: DegradationAction | None = None,
    status: DegradationPolicyStatus | None = None,
    project_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DegradationPolicy]:
    """List degradation policies with optional filters."""
    conditions: list[str] = []
    params: list = []
    idx = 1

    if trigger_type is not None:
        conditions.append(f"trigger_type = ${idx}")
        params.append(trigger_type.value)
        idx += 1

    if action is not None:
        conditions.append(f"action = ${idx}")
        params.append(action.value)
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
        SELECT * FROM degradation_policies {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_policy(r) for r in rows]


async def update_policy(
    pool: asyncpg.Pool,
    policy_id: str,
    *,
    name: object = _UNSET,
    description: object = _UNSET,
    status: DegradationPolicyStatus | object = _UNSET,
    cooldown_minutes: float | None | object = _UNSET,
    max_fires: int | None | object = _UNSET,
    enabled: bool | object = _UNSET,
) -> DegradationPolicy | None:
    """Update a degradation policy. Returns updated policy or None if not found."""
    sets: list[str] = []
    params: list = []
    idx = 1

    if name is not _UNSET:
        sets.append(f"name = ${idx}")
        params.append(name)
        idx += 1

    if description is not _UNSET:
        sets.append(f"description = ${idx}")
        params.append(description)
        idx += 1

    if status is not _UNSET:
        sets.append(f"status = ${idx}")
        params.append(status.value)  # type: ignore[union-attr]
        idx += 1

    if cooldown_minutes is not _UNSET:
        sets.append(f"cooldown_minutes = ${idx}")
        params.append(cooldown_minutes)
        idx += 1

    if max_fires is not _UNSET:
        sets.append(f"max_fires = ${idx}")
        params.append(max_fires)
        idx += 1

    if not sets:
        return await get_policy(pool, policy_id)

    sets.append("updated_at = now()")
    params.append(policy_id)
    set_clause = ", ".join(sets)
    query = f"""
        UPDATE degradation_policies
        SET {set_clause}
        WHERE id = ${idx}
        RETURNING *
    """

    row = await get_db(pool).fetchrow(query, *params)
    return _row_to_policy(row) if row else None


async def delete_policy(pool: asyncpg.Pool, policy_id: str) -> bool:
    """Delete a degradation policy. Returns True if deleted."""
    result = await get_db(pool).execute(
        "DELETE FROM degradation_policies WHERE id = $1",
        policy_id,
    )
    return result.split()[-1] != "0"


# --- Firing ---


async def record_fire(
    pool: asyncpg.Pool,
    policy_id: str,
) -> DegradationPolicy | None:
    """Record a policy firing — increments fire_count and updates last_fired_at.

    If max_fires is reached, status transitions to 'fired'.
    Returns the updated policy or None if not found.
    """
    row = await get_db(pool).fetchrow(
        """
        UPDATE degradation_policies
        SET fire_count = fire_count + 1,
            last_fired_at = now(),
            status = CASE
                WHEN max_fires IS NOT NULL AND fire_count + 1 >= max_fires
                THEN 'fired'
                ELSE status
            END,
            updated_at = now()
        WHERE id = $1
        RETURNING *
        """,
        policy_id,
    )
    return _row_to_policy(row) if row else None


def is_cooldown_elapsed(policy: DegradationPolicy) -> bool:
    """Check if enough time has passed since the last firing."""
    if policy.cooldown_minutes is None:
        return True
    if policy.last_fired_at is None:
        return True
    elapsed = datetime.now(timezone.utc) - policy.last_fired_at
    return elapsed >= timedelta(minutes=policy.cooldown_minutes)


# --- Active policy queries ---


async def get_active_policies(
    pool: asyncpg.Pool,
    *,
    trigger_type: DegradationTriggerType | None = None,
    project_id: str | None = None,
) -> list[DegradationPolicy]:
    """Return active policies that are eligible to fire.

    Filters out:
    - Non-active policies
    - Policies still in cooldown
    - Policies that have reached max_fires
    """
    conditions = ["status = 'active'"]
    params: list = []
    idx = 1

    # Push cooldown into SQL (use secs with multiplication to avoid type issues)
    conditions.append(
        "(cooldown_minutes IS NULL OR last_fired_at IS NULL "
        "OR last_fired_at + make_interval(secs => cooldown_minutes * 60) <= now())"
    )

    # Push max_fires into SQL
    conditions.append(
        "(max_fires IS NULL OR fire_count < max_fires)"
    )

    if trigger_type is not None:
        conditions.append(f"trigger_type = ${idx}")
        params.append(trigger_type.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT * FROM degradation_policies {where}
        ORDER BY created_at ASC
    """

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_policy(r) for r in rows]


# --- State evaluation ---


async def update_degradation_state(
    pool: asyncpg.Pool,
    *,
    metrics: dict[str, float | int],
    project_id: str | None = None,
) -> list[dict]:
    """Evaluate current metrics against active degradation policies.

    Checks each eligible policy's condition against the provided metrics
    and fires any that match. Returns a list of triggered policy dicts
    with the action to take.

    metrics keys should match condition requirements:
      - "confidence": current confidence level (0.0-1.0)
      - "error_count": number of API errors in window
      - "context_age_hours": hours since context was fresh
      - "tokens_used": total tokens consumed

    Returns list of dicts: [{policy_id, name, action, reason, ...}, ...]
    """
    active = await get_active_policies(pool, project_id=project_id)
    triggered: list[dict] = []

    for policy in active:
        match = _evaluate_condition(policy, metrics)
        if match is None:
            continue

        fired = await record_fire(pool, policy.id)
        if fired is None:
            continue

        triggered.append({
            "policy_id": policy.id,
            "name": policy.name,
            "trigger_type": policy.trigger_type.value,
            "action": policy.action.value,
            "reason": match,
            "fire_count": fired.fire_count,
            "status": fired.status.value,
        })
        logger.info(
            "Degradation policy %s fired: %s → %s (%s)",
            policy.id, policy.name, policy.action.value, match,
        )

    return triggered


def _evaluate_condition(
    policy: DegradationPolicy,
    metrics: dict[str, float | int],
) -> str | None:
    """Check if a policy's condition is met by the current metrics.

    Returns a reason string if triggered, None otherwise.
    """
    tt = policy.trigger_type
    c = policy.condition

    if tt == DegradationTriggerType.low_confidence:
        threshold = c.get("threshold", 0.3)
        current = metrics.get("confidence")
        if current is not None and current < threshold:
            return f"confidence {current:.2f} < threshold {threshold:.2f}"

    elif tt == DegradationTriggerType.api_error:
        max_errors = c.get("max_errors", 5)
        current = metrics.get("error_count")
        if current is not None and current >= max_errors:
            return f"error_count {current} >= max_errors {max_errors}"

    elif tt == DegradationTriggerType.context_decay:
        max_age = c.get("max_age_hours", 24)
        current = metrics.get("context_age_hours")
        if current is not None and current >= max_age:
            return f"context_age_hours {current:.1f} >= max_age {max_age}"

    elif tt == DegradationTriggerType.budget_breach:
        max_tokens = c.get("max_tokens", 100000)
        current = metrics.get("tokens_used")
        if current is not None and current >= max_tokens:
            return f"tokens_used {current} >= max_tokens {max_tokens}"

    elif tt == DegradationTriggerType.cost_breach:
        threshold_pct = c.get("pct_used", 100.0)
        current = metrics.get("cost_pct_used")
        if current is not None and current >= threshold_pct:
            return f"cost_pct_used {current:.1f} >= pct_used {threshold_pct:.1f}"

    return None
