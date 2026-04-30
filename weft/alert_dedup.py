"""Alert deduplication and suppression — replaces the V1 alert_type-only dedup.

V1 (loom_alerts.py / memory_hygiene_alerts.py each had their own copy of
``_recent_alert_types``) deduped by alert_type alone over a fixed 24h
window. Consequence: when ``loom_stale_claim`` fired for project A,
project B couldn't get one for 24h.

V2 keys dedup state per ``(alert_type, dedup_key)`` where dedup_key is
producer-chosen (``"task:abc"``, ``"project:xyz"``, ``"memory:m-12"``,
or ``"global"`` for singletons). Per-type cooldowns are configurable in
:class:`AlertCooldownConfig`. Manual mutes via
:func:`suppress` outrank cooldowns in both directions.

Federation note: state is per-user via RLS. ``suppress`` and
``clear_suppression`` only affect the current user; agents acting as
different users have independent cooldown state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

from weft.db.connection import get_db
from weft.models import AlertType, _now, _weft_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AlertState(BaseModel):
    """Per-(alert_type, dedup_key, user) cooldown + suppression state."""

    id: str = Field(default_factory=_weft_id)
    alert_type: AlertType
    dedup_key: str
    last_fired_at: datetime | None = None
    last_alert_id: str | None = None
    fire_count: int = 0
    suppressed_until: datetime | None = None
    suppression_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["alert_type"] = self.alert_type.value
        return d


def _row_to_state(row: asyncpg.Record) -> AlertState:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return AlertState(
        id=row["id"],
        alert_type=AlertType(row["alert_type"]),
        dedup_key=row["dedup_key"],
        last_fired_at=row["last_fired_at"],
        last_alert_id=row["last_alert_id"],
        fire_count=row["fire_count"],
        suppressed_until=row["suppressed_until"],
        suppression_reason=row["suppression_reason"],
        metadata=metadata or {},
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# Core decision: should we fire?
# ---------------------------------------------------------------------------


async def _get_state(
    pool: asyncpg.Pool, alert_type: AlertType, dedup_key: str,
) -> AlertState | None:
    """Fetch the state row for an (alert_type, dedup_key) for the current user."""
    row = await get_db(pool).fetchrow(
        """
        SELECT * FROM alert_state
        WHERE alert_type = $1 AND dedup_key = $2
        """,
        alert_type.value,
        dedup_key,
    )
    return _row_to_state(row) if row else None


async def should_fire(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    dedup_key: str,
    *,
    cooldown_minutes: float,
    now: datetime | None = None,
) -> bool:
    """Return True if a new alert should fire for this (type, key).

    Order of checks:
      1. ``suppressed_until > now`` — manual mute beats everything (False).
      2. No state row OR ``last_fired_at`` is None — first fire (True).
      3. ``now - last_fired_at >= cooldown_minutes`` — cooldown elapsed (True).
      4. Otherwise — within cooldown (False).
    """
    ref_now = now or datetime.now(timezone.utc)
    state = await _get_state(pool, alert_type, dedup_key)
    if state is None:
        return True

    if state.suppressed_until is not None and state.suppressed_until > ref_now:
        return False

    if state.last_fired_at is None:
        return True

    elapsed = ref_now - state.last_fired_at
    return elapsed >= timedelta(minutes=cooldown_minutes)


# ---------------------------------------------------------------------------
# Recording a fire
# ---------------------------------------------------------------------------


async def record_fire(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    dedup_key: str,
    *,
    alert_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> AlertState:
    """Upsert state row with new fire timestamp and increment fire_count.

    Does NOT touch ``suppressed_until``: a manual mute set previously must
    survive a fire that slipped through (e.g., if the producer ignored
    should_fire). This separation is by design — record_fire describes
    history, suppress() / clear_suppression() describe policy.
    """
    ref_now = now or datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {})

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO alert_state (
            id, alert_type, dedup_key, last_fired_at, last_alert_id,
            fire_count, metadata, user_id, created_at, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, 1, $6::jsonb,
            nullif(current_setting('app.user_id', true), ''),
            $4, $4
        )
        ON CONFLICT (alert_type, dedup_key, user_id) DO UPDATE
            SET last_fired_at = EXCLUDED.last_fired_at,
                last_alert_id = EXCLUDED.last_alert_id,
                fire_count    = alert_state.fire_count + 1,
                metadata      = alert_state.metadata || EXCLUDED.metadata,
                updated_at    = EXCLUDED.last_fired_at
        RETURNING *
        """,
        _weft_id(),
        alert_type.value,
        dedup_key,
        ref_now,
        alert_id,
        metadata_json,
    )
    return _row_to_state(row)


# ---------------------------------------------------------------------------
# Manual mute / unmute
# ---------------------------------------------------------------------------


async def suppress(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    dedup_key: str,
    *,
    until: datetime,
    reason: str | None = None,
) -> AlertState:
    """Mute (alert_type, dedup_key) until *until*.

    Creates a state row if none exists. Does NOT change last_fired_at /
    fire_count — suppression is policy, not history. Subsequent
    should_fire calls return False until either *until* passes or
    clear_suppression is called.
    """
    if until <= datetime.now(timezone.utc):
        raise ValueError("suppress until= must be in the future")

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO alert_state (
            id, alert_type, dedup_key, suppressed_until, suppression_reason,
            user_id, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5,
            nullif(current_setting('app.user_id', true), ''),
            now()
        )
        ON CONFLICT (alert_type, dedup_key, user_id) DO UPDATE
            SET suppressed_until    = EXCLUDED.suppressed_until,
                suppression_reason  = EXCLUDED.suppression_reason,
                updated_at          = now()
        RETURNING *
        """,
        _weft_id(),
        alert_type.value,
        dedup_key,
        until,
        reason,
    )
    return _row_to_state(row)


async def clear_suppression(
    pool: asyncpg.Pool,
    alert_type: AlertType,
    dedup_key: str,
) -> bool:
    """Lift any active suppression for (alert_type, dedup_key).

    Returns True if a row existed (and was cleared), False otherwise.
    Does not delete the state row — last_fired_at / fire_count are
    history and remain for cooldown calculations.
    """
    result = await get_db(pool).execute(
        """
        UPDATE alert_state
        SET suppressed_until = NULL,
            suppression_reason = NULL,
            updated_at = now()
        WHERE alert_type = $1 AND dedup_key = $2
        """,
        alert_type.value,
        dedup_key,
    )
    return int(result.rsplit(" ", 1)[-1]) > 0


# ---------------------------------------------------------------------------
# Listing / inspection
# ---------------------------------------------------------------------------


async def list_state(
    pool: asyncpg.Pool,
    *,
    alert_type: AlertType | None = None,
    suppressed_only: bool = False,
    limit: int = 100,
) -> list[AlertState]:
    """List state rows, newest-fired first.

    suppressed_only=True restricts to rows whose suppressed_until is
    still in the future.
    """
    clauses: list[str] = []
    params: list[Any] = []
    idx = 1

    if alert_type is not None:
        clauses.append(f"alert_type = ${idx}")
        params.append(alert_type.value)
        idx += 1

    if suppressed_only:
        clauses.append("suppressed_until IS NOT NULL AND suppressed_until > now()")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM alert_state
        {where}
        ORDER BY last_fired_at DESC NULLS LAST
        LIMIT ${idx}
        """,
        *params,
    )
    return [_row_to_state(r) for r in rows]
