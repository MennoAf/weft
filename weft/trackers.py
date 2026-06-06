"""Tracker store — single writer for the trackers table.

Trackers are the lifecycle-aware open-loop primitive (Wick Phase 3).
Where memories are blob-shaped facts, trackers carry state that changes
over time, can be nudged on schedule, snoozed, dismissed, or closed.

State machine:
    open ∈ {in_progress, awaiting_reply, blocked}
    terminal ∈ {done, abandoned}
    open → open allowed (any direction); open → terminal allowed;
    terminal → anything is rejected (re-open by creating a new tracker).

Nudge modes (V1):
    none   — query-only; never appears in due()
    once   — fires once at nudge_after, then silent (mode flips to none on dismiss)
    recur  — fires every nudge_interval; dismiss bumps last_touch and rolls
             nudge_after forward by nudge_interval

snooze_until temporarily suppresses due-results without changing mode.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from weft.auth import get_caller_mode
from weft.db.connection import get_db
from weft.models import (
    NudgeMode,
    Tracker,
    TrackerCreate,
    TrackerKind,
    TrackerState,
    _tracker_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def create_tracker(
    pool: asyncpg.Pool, create: TrackerCreate,
) -> Tracker:
    """Create a tracker.

    Phase 2 / Layer 1:
    * ``provenance`` is stamped from the caller-mode contextvar — the
      ``create.provenance`` field is intentionally ignored so untrusted
      callers can't self-attest as 'supervisor'.
    * ``kind=trace`` is supervisor-only. Trace trackers are Orchestrator
      promotions of long-running agent context windows; they implicitly
      exit the trust boundary, so an agent-mode caller minting one would
      let an attacker write directly into the supervisor's working set.
    """
    write_provenance = get_caller_mode()
    if create.kind == TrackerKind.trace and write_provenance != "supervisor":
        raise PermissionError(
            "kind=trace trackers are supervisor-only — the Orchestrator is "
            "the trusted writer (Phase 2 / Layer 1)",
        )

    tid = _tracker_id()
    now = datetime.now(timezone.utc)
    history = [{
        "from": None,
        "to": create.state.value,
        "at": now.isoformat(),
        "note": "created",
    }]

    db = get_db(pool)
    row = await db.fetchrow(
        """
        INSERT INTO trackers (
            id, project_id, entity_id, kind, title, state,
            state_history, context, last_touch,
            nudge_mode, nudge_after, nudge_interval,
            provenance, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7::jsonb, $8::jsonb, $9,
            $10, $11, $12,
            $13, $9, $9
        )
        RETURNING *
        """,
        tid,
        create.project_id,
        create.entity_id,
        create.kind.value,
        create.title,
        create.state.value,
        json.dumps(history),
        json.dumps(create.context),
        now,
        create.nudge_mode.value,
        create.nudge_after,
        create.nudge_interval,
        write_provenance,
    )
    return _row_to_tracker(row)


async def get_tracker(pool: asyncpg.Pool, tracker_id: str) -> Tracker | None:
    row = await get_db(pool).fetchrow(
        "SELECT * FROM trackers WHERE id = $1", tracker_id,
    )
    return _row_to_tracker(row) if row else None


async def update_tracker(
    pool: asyncpg.Pool,
    tracker_id: str,
    *,
    title: str | None = None,
    state: TrackerState | None = None,
    state_note: str | None = None,
    context: dict[str, Any] | None = None,
    nudge_mode: NudgeMode | None = None,
    nudge_after: datetime | None = None,
    nudge_interval: timedelta | None = None,
    bump_last_touch: bool = True,
) -> Tracker:
    """Generic patch. State transitions append to state_history.

    Refuses transitions out of terminal states — re-opening a closed
    tracker is intentionally not supported (create a new one). Refuses
    transitions to invalid state values (CHECK constraint also catches it).
    """
    existing = await get_tracker(pool, tracker_id)
    if existing is None:
        raise LookupError(f"tracker not found: {tracker_id}")

    if state is not None and existing.state != state:
        if not existing.is_open():
            raise ValueError(
                f"cannot transition out of terminal state {existing.state.value}"
                f" (tracker={tracker_id})"
            )

    now = datetime.now(timezone.utc)
    sets: list[str] = ["updated_at = $1"]
    params: list[Any] = [now]
    idx = 2

    if title is not None:
        sets.append(f"title = ${idx}")
        params.append(title)
        idx += 1

    if state is not None and state != existing.state:
        sets.append(f"state = ${idx}")
        params.append(state.value)
        idx += 1
        sets.append(f"state_history = state_history || ${idx}::jsonb")
        params.append(json.dumps([{
            "from": existing.state.value,
            "to": state.value,
            "at": now.isoformat(),
            "note": state_note,
        }]))
        idx += 1

    if context is not None:
        # Replace, don't merge — callers wanting merge use list_append helpers.
        sets.append(f"context = ${idx}::jsonb")
        params.append(json.dumps(context))
        idx += 1

    if nudge_mode is not None:
        sets.append(f"nudge_mode = ${idx}")
        params.append(nudge_mode.value)
        idx += 1

    if nudge_after is not None:
        sets.append(f"nudge_after = ${idx}")
        params.append(nudge_after)
        idx += 1

    if nudge_interval is not None:
        sets.append(f"nudge_interval = ${idx}")
        params.append(nudge_interval)
        idx += 1

    if bump_last_touch:
        sets.append(f"last_touch = ${idx}")
        params.append(now)
        idx += 1

    params.append(tracker_id)
    row = await get_db(pool).fetchrow(
        f"UPDATE trackers SET {', '.join(sets)} WHERE id = ${idx} RETURNING *",
        *params,
    )
    if row is None:
        raise LookupError(f"tracker not found: {tracker_id}")
    return _row_to_tracker(row)


async def close_tracker(
    pool: asyncpg.Pool,
    tracker_id: str,
    *,
    final_state: TrackerState = TrackerState.done,
    note: str | None = None,
) -> Tracker:
    """Terminal close. final_state must be done or abandoned."""
    if final_state.value not in TrackerState.terminal_states():
        raise ValueError(
            f"close requires a terminal state; got {final_state.value}"
        )
    return await update_tracker(
        pool, tracker_id, state=final_state, state_note=note,
    )


async def dismiss_tracker(pool: asyncpg.Pool, tracker_id: str) -> Tracker:
    """One-click "thanks, I know" — bumps last_touch, rolls nudge_after
    forward by nudge_interval (recur), or silences (once → none)."""
    existing = await get_tracker(pool, tracker_id)
    if existing is None:
        raise LookupError(f"tracker not found: {tracker_id}")

    now = datetime.now(timezone.utc)

    if existing.nudge_mode == NudgeMode.recur and existing.nudge_interval:
        next_after = now + existing.nudge_interval
        return await update_tracker(
            pool, tracker_id,
            nudge_after=next_after,
            bump_last_touch=True,
        )
    if existing.nudge_mode == NudgeMode.once:
        # Flip to silent once dismissed.
        return await update_tracker(
            pool, tracker_id,
            nudge_mode=NudgeMode.none,
            bump_last_touch=True,
        )
    # No-nudge tracker: just bump last_touch.
    return await update_tracker(pool, tracker_id, bump_last_touch=True)


async def snooze_tracker(
    pool: asyncpg.Pool, tracker_id: str, until: datetime,
) -> Tracker:
    """Suppress due() results until ``until``. Mode unchanged."""
    db = get_db(pool)
    now = datetime.now(timezone.utc)
    row = await db.fetchrow(
        """
        UPDATE trackers SET snooze_until = $1, updated_at = $2
        WHERE id = $3 RETURNING *
        """,
        until, now, tracker_id,
    )
    if row is None:
        raise LookupError(f"tracker not found: {tracker_id}")
    return _row_to_tracker(row)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


async def list_trackers(
    pool: asyncpg.Pool,
    *,
    kind: TrackerKind | None = None,
    state: TrackerState | None = None,
    open_only: bool = False,
    project_id: str | None = None,
    entity_id: str | None = None,
    context_filter: dict[str, str] | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Tracker]:
    """List trackers, newest-touch first, narrowed by the given filters.

    ``context_filter`` matches top-level keys in the JSONB ``context``
    column by string equality — one ``context ->> key = value`` clause per
    pair, ANDed together. This is the read counterpart to the generic
    ``context`` writes that Wick's operational records use: every Wick
    interchange record collapses to ``kind=trace`` and carries its true
    kind under ``context.wick_kind``, so a consumer isolates (say)
    authority-skip events with ``context_filter={"wick_kind":
    "authority_skip"}``. Only string-valued equality is supported;
    callers needing range/containment semantics should add a dedicated
    surface rather than overloading this.

    ``since`` bounds results to ``created_at >= since`` (timezone-aware
    UTC datetime).
    """
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if kind is not None:
        conditions.append(f"kind = ${idx}")
        params.append(kind.value)
        idx += 1

    if state is not None:
        conditions.append(f"state = ${idx}")
        params.append(state.value)
        idx += 1
    elif open_only:
        conditions.append(
            "state IN ('in_progress', 'awaiting_reply', 'blocked')"
        )

    if project_id is not None:
        conditions.append(f"project_id = ${idx}")
        params.append(project_id)
        idx += 1

    if entity_id is not None:
        conditions.append(f"entity_id = ${idx}")
        params.append(entity_id)
        idx += 1

    if context_filter:
        for key, value in context_filter.items():
            conditions.append(f"context ->> ${idx} = ${idx + 1}")
            params.append(key)
            params.append(value)
            idx += 2

    if since is not None:
        conditions.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM trackers {where}
        ORDER BY last_touch DESC
        LIMIT ${idx}
        """,
        *params, limit,
    )
    return [_row_to_tracker(r) for r in rows]


async def due_trackers(
    pool: asyncpg.Pool, *, now: datetime | None = None, limit: int = 100,
) -> list[Tracker]:
    """Trackers whose nudge is due. Open-state, non-snoozed, past nudge_after."""
    now = now or datetime.now(timezone.utc)
    rows = await get_db(pool).fetch(
        """
        SELECT * FROM trackers
        WHERE state IN ('in_progress', 'awaiting_reply', 'blocked')
          AND nudge_mode <> 'none'
          AND nudge_after IS NOT NULL
          AND nudge_after <= $1
          AND (snooze_until IS NULL OR snooze_until <= $1)
        ORDER BY nudge_after ASC
        LIMIT $2
        """,
        now, limit,
    )
    return [_row_to_tracker(r) for r in rows]


# ---------------------------------------------------------------------------
# List sugar (over context.items)
# ---------------------------------------------------------------------------


async def list_append(
    pool: asyncpg.Pool, tracker_id: str, item: dict[str, Any],
) -> Tracker:
    """Append an item to context.items. Item shape: {text, checked?, link?}."""
    existing = await get_tracker(pool, tracker_id)
    if existing is None:
        raise LookupError(f"tracker not found: {tracker_id}")
    items = list(existing.context.get("items", []))
    items.append(item)
    new_ctx = {**existing.context, "items": items}
    return await update_tracker(pool, tracker_id, context=new_ctx)


async def list_check(
    pool: asyncpg.Pool, tracker_id: str, index: int, checked: bool = True,
) -> Tracker:
    """Toggle checked on the item at ``index``."""
    existing = await get_tracker(pool, tracker_id)
    if existing is None:
        raise LookupError(f"tracker not found: {tracker_id}")
    items = list(existing.context.get("items", []))
    if not 0 <= index < len(items):
        raise ValueError(f"index {index} out of range (0..{len(items) - 1})")
    items[index] = {**items[index], "checked": checked}
    new_ctx = {**existing.context, "items": items}
    return await update_tracker(pool, tracker_id, context=new_ctx)


async def list_remove(
    pool: asyncpg.Pool, tracker_id: str, index: int,
) -> Tracker:
    """Remove the item at ``index`` from context.items."""
    existing = await get_tracker(pool, tracker_id)
    if existing is None:
        raise LookupError(f"tracker not found: {tracker_id}")
    items = list(existing.context.get("items", []))
    if not 0 <= index < len(items):
        raise ValueError(f"index {index} out of range (0..{len(items) - 1})")
    items.pop(index)
    new_ctx = {**existing.context, "items": items}
    return await update_tracker(pool, tracker_id, context=new_ctx)


# ---------------------------------------------------------------------------
# Row hydration
# ---------------------------------------------------------------------------


def _row_to_tracker(row: asyncpg.Record) -> Tracker:
    history = row["state_history"]
    if isinstance(history, str):
        history = json.loads(history)
    context = row["context"]
    if isinstance(context, str):
        context = json.loads(context)
    return Tracker(
        id=row["id"],
        user_id=row["user_id"],
        project_id=row["project_id"],
        entity_id=row["entity_id"],
        kind=TrackerKind(row["kind"]),
        title=row["title"],
        state=TrackerState(row["state"]),
        state_history=list(history) if history else [],
        context=dict(context) if context else {},
        last_touch=row["last_touch"],
        nudge_mode=NudgeMode(row["nudge_mode"]),
        nudge_after=row["nudge_after"],
        nudge_interval=row["nudge_interval"],
        snooze_until=row["snooze_until"],
        trigger_ids=list(row["trigger_ids"]) if row["trigger_ids"] else [],
        provenance=row["provenance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
