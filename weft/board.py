"""Weft board — unified open-items contract, pure functions for bucketing and
ranking, and the assemble_board() fan-out orchestrator.

This module defines the Item model (normalized across five sources: trackers,
alerts, triggers, task-memories, review queue), pure functions for urgency
bucketing and ranking, per-source adapters, and assemble_board() — the
concurrent-fan-out read that turns the five sources into one board response.
assemble_board() is read-only (PRD Validation V6): it never mutates a source
row.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

import asyncpg

from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.models import TriggerConditionType

if TYPE_CHECKING:
    from weft.models import Alert, Memory, Tracker, Trigger
    from weft.skills import TaskEntry

logger = logging.getLogger(__name__)


ItemSource = Literal["tracker", "alert", "trigger", "task", "review"]
Urgency = Literal["overdue", "due_soon", "pending", "no_date"]

# PRD §Behavior default lookahead window (R3 notes this may need to vary per
# source later; v1 ships one horizon for all five sources).
DEFAULT_HORIZON_DAYS = 7

# PRD §Constraints Touched: per-source read cap, well above realistic
# single-user volume. Hitting it emits a `warnings` truncation entry (V7) —
# never silent.
DEFAULT_PER_SOURCE_CAP = 200

_URGENCY_BUCKETS: tuple[Urgency, ...] = ("overdue", "due_soon", "pending", "no_date")


@dataclass
class Action:
    """A self-describing triage action: verb + tool name + args for that item."""

    verb: str
    tool: str
    args: dict


@dataclass
class Item:
    """Normalized open-items contract across all five sources.

    Schema fields per PRD §Interfaces:
    - id: unique identifier within the source
    - source: one of {tracker, alert, trigger, task, review}
    - kind: tracker kind / alert_type / condition_type / task priority / "review"
    - title: display string
    - state: current state (tracker/alert state), null if not applicable
    - due_at: when it becomes actionable (datetime ISO8601 or None)
    - snoozed_until: if a tracker, when snooze expires (ISO8601 or None)
    - age_days: days since created (computed at model time, not after)
    - urgency: bucket derived from due_at vs now + horizon
    - project_id: associated project, if any
    - entity_id: associated entity, if any
    - actions: self-describing triage actions (close, snooze, dismiss, etc.)
    """

    id: str
    source: ItemSource
    kind: str
    title: str
    state: str | None
    due_at: datetime | None
    snoozed_until: datetime | None
    age_days: float
    urgency: Urgency
    project_id: str | None = None
    entity_id: str | None = None
    actions: list[Action] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict, matching PRD schema."""
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "title": self.title,
            "state": self.state,
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "snoozed_until": (
                self.snoozed_until.isoformat() if self.snoozed_until else None
            ),
            "age_days": self.age_days,
            "urgency": self.urgency,
            "project_id": self.project_id,
            "entity_id": self.entity_id,
            "actions": [
                {
                    "verb": a.verb,
                    "tool": a.tool,
                    "args": a.args,
                }
                for a in self.actions
            ],
        }


def calculate_urgency(
    due_at: datetime | None,
    now: datetime | None = None,
    horizon_days: int = 7,
) -> Urgency:
    """Derive urgency bucket from due_at vs configurable horizon.

    Pure function, deterministic and idempotent.

    Args:
        due_at: when the item becomes actionable (datetime with UTC assumed,
                or None if no due date)
        now: current time (defaults to UTC now if not provided)
        horizon_days: lookahead window in days (default 7)

    Returns:
        one of {overdue, due_soon, pending, no_date}

    Validation V2 (PRD):
    - past due_at → overdue
    - within horizon → due_soon
    - open with due_at beyond horizon → pending
    - null due_at → no_date
    """
    if due_at is None:
        return "no_date"

    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure both are timezone-aware for comparison
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if due_at < now:
        return "overdue"

    horizon_cutoff = now + timedelta(days=horizon_days)
    if due_at <= horizon_cutoff:
        return "due_soon"

    return "pending"


def rank_items(
    items: list[Item],
) -> list[Item]:
    """Sort items within a bucket by urgency rules.

    Pure function, does not mutate input list.

    Sorting order per PRD:
    1. oldest due_at first (ascending due_at, None sorts last)
    2. ties broken by age_days descending (older items first)
    3. final tie-break by title (ascending, alphabetical)

    Args:
        items: unsorted items, all assumed to be in the same urgency bucket

    Returns:
        sorted copy of items
    """

    def sort_key(item: Item):
        # due_at: None values sort last (float('inf')), earlier dates sort first
        due_at_key = (
            item.due_at.timestamp() if item.due_at else float("inf")
        )
        # age_days: negate so older (higher age_days) sorts first
        age_days_key = -item.age_days
        # title: alphabetical ascending
        title_key = item.title.lower()

        return (due_at_key, age_days_key, title_key)

    return sorted(items, key=sort_key)


def tracker_adapter(
    trackers: list[Tracker],
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[Item]:
    """Map Tracker objects to Items for the board.

    Reuses due_trackers filtering (snooze logic already applied upstream).
    Each tracker is converted to an Item with source="tracker", using nudge_after
    as the due_at (when the next nudge is due). Urgency computed from nudge_after.

    Args:
        trackers: list of Tracker objects from due_trackers()
        now: current time for age_days calculation (defaults to UTC now)
        horizon_days: due_soon lookahead window forwarded to calculate_urgency
            (PRD V2 "configured horizon")

    Returns:
        list of Item objects ready for bucketing
    """
    if now is None:
        now = datetime.now(timezone.utc)

    items: list[Item] = []
    for tracker in trackers:
        # Compute age_days from created_at
        if tracker.created_at.tzinfo is None:
            created_at = tracker.created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = tracker.created_at

        age_delta = now - created_at
        age_days = age_delta.total_seconds() / 86400.0

        # Map to Item
        item = Item(
            id=tracker.id,
            source="tracker",
            kind=tracker.kind.value,
            title=tracker.title,
            state=tracker.state.value,
            due_at=tracker.nudge_after,
            snoozed_until=tracker.snooze_until,
            age_days=age_days,
            urgency=calculate_urgency(tracker.nudge_after, now, horizon_days),
            project_id=tracker.project_id,
            entity_id=tracker.entity_id,
            actions=[],
        )
        items.append(item)

    return items


def alert_adapter(
    alerts: list[Alert],
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[Item]:
    """Map Alert objects to Items for the board.

    Each alert is converted to an Item with source="alert", using trigger_at
    as the due_at. Only maps pending alerts; other statuses excluded by list_alerts
    filtering. Urgency computed from trigger_at.

    Args:
        alerts: list of Alert objects, typically filtered by status=pending
        now: current time for age_days calculation (defaults to UTC now)
        horizon_days: due_soon lookahead window forwarded to calculate_urgency
            (PRD V2 "configured horizon")

    Returns:
        list of Item objects ready for bucketing
    """
    if now is None:
        now = datetime.now(timezone.utc)

    items: list[Item] = []
    for alert in alerts:
        # Compute age_days from created_at
        if alert.created_at.tzinfo is None:
            created_at = alert.created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = alert.created_at

        age_delta = now - created_at
        age_days = age_delta.total_seconds() / 86400.0

        # Map to Item
        item = Item(
            id=alert.id,
            source="alert",
            kind=alert.alert_type.value,
            title=alert.title,
            state=alert.status.value,
            due_at=alert.trigger_at,
            snoozed_until=None,  # Alerts do not have snooze capability
            age_days=age_days,
            urgency=calculate_urgency(alert.trigger_at, now, horizon_days),
            project_id=alert.project_id,
            entity_id=None,  # Alerts do not have entity association
            actions=[],
        )
        items.append(item)

    return items


# Conservative cold-start hide-list (PRD weft-board §Source-Semantics / R1):
# trigger *names* containing any of these case-insensitive substrings are
# excluded from the board so v1 isn't noisy with system-internal triggers
# (e.g. recall-canary health checks, mood/sleep check-in reminders). This is
# deliberately a config list, not a hard-coded taxonomy — Trigger has no
# "kind" field of its own (condition_type is only time/threshold/event/
# absence, which says nothing about whether a trigger is system-internal),
# so name substrings are the only pre-mapping signal available. The Epic's
# L1 `hidden_kinds` feedback loop is meant to tune/replace this list from
# real dismiss patterns once triage volume exists — this seed is intentionally
# conservative, not exhaustive.
_TRIGGER_HIDE_KINDS: list[str] = ["canary", "check_in", "check-in"]


def _is_hidden_trigger(trigger: Trigger) -> bool:
    """True if trigger.name matches a cold-start hide-list entry."""
    name_lower = trigger.name.lower()
    return any(hidden in name_lower for hidden in _TRIGGER_HIDE_KINDS)


def trigger_adapter(
    triggers: list[Trigger],
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[Item]:
    """Map Trigger objects to Items for the board.

    Each trigger is converted to an Item with source="trigger". `due_at` is
    populated ONLY for condition_type="time" triggers, parsed from the ISO
    datetime string nested at `condition["trigger_at"]` (see
    `weft.models.TriggerCreate._validate_condition` and
    `weft.triggers.get_triggers_due` for the same nesting). Non-time
    conditions (threshold/event/absence) have no meaningful due date, so
    `due_at` stays None and `calculate_urgency` naturally buckets them as
    "no_date" — no special-casing needed beyond the due_at computation.

    Triggers matching `_TRIGGER_HIDE_KINDS` (by name substring) are excluded
    entirely — a conservative cold-start filter for system-internal triggers.

    Args:
        triggers: list of Trigger objects (pre-filtered upstream, e.g. by
            status/project as the caller's query decides)
        now: current time for age_days calculation (defaults to UTC now)
        horizon_days: due_soon lookahead window forwarded to calculate_urgency
            (PRD V2 "configured horizon")

    Returns:
        list of Item objects ready for bucketing (hidden-kind triggers omitted)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    items: list[Item] = []
    for trigger in triggers:
        if _is_hidden_trigger(trigger):
            continue

        # Compute age_days from created_at
        if trigger.created_at.tzinfo is None:
            created_at = trigger.created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = trigger.created_at

        age_delta = now - created_at
        age_days = age_delta.total_seconds() / 86400.0

        # due_at exists only for time-condition triggers, nested in condition JSON
        due_at: datetime | None = None
        if trigger.condition_type == TriggerConditionType.time:
            trigger_at_raw = trigger.condition.get("trigger_at")
            if trigger_at_raw:
                try:
                    due_at = datetime.fromisoformat(trigger_at_raw)
                except (ValueError, TypeError):
                    due_at = None  # malformed trigger_at — treat as no_date

        # Map to Item
        item = Item(
            id=trigger.id,
            source="trigger",
            kind=trigger.condition_type.value,
            title=trigger.name,
            state=trigger.status.value,
            due_at=due_at,
            snoozed_until=None,  # Triggers do not have snooze capability
            age_days=age_days,
            urgency=calculate_urgency(due_at, now, horizon_days),
            project_id=trigger.project_id,
            entity_id=None,  # Triggers do not have entity association
            actions=[],
        )
        items.append(item)

    return items


def task_adapter(
    entries: list[TaskEntry],
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[Item]:
    """Map task-memory TaskEntry objects to Items for the board.

    Consumes `weft.skills.fetch_task_entries` — the shared up_next core — so
    task selection ('tasks' topic memories) and due-date/priority parsing
    can never fork between `weft_up_next` and the board (weft-board
    subsumption contract, PRD Critical Implementation Notes).

    Read-only source (PRD Non-Goals): task-memory writes (`weft_revise`,
    `weft_forget`) are out of scope for v1 board triage, so items carry no
    actions.

    Args:
        entries: list of TaskEntry objects from `weft.skills.fetch_task_entries`
        now: current time for age_days calculation (defaults to UTC now)
        horizon_days: due_soon lookahead window forwarded to calculate_urgency
            (PRD V2 "configured horizon")

    Returns:
        list of Item objects ready for bucketing
    """
    if now is None:
        now = datetime.now(timezone.utc)

    items: list[Item] = []
    for entry in entries:
        # Compute age_days from created_at
        if entry.created_at.tzinfo is None:
            created_at = entry.created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = entry.created_at

        age_delta = now - created_at
        age_days = age_delta.total_seconds() / 86400.0

        # due_at parsed from the "due:YYYY-MM-DD" topic tag, midnight UTC
        due_at: datetime | None = None
        if entry.due_date:
            try:
                due_at = datetime.strptime(entry.due_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                due_at = None  # malformed due-date tag — treat as no_date

        # Map to Item
        item = Item(
            id=entry.id,
            source="task",
            kind=entry.priority or "task",
            title=entry.content,
            state=None,  # task-memories have no tracker/alert-style state
            due_at=due_at,
            snoozed_until=None,  # Task-memories do not have snooze capability
            age_days=age_days,
            urgency=calculate_urgency(due_at, now, horizon_days),
            project_id=None,  # TaskEntry does not carry project association
            entity_id=None,  # TaskEntry does not carry entity association
            actions=[],
        )
        items.append(item)

    return items


def review_adapter(
    memories: list[Memory],
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[Item]:
    """Map Memory objects to Items for the board's review queue.

    Each memory is converted to an Item with source="review", using
    `review_after` as the due_at (when the memory is due for re-verification).
    Callers are expected to pass memories whose `review_after` is set (e.g.
    filtered the same way as `daily_brief._query_review_queue`:
    `review_after IS NOT NULL`) — this adapter does not re-filter, mirroring
    how tracker_adapter/alert_adapter trust upstream filtering. A memory with
    a null review_after simply buckets to "no_date" rather than erroring.

    Read-only source (PRD Non-Goals): review-queue writes (`weft_revise`,
    `weft_forget`) are out of scope for v1 board triage, so items carry no
    actions.

    Args:
        memories: list of Memory objects, typically filtered by
            review_after IS NOT NULL
        now: current time for age_days calculation (defaults to UTC now)
        horizon_days: due_soon lookahead window forwarded to calculate_urgency
            (PRD V2 "configured horizon")

    Returns:
        list of Item objects ready for bucketing
    """
    if now is None:
        now = datetime.now(timezone.utc)

    items: list[Item] = []
    for memory in memories:
        # Compute age_days from created_at
        if memory.created_at.tzinfo is None:
            created_at = memory.created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = memory.created_at

        age_delta = now - created_at
        age_days = age_delta.total_seconds() / 86400.0

        # Map to Item
        item = Item(
            id=memory.id,
            source="review",
            kind="review",
            title=memory.content,
            state=None,  # memories have no tracker/alert-style state
            due_at=memory.review_after,
            snoozed_until=None,  # Memories do not have snooze capability
            age_days=age_days,
            urgency=calculate_urgency(memory.review_after, now, horizon_days),
            project_id=memory.project_id,
            entity_id=None,  # Memories do not have entity association
            actions=[],
        )
        items.append(item)

    return items


async def _fetch_review_memories(
    conn: asyncpg.Connection, limit: int,
) -> list[Memory]:
    """Fetch memories due for review, oldest-review-due first.

    Filter mirrors `daily_brief._query_review_queue` (status='active' AND
    review_after IS NOT NULL) — see review_adapter's docstring — but returns
    structured Memory objects rather than display strings, since
    review_adapter consumes rows, not rendered text.
    """
    from weft.store import _row_to_memory

    rows = await conn.fetch(
        """
        SELECT * FROM memories
        WHERE status = 'active'
          AND review_after IS NOT NULL
        ORDER BY review_after ASC
        LIMIT $1
        """,
        limit,
    )
    return [_row_to_memory(r) for r in rows]


async def assemble_board(
    pool: asyncpg.Pool,
    *,
    days: int = DEFAULT_HORIZON_DAYS,
    per_source_cap: int = DEFAULT_PER_SOURCE_CAP,
    now: datetime | None = None,
    user_id: str | None = None,
) -> dict:
    """Fan out concurrently across the five open-item sources and assemble
    the unified board response (PRD §Behavior, §Interfaces).

    Each source is fetched THEN adapted (fetch-then-map, per Epic Task 4):
    tracker -> due_trackers -> tracker_adapter; alert -> list_alerts
    (status=pending) -> alert_adapter; trigger -> list_triggers
    (status=enabled) -> trigger_adapter; task -> fetch_task_entries ->
    task_adapter; review -> memories with review_after set -> review_adapter.

    Isolation (V5): each source runs inside a `_safe` wrapper mirroring
    `daily_brief.py:621` — a single source raising is caught, logged, and
    recorded as a `warnings` entry naming the source; the other four sources'
    items are still returned. The call itself never raises for a
    single-source failure.

    Per-source cap + truncation (V7): each source is read up to
    `per_source_cap` rows (default 200). When a source's read would exceed
    the cap, it is truncated to the cap and a `warnings` entry
    `{source, truncated: True, cap}` is recorded — truncation is never silent.

    Bucketing (V2): every collected item already carries a `urgency` bucket
    computed by its adapter (via `calculate_urgency`); this function only
    groups items by that pre-computed bucket and ranks each bucket with
    `rank_items` — urgency is never re-derived here.

    Read-purity (V6): every source read below is a SELECT. assemble_board
    performs no INSERT/UPDATE/DELETE and calls no write-path function, so
    invoking it never mutates a tracker/alert/trigger/task-memory/review row.

    RLS/GUC user scoping (Epic Critical Implementation Note / PRD R5): each
    per-source fetch runs inside `weft.db.connection.acquire(pool)`, which
    issues `SET LOCAL app.user_id` for the connection it hands out from the
    `current_user_id` contextvar (see `weft/auth.py`, `weft/topic_gather.py`
    for the same convention). `user_id` defaults to `WEFT_DEFAULT_USER_ID`
    (the deployment owner) — the same env-var convention `canary_audit_loop`
    uses (`weft/scheduler.py:964`) — so a caller can still override it
    explicitly rather than the board baking in a single-user assumption.
    Each source acquires its OWN connection (not one shared connection)
    because asyncpg connections cannot serve concurrent queries — sharing
    one across the concurrent fan-out would raise "another operation is in
    progress" under real concurrency.

    Args:
        pool: asyncpg connection pool.
        days: due_soon horizon in days, forwarded to every adapter as
            `horizon_days` (PRD `weft_board(days=7, ...)`).
        per_source_cap: max rows read per source before truncation (V7).
        now: reference time for urgency/age computation (defaults to UTC now).
        user_id: RLS scope override. Defaults to `WEFT_DEFAULT_USER_ID`.

    Returns:
        dict matching PRD §Interfaces: generated_at, horizon_days, buckets
        (per-urgency lists of Item dicts), items (flat, bucket-ordered),
        counts (per-bucket + total), warnings (per-source error/truncation
        entries).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if user_id is None:
        user_id = os.environ.get("WEFT_DEFAULT_USER_ID")
        if user_id is None:
            logger.warning(
                "board.no_default_user — set WEFT_DEFAULT_USER_ID for the "
                "deployment owner, or pass user_id= explicitly; RLS will "
                "scope reads to whatever app.user_id the connection already "
                "carries (likely none, yielding an empty board)"
            )

    warnings: list[dict] = []

    def _cap(source: ItemSource, rows: list) -> list:
        """Truncate rows to per_source_cap, recording a warning if hit (V7)."""
        if len(rows) > per_source_cap:
            warnings.append({"source": source, "truncated": True, "cap": per_source_cap})
            return rows[:per_source_cap]
        return rows

    async def _safe(source: ItemSource, coro) -> list[Item]:
        """Isolation wrapper (V5) — mirrors daily_brief.py:621 `_safe`."""
        try:
            return await coro
        except Exception as exc:
            logger.exception(f"board.{source}_error")
            warnings.append({"source": source, "error": str(exc)})
            return []

    async def _tracker_items() -> list[Item]:
        from weft.trackers import due_trackers

        async with acquire(pool) as conn:
            trackers = await due_trackers(conn, now=now, limit=per_source_cap + 1)
        trackers = _cap("tracker", trackers)
        return tracker_adapter(trackers, now, days)

    async def _alert_items() -> list[Item]:
        from weft.alerts import list_alerts
        from weft.models import AlertStatus

        async with acquire(pool) as conn:
            alerts = await list_alerts(
                conn, status=AlertStatus.pending, limit=per_source_cap + 1,
            )
        alerts = _cap("alert", alerts)
        return alert_adapter(alerts, now, days)

    async def _trigger_items() -> list[Item]:
        from weft.triggers import list_triggers
        from weft.models import TriggerStatus

        async with acquire(pool) as conn:
            triggers = await list_triggers(
                conn, status=TriggerStatus.enabled, limit=per_source_cap + 1,
            )
        triggers = _cap("trigger", triggers)
        return trigger_adapter(triggers, now, days)

    async def _task_items() -> list[Item]:
        from weft.skills import fetch_task_entries

        async with acquire(pool) as conn:
            entries = await fetch_task_entries(conn, per_source_cap + 1)
        entries = _cap("task", entries)
        return task_adapter(entries, now, days)

    async def _review_items() -> list[Item]:
        async with acquire(pool) as conn:
            memories = await _fetch_review_memories(conn, per_source_cap + 1)
        memories = _cap("review", memories)
        return review_adapter(memories, now, days)

    # Bind identity BEFORE spawning the concurrent fan-out so each task's
    # copied context inherits it; each task's own acquire() call then
    # acquires its OWN connection (see docstring — one shared connection
    # can't serve five concurrent queries).
    token = current_user_id.set(user_id)
    try:
        tracker_items, alert_items, trigger_items, task_items, review_items = (
            await asyncio.gather(
                _safe("tracker", _tracker_items()),
                _safe("alert", _alert_items()),
                _safe("trigger", _trigger_items()),
                _safe("task", _task_items()),
                _safe("review", _review_items()),
            )
        )
    finally:
        current_user_id.reset(token)

    all_items = tracker_items + alert_items + trigger_items + task_items + review_items

    buckets: dict[Urgency, list[Item]] = {bucket: [] for bucket in _URGENCY_BUCKETS}
    for item in all_items:
        buckets[item.urgency].append(item)
    for bucket in _URGENCY_BUCKETS:
        buckets[bucket] = rank_items(buckets[bucket])

    flat_items: list[Item] = [
        item for bucket in _URGENCY_BUCKETS for item in buckets[bucket]
    ]

    counts = {bucket: len(buckets[bucket]) for bucket in _URGENCY_BUCKETS}
    counts["total"] = len(flat_items)

    return {
        "generated_at": now.isoformat(),
        "horizon_days": days,
        "buckets": {
            bucket: [item.to_dict() for item in buckets[bucket]]
            for bucket in _URGENCY_BUCKETS
        },
        "items": [item.to_dict() for item in flat_items],
        "counts": counts,
        "warnings": warnings,
    }
