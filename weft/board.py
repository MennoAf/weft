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
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Literal
from zoneinfo import ZoneInfo

import asyncpg

from weft.auth import current_user_id
from weft.db.connection import _current_conn, acquire
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

# All five sources, in the board's canonical fan-out order (Epic Task 5 /
# weft_board `sources` param, PRD §Interfaces). Kept as the single ordering
# source of truth for both the default fan-out and `sources=` filtering.
_ALL_ITEM_SOURCES: tuple[ItemSource, ...] = (
    "tracker", "alert", "trigger", "task", "review",
)

# Ghost-F3 fix: a task-memory `due:YYYY-MM-DD` tag has no time-of-day, so it
# must be interpreted as a calendar date in the OWNER's timezone, not
# midnight UTC — otherwise a task due "today" reads as already-past the
# instant UTC crosses midnight, bucketing it `overdue` while `up_next`
# (date-granularity comparison) still calls it `due_soon`, and the two
# surfaces disagree for most of the actual due day. Named-zone (not a fixed
# UTC offset) so DST (EDT/EST) is handled correctly. Single module-level
# seam so a later per-user timezone becomes a one-line change here instead
# of a hunt through every due-date call site.
BOARD_TIMEZONE = ZoneInfo("America/New_York")


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
                else:
                    # TriggerCreate._validate_condition (weft/models.py) does
                    # NOT require trigger_at to be tz-aware (unlike
                    # AlertCreate.trigger_at), so a stored offset-less ISO
                    # string yields a NAIVE datetime here. Normalize to UTC
                    # now, at adapter-construction time, so every downstream
                    # consumer (rank_items' due_at.timestamp(), to_dict's
                    # isoformat()) sees a consistent tz-aware value instead
                    # of naive-as-local-time behavior on non-UTC hosts.
                    if due_at.tzinfo is None:
                        due_at = due_at.replace(tzinfo=timezone.utc)

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


def task_due_at(due_date: str | None) -> datetime | None:
    """Parse a task-memory `due:YYYY-MM-DD` tag into an aware `due_at`.

    Ghost-F3 fix: a date-only tag carries no time-of-day, so it is
    interpreted as END OF DAY (23:59:59) in `BOARD_TIMEZONE` — the owner's
    calendar day, not midnight UTC. A task due "today" therefore stays
    `due_soon` until midnight in that timezone, matching `up_next`'s
    date-granularity rollover instead of flipping to `overdue` the instant
    UTC crosses midnight.

    This is the SINGLE due-date-parsing implementation shared by
    `task_adapter` (below) and `weft.skills.up_next` — the weft-board
    subsumption contract requires the two callers' bucketing to be
    computed from one function, not two independent reimplementations,
    so they cannot silently fork (Epic Task 5 Critical Implementation
    Notes).

    Returns None for a missing or malformed date string — callers treat
    that as `no_date` via `calculate_urgency(None, ...)`.
    """
    if not due_date:
        return None
    try:
        due_date_only = datetime.strptime(due_date, "%Y-%m-%d").date()
    except ValueError:
        return None  # malformed due-date tag — treat as no_date
    return datetime.combine(due_date_only, time(23, 59, 59), tzinfo=BOARD_TIMEZONE)


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

        # due_at parsed from the "due:YYYY-MM-DD" topic tag, end-of-day in
        # BOARD_TIMEZONE (Ghost-F3 fix — see task_due_at docstring).
        due_at = task_due_at(entry.due_date)

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


async def _fetch_snoozed_due_trackers(
    conn: asyncpg.Connection, *, now: datetime, limit: int,
) -> list[Tracker]:
    """Trackers that would be due except an active `snooze_until` is
    suppressing them — the exact complement of what
    `weft.trackers.due_trackers` returns. Included on the board only when
    `include_snoozed=True` (PRD §Validation V3).

    This mirrors `due_trackers`'s query (same open-state/nudge_mode/
    nudge_after filter) with the snooze clause inverted, rather than adding
    an `include_snoozed` parameter to `weft.trackers.due_trackers` itself —
    that module is outside this task's anchored write scope (leaf task
    loom-0a001c5a: weft/board.py, weft/skills.py, weft/mcp/tools.py only),
    and keeping `due_trackers`'s own signature/behavior untouched preserves
    the existing monkeypatch-based isolation test
    (`test_one_source_failure_isolated_others_still_return`), which patches
    `weft.trackers.due_trackers` directly.
    """
    from weft.trackers import _row_to_tracker

    rows = await conn.fetch(
        """
        SELECT * FROM trackers
        WHERE state IN ('in_progress', 'awaiting_reply', 'blocked')
          AND nudge_mode <> 'none'
          AND nudge_after IS NOT NULL
          AND nudge_after <= $1
          AND snooze_until IS NOT NULL AND snooze_until > $1
        ORDER BY nudge_after ASC
        LIMIT $2
        """,
        now, limit,
    )
    return [_row_to_tracker(r) for r in rows]


async def assemble_board(
    pool: asyncpg.Pool,
    *,
    days: int = DEFAULT_HORIZON_DAYS,
    per_source_cap: int = DEFAULT_PER_SOURCE_CAP,
    now: datetime | None = None,
    user_id: str | None = None,
    include_snoozed: bool = False,
    sources: list[str] | None = None,
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
    for the same convention). `user_id` defaults to the ambient authenticated
    caller (`current_user_id` contextvar, mirroring
    `weft.auth.resolve_caller_user_id`'s precedence) and falls back to
    `WEFT_DEFAULT_USER_ID` (the deployment owner) only when no caller is
    bound — the same env-var convention `canary_audit_loop` uses
    (`weft/scheduler.py:964`) — so a caller can still override it explicitly
    rather than the board baking in a single-user assumption. Each source
    acquires its OWN connection (not one shared connection) because asyncpg
    connections cannot serve concurrent queries — sharing one across the
    concurrent fan-out would raise "another operation is in progress" under
    real concurrency. Because `weft.db.connection.acquire()` is idempotent
    (it reuses an already-bound connection rather than acquiring a new one),
    this function clears the `_current_conn` contextvar in its own context
    right before `asyncio.gather()` so each fan-out task's copied context
    takes the fresh-acquire branch even when assemble_board is itself
    invoked from inside an outer `acquire()` scope (e.g. an MCP tool
    handler).

    Snoozing (V3): a tracker whose `snooze_until` is still in the future is
    excluded from the board by default (`due_trackers` already filters it
    out at the SQL level) and included only when `include_snoozed=True`. The
    snoozed-but-otherwise-due trackers are fetched via a small supplementary
    query local to this module (`_fetch_snoozed_due_trackers`) rather than
    changing `weft.trackers.due_trackers`'s own filtering — this keeps
    `due_trackers`'s existing default-path contract (and the tests that
    monkeypatch it) untouched while still surfacing snoozed items on request.

    Source selection: `sources`, when given, restricts the fan-out to that
    subset of {"tracker", "alert", "trigger", "task", "review"} — unknown
    values are silently ignored. Sources not selected are simply not
    fetched (no warning); this is a scope filter, not a failure.

    Args:
        pool: asyncpg connection pool.
        days: due_soon horizon in days, forwarded to every adapter as
            `horizon_days` (PRD `weft_board(days=7, ...)`).
        per_source_cap: max rows read per source before truncation (V7).
        now: reference time for urgency/age computation (defaults to UTC now).
        user_id: RLS scope override. Defaults to the ambient authenticated
            caller (`current_user_id` contextvar), falling back to
            `WEFT_DEFAULT_USER_ID` when no caller is bound.
        include_snoozed: include trackers currently suppressed by an active
            `snooze_until` (default False — matches PRD §Validation V3).
        sources: restrict the fan-out to this subset of source names
            (default None = all five sources).

    Returns:
        dict matching PRD §Interfaces: generated_at, horizon_days, buckets
        (per-urgency lists of Item dicts), items (flat, bucket-ordered),
        counts (per-bucket + total), warnings (per-source error/truncation
        entries, plus a structured `{"source": "identity", ...}` entry when
        no caller identity could be resolved — see below).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    warnings: list[dict] = []

    if user_id is None:
        # Ambient authenticated caller wins over the deployment-owner env
        # fallback — matches every other read path's identity precedence
        # (weft.auth.resolve_caller_user_id: contextvar first, fallback
        # second). Jumping straight to WEFT_DEFAULT_USER_ID here would
        # silently ignore a real caller already bound via current_user_id
        # (e.g. an authenticated MCP tool invocation), scoping the board to
        # the deployment owner instead of the actual requester.
        user_id = current_user_id.get() or os.environ.get("WEFT_DEFAULT_USER_ID")
        if user_id is None:
            logger.warning(
                "board.no_default_user — set WEFT_DEFAULT_USER_ID for the "
                "deployment owner, or pass user_id= explicitly; RLS will "
                "scope reads to whatever app.user_id the connection already "
                "carries (likely none, yielding an empty board)"
            )
            # Ratified decision (weft-64c14697): the log line alone left
            # "misconfigured — no identity resolved" indistinguishable from
            # "resolved fine, genuinely nothing is due" once the response
            # left this process. Surface it structurally too so a caller
            # (UI or agent) can tell the two apart without grepping logs.
            warnings.append({
                "source": "identity",
                "error": "no_default_user",
                "message": (
                    "No caller identity resolved (current_user_id unset and "
                    "WEFT_DEFAULT_USER_ID not configured) — board results "
                    "may be scoped to no rows rather than reflecting "
                    "genuinely empty state."
                ),
            })

    def _cap(source: ItemSource, rows: list) -> list:
        """Truncate rows to per_source_cap, recording a warning if hit (V7)."""
        if len(rows) > per_source_cap:
            warnings.append({"source": source, "truncated": True, "cap": per_source_cap})
            return rows[:per_source_cap]
        return rows

    async def _safe(source: ItemSource, coro) -> list[Item]:
        """Isolation wrapper (V5) — mirrors daily_brief.py:621 `_safe`.

        Records only the exception's class name in `warnings`, never
        `str(exc)` — the full detail (which for asyncpg failures can
        include schema/query internals) is already captured by
        `logger.exception` below; the response payload should not leak it.
        """
        try:
            return await coro
        except Exception as exc:
            logger.exception(f"board.{source}_error")
            warnings.append({"source": source, "error": type(exc).__name__})
            return []

    async def _tracker_items() -> list[Item]:
        from weft.trackers import due_trackers

        async with acquire(pool) as conn:
            trackers = await due_trackers(conn, now=now, limit=per_source_cap + 1)
            if include_snoozed:
                trackers = trackers + await _fetch_snoozed_due_trackers(
                    conn, now=now, limit=per_source_cap + 1,
                )
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

    _SOURCE_FETCHERS: dict[ItemSource, Callable[[], Awaitable[list[Item]]]] = {
        "tracker": _tracker_items,
        "alert": _alert_items,
        "trigger": _trigger_items,
        "task": _task_items,
        "review": _review_items,
    }

    # `sources`, when given, narrows the fan-out to that subset (order fixed
    # by _ALL_ITEM_SOURCES regardless of the caller's list order); unknown
    # names are silently dropped rather than erroring — a scope filter, not
    # a validated enum.
    selected_sources: tuple[ItemSource, ...] = (
        _ALL_ITEM_SOURCES
        if sources is None
        else tuple(s for s in _ALL_ITEM_SOURCES if s in sources)
    )

    # Bind identity BEFORE spawning the concurrent fan-out so each task's
    # copied context inherits it; each task's own acquire() call then
    # acquires its OWN connection (see docstring — one shared connection
    # can't serve five concurrent queries).
    #
    # acquire() (weft.db.connection) is IDEMPOTENT: if `_current_conn` is
    # already set (e.g. because assemble_board is called from inside an MCP
    # tool handler, which wraps its whole body in `async with
    # acquire(app.pool):`), it yields the EXISTING connection instead of
    # acquiring a new one. asyncio.gather's five tasks each get a *copy* of
    # the current context, so all five would inherit that same connection
    # and run `conn.fetch()` concurrently on it — asyncpg raises "another
    # operation is in progress", which `_safe` swallows into `warnings`,
    # silently starving the board. Force the fresh-acquire branch for the
    # fan-out by clearing `_current_conn` in *this* context right before
    # gathering; each task's copied context then sees no existing
    # connection and calls `pool.acquire()` for its own. `current_user_id`
    # is untouched by this reset, so `acquire()`'s `SET LOCAL app.user_id`
    # still binds the correct RLS identity on every fresh connection.
    token = current_user_id.set(user_id)
    conn_token = _current_conn.set(None)
    try:
        results = await asyncio.gather(
            *(
                _safe(source, _SOURCE_FETCHERS[source]())
                for source in selected_sources
            )
        )
    finally:
        _current_conn.reset(conn_token)
        current_user_id.reset(token)

    items_by_source: dict[ItemSource, list[Item]] = dict(zip(selected_sources, results))
    all_items: list[Item] = [
        item
        for source in _ALL_ITEM_SOURCES
        for item in items_by_source.get(source, [])
    ]

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


# ---------------------------------------------------------------------------
# L1 feedback engine — SHADOW mode (weft-board-epic Task 7; PRD §Compounding
# Loops "Triage Correction Ratchet").
#
# The board discards its own exhaust unless something records it: every
# triage action fired through `act()` below is a datapoint about what
# actually deserved attention. This section ships:
#   1. `record_triage_event()` — the SIGNAL append (board_triage_events).
#   2. `act()` — fires an existing Weft write tool for one triage action AND
#      appends the triage event as a side effect (the non-HTTP analog of the
#      PRD's `POST /act`; the future localhost overlay, weft-board-epic
#      Task 9, is expected to front this same function rather than
#      reimplementing the dispatch/append).
#   3. `RULE_REGISTRY` — a data-described `{name, signal_predicate,
#      proposed_action}` list (two rules: repeat-snooze, repeat-dismiss).
#   4. `run_feedback_pass()` — evaluates the registry and writes proposals
#      (board_feedback_proposals). SHADOW mode (the v1 default) writes
#      proposals ONLY; mutation is possible only in `active` mode, and only
#      through the single, explicitly-gated call site in this function
#      (PRD Validation V8 / Epic Critical Implementation Note — the shadow
#      no-write gate is a hard requirement, not a convention).
# ---------------------------------------------------------------------------

BoardFeedbackMode = Literal["off", "shadow", "active"]

# Config: which mode the L1 feedback engine runs in. Shadow-first rollout is
# the ratified default (weft-64c14697) — the engine records every proposal
# it WOULD make without mutating anything, so the theorem ("would-have-
# proposed" vs "what you actually did") is proven before activation.
# Activation to `active` is a later, explicit config flip (gated on a
# mechanical data threshold per the loop blueprint's Phasing — see PRD
# §Compounding Loops Research Item R7), not a code change.
_BOARD_FEEDBACK_MODE_ENV_VAR = "WEFT_BOARD_FEEDBACK_MODE"
DEFAULT_BOARD_FEEDBACK_MODE: BoardFeedbackMode = "shadow"


def get_board_feedback_mode() -> BoardFeedbackMode:
    """Resolve the L1 feedback engine's operating mode.

    Reads `WEFT_BOARD_FEEDBACK_MODE` (mirrors the `WEFT_*` env-var config
    convention used elsewhere, e.g. `WEFT_DEFAULT_USER_ID`,
    `WEFT_TURN_RERANK_DISABLE`), defaulting to `"shadow"` — never to
    `"active"` — so an unset/misconfigured deployment never mutates state
    from the feedback loop by accident.
    """
    raw = os.environ.get(_BOARD_FEEDBACK_MODE_ENV_VAR, DEFAULT_BOARD_FEEDBACK_MODE)
    if raw not in ("off", "shadow", "active"):
        logger.warning(
            f"board.invalid_feedback_mode — {raw!r} is not one of "
            f"off/shadow/active; falling back to {DEFAULT_BOARD_FEEDBACK_MODE!r}"
        )
        return DEFAULT_BOARD_FEEDBACK_MODE
    return raw  # type: ignore[return-value]


# R7 (PRD §Compounding Loops / Research Items): tuning thresholds are
# ASSUMED, not derived from real triage volume yet. Named config constants
# — not literals buried in a query — so they move without a schema change
# once real data informs them.
SNOOZE_REPEAT_THRESHOLD = 3
DISMISS_DISTINCT_ITEM_THRESHOLD = 3


async def record_triage_event(
    pool: asyncpg.Pool,
    *,
    item_id: str,
    source: str,
    kind: str,
    urgency_at_surface: str,
    age_days_at_surface: float,
    verb: str,
    snooze_duration_days: float | None = None,
) -> None:
    """Append one row to `board_triage_events` — the L1 loop's SIGNAL.

    This is the single append point for the triage-correction loop's raw
    signal (loop blueprint: "NEEDS instrumentation — one append in the
    action-dispatch path"). `act()` below calls this as a side effect of
    firing a triage action; the future overlay's `POST /act`
    (weft-board-epic Task 9) is expected to call this same function rather
    than re-implementing the insert, so the two callers can't fork.
    """
    async with acquire(pool) as conn:
        await conn.execute(
            """
            INSERT INTO board_triage_events
                (item_id, source, kind, urgency_at_surface,
                 age_days_at_surface, verb, snooze_duration_days)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            item_id, source, kind, urgency_at_surface,
            age_days_at_surface, verb, snooze_duration_days,
        )


# --- /act: fire an existing write tool + record the triage event ----------
#
# Allowlist: exactly the existing Weft write tools the PRD names for triage
# (§Ground Truth / §Behavior "triage without a new write path") — never
# expanded ad hoc; the overlay's security boundary (PRD §Critical
# Implementation Notes: "must reject any tool not on the write-tool
# allowlist BEFORE dispatch") is this same dict. Each entry wraps the
# store-layer function directly (not the MCP-tool-decorated function, which
# requires an `mcp.Context`) so this module carries no MCP dependency.

async def _act_tracker_close(pool: asyncpg.Pool, args: dict):
    from weft.models import TrackerState
    from weft.trackers import close_tracker

    return await close_tracker(
        pool, args["tracker_id"],
        final_state=TrackerState(args.get("final_state", "done")),
        note=args.get("note"),
    )


async def _act_tracker_snooze(pool: asyncpg.Pool, args: dict):
    from weft.trackers import snooze_tracker

    until = args["until"]
    if isinstance(until, str):
        until = datetime.fromisoformat(until)
    return await snooze_tracker(pool, args["tracker_id"], until)


async def _act_tracker_dismiss(pool: asyncpg.Pool, args: dict):
    from weft.trackers import dismiss_tracker

    return await dismiss_tracker(pool, args["tracker_id"])


async def _act_tracker_update(pool: asyncpg.Pool, args: dict):
    from weft.models import NudgeMode, TrackerState
    from weft.trackers import update_tracker

    nudge_after = args.get("nudge_after")
    nudge_interval_seconds = args.get("nudge_interval_seconds")
    return await update_tracker(
        pool, args["tracker_id"],
        title=args.get("title"),
        state=TrackerState(args["state"]) if args.get("state") else None,
        state_note=args.get("state_note"),
        context=args.get("context"),
        nudge_mode=NudgeMode(args["nudge_mode"]) if args.get("nudge_mode") else None,
        nudge_after=(
            datetime.fromisoformat(nudge_after) if nudge_after else None
        ),
        nudge_interval=(
            timedelta(seconds=nudge_interval_seconds)
            if nudge_interval_seconds is not None else None
        ),
    )


async def _act_alert_dismiss(pool: asyncpg.Pool, args: dict):
    from weft.alerts import dismiss_alert

    return await dismiss_alert(pool, args["alert_id"])


async def _act_trigger_delete(pool: asyncpg.Pool, args: dict):
    from weft.triggers import delete_trigger

    return await delete_trigger(pool, args["trigger_id"])


async def _act_trigger_fire(pool: asyncpg.Pool, args: dict):
    from weft.triggers import record_fire

    return await record_fire(pool, args["trigger_id"])


ACT_ALLOWLIST: dict[str, Callable[[asyncpg.Pool, dict], Awaitable]] = {
    "weft_tracker_close": _act_tracker_close,
    "weft_tracker_snooze": _act_tracker_snooze,
    "weft_tracker_dismiss": _act_tracker_dismiss,
    "weft_tracker_update": _act_tracker_update,
    "weft_alert_dismiss": _act_alert_dismiss,
    "weft_trigger_delete": _act_trigger_delete,
    "weft_trigger_fire": _act_trigger_fire,
}


async def act(
    pool: asyncpg.Pool,
    *,
    tool: str,
    args: dict,
    item_id: str,
    source: str,
    kind: str,
    urgency_at_surface: str,
    age_days_at_surface: float,
    verb: str,
    snooze_duration_days: float | None = None,
) -> dict:
    """Fire one triage action through an existing Weft write tool AND record
    it as L1 feedback-loop signal (PRD §Compounding Loops "Triage
    Correction Ratchet"). This is the non-HTTP analog of the PRD's
    `POST /act`; the future localhost overlay (weft-board-epic Task 9, not
    yet built) is expected to front this exact function with an HTTP
    handler instead of re-implementing the dispatch or the event-append —
    "an agent firing a weft_board action" (the loop blueprint's SIGNAL
    description) is this call, made directly.

    `tool` must be a name in `ACT_ALLOWLIST` — anything else is rejected
    with no write and no triage-event append (mirrors the PRD's overlay
    security boundary). The triage event is appended only after the
    underlying write succeeds, so a failed action is never recorded as a
    triage datapoint.

    `urgency_at_surface` / `age_days_at_surface` are the item's bucket/age
    AT THE TIME it was surfaced by `weft_board` — the caller (an agent or
    the future overlay) already holds these from the board response; `act`
    does not re-derive them, since re-deriving "at surface time" values at
    act time would silently answer a different question.
    """
    dispatch = ACT_ALLOWLIST.get(tool)
    if dispatch is None:
        raise ValueError(
            f"board.act: tool {tool!r} is not on the triage write-tool "
            f"allowlist ({sorted(ACT_ALLOWLIST)})"
        )

    result = await dispatch(pool, args)

    await record_triage_event(
        pool,
        item_id=item_id,
        source=source,
        kind=kind,
        urgency_at_surface=urgency_at_surface,
        age_days_at_surface=age_days_at_surface,
        verb=verb,
        snooze_duration_days=snooze_duration_days,
    )

    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"result": result}


# --- Rule registry: {name, signal_predicate, proposed_action} -------------
#
# A new rule is a registry entry, not new engine code (PRD Non-Goals: no
# plugin framework, just this data-described seam). `signal_predicate` reads
# `board_triage_events` and returns the list of targets that currently
# satisfy the rule; `proposed_action` maps one target to the data-described
# change that would be applied in `active` mode.


@dataclass
class TriageRule:
    """One rule registry entry: `{name, signal_predicate, proposed_action}`
    per PRD §Interfaces / Epic Core Decisions."""

    name: str
    signal_predicate: Callable[[asyncpg.Connection], Awaitable[list[dict]]]
    proposed_action: Callable[[dict], dict]


async def _repeat_snooze_signal(conn: asyncpg.Connection) -> list[dict]:
    """Targets: item_ids snoozed >= SNOOZE_REPEAT_THRESHOLD times
    (ROBUSTNESS GAP rule — loop blueprint)."""
    rows = await conn.fetch(
        """
        SELECT item_id, source, kind, count(*) AS snooze_count
        FROM board_triage_events
        WHERE verb = 'snooze'
        GROUP BY item_id, source, kind
        HAVING count(*) >= $1
        """,
        SNOOZE_REPEAT_THRESHOLD,
    )
    return [dict(r) for r in rows]


def _repeat_snooze_action(target: dict) -> dict:
    """Proposed change: extend that item's nudge_interval one step. The
    exact new value is computed by the ACTIVE apply path (weft-board-epic
    Task loom-692f59d4, R6) — this data shape is deliberately abstract
    ("extend one step") rather than inventing a concrete new_value here."""
    return {
        "field": "nudge_interval",
        "action": "extend_one_step",
        "item_id": target["item_id"],
        "snooze_count": target["snooze_count"],
    }


async def _repeat_dismiss_signal(conn: asyncpg.Connection) -> list[dict]:
    """Targets: (source, kind) pairs dismissed across
    >= DISMISS_DISTINCT_ITEM_THRESHOLD distinct items (FEATURE SIGNAL rule —
    loop blueprint)."""
    rows = await conn.fetch(
        """
        SELECT source, kind, count(DISTINCT item_id) AS distinct_dismissed
        FROM board_triage_events
        WHERE verb = 'dismiss'
        GROUP BY source, kind
        HAVING count(DISTINCT item_id) >= $1
        """,
        DISMISS_DISTINCT_ITEM_THRESHOLD,
    )
    return [dict(r) for r in rows]


def _repeat_dismiss_action(target: dict) -> dict:
    """Proposed change: add this (source, kind) pair to hidden_kinds.

    Ratified decision (weft-64c14697): hidden_kinds matches by KIND/ID
    equality, never name-substring — `kind` here is the exact `Item.kind`
    identifier (e.g. a TriggerConditionType value, a TrackerKind value),
    not a name fragment to substring-match against titles."""
    return {
        "add_to": "hidden_kinds",
        "source": target["source"],
        "kind": target["kind"],
        "distinct_dismissed": target["distinct_dismissed"],
    }


RULE_REGISTRY: list[TriageRule] = [
    TriageRule(
        name="repeat-snooze",
        signal_predicate=_repeat_snooze_signal,
        proposed_action=_repeat_snooze_action,
    ),
    TriageRule(
        name="repeat-dismiss",
        signal_predicate=_repeat_dismiss_signal,
        proposed_action=_repeat_dismiss_action,
    ),
]


def _proposal_target_id(rule_name: str, target: dict) -> str:
    """Opaque `target_id` for `board_feedback_proposals` — a single-item id
    for repeat-snooze, or the `source:kind` pair for repeat-dismiss (kept
    together rather than bare `kind`, since kind vocabularies are not
    guaranteed unique across the five sources)."""
    if rule_name == "repeat-snooze":
        return target["item_id"]
    return f"{target['source']}:{target['kind']}"


async def _apply_proposal(pool: asyncpg.Pool, rule_name: str, proposal: dict) -> None:
    """ACTIVE-mode apply path — INTENTIONAL NO-OP STUB in this task.

    weft-board-epic Task loom-692f59d4 (L1 ACTIVE apply path) fills this
    in: extending a tracker's nudge_interval for repeat-snooze (via
    `weft.trackers.update_tracker`, reusing existing internals per R6), and
    firing a `board_feedback` alert for repeat-dismiss for human review.
    This task ships the registry + the mode gate at the single call site in
    `run_feedback_pass` — not the mutation itself.
    """
    return None


async def run_feedback_pass(
    pool: asyncpg.Pool,
    *,
    mode: BoardFeedbackMode | None = None,
) -> list[dict]:
    """Evaluate `RULE_REGISTRY` over `board_triage_events` and record
    proposals to `board_feedback_proposals` (PRD §Compounding Loops L1).

    SHADOW NO-WRITE GATE (hard requirement, PRD Validation V8 / Epic
    Critical Implementation Note — NOT a convention): this function's ONLY
    mutation of tracker/alert state is the call to `_apply_proposal` below,
    and that call is reached ONLY when `mode == "active"`. In `shadow` mode
    (the v1 default) or `off` mode, no tracker row and no alert is ever
    touched by this function — it writes exclusively to
    `board_feedback_proposals`. The foot-gun this guards against (per the
    Epic): applying `proposed_action` inline during rule evaluation instead
    of behind the mode check.

    Returns the list of proposal rows written (as dicts), for both modes.
    """
    if mode is None:
        mode = get_board_feedback_mode()

    if mode == "off":
        return []

    proposals: list[dict] = []
    async with acquire(pool) as conn:
        for rule in RULE_REGISTRY:
            targets = await rule.signal_predicate(conn)
            for target in targets:
                proposed_change = rule.proposed_action(target)
                target_id = _proposal_target_id(rule.name, target)

                row = await conn.fetchrow(
                    """
                    INSERT INTO board_feedback_proposals
                        (rule, target_id, proposed_change, mode)
                    VALUES ($1, $2, $3::jsonb, $4)
                    RETURNING id, rule, target_id, proposed_change, mode, created_at
                    """,
                    rule.name, target_id, json.dumps(proposed_change), mode,
                )
                proposal = dict(row)
                if isinstance(proposal["proposed_change"], str):
                    proposal["proposed_change"] = json.loads(proposal["proposed_change"])
                proposals.append(proposal)

                # SHADOW NO-WRITE GATE — see docstring. `mode == "shadow"`
                # (and `"off"`, handled above) never reach this branch.
                if mode == "active":
                    await _apply_proposal(pool, rule.name, proposal)

    return proposals
