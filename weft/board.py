"""Weft board — unified open-items contract, pure functions for bucketing and ranking.

This module defines the Item model (normalized across five sources: trackers,
alerts, triggers, taREDACTED, review queue) and pure functions for urgency
bucketing and ranking. No database access; composable into adapters that feed
the assemble_board orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

from weft.models import TriggerConditionType

if TYPE_CHECKING:
    from weft.models import Alert, Memory, Tracker, Trigger
    from weft.skills import TaskEntry


ItemSource = Literal["tracker", "alert", "trigger", "task", "review"]
Urgency = Literal["overdue", "due_soon", "pending", "no_date"]


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


def tracker_adapter(trackers: list[Tracker], now: datetime | None = None) -> list[Item]:
    """Map Tracker objects to Items for the board.

    Reuses due_trackers filtering (snooze logic already applied upstream).
    Each tracker is converted to an Item with source="tracker", using nudge_after
    as the due_at (when the next nudge is due). Urgency computed from nudge_after.

    Args:
        trackers: list of Tracker objects from due_trackers()
        now: current time for age_days calculation (defaults to UTC now)

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
            urgency=calculate_urgency(tracker.nudge_after, now),
            project_id=tracker.project_id,
            entity_id=tracker.entity_id,
            actions=[],
        )
        items.append(item)

    return items


def alert_adapter(alerts: list[Alert], now: datetime | None = None) -> list[Item]:
    """Map Alert objects to Items for the board.

    Each alert is converted to an Item with source="alert", using trigger_at
    as the due_at. Only maps pending alerts; other statuses excluded by list_alerts
    filtering. Urgency computed from trigger_at.

    Args:
        alerts: list of Alert objects, typically filtered by status=pending
        now: current time for age_days calculation (defaults to UTC now)

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
            urgency=calculate_urgency(alert.trigger_at, now),
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


def trigger_adapter(triggers: list[Trigger], now: datetime | None = None) -> list[Item]:
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
            urgency=calculate_urgency(due_at, now),
            project_id=trigger.project_id,
            entity_id=None,  # Triggers do not have entity association
            actions=[],
        )
        items.append(item)

    return items


def task_adapter(entries: list[TaskEntry], now: datetime | None = None) -> list[Item]:
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
            state=None,  # taREDACTED have no tracker/alert-style state
            due_at=due_at,
            snoozed_until=None,  # TaREDACTED do not have snooze capability
            age_days=age_days,
            urgency=calculate_urgency(due_at, now),
            project_id=None,  # TaskEntry does not carry project association
            entity_id=None,  # TaskEntry does not carry entity association
            actions=[],
        )
        items.append(item)

    return items


def review_adapter(memories: list[Memory], now: datetime | None = None) -> list[Item]:
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
            urgency=calculate_urgency(memory.review_after, now),
            project_id=memory.project_id,
            entity_id=None,  # Memories do not have entity association
            actions=[],
        )
        items.append(item)

    return items
