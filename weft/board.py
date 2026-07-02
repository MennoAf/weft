"""Weft board — unified open-items contract, pure functions for bucketing and ranking.

This module defines the Item model (normalized across five sources: trackers,
alerts, triggers, task-memories, review queue) and pure functions for urgency
bucketing and ranking. No database access; composable into adapters that feed
the assemble_board orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


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
