#!/usr/bin/env python3
"""
agenda_manifest.py — PAAH ground truth for the agenda shape (trackers).

The personal-agent query: "what's on my plate" / "what do I keep pushing." The
answer is the set of OPEN LOOPS that need attention right now — trackers that are
past their nudge, still open, and not snoozed. Unlike the enumeration/entity/
temporal shapes, this shape has no stochastic embedding path: trackers are
retrieved by deterministic SQL predicate, not semantic search. So the "never-miss
spread over k phrasings" rule (banned single-run comparisons, weft-b015d16a) does
not apply here — ``weft_tracker_due`` is a pure, reproducible query and a single
run is authoritative. The measurement is instead structural: does the due query
return EXACTLY the seeded open loops, ordered so the thing you keep pushing
surfaces first, and does the agent-facing digest actually carry them?

Ground truth is DERIVED, not hand-flagged: each spec's ``expected_due`` recomputes
the exact ``due_trackers`` predicate (open state ∧ nudge_mode≠none ∧ nudge_after≤now
∧ not-snoozed), so the manifest can't drift out of sync with the rule it asserts.

Dates are RELATIVE. "This week" only means anything relative to the moment the
harness runs, so specs carry day-offsets from now and the seeder resolves them to
absolute timestamps at seed time (the temporal shape could hardcode ISO dates
because it asks "when did X"; the agenda shape asks "what's due" and must move
with the clock).

Synthetic persona only (house rule): the user is Jim Boblaw. No real names.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.personal_agent.manifest import PAAH_USER_ID  # re-export identity

PAAH_AGENDA_PROJECT_ID = "paah-personal-agent-agenda"

# Open states that keep a tracker in the due queue (mirrors TrackerState.open_states).
_OPEN_STATES = frozenset({"in_progress", "awaiting_reply", "blocked"})

__all__ = [
    "PAAH_USER_ID",
    "PAAH_AGENDA_PROJECT_ID",
    "AgendaSpec",
    "get_agenda_specs",
]


@dataclass(frozen=True)
class AgendaSpec:
    """One seeded tracker with a known due-fate.

    key: stable handle so the harness can name trackers by id.
    kind/title/state: the tracker's shape at creation. ``state`` is always an
        OPEN state; terminal trackers are created open and then closed via
        ``close_as`` so the real lifecycle (state_history append) is exercised.
    nudge_mode: 'once' | 'recur' | 'none'. 'none' is query-only — never due.
    nudge_after_days: offset from now, in days. Negative = already past (overdue);
        positive = not yet due; None = no nudge scheduled (paired with mode 'none').
    nudge_interval: shorthand ('2w') required by 'recur' mode; ignored otherwise.
    snooze_days: if set, the tracker is snoozed until now+snooze_days — suppressed
        from the due queue even when otherwise overdue.
    close_as: 'done' | 'abandoned' to close the tracker after creation (terminal).
    """

    key: str
    kind: str
    title: str
    state: str
    nudge_mode: str
    nudge_after_days: float | None
    nudge_interval: str | None = None
    snooze_days: float | None = None
    close_as: str | None = None

    @property
    def expected_due(self) -> bool:
        """Re-derives the ``due_trackers`` predicate — this IS the oracle.

        Open-state ∧ nudge scheduled ∧ nudge_after in the past ∧ not currently
        snoozed. Kept in lock-step with weft/trackers.py::due_trackers so the
        manifest's ground truth is the query's own contract, not a copy of it.
        """
        is_open = self.close_as is None and self.state in _OPEN_STATES
        has_nudge = self.nudge_mode != "none" and self.nudge_after_days is not None
        past_due = has_nudge and self.nudge_after_days <= 0
        not_snoozed = self.snooze_days is None or self.snooze_days <= 0
        return is_open and past_due and not_snoozed


# ── The plate: 10 trackers, 4 due open loops + 6 that must be excluded ──
#
# Every excluded tracker exercises a DISTINCT clause of the due predicate, so a
# regression that drops one clause (e.g. stops honoring snooze) trips precision.
def get_agenda_specs() -> list[AgendaSpec]:
    return [
        # ---- DUE now: the open loops an agenda answer must surface ----
        # taxes is the longest overdue → the "on my plate I keep pushing" anchor;
        # due_trackers orders by nudge_after ASC, so it must come back first.
        AgendaSpec(
            key="taxes", kind="follow_up",
            title="File the amended 2025 tax return",
            state="in_progress", nudge_mode="once", nudge_after_days=-21,
        ),
        AgendaSpec(
            key="dentist", kind="task",
            title="Schedule Jim's overdue dentist cleaning",
            state="in_progress", nudge_mode="once", nudge_after_days=-7,
        ),
        AgendaSpec(
            key="plumber", kind="task",
            title="Call the plumber about the basement leak",
            state="awaiting_reply", nudge_mode="once", nudge_after_days=-3,
        ),
        AgendaSpec(
            key="membership", kind="follow_up",
            title="Renew the community garden co-op membership",
            state="blocked", nudge_mode="once", nudge_after_days=-1,
        ),
        # ---- EXCLUDED: each fails a different clause of the due predicate ----
        # future nudge (not yet due) — even though it's "this week", it's not on
        # the plate YET, and due_trackers is a past-due query.
        AgendaSpec(
            key="book_club", kind="task",
            title="Pick the next book-club selection",
            state="in_progress", nudge_mode="once", nudge_after_days=3,
        ),
        # far-future recurring nudge — not due.
        AgendaSpec(
            key="conference", kind="follow_up",
            title="Submit the urban-gardening conference proposal",
            state="in_progress", nudge_mode="recur", nudge_after_days=21,
            nudge_interval="2w",
        ),
        # overdue BUT snoozed — the user parked it; must stay off the plate.
        AgendaSpec(
            key="insurance", kind="follow_up",
            title="Compare home-insurance renewal quotes",
            state="in_progress", nudge_mode="once", nudge_after_days=-5,
            snooze_days=6,
        ),
        # terminal (done) — finished loops don't nag.
        AgendaSpec(
            key="donate", kind="task",
            title="Drop off the winter clothing donation",
            state="in_progress", nudge_mode="once", nudge_after_days=-10,
            close_as="done",
        ),
        # terminal (abandoned) — dropped loops don't nag either.
        AgendaSpec(
            key="oldbike", kind="task",
            title="Restore the vintage road bike",
            state="in_progress", nudge_mode="once", nudge_after_days=-30,
            close_as="abandoned",
        ),
        # query-only (nudge_mode none) — a someday/maybe, never in the due queue.
        AgendaSpec(
            key="someday_sail", kind="watch",
            title="Someday: learn to sail",
            state="in_progress", nudge_mode="none", nudge_after_days=None,
        ),
    ]
