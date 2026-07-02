"""Migration 67: board_triage_events + board_feedback_proposals — L1 triage
correction loop substrate (weft-board-epic Task 6).

## Why these tables exist

`weft_board` (weft/board.py) unifies five open-item sources into one
triage-able contract, but the base design discards the exhaust: every triage
action (close/snooze/dismiss/update fired via an item's `actions[]`
descriptor) is a datapoint about what actually deserved attention, and
without a store that signal evaporates. This migration ships the SIGNAL +
STORE half of the "Triage Correction Ratchet" compounding loop
(documents/prds/weft-board.md — Compounding Loops), landing the schema ahead
of the SHADOW-mode feedback engine (weft-board-epic Task 7) that reads and
writes it.

## What these tables are

`board_triage_events` — append-only log, one row per triage action fired
through the board (`POST /act` or an agent calling a `weft_board` action).
Captures `(item_id, source, kind, urgency_at_surface, age_days_at_surface,
verb, snooze_duration_days, ts)` per the loop's SIGNAL spec. Queryable by
`item_id` (has this specific item been repeatedly snoozed?) and by
`(source, kind)` (is this kind of item chronically dismissed?) — the two
axes the two v1 rules (repeat-snooze, repeat-dismiss) evaluate.

`board_feedback_proposals` — the feedback engine's proposals log. Every rule
firing records `{rule, target_id, proposed_change, mode, ts}`. In `shadow`
mode (the v1 default) this is the ONLY write the engine makes — the shadow
no-write gate (PRD Validation V8) depends on tracker/alert state being
unchanged while this table accumulates proposals. `mode` is stamped per row
so a shadow-vs-active proposal history is distinguishable after the engine
is later flipped to `active`.

Both tables are populated starting with Task 7 (the feedback engine); this
migration only lands the schema so that work — and its tests
(`tests/test_board_triage_loop.py`) — can build directly on it.

## Retention

`board_triage_events` is append-only and windowed, mirroring the
`recall_canary_audit` pattern (v66) and its `_CANARY_AUDIT_RETENTION_DAYS`
constant in weft/canary.py. `BOARD_TRIAGE_RETENTION_DAYS` below is the named
config constant for that horizon (Research Item R8: 90 days, matching the
loop blueprint's "windowed retention 90d" — long enough to cover the L1
snooze/dismiss thresholds' lookback with room to spare, short enough to keep
the log bounded). It is NOT a hard-coded literal buried in a query: the
future feedback engine (Task 7) prunes rows older than this constant on each
pass, the same way `canary_audit()` prunes `recall_canary_audit` today (see
weft/canary.py:672-676). The migration itself does not prune — Postgres has
no expiry here, only the index that makes a windowed prune/query cheap
(`idx_board_triage_events_window`).

## Schema notes

* Both tables use `BIGSERIAL` surrogate PKs — events are append-only, no
  natural key needed.
* `user_id` follows the post-v36 convention: `NOT NULL DEFAULT` the
  `app.user_id` GUC, so callers that go through the pooled connection (which
  sets the GUC on every acquire, e.g. `weft/board_server.py`'s `/act`
  handler) don't need to pass it explicitly.
* No FK from `item_id` / `target_id` to a single source table — board items
  span five heterogeneous sources (trackers/alerts/triggers/task-memories/
  review rows), so these are opaque cross-source identifiers, not a
  relational reference. Mirrors how `Item.id` (weft/board.py) is already a
  cross-source string, not a foreign key.
* `verb` / `source` / `kind` / `mode` are plain `TEXT`, not a Postgres enum
  type or `CHECK` constraint — same convention as `alerts.alert_type`
  (`weft/db/migrations/v21_alerts_table.py:11`, verified plain
  `TEXT NOT NULL` with no enum/CHECK anywhere in the migration history).
  Validity is enforced at the Python layer (`ItemSource` / `Urgency`
  Literals in weft/board.py, the rule registry's verb vocabulary in the
  future feedback engine), consistent with how `AlertType` is validated.
* `snooze_duration_days` is nullable — only populated when `verb == "snooze"`
  (the loop blueprint's `snooze_duration|null` field); other verbs leave it
  NULL.
* `proposed_change` is `JSONB` — the rule registry's `proposed_action` is a
  data-described change (e.g. `{"field": "nudge_interval", "new_value": ...}`
  or `{"add_to": "hidden_kinds", "kind": ...}`), not a fixed set of columns,
  so new rules stay additive data per the PRD's registry design.

## RLS

Mirrors the `recall_canary_audit` policy set (v66): SELECT admits the
system sentinel (`__system_global_zathras__`) for service-side callers;
INSERT / UPDATE / DELETE gate strictly on
`user_id = nullif(current_setting('app.user_id', true), '')`.

Spec: weft-board-epic Task 6 (documents/epics/weft-board-epic.md).
Depends on: none (new, independent tables).
"""

from __future__ import annotations

# Retention horizon (days) for board_triage_events rows. Named config
# constant per weft-board-epic Task 6 + PRD Research Item R8 — see the
# module docstring "Retention" section. Consumed by the future L1 feedback
# engine (Task 7), which prunes rows older than this on each pass, mirroring
# `_CANARY_AUDIT_RETENTION_DAYS` in weft/canary.py.
BOARD_TRIAGE_RETENTION_DAYS = 90

VERSION = 67
DESCRIPTION = "board_triage_events + board_feedback_proposals: L1 triage correction loop substrate"
SQL = r"""
CREATE TABLE IF NOT EXISTS board_triage_events (
    id                    BIGSERIAL PRIMARY KEY,
    user_id               TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    item_id               TEXT NOT NULL,
    source                TEXT NOT NULL,
    kind                  TEXT NOT NULL,
    urgency_at_surface    TEXT NOT NULL,
    age_days_at_surface   DOUBLE PRECISION NOT NULL,
    verb                  TEXT NOT NULL,
    snooze_duration_days  DOUBLE PRECISION,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Repeat-action lookups for a single item (e.g. "has this tracker been
-- snoozed >=3 times?" — the ROBUSTNESS GAP rule).
CREATE INDEX IF NOT EXISTS idx_board_triage_events_item
    ON board_triage_events (item_id);

-- Cross-item lookups by source+kind (e.g. "has this kind been dismissed
-- across >=3 distinct items?" — the FEATURE SIGNAL rule).
CREATE INDEX IF NOT EXISTS idx_board_triage_events_source_kind
    ON board_triage_events (source, kind);

-- Windowed retention + per-user history scans (created_at DESC scan),
-- mirroring idx_recall_canary_audit_window (v66).
CREATE INDEX IF NOT EXISTS idx_board_triage_events_window
    ON board_triage_events (user_id, created_at);

ALTER TABLE board_triage_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS board_triage_events_select ON board_triage_events;
CREATE POLICY board_triage_events_select ON board_triage_events FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_triage_events_insert ON board_triage_events;
CREATE POLICY board_triage_events_insert ON board_triage_events FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_triage_events_update ON board_triage_events;
CREATE POLICY board_triage_events_update ON board_triage_events FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_triage_events_delete ON board_triage_events;
CREATE POLICY board_triage_events_delete ON board_triage_events FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));


CREATE TABLE IF NOT EXISTS board_feedback_proposals (
    id                BIGSERIAL PRIMARY KEY,
    user_id           TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    rule              TEXT NOT NULL,
    target_id         TEXT NOT NULL,
    proposed_change   JSONB NOT NULL DEFAULT '{}',
    mode              TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotency / "has this rule already proposed for this target?" lookups.
CREATE INDEX IF NOT EXISTS idx_board_feedback_proposals_rule_target
    ON board_feedback_proposals (rule, target_id);

-- Per-user history scans + "reviewed batch" activation-gate queries
-- (created_at DESC scan).
CREATE INDEX IF NOT EXISTS idx_board_feedback_proposals_window
    ON board_feedback_proposals (user_id, created_at);

ALTER TABLE board_feedback_proposals ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS board_feedback_proposals_select ON board_feedback_proposals;
CREATE POLICY board_feedback_proposals_select ON board_feedback_proposals FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_feedback_proposals_insert ON board_feedback_proposals;
CREATE POLICY board_feedback_proposals_insert ON board_feedback_proposals FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_feedback_proposals_update ON board_feedback_proposals;
CREATE POLICY board_feedback_proposals_update ON board_feedback_proposals FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS board_feedback_proposals_delete ON board_feedback_proposals;
CREATE POLICY board_feedback_proposals_delete ON board_feedback_proposals FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
