"""Migration 38: Trackers v1: open-loop primitive (Wick Phase 3)"""

from __future__ import annotations

VERSION = 38
DESCRIPTION = 'Trackers v1: open-loop primitive (Wick Phase 3)'
SQL = r"""
-- Trackers are the lifecycle-aware primitive that memories aren't.
-- Where memories are blob-shaped facts ("I pitched Tory Burch"),
-- trackers carry state that changes over time, can be nudged, can
-- be snoozed, and surface in the daily brief as "open loops."
--
-- Kinds (per weft_v2_spec.md §3, §4):
--   outreach       — contact made, awaiting reply
--   task           — work item not yet shaped as a Loom task
--   follow_up      — generic follow-up
--   meal_plan      — weekly menu (context.items shape)
--   shopping_list  — grocery list (context.items shape)
--   pantry         — what's in the pantry (context.items shape)
--   watch          — passive monitor (e.g. "watch the AAPL earnings call")
--   list           — generic list (sugar tools weft_list_*)
--   trace          — Orchestrator long-running trace promotion
--
-- States: open ∈ {in_progress, awaiting_reply, blocked};
--         terminal ∈ {done, abandoned}.
-- state_history is an append-only JSONB array of
--   {"from": "...", "to": "...", "at": "...", "note": "..."}.
--
-- Nudge modes (V1): none (query-only), once (fire at nudge_after,
-- then silent), recur (fire every nudge_interval until closed).
-- snooze_until temporarily suppresses due-results without changing
-- mode. dismiss = bump last_touch (silence until next interval);
-- close = terminal state.
--
-- provenance: 'supervisor' | 'agent'. Schema-only for V1; Phase 2
-- will wire app-layer enforcement (Layer 1 of poisoning defense).

CREATE TABLE IF NOT EXISTS trackers (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT
                    nullif(current_setting('app.user_id', true), ''),
    project_id      TEXT,
    entity_id       TEXT REFERENCES entities(id) ON DELETE SET NULL,
    kind            TEXT NOT NULL,
    title           TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'in_progress',
    state_history   JSONB NOT NULL DEFAULT '[]'::jsonb,
    context         JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_touch      TIMESTAMPTZ NOT NULL DEFAULT now(),
    nudge_mode      TEXT NOT NULL DEFAULT 'none',
    nudge_after     TIMESTAMPTZ,
    nudge_interval  INTERVAL,
    snooze_until    TIMESTAMPTZ,
    trigger_ids     TEXT[] NOT NULL DEFAULT '{}',
    provenance      TEXT NOT NULL DEFAULT 'supervisor',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT trackers_kind_check CHECK (kind IN (
        'outreach', 'task', 'follow_up', 'meal_plan',
        'shopping_list', 'pantry', 'watch', 'list', 'trace'
    )),
    CONSTRAINT trackers_state_check CHECK (state IN (
        'in_progress', 'awaiting_reply', 'blocked', 'done', 'abandoned'
    )),
    CONSTRAINT trackers_nudge_mode_check CHECK (nudge_mode IN (
        'none', 'once', 'recur'
    )),
    CONSTRAINT trackers_provenance_check CHECK (provenance IN (
        'supervisor', 'agent'
    ))
);

-- Hot-path index for ``weft_tracker_due``: open-state rows with a
-- due nudge time, optionally snoozed. Filtered partial index keeps
-- it tiny — terminal states never appear in the due query.
CREATE INDEX IF NOT EXISTS idx_trackers_due
    ON trackers (user_id, nudge_after)
    WHERE state IN ('in_progress', 'awaiting_reply', 'blocked')
      AND nudge_mode <> 'none';

CREATE INDEX IF NOT EXISTS idx_trackers_user_kind
    ON trackers (user_id, kind);
CREATE INDEX IF NOT EXISTS idx_trackers_user_state
    ON trackers (user_id, state);
CREATE INDEX IF NOT EXISTS idx_trackers_project
    ON trackers (project_id) WHERE project_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trackers_entity
    ON trackers (entity_id) WHERE entity_id IS NOT NULL;

-- RLS: sentinel-or-self, same pattern as the 16 other user-scoped
-- tables (see migration 36's DO block). No workspace branch yet —
-- v1 trackers are user-private. If shared trackers become a need,
-- mirror memories' workspace_id branch.
ALTER TABLE trackers ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS trackers_select ON trackers;
CREATE POLICY trackers_select ON trackers FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));
DROP POLICY IF EXISTS trackers_insert ON trackers;
CREATE POLICY trackers_insert ON trackers FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));
DROP POLICY IF EXISTS trackers_update ON trackers;
CREATE POLICY trackers_update ON trackers FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
DROP POLICY IF EXISTS trackers_delete ON trackers;
CREATE POLICY trackers_delete ON trackers FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
