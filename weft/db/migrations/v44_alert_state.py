"""Migration 44: alert_state table — per-(alert_type, dedup_key) cooldown + suppression.

Replaces the V1 dedup pattern (scan last 200 alerts, dedup by alert_type
alone) with proper per-key state. The dedup_key is producer-chosen and
type-specific:

  - loom_stale_claim          -> "task:<loom_task_id>"
  - loom_blocked_pile_up      -> "project:<loom_project_id>"
  - loom_epic_ready           -> "epic:<loom_task_id>"
  - stale_decision            -> "memory:<memory_id>"
  - memory_consolidation_overdue -> "global"
  - memory_count_threshold    -> "global"
  - memory_contradiction      -> "memory:<memory_id>"

Per-(alert_type, dedup_key, user_id) row holds:
  - last_fired_at, last_alert_id, fire_count: cooldown signal
  - suppressed_until + suppression_reason: manual / structured mute

Federation note: dedup state is per-user via RLS. Two users can
independently fire the same alert_type/dedup_key without interference;
suppress() and clear_suppression() only affect the current user.
Same post-mig-36 contract as mig 43 (NOT NULL user_id, GUC default,
sentinel-or-self policies).
"""

from __future__ import annotations

VERSION = 44
DESCRIPTION = (
    "alert_state table for per-(alert_type, dedup_key) cooldown + suppression"
)
SQL = r"""
CREATE TABLE IF NOT EXISTS alert_state (
    id                  TEXT PRIMARY KEY,
    alert_type          TEXT NOT NULL,
    dedup_key           TEXT NOT NULL,
    last_fired_at       TIMESTAMPTZ,
    last_alert_id       TEXT,
    fire_count          INTEGER NOT NULL DEFAULT 0,
    suppressed_until    TIMESTAMPTZ,
    suppression_reason  TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    user_id             TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Identity is per-user (RLS scope) so the unique key includes user_id.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_alert_state_type_key_user
    ON alert_state (alert_type, dedup_key, user_id);

CREATE INDEX IF NOT EXISTS idx_alert_state_last_fired
    ON alert_state (last_fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_state_suppressed
    ON alert_state (suppressed_until);
CREATE INDEX IF NOT EXISTS idx_alert_state_user
    ON alert_state (user_id);

ALTER TABLE alert_state ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS alert_state_select ON alert_state;
CREATE POLICY alert_state_select ON alert_state FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alert_state_insert ON alert_state;
CREATE POLICY alert_state_insert ON alert_state FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alert_state_update ON alert_state;
CREATE POLICY alert_state_update ON alert_state FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alert_state_delete ON alert_state;
CREATE POLICY alert_state_delete ON alert_state FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
