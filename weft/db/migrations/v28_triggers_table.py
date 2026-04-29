"""Migration 28: Create triggers table for proactive condition-driven rules"""

from __future__ import annotations

VERSION = 28
DESCRIPTION = 'Create triggers table for proactive condition-driven rules'
SQL = r"""
CREATE TABLE IF NOT EXISTS triggers (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    condition_type  TEXT NOT NULL,
    condition       JSONB NOT NULL DEFAULT '{}'::jsonb,
    action          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'enabled',
    cooldown_hours  REAL,
    max_fires       INTEGER,
    fire_count      INTEGER NOT NULL DEFAULT 0,
    last_fired_at   TIMESTAMPTZ,
    project_id      TEXT,
    agent_id        TEXT,
    user_id         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_triggers_condition_type
    ON triggers (condition_type);
CREATE INDEX IF NOT EXISTS idx_triggers_status
    ON triggers (status) WHERE status = 'enabled';
CREATE INDEX IF NOT EXISTS idx_triggers_project
    ON triggers (project_id);
CREATE INDEX IF NOT EXISTS idx_triggers_user
    ON triggers (user_id);

ALTER TABLE triggers ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS triggers_select ON triggers;
CREATE POLICY triggers_select ON triggers FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS triggers_insert ON triggers;
CREATE POLICY triggers_insert ON triggers FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS triggers_update ON triggers;
CREATE POLICY triggers_update ON triggers FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS triggers_delete ON triggers;
CREATE POLICY triggers_delete ON triggers FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
