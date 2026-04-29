"""Migration 25: Create cost_entries table for token/cost tracking"""

from __future__ import annotations

VERSION = 25
DESCRIPTION = 'Create cost_entries table for token/cost tracking'
SQL = r"""
CREATE TABLE IF NOT EXISTS cost_entries (
    id                  TEXT PRIMARY KEY,
    entry_type          TEXT NOT NULL DEFAULT 'session',
    reference_id        TEXT,
    model               TEXT,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL NOT NULL DEFAULT 0,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    project_id          TEXT,
    agent_id            TEXT,
    user_id             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cost_entries_type
    ON cost_entries (entry_type);
CREATE INDEX IF NOT EXISTS idx_cost_entries_reference
    ON cost_entries (reference_id);
CREATE INDEX IF NOT EXISTS idx_cost_entries_user
    ON cost_entries (user_id);
CREATE INDEX IF NOT EXISTS idx_cost_entries_project
    ON cost_entries (project_id);
CREATE INDEX IF NOT EXISTS idx_cost_entries_created
    ON cost_entries (created_at DESC);

ALTER TABLE cost_entries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS cost_entries_select ON cost_entries;
CREATE POLICY cost_entries_select ON cost_entries FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_entries_insert ON cost_entries;
CREATE POLICY cost_entries_insert ON cost_entries FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_entries_update ON cost_entries;
CREATE POLICY cost_entries_update ON cost_entries FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_entries_delete ON cost_entries;
CREATE POLICY cost_entries_delete ON cost_entries FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
