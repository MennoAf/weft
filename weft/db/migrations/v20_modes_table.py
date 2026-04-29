"""Migration 20: Create modes table for named retrieval personas with weight overrides"""

from __future__ import annotations

VERSION = 20
DESCRIPTION = 'Create modes table for named retrieval personas with weight overrides'
SQL = r"""
CREATE TABLE IF NOT EXISTS modes (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    name        TEXT NOT NULL,
    description TEXT,
    weights     JSONB NOT NULL DEFAULT '{}',
    project_id  TEXT,
    agent_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_modes_user_id_name UNIQUE (user_id, name)
);

-- Partial unique index for NULL user_id (PostgreSQL treats NULLs as
-- distinct in regular UNIQUE constraints, so global modes need this)
CREATE UNIQUE INDEX IF NOT EXISTS uq_modes_null_user_name
ON modes (name) WHERE user_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_modes_user ON modes (user_id);

-- RLS: same pattern as all other user-scoped tables
ALTER TABLE modes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS modes_select ON modes;
CREATE POLICY modes_select ON modes FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS modes_insert ON modes;
CREATE POLICY modes_insert ON modes FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS modes_update ON modes;
CREATE POLICY modes_update ON modes FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS modes_delete ON modes;
CREATE POLICY modes_delete ON modes FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
