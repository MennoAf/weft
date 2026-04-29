"""Migration 30: Create degradation_policies table"""

from __future__ import annotations

VERSION = 30
DESCRIPTION = 'Create degradation_policies table'
SQL = r"""
CREATE TABLE IF NOT EXISTS degradation_policies (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    trigger_type      TEXT NOT NULL,
    condition         JSONB NOT NULL DEFAULT '{}'::jsonb,
    action            TEXT NOT NULL,
    description       TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    cooldown_minutes  DOUBLE PRECISION,
    max_fires         INTEGER,
    fire_count        INTEGER NOT NULL DEFAULT 0,
    last_fired_at     TIMESTAMPTZ,
    project_id        TEXT,
    agent_id          TEXT,
    user_id           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_degradation_policies_trigger_type
    ON degradation_policies (trigger_type);
CREATE INDEX IF NOT EXISTS idx_degradation_policies_action
    ON degradation_policies (action);
CREATE INDEX IF NOT EXISTS idx_degradation_policies_status
    ON degradation_policies (status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_degradation_policies_user
    ON degradation_policies (user_id);
CREATE INDEX IF NOT EXISTS idx_degradation_policies_created
    ON degradation_policies (created_at DESC);

ALTER TABLE degradation_policies ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS degradation_policies_select ON degradation_policies;
CREATE POLICY degradation_policies_select ON degradation_policies FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS degradation_policies_insert ON degradation_policies;
CREATE POLICY degradation_policies_insert ON degradation_policies FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS degradation_policies_update ON degradation_policies;
CREATE POLICY degradation_policies_update ON degradation_policies FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS degradation_policies_delete ON degradation_policies;
CREATE POLICY degradation_policies_delete ON degradation_policies FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
