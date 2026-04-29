"""Migration 24: Create autonomy_policies and policy_calibration_events tables"""

from __future__ import annotations

VERSION = 24
DESCRIPTION = 'Create autonomy_policies and policy_calibration_events tables'
SQL = r"""
CREATE TABLE IF NOT EXISTS autonomy_policies (
    id              TEXT PRIMARY KEY,
    action          TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'never',
    description     TEXT,
    conditions      JSONB NOT NULL DEFAULT '{}'::jsonb,
    project_id      TEXT,
    agent_id        TEXT,
    user_id         TEXT,
    enabled         BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_autonomy_policies_action
    ON autonomy_policies (action);
CREATE INDEX IF NOT EXISTS idx_autonomy_policies_tier
    ON autonomy_policies (tier);
CREATE INDEX IF NOT EXISTS idx_autonomy_policies_user
    ON autonomy_policies (user_id);
CREATE INDEX IF NOT EXISTS idx_autonomy_policies_project
    ON autonomy_policies (project_id);
CREATE INDEX IF NOT EXISTS idx_autonomy_policies_agent
    ON autonomy_policies (agent_id);

ALTER TABLE autonomy_policies ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS autonomy_policies_select ON autonomy_policies;
CREATE POLICY autonomy_policies_select ON autonomy_policies FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_policies_insert ON autonomy_policies;
CREATE POLICY autonomy_policies_insert ON autonomy_policies FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_policies_update ON autonomy_policies;
CREATE POLICY autonomy_policies_update ON autonomy_policies FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_policies_delete ON autonomy_policies;
CREATE POLICY autonomy_policies_delete ON autonomy_policies FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

CREATE TABLE IF NOT EXISTS policy_calibration_events (
    id              TEXT PRIMARY KEY,
    policy_id       TEXT NOT NULL REFERENCES autonomy_policies(id) ON DELETE CASCADE,
    previous_tier   TEXT NOT NULL,
    new_tier        TEXT NOT NULL,
    reason          TEXT,
    agent_id        TEXT,
    user_id         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calibration_events_policy
    ON policy_calibration_events (policy_id);
CREATE INDEX IF NOT EXISTS idx_calibration_events_user
    ON policy_calibration_events (user_id);
CREATE INDEX IF NOT EXISTS idx_calibration_events_created
    ON policy_calibration_events (created_at DESC);

ALTER TABLE policy_calibration_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS calibration_events_select ON policy_calibration_events;
CREATE POLICY calibration_events_select ON policy_calibration_events FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_events_insert ON policy_calibration_events;
CREATE POLICY calibration_events_insert ON policy_calibration_events FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_events_update ON policy_calibration_events;
CREATE POLICY calibration_events_update ON policy_calibration_events FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_events_delete ON policy_calibration_events;
CREATE POLICY calibration_events_delete ON policy_calibration_events FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
