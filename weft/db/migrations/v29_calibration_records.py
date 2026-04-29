"""Migration 29: Create calibration_records table for agent action outcome tracking"""

from __future__ import annotations

VERSION = 29
DESCRIPTION = 'Create calibration_records table for agent action outcome tracking'
SQL = r"""
CREATE TABLE IF NOT EXISTS calibration_records (
    id                  TEXT PRIMARY KEY,
    action_category     TEXT NOT NULL,
    action_description  TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    agent_id            TEXT,
    project_id          TEXT,
    context             JSONB NOT NULL DEFAULT '{}'::jsonb,
    user_id             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calibration_records_category
    ON calibration_records (action_category);
CREATE INDEX IF NOT EXISTS idx_calibration_records_outcome
    ON calibration_records (outcome);
CREATE INDEX IF NOT EXISTS idx_calibration_records_user
    ON calibration_records (user_id);
CREATE INDEX IF NOT EXISTS idx_calibration_records_project
    ON calibration_records (project_id);
CREATE INDEX IF NOT EXISTS idx_calibration_records_created
    ON calibration_records (created_at DESC);

ALTER TABLE calibration_records ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS calibration_records_select ON calibration_records;
CREATE POLICY calibration_records_select ON calibration_records FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_records_insert ON calibration_records;
CREATE POLICY calibration_records_insert ON calibration_records FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_records_update ON calibration_records;
CREATE POLICY calibration_records_update ON calibration_records FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS calibration_records_delete ON calibration_records;
CREATE POLICY calibration_records_delete ON calibration_records FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
