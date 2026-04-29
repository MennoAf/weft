"""Migration 21: Create alerts table for proactive push notifications"""

from __future__ import annotations

VERSION = 21
DESCRIPTION = 'Create alerts table for proactive push notifications'
SQL = r"""
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT,
    alert_type      TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT,
    trigger_at      TIMESTAMPTZ NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    channel         TEXT NOT NULL DEFAULT 'log',
    channel_target  TEXT,
    payload         JSONB NOT NULL DEFAULT '{}',
    project_id      TEXT,
    agent_id        TEXT,
    fired_at        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Partial index for scheduler polling: only pending alerts matter
CREATE INDEX IF NOT EXISTS idx_alerts_poll
ON alerts (user_id, status, trigger_at) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts (user_id);

-- RLS: same pattern as all other user-scoped tables
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS alerts_select ON alerts;
CREATE POLICY alerts_select ON alerts FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alerts_insert ON alerts;
CREATE POLICY alerts_insert ON alerts FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alerts_update ON alerts;
CREATE POLICY alerts_update ON alerts FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS alerts_delete ON alerts;
CREATE POLICY alerts_delete ON alerts FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
