"""Migration 22: Create check_ins table for mood/sleep/energy tracking"""

from __future__ import annotations

VERSION = 22
DESCRIPTION = 'Create check_ins table for mood/sleep/energy tracking'
SQL = r"""
CREATE TABLE IF NOT EXISTS check_ins (
    id              TEXT PRIMARY KEY,
    user_id         TEXT,
    mood            SMALLINT,
    sleep_hours     REAL,
    energy          SMALLINT,
    notes           TEXT,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_check_ins_user_logged
    ON check_ins (user_id, logged_at DESC);

ALTER TABLE check_ins ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS check_ins_select ON check_ins;
CREATE POLICY check_ins_select ON check_ins FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS check_ins_insert ON check_ins;
CREATE POLICY check_ins_insert ON check_ins FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS check_ins_update ON check_ins;
CREATE POLICY check_ins_update ON check_ins FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS check_ins_delete ON check_ins;
CREATE POLICY check_ins_delete ON check_ins FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
