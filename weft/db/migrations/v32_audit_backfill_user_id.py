"""Migration 32: Create audit_backfill_user_id table for Phase 1 backfill audit trail"""

from __future__ import annotations

VERSION = 32
DESCRIPTION = 'Create audit_backfill_user_id table for Phase 1 backfill audit trail'
SQL = r"""
CREATE TABLE IF NOT EXISTS audit_backfill_user_id (
    id           SERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,
    row_id       TEXT NOT NULL,
    old_scope    TEXT NOT NULL,
    new_scope    TEXT NOT NULL,
    migrated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_backfill_source
    ON audit_backfill_user_id (source_table, row_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
