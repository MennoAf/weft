"""Migration 3: Create schema_migrations tracking table"""

from __future__ import annotations

VERSION = 3
DESCRIPTION = 'Create schema_migrations tracking table'
SQL = r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
