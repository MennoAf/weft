"""Migration 15: Add weft_metadata table for system-level key-value storage"""

from __future__ import annotations

VERSION = 15
DESCRIPTION = 'Add weft_metadata table for system-level key-value storage'
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_metadata (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
