"""Migration 27: Add graduated_memory_id column to episodes for graduation path"""

from __future__ import annotations

VERSION = 27
DESCRIPTION = 'Add graduated_memory_id column to episodes for graduation path'
SQL = r"""
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS graduated_memory_id TEXT
    REFERENCES memories(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_episodes_graduated
    ON episodes (graduated_memory_id)
    WHERE graduated_memory_id IS NOT NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
