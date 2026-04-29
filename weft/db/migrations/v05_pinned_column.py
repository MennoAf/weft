"""Migration 5: Add pinned column to memories"""

from __future__ import annotations

VERSION = 5
DESCRIPTION = 'Add pinned column to memories'
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories (pinned) WHERE pinned = TRUE;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
