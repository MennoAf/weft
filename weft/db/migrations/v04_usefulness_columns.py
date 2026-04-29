"""Migration 4: Add usefulness_score and usefulness_count to memories"""

from __future__ import annotations

VERSION = 4
DESCRIPTION = 'Add usefulness_score and usefulness_count to memories'
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS usefulness_score FLOAT DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS usefulness_count INTEGER DEFAULT 0;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
