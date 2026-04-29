"""Migration 6: Add review_after column to memories"""

from __future__ import annotations

VERSION = 6
DESCRIPTION = 'Add review_after column to memories'
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS review_after TIMESTAMPTZ;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
