"""Migration 17: Add last_boosted_at to memories for usefulness time decay"""

from __future__ import annotations

VERSION = 17
DESCRIPTION = 'Add last_boosted_at to memories for usefulness time decay'
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS
    last_boosted_at TIMESTAMPTZ DEFAULT NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
