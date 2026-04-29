"""Migration 26: Add expires_at column to episodes for working memory TTL"""

from __future__ import annotations

VERSION = 26
DESCRIPTION = 'Add expires_at column to episodes for working memory TTL'
SQL = r"""
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_episodes_expires
    ON episodes (expires_at)
    WHERE expires_at IS NOT NULL AND status = 'open';
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
