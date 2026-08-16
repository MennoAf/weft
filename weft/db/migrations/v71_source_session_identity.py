"""Add provider-namespaced source identity to episode turns."""

from __future__ import annotations

VERSION = 71
DESCRIPTION = "provider-namespaced source identity on episode turns"
SQL = r"""
ALTER TABLE episode_turns
    ADD COLUMN IF NOT EXISTS source_session_id TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_episode_turns_source_session
    ON episode_turns (source_session_id)
    WHERE source_session_id IS NOT NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
