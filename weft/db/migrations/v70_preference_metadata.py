"""Add validated semantic annotations for preference memories."""

from __future__ import annotations

VERSION = 70
DESCRIPTION = "nullable preference metadata JSONB on memories"
SQL = r"""
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS preference_metadata JSONB NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
