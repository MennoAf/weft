"""Migration 2: Create memory_relationships table"""

from __future__ import annotations

VERSION = 2
DESCRIPTION = 'Create memory_relationships table'
SQL = r"""
CREATE TABLE IF NOT EXISTS memory_relationships (
    source_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_memrel_source ON memory_relationships (source_id);
CREATE INDEX IF NOT EXISTS idx_memrel_target ON memory_relationships (target_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
