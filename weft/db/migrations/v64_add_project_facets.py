"""Migration 64: Add project_facets column and GIN index to memories table.

Facet-based recall requires a denormalized project_facets column (TEXT array)
to improve recall ranking by project scope. This is Phase 0 of the facet system:
schema + model only, non-destructive. The project_id column remains untouched
as the 'origin' anchor for soft-boost ranking.

Backfill: for existing rows, project_facets is set to:
  - '{}' (empty array) if project_id IS NULL
  - ARRAY[lower(project_id)] if project_id is NOT NULL

This ensures all rows satisfy the NOT NULL constraint and are ready for
facet-based recall queries (L2/L3 work).

Spec: loom-ca3b3893 (L1 — Schema: add project_facets)
Parent epic: loom-1d6c8e5c (EPIC: Facet-based recall)
"""

from __future__ import annotations

VERSION = 64
DESCRIPTION = "Add project_facets TEXT[] column + GIN index + backfill to memories"

SQL = r"""
-- Add column with NOT NULL constraint, default to empty array
ALTER TABLE memories
ADD COLUMN project_facets TEXT[] NOT NULL DEFAULT '{}';

-- Backfill existing rows: project_facets = ARRAY[lower(project_id)] or '{}'
UPDATE memories
SET project_facets = CASE
    WHEN project_id IS NULL THEN '{}'::text[]
    ELSE ARRAY[lower(project_id)]
END;

-- Create GIN index for efficient facet-based queries
CREATE INDEX IF NOT EXISTS idx_memories_project_facets
ON memories USING gin (project_facets);
"""

# For future migration frameworks that support explicit up/down.
def up() -> str:
    """Forward migration: add project_facets column, backfill, create index."""
    return SQL


def down() -> str:
    """Rollback: drop index and column."""
    return """
    DROP INDEX IF EXISTS idx_memories_project_facets;
    ALTER TABLE memories DROP COLUMN IF EXISTS project_facets;
    """


MIGRATION = (VERSION, DESCRIPTION, SQL)
