"""Migration 19: Add tsvector column for full-text search (hybrid BM25 + vector)"""

from __future__ import annotations

VERSION = 19
DESCRIPTION = 'Add tsvector column for full-text search (hybrid BM25 + vector)'
SQL = r"""
-- Add tsvector column for keyword/BM25 search
ALTER TABLE memories ADD COLUMN IF NOT EXISTS
    search_tsv tsvector;

-- Backfill existing memories
UPDATE memories
SET search_tsv = to_tsvector('english',
    coalesce(content, '') || ' ' || coalesce(array_to_string(topic, ' '), '')
)
WHERE search_tsv IS NULL;

-- GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_memories_search_tsv
ON memories USING gin (search_tsv);

-- Trigger to auto-update search_tsv on INSERT or UPDATE
CREATE OR REPLACE FUNCTION memories_search_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_tsv := to_tsvector('english',
        coalesce(NEW.content, '') || ' ' || coalesce(array_to_string(NEW.topic, ' '), '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memories_search_tsv ON memories;
CREATE TRIGGER trg_memories_search_tsv
BEFORE INSERT OR UPDATE OF content, topic ON memories
FOR EACH ROW EXECUTE FUNCTION memories_search_tsv_trigger();
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
