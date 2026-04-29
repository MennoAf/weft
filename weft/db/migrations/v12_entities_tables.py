"""Migration 12: Create entities and entity_mentions tables for entity graph"""

from __future__ import annotations

VERSION = 12
DESCRIPTION = 'Create entities and entity_mentions tables for entity graph'
SQL = r"""
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    description TEXT,
    project_id  TEXT,
    agent_id    TEXT,
    status      TEXT NOT NULL DEFAULT 'active',
    mention_count INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(384)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_project ON entities (project_id);
CREATE INDEX IF NOT EXISTS idx_entities_status ON entities (status);
CREATE INDEX IF NOT EXISTS idx_entities_embedding_hnsw
ON entities USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    mentioned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_entmem_memory ON entity_mentions (memory_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
