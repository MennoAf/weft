"""Migration 1: Create memories table with pgvector"""

from __future__ import annotations

VERSION = 1
DESCRIPTION = 'Create memories table with pgvector'
SQL = r"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    topic           TEXT[] NOT NULL DEFAULT '{}',
    content         TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'conversation',
    confidence      REAL NOT NULL DEFAULT 0.7,
    token_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count    INTEGER NOT NULL DEFAULT 0,
    project_id      TEXT,
    agent_id        TEXT,
    embedding       vector,
    status          TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (type);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories (project_id);
CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories USING gin (topic);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories (accessed_at DESC);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
