"""Migration 10: Create behaviors table for persistent agent rules and strategies"""

from __future__ import annotations

VERSION = 10
DESCRIPTION = 'Create behaviors table for persistent agent rules and strategies'
SQL = r"""
CREATE TABLE IF NOT EXISTS behaviors (
    id              TEXT PRIMARY KEY,
    trigger_pattern TEXT NOT NULL,
    action          TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.7,
    scope           TEXT NOT NULL DEFAULT 'global',
    project_id      TEXT,
    agent_id        TEXT,
    user_id         TEXT,
    priority        INTEGER NOT NULL DEFAULT 0,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    access_count    INTEGER NOT NULL DEFAULT 0,
    token_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(384),
    status          TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_behaviors_scope ON behaviors (scope);
CREATE INDEX IF NOT EXISTS idx_behaviors_project ON behaviors (project_id);
CREATE INDEX IF NOT EXISTS idx_behaviors_agent ON behaviors (agent_id);
CREATE INDEX IF NOT EXISTS idx_behaviors_enabled ON behaviors (enabled) WHERE enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_behaviors_status ON behaviors (status);
CREATE INDEX IF NOT EXISTS idx_behaviors_embedding_hnsw
ON behaviors USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
