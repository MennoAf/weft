"""Migration 11: Create episodes and episode_memories tables for episodic timeline"""

from __future__ import annotations

VERSION = 11
DESCRIPTION = 'Create episodes and episode_memories tables for episodic timeline'
SQL = r"""
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    summary     TEXT,
    project_id  TEXT,
    agent_id    TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'open',
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes (project_id);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes (status);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes (started_at DESC);

CREATE TABLE IF NOT EXISTS episode_memories (
    episode_id  TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL DEFAULT 0,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (episode_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_epmem_memory ON episode_memories (memory_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
