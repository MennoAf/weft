"""Migration 13: Add user_id column to all user-scoped tables (RLS prerequisite)"""

from __future__ import annotations

VERSION = 13
DESCRIPTION = 'Add user_id column to all user-scoped tables (RLS prerequisite)'
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE memory_relationships ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE episode_memories ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memory_relationships_user ON memory_relationships (user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes (user_id);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities (user_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
