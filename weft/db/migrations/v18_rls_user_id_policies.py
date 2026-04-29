"""Migration 18: Enable RLS and create user_id isolation policies on all user-scoped tables"""

from __future__ import annotations

VERSION = 18
DESCRIPTION = 'Enable RLS and create user_id isolation policies on all user-scoped tables'
SQL = r"""
-- Enable Row Level Security on all user-scoped tables
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE behaviors ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE episode_memories ENABLE ROW LEVEL SECURITY;

-- memories policies
DROP POLICY IF EXISTS memories_select ON memories;
CREATE POLICY memories_select ON memories FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memories_insert ON memories;
CREATE POLICY memories_insert ON memories FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memories_update ON memories;
CREATE POLICY memories_update ON memories FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memories_delete ON memories;
CREATE POLICY memories_delete ON memories FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- memory_relationships policies
DROP POLICY IF EXISTS memory_relationships_select ON memory_relationships;
CREATE POLICY memory_relationships_select ON memory_relationships FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memory_relationships_insert ON memory_relationships;
CREATE POLICY memory_relationships_insert ON memory_relationships FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memory_relationships_update ON memory_relationships;
CREATE POLICY memory_relationships_update ON memory_relationships FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS memory_relationships_delete ON memory_relationships;
CREATE POLICY memory_relationships_delete ON memory_relationships FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- behaviors policies
DROP POLICY IF EXISTS behaviors_select ON behaviors;
CREATE POLICY behaviors_select ON behaviors FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS behaviors_insert ON behaviors;
CREATE POLICY behaviors_insert ON behaviors FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS behaviors_update ON behaviors;
CREATE POLICY behaviors_update ON behaviors FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS behaviors_delete ON behaviors;
CREATE POLICY behaviors_delete ON behaviors FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- entities policies
DROP POLICY IF EXISTS entities_select ON entities;
CREATE POLICY entities_select ON entities FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entities_insert ON entities;
CREATE POLICY entities_insert ON entities FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entities_update ON entities;
CREATE POLICY entities_update ON entities FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entities_delete ON entities;
CREATE POLICY entities_delete ON entities FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- entity_mentions policies
DROP POLICY IF EXISTS entity_mentions_select ON entity_mentions;
CREATE POLICY entity_mentions_select ON entity_mentions FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entity_mentions_insert ON entity_mentions;
CREATE POLICY entity_mentions_insert ON entity_mentions FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entity_mentions_update ON entity_mentions;
CREATE POLICY entity_mentions_update ON entity_mentions FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS entity_mentions_delete ON entity_mentions;
CREATE POLICY entity_mentions_delete ON entity_mentions FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- episodes policies
DROP POLICY IF EXISTS episodes_select ON episodes;
CREATE POLICY episodes_select ON episodes FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episodes_insert ON episodes;
CREATE POLICY episodes_insert ON episodes FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episodes_update ON episodes;
CREATE POLICY episodes_update ON episodes FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episodes_delete ON episodes;
CREATE POLICY episodes_delete ON episodes FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

-- episode_memories policies
DROP POLICY IF EXISTS episode_memories_select ON episode_memories;
CREATE POLICY episode_memories_select ON episode_memories FOR SELECT
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_memories_insert ON episode_memories;
CREATE POLICY episode_memories_insert ON episode_memories FOR INSERT
    WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_memories_update ON episode_memories;
CREATE POLICY episode_memories_update ON episode_memories FOR UPDATE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_memories_delete ON episode_memories;
CREATE POLICY episode_memories_delete ON episode_memories FOR DELETE
    USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
