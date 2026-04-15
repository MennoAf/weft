"""Database migrations — sequential, append-only.

Each migration is a (version, description, sql) tuple.
Never modify existing migrations; always add new numbered ones.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Create memories table with pgvector",
        """
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
        """,
    ),
    (
        2,
        "Create memory_relationships table",
        """
        CREATE TABLE IF NOT EXISTS memory_relationships (
            source_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            target_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            relation    TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (source_id, target_id, relation)
        );

        CREATE INDEX IF NOT EXISTS idx_memrel_source ON memory_relationships (source_id);
        CREATE INDEX IF NOT EXISTS idx_memrel_target ON memory_relationships (target_id);
        """,
    ),
    (
        3,
        "Create schema_migrations tracking table",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
    (
        4,
        "Add usefulness_score and usefulness_count to memories",
        """
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS usefulness_score FLOAT DEFAULT 1.0;
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS usefulness_count INTEGER DEFAULT 0;
        """,
    ),
    (
        5,
        "Add pinned column to memories",
        """
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;
        CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories (pinned) WHERE pinned = TRUE;
        """,
    ),
    (
        6,
        "Add review_after column to memories",
        """
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS review_after TIMESTAMPTZ;
        """,
    ),
    (
        7,
        "Set embedding dimension and add HNSW index for vector search",
        """
        ALTER TABLE memories ALTER COLUMN embedding TYPE vector(384);

        CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
        ON memories USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """,
    ),
    (
        8,
        "Lower default usefulness_score to 0.7 and backfill existing rows",
        """
        ALTER TABLE memories ALTER COLUMN usefulness_score SET DEFAULT 0.7;

        UPDATE memories
        SET usefulness_score = 0.7
        WHERE usefulness_score >= 0.9999
          AND usefulness_score <= 1.0001
          AND (usefulness_count = 0 OR usefulness_count IS NULL);
        """,
    ),
    (
        9,
        "Add index on agent_id for scoped queries",
        """
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories (agent_id);
        """,
    ),
    (
        10,
        "Create behaviors table for persistent agent rules and strategies",
        """
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
        """,
    ),
    (
        11,
        "Create episodes and episode_memories tables for episodic timeline",
        """
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
        """,
    ),
    (
        12,
        "Create entities and entity_mentions tables for entity graph",
        """
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
        """,
    ),
    (
        13,
        "Add user_id column to all user-scoped tables (RLS prerequisite)",
        """
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
        """,
    ),
    (
        14,
        "Widen embedding columns from vector(384) to vector(768) for OpenAI provider",
        """
        -- Drop existing HNSW indexes (cannot ALTER type with index present)
        DROP INDEX IF EXISTS idx_memories_embedding_hnsw;
        DROP INDEX IF EXISTS idx_behaviors_embedding_hnsw;
        DROP INDEX IF EXISTS idx_entities_embedding_hnsw;

        -- Null out existing embeddings (384-dim vectors can't be cast to 768-dim;
        -- run `weft re-embed` after this migration to regenerate)
        UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL;
        UPDATE behaviors SET embedding = NULL WHERE embedding IS NOT NULL;
        UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL;

        -- Widen columns: 384 → 768 (Matryoshka-truncated OpenAI embeddings)
        ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);
        ALTER TABLE behaviors ALTER COLUMN embedding TYPE vector(768);
        ALTER TABLE entities ALTER COLUMN embedding TYPE vector(768);

        -- Recreate HNSW indexes at new dimension
        CREATE INDEX idx_memories_embedding_hnsw
        ON memories USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

        CREATE INDEX idx_behaviors_embedding_hnsw
        ON behaviors USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

        CREATE INDEX idx_entities_embedding_hnsw
        ON entities USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """,
    ),
    (
        15,
        "Add weft_metadata table for system-level key-value storage",
        """
        CREATE TABLE IF NOT EXISTS weft_metadata (
            key TEXT PRIMARY KEY,
            value JSONB NOT NULL DEFAULT '{}',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    ),
    (
        16,
        "Add memory_access_log for session-scoped access tracking",
        """
        CREATE TABLE IF NOT EXISTS memory_access_log (
            session_id  TEXT NOT NULL,
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            tool_name   TEXT NOT NULL,
            accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, memory_id)
        );

        CREATE INDEX IF NOT EXISTS idx_access_log_session
        ON memory_access_log (session_id);

        CREATE INDEX IF NOT EXISTS idx_access_log_accessed
        ON memory_access_log (accessed_at);
        """,
    ),
    (
        17,
        "Add last_boosted_at to memories for usefulness time decay",
        """
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS
            last_boosted_at TIMESTAMPTZ DEFAULT NULL;
        """,
    ),
    (
        18,
        "Enable RLS and create user_id isolation policies on all user-scoped tables",
        """
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
        """,
    ),
    (
        19,
        "Add tsvector column for full-text search (hybrid BM25 + vector)",
        """
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
        """,
    ),
    (
        20,
        "Create modes table for named retrieval personas with weight overrides",
        """
        CREATE TABLE IF NOT EXISTS modes (
            id          TEXT PRIMARY KEY,
            user_id     TEXT,
            name        TEXT NOT NULL,
            description TEXT,
            weights     JSONB NOT NULL DEFAULT '{}',
            project_id  TEXT,
            agent_id    TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_modes_user_id_name UNIQUE (user_id, name)
        );

        -- Partial unique index for NULL user_id (PostgreSQL treats NULLs as
        -- distinct in regular UNIQUE constraints, so global modes need this)
        CREATE UNIQUE INDEX IF NOT EXISTS uq_modes_null_user_name
        ON modes (name) WHERE user_id IS NULL;

        CREATE INDEX IF NOT EXISTS idx_modes_user ON modes (user_id);

        -- RLS: same pattern as all other user-scoped tables
        ALTER TABLE modes ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS modes_select ON modes;
        CREATE POLICY modes_select ON modes FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS modes_insert ON modes;
        CREATE POLICY modes_insert ON modes FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS modes_update ON modes;
        CREATE POLICY modes_update ON modes FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS modes_delete ON modes;
        CREATE POLICY modes_delete ON modes FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        21,
        "Create alerts table for proactive push notifications",
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id              TEXT PRIMARY KEY,
            user_id         TEXT,
            alert_type      TEXT NOT NULL,
            title           TEXT NOT NULL,
            body            TEXT,
            trigger_at      TIMESTAMPTZ NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            channel         TEXT NOT NULL DEFAULT 'log',
            channel_target  TEXT,
            payload         JSONB NOT NULL DEFAULT '{}',
            project_id      TEXT,
            agent_id        TEXT,
            fired_at        TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        -- Partial index for scheduler polling: only pending alerts matter
        CREATE INDEX IF NOT EXISTS idx_alerts_poll
        ON alerts (user_id, status, trigger_at) WHERE status = 'pending';

        CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts (user_id);

        -- RLS: same pattern as all other user-scoped tables
        ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS alerts_select ON alerts;
        CREATE POLICY alerts_select ON alerts FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS alerts_insert ON alerts;
        CREATE POLICY alerts_insert ON alerts FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS alerts_update ON alerts;
        CREATE POLICY alerts_update ON alerts FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS alerts_delete ON alerts;
        CREATE POLICY alerts_delete ON alerts FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    # 22: Check-ins table for mood/sleep/energy tracking
    (
        22,
        "Create check_ins table for mood/sleep/energy tracking",
        """
        CREATE TABLE IF NOT EXISTS check_ins (
            id              TEXT PRIMARY KEY,
            user_id         TEXT,
            mood            SMALLINT,
            sleep_hours     REAL,
            energy          SMALLINT,
            notes           TEXT,
            logged_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_check_ins_user_logged
            ON check_ins (user_id, logged_at DESC);

        ALTER TABLE check_ins ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS check_ins_select ON check_ins;
        CREATE POLICY check_ins_select ON check_ins FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS check_ins_insert ON check_ins;
        CREATE POLICY check_ins_insert ON check_ins FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS check_ins_update ON check_ins;
        CREATE POLICY check_ins_update ON check_ins FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS check_ins_delete ON check_ins;
        CREATE POLICY check_ins_delete ON check_ins FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        23,
        "Enable RLS on system tables (schema_migrations, weft_metadata, memory_access_log)",
        """
        -- schema_migrations: service-role only, no user data
        ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS schema_migrations_service ON schema_migrations;
        CREATE POLICY schema_migrations_service ON schema_migrations
            USING (true) WITH CHECK (true);

        -- weft_metadata: service-role only, no user data
        ALTER TABLE weft_metadata ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS weft_metadata_service ON weft_metadata;
        CREATE POLICY weft_metadata_service ON weft_metadata
            USING (true) WITH CHECK (true);

        -- memory_access_log: service-role only, session tracking
        ALTER TABLE memory_access_log ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS memory_access_log_service ON memory_access_log;
        CREATE POLICY memory_access_log_service ON memory_access_log
            USING (true) WITH CHECK (true);
        """,
    ),
    (
        24,
        "Create autonomy_policies and policy_calibration_events tables",
        """
        CREATE TABLE IF NOT EXISTS autonomy_policies (
            id              TEXT PRIMARY KEY,
            action          TEXT NOT NULL,
            tier            TEXT NOT NULL DEFAULT 'never',
            description     TEXT,
            conditions      JSONB NOT NULL DEFAULT '{}'::jsonb,
            project_id      TEXT,
            agent_id        TEXT,
            user_id         TEXT,
            enabled         BOOLEAN NOT NULL DEFAULT true,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_action
            ON autonomy_policies (action);
        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_tier
            ON autonomy_policies (tier);
        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_user
            ON autonomy_policies (user_id);
        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_project
            ON autonomy_policies (project_id);
        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_agent
            ON autonomy_policies (agent_id);

        ALTER TABLE autonomy_policies ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS autonomy_policies_select ON autonomy_policies;
        CREATE POLICY autonomy_policies_select ON autonomy_policies FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS autonomy_policies_insert ON autonomy_policies;
        CREATE POLICY autonomy_policies_insert ON autonomy_policies FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS autonomy_policies_update ON autonomy_policies;
        CREATE POLICY autonomy_policies_update ON autonomy_policies FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS autonomy_policies_delete ON autonomy_policies;
        CREATE POLICY autonomy_policies_delete ON autonomy_policies FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        CREATE TABLE IF NOT EXISTS policy_calibration_events (
            id              TEXT PRIMARY KEY,
            policy_id       TEXT NOT NULL REFERENCES autonomy_policies(id) ON DELETE CASCADE,
            previous_tier   TEXT NOT NULL,
            new_tier        TEXT NOT NULL,
            reason          TEXT,
            agent_id        TEXT,
            user_id         TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_calibration_events_policy
            ON policy_calibration_events (policy_id);
        CREATE INDEX IF NOT EXISTS idx_calibration_events_user
            ON policy_calibration_events (user_id);
        CREATE INDEX IF NOT EXISTS idx_calibration_events_created
            ON policy_calibration_events (created_at DESC);

        ALTER TABLE policy_calibration_events ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS calibration_events_select ON policy_calibration_events;
        CREATE POLICY calibration_events_select ON policy_calibration_events FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_events_insert ON policy_calibration_events;
        CREATE POLICY calibration_events_insert ON policy_calibration_events FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_events_update ON policy_calibration_events;
        CREATE POLICY calibration_events_update ON policy_calibration_events FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_events_delete ON policy_calibration_events;
        CREATE POLICY calibration_events_delete ON policy_calibration_events FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        25,
        "Create cost_entries table for token/cost tracking",
        """
        CREATE TABLE IF NOT EXISTS cost_entries (
            id                  TEXT PRIMARY KEY,
            entry_type          TEXT NOT NULL DEFAULT 'session',
            reference_id        TEXT,
            model               TEXT,
            input_tokens        INTEGER NOT NULL DEFAULT 0,
            output_tokens       INTEGER NOT NULL DEFAULT 0,
            total_tokens        INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd  REAL NOT NULL DEFAULT 0,
            metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
            project_id          TEXT,
            agent_id            TEXT,
            user_id             TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_cost_entries_type
            ON cost_entries (entry_type);
        CREATE INDEX IF NOT EXISTS idx_cost_entries_reference
            ON cost_entries (reference_id);
        CREATE INDEX IF NOT EXISTS idx_cost_entries_user
            ON cost_entries (user_id);
        CREATE INDEX IF NOT EXISTS idx_cost_entries_project
            ON cost_entries (project_id);
        CREATE INDEX IF NOT EXISTS idx_cost_entries_created
            ON cost_entries (created_at DESC);

        ALTER TABLE cost_entries ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS cost_entries_select ON cost_entries;
        CREATE POLICY cost_entries_select ON cost_entries FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS cost_entries_insert ON cost_entries;
        CREATE POLICY cost_entries_insert ON cost_entries FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS cost_entries_update ON cost_entries;
        CREATE POLICY cost_entries_update ON cost_entries FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS cost_entries_delete ON cost_entries;
        CREATE POLICY cost_entries_delete ON cost_entries FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        26,
        "Add expires_at column to episodes for working memory TTL",
        """
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

        CREATE INDEX IF NOT EXISTS idx_episodes_expires
            ON episodes (expires_at)
            WHERE expires_at IS NOT NULL AND status = 'open';
        """,
    ),
    (
        27,
        "Add graduated_memory_id column to episodes for graduation path",
        """
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS graduated_memory_id TEXT
            REFERENCES memories(id) ON DELETE SET NULL;

        CREATE INDEX IF NOT EXISTS idx_episodes_graduated
            ON episodes (graduated_memory_id)
            WHERE graduated_memory_id IS NOT NULL;
        """,
    ),
    (
        28,
        "Create triggers table for proactive condition-driven rules",
        """
        CREATE TABLE IF NOT EXISTS triggers (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            condition_type  TEXT NOT NULL,
            condition       JSONB NOT NULL DEFAULT '{}'::jsonb,
            action          TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'enabled',
            cooldown_hours  REAL,
            max_fires       INTEGER,
            fire_count      INTEGER NOT NULL DEFAULT 0,
            last_fired_at   TIMESTAMPTZ,
            project_id      TEXT,
            agent_id        TEXT,
            user_id         TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_triggers_condition_type
            ON triggers (condition_type);
        CREATE INDEX IF NOT EXISTS idx_triggers_status
            ON triggers (status) WHERE status = 'enabled';
        CREATE INDEX IF NOT EXISTS idx_triggers_project
            ON triggers (project_id);
        CREATE INDEX IF NOT EXISTS idx_triggers_user
            ON triggers (user_id);

        ALTER TABLE triggers ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS triggers_select ON triggers;
        CREATE POLICY triggers_select ON triggers FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS triggers_insert ON triggers;
        CREATE POLICY triggers_insert ON triggers FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS triggers_update ON triggers;
        CREATE POLICY triggers_update ON triggers FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS triggers_delete ON triggers;
        CREATE POLICY triggers_delete ON triggers FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        29,
        "Create calibration_records table for agent action outcome tracking",
        """
        CREATE TABLE IF NOT EXISTS calibration_records (
            id                  TEXT PRIMARY KEY,
            action_category     TEXT NOT NULL,
            action_description  TEXT NOT NULL,
            outcome             TEXT NOT NULL,
            agent_id            TEXT,
            project_id          TEXT,
            context             JSONB NOT NULL DEFAULT '{}'::jsonb,
            user_id             TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_calibration_records_category
            ON calibration_records (action_category);
        CREATE INDEX IF NOT EXISTS idx_calibration_records_outcome
            ON calibration_records (outcome);
        CREATE INDEX IF NOT EXISTS idx_calibration_records_user
            ON calibration_records (user_id);
        CREATE INDEX IF NOT EXISTS idx_calibration_records_project
            ON calibration_records (project_id);
        CREATE INDEX IF NOT EXISTS idx_calibration_records_created
            ON calibration_records (created_at DESC);

        ALTER TABLE calibration_records ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS calibration_records_select ON calibration_records;
        CREATE POLICY calibration_records_select ON calibration_records FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_records_insert ON calibration_records;
        CREATE POLICY calibration_records_insert ON calibration_records FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_records_update ON calibration_records;
        CREATE POLICY calibration_records_update ON calibration_records FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS calibration_records_delete ON calibration_records;
        CREATE POLICY calibration_records_delete ON calibration_records FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        30,
        "Create degradation_policies table",
        """
        CREATE TABLE IF NOT EXISTS degradation_policies (
            id                TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            trigger_type      TEXT NOT NULL,
            condition         JSONB NOT NULL DEFAULT '{}'::jsonb,
            action            TEXT NOT NULL,
            description       TEXT,
            status            TEXT NOT NULL DEFAULT 'active',
            cooldown_minutes  DOUBLE PRECISION,
            max_fires         INTEGER,
            fire_count        INTEGER NOT NULL DEFAULT 0,
            last_fired_at     TIMESTAMPTZ,
            project_id        TEXT,
            agent_id          TEXT,
            user_id           TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_degradation_policies_trigger_type
            ON degradation_policies (trigger_type);
        CREATE INDEX IF NOT EXISTS idx_degradation_policies_action
            ON degradation_policies (action);
        CREATE INDEX IF NOT EXISTS idx_degradation_policies_status
            ON degradation_policies (status) WHERE status = 'active';
        CREATE INDEX IF NOT EXISTS idx_degradation_policies_user
            ON degradation_policies (user_id);
        CREATE INDEX IF NOT EXISTS idx_degradation_policies_created
            ON degradation_policies (created_at DESC);

        ALTER TABLE degradation_policies ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS degradation_policies_select ON degradation_policies;
        CREATE POLICY degradation_policies_select ON degradation_policies FOR SELECT
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS degradation_policies_insert ON degradation_policies;
        CREATE POLICY degradation_policies_insert ON degradation_policies FOR INSERT
            WITH CHECK (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS degradation_policies_update ON degradation_policies;
        CREATE POLICY degradation_policies_update ON degradation_policies FOR UPDATE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));

        DROP POLICY IF EXISTS degradation_policies_delete ON degradation_policies;
        CREATE POLICY degradation_policies_delete ON degradation_policies FOR DELETE
            USING (user_id IS NULL OR user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
    (
        31,
        "Create oauth_storage table for persistent OAuth state on Fly.io",
        """
        CREATE TABLE IF NOT EXISTS oauth_storage (
            collection  TEXT NOT NULL DEFAULT '',
            key         TEXT NOT NULL,
            value       JSONB NOT NULL,
            expires_at  TIMESTAMPTZ,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (collection, key)
        );

        CREATE INDEX IF NOT EXISTS idx_oauth_storage_expires
            ON oauth_storage (expires_at)
            WHERE expires_at IS NOT NULL;
        """,
    ),
]


async def _get_applied_versions(pool: asyncpg.Pool) -> set[int]:
    """Get set of already-applied migration versions."""
    # Check if schema_migrations table exists
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'schema_migrations'
        )
        """
    )
    if not exists:
        return set()
    rows = await pool.fetch("SELECT version FROM schema_migrations")
    return {r["version"] for r in rows}


# Advisory lock ID for serializing migrations across processes
_MIGRATION_LOCK_ID = 839271  # arbitrary unique int


async def run_migrations(pool: asyncpg.Pool) -> list[int]:
    """Run all pending migrations in order. Returns list of applied versions.

    Uses a Postgres advisory lock to serialize concurrent migration runs.
    """
    applied: list[int] = []

    async with pool.acquire() as lock_conn:
        # Acquire session-level advisory lock (blocks until available)
        await lock_conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_ID)
        try:
            existing = await _get_applied_versions(pool)

            for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
                if version in existing:
                    continue
                async with lock_conn.transaction():
                    await lock_conn.execute(sql)
                applied.append(version)
                logger.info("Applied migration %d: %s", version, description)

            # Record newly applied migrations (schema_migrations table
            # exists after migration 3 runs)
            if applied:
                for version, description, _ in MIGRATIONS:
                    if version not in existing:
                        await lock_conn.execute(
                            "INSERT INTO schema_migrations (version, description) "
                            "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                            version,
                            description,
                        )
        finally:
            await lock_conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID,
            )

    return applied
