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
        "Ensure user_id column exists and is nullable on all user-scoped tables",
        """
        -- Add user_id to behaviors if missing (created in migration 10 but ensure it exists)
        ALTER TABLE behaviors ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to entities if missing
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to episodes if missing
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to modes if missing (created in migration 20 but ensure it exists)
        ALTER TABLE modes ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to autonomy_policies if missing (created in migration 24 but ensure it exists)
        ALTER TABLE autonomy_policies ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to calibration_records if missing (created in migration 29 but ensure it exists)
        ALTER TABLE calibration_records ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Add user_id to degradation_policies if missing (created in migration 30 but ensure it exists)
        ALTER TABLE degradation_policies ADD COLUMN IF NOT EXISTS user_id TEXT;

        -- Create indexes for efficient user_id filtering (idempotent)
        CREATE INDEX IF NOT EXISTS idx_behaviors_user ON behaviors (user_id);
        CREATE INDEX IF NOT EXISTS idx_entities_user ON entities (user_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes (user_id);
        CREATE INDEX IF NOT EXISTS idx_modes_user ON modes (user_id);
        CREATE INDEX IF NOT EXISTS idx_autonomy_policies_user ON autonomy_policies (user_id);
        CREATE INDEX IF NOT EXISTS idx_calibration_records_user ON calibration_records (user_id);
        CREATE INDEX IF NOT EXISTS idx_degradation_policies_user ON degradation_policies (user_id);
        """,
    ),
    (
        32,
        "Create audit_backfill_user_id table for Phase 1 backfill audit trail",
        """
        CREATE TABLE IF NOT EXISTS audit_backfill_user_id (
            id           SERIAL PRIMARY KEY,
            source_table TEXT NOT NULL,
            row_id       TEXT NOT NULL,
            old_scope    TEXT NOT NULL,
            new_scope    TEXT NOT NULL,
            migrated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_audit_backfill_source
            ON audit_backfill_user_id (source_table, row_id);
        """,
    ),
    (
        33,
        "Create OAuth 2.1 tables (clients, codes, refresh tokens, revocations)",
        """
        -- OAuth dynamic-client registrations (RFC 7591)
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id                  TEXT PRIMARY KEY,
            client_name                TEXT,
            redirect_uris              TEXT[] NOT NULL,
            grant_types                TEXT[] NOT NULL DEFAULT '{authorization_code,refresh_token}',
            response_types             TEXT[] NOT NULL DEFAULT '{code}',
            token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',
            scope                      TEXT NOT NULL DEFAULT 'mcp.read mcp.write',
            software_id                TEXT,
            software_version           TEXT,
            created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_used_at               TIMESTAMPTZ
        );

        -- Pending authorization codes (short-lived, ~10 min)
        CREATE TABLE IF NOT EXISTS oauth_authorization_codes (
            code                  TEXT PRIMARY KEY,
            client_id             TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
            user_sub              TEXT NOT NULL,
            redirect_uri          TEXT NOT NULL,
            scope                 TEXT NOT NULL,
            code_challenge        TEXT NOT NULL,
            code_challenge_method TEXT NOT NULL,
            expires_at            TIMESTAMPTZ NOT NULL,
            consumed_at           TIMESTAMPTZ,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_codes_expires
            ON oauth_authorization_codes (expires_at);

        -- Refresh tokens (longer-lived, rotated on use)
        CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
            jti         TEXT PRIMARY KEY,
            client_id   TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
            user_sub    TEXT NOT NULL,
            scope       TEXT NOT NULL,
            token_hash  TEXT NOT NULL,
            issued_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at  TIMESTAMPTZ NOT NULL,
            revoked_at  TIMESTAMPTZ,
            rotated_to  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_refresh_user
            ON oauth_refresh_tokens (user_sub);
        CREATE INDEX IF NOT EXISTS idx_oauth_refresh_expires
            ON oauth_refresh_tokens (expires_at);

        -- Access-token revocation blocklist (rare; most access tokens expire
        -- before revoke)
        CREATE TABLE IF NOT EXISTS oauth_access_revocations (
            jti         TEXT PRIMARY KEY,
            revoked_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at  TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_access_rev_expires
            ON oauth_access_revocations (expires_at);

        -- OAuth tables are service-role only — not user-scoped via RLS.
        -- Pattern mirrors migration 23 (schema_migrations, weft_metadata,
        -- memory_access_log): USING (true) WITH CHECK (true) means RLS is
        -- enforced structurally by restricting which connections can reach
        -- these tables (service role, app.user_id='').
        ALTER TABLE oauth_clients ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS oauth_clients_service ON oauth_clients;
        CREATE POLICY oauth_clients_service ON oauth_clients
            USING (true) WITH CHECK (true);

        ALTER TABLE oauth_authorization_codes ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS oauth_authorization_codes_service ON oauth_authorization_codes;
        CREATE POLICY oauth_authorization_codes_service ON oauth_authorization_codes
            USING (true) WITH CHECK (true);

        ALTER TABLE oauth_refresh_tokens ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS oauth_refresh_tokens_service ON oauth_refresh_tokens;
        CREATE POLICY oauth_refresh_tokens_service ON oauth_refresh_tokens
            USING (true) WITH CHECK (true);

        ALTER TABLE oauth_access_revocations ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS oauth_access_revocations_service ON oauth_access_revocations;
        CREATE POLICY oauth_access_revocations_service ON oauth_access_revocations
            USING (true) WITH CHECK (true);
        """,
    ),
    (
        34,
        "Schema v1: add federated-future columns to memories (additive)",
        """
        -- Federated-Future Schema v1: additive only. No behavior change.
        -- See 2026-04-26-weft-federated-future-schema-v1.md.
        --
        -- schema_version  : dispatch field for future upgrade chain
        -- author_identity : who wrote this row (survives sharing)
        -- visibility      : private | global  ('shared' rejected by CHECK
        --                   until workspace logic ships in v2)
        -- provenance      : audit trail; lives ALONGSIDE the existing
        --                   `source` column (do not unify — see opinion)
        -- sharing_metadata: escape hatch for federation/ACL/expiry; empty in v1
        -- workspace_id    : FK target for migration 35's workspaces table

        ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS schema_version  INTEGER NOT NULL DEFAULT 1,
            ADD COLUMN IF NOT EXISTS author_identity JSONB   NOT NULL DEFAULT '{"kind":"unknown"}'::jsonb,
            ADD COLUMN IF NOT EXISTS visibility      TEXT    NOT NULL DEFAULT 'private',
            ADD COLUMN IF NOT EXISTS provenance      JSONB   NOT NULL DEFAULT '{"source":"self"}'::jsonb,
            ADD COLUMN IF NOT EXISTS sharing_metadata JSONB  NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS workspace_id    TEXT;

        -- Backfill author_identity + visibility from existing user_id state.
        -- Rows with user_id IS NULL = current "global by convention" rows
        -- (system seeds, shared modes, etc.). Migration 36 will replace
        -- the NULL convention with a SYSTEM_GLOBAL sentinel.
        UPDATE memories
        SET
            author_identity = CASE
                WHEN user_id IS NULL THEN '{"kind":"system","component":"seed"}'::jsonb
                ELSE jsonb_build_object('kind', 'local_user', 'user_id', user_id)
            END,
            visibility = CASE
                WHEN user_id IS NULL THEN 'global'
                ELSE 'private'
            END
        WHERE author_identity = '{"kind":"unknown"}'::jsonb;

        -- Fail-loud constraint: 'shared' is reserved but not honored by v1
        -- logic. Reject writes until workspace primitive ships, so we never
        -- silently treat would-be-shared rows as private.
        ALTER TABLE memories
            DROP CONSTRAINT IF EXISTS memories_visibility_check;
        ALTER TABLE memories
            ADD CONSTRAINT memories_visibility_check
            CHECK (visibility IN ('private', 'global'));

        CREATE INDEX IF NOT EXISTS idx_memories_workspace
            ON memories (workspace_id) WHERE workspace_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_memories_visibility
            ON memories (visibility);
        """,
    ),
    (
        35,
        "Schema v1: workspaces + workspace_members tables",
        """
        -- Minimum workspace primitive. Two tables; no role hierarchy beyond
        -- 'member'; no invitation flow; no expiry. Owner inserts members
        -- directly. v1 use case: AIO Cleanroom shared brain with Brandon.
        --
        -- member_identity is JSONB so it can carry remote-install members
        -- once federation lands ({"kind":"remote_install","install_pubkey":
        -- "...","member_uuid":"..."}). For local users it's
        -- {"kind":"local_user","user_id":"<uuid>"}.

        CREATE TABLE IF NOT EXISTS workspaces (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            description     TEXT,
            created_by      TEXT NOT NULL,
            install_pubkey  TEXT,
            metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_workspaces_created_by
            ON workspaces (created_by);

        CREATE TABLE IF NOT EXISTS workspace_members (
            workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            member_identity JSONB NOT NULL,
            role            TEXT NOT NULL DEFAULT 'member',
            added_by        TEXT NOT NULL,
            added_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (workspace_id, member_identity)
        );
        CREATE INDEX IF NOT EXISTS idx_workspace_members_user
            ON workspace_members ((member_identity->>'user_id'))
            WHERE member_identity->>'user_id' IS NOT NULL;

        -- FK from memories.workspace_id (added in migration 34) to workspaces.
        -- ON DELETE SET NULL: deleting a workspace orphans the memories back
        -- to private/global scope rather than destroying them.
        ALTER TABLE memories
            DROP CONSTRAINT IF EXISTS memories_workspace_fk;
        ALTER TABLE memories
            ADD CONSTRAINT memories_workspace_fk
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE SET NULL;

        -- RLS: service-role for now. App layer enforces "who can read
        -- workspace metadata" via tool-level checks. A v2 tightening pass
        -- can scope these to "members only" via subquery policies.
        ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS workspaces_service ON workspaces;
        CREATE POLICY workspaces_service ON workspaces
            USING (true) WITH CHECK (true);

        ALTER TABLE workspace_members ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS workspace_members_service ON workspace_members;
        CREATE POLICY workspace_members_service ON workspace_members
            USING (true) WITH CHECK (true);

        -- Extend memories RLS to include workspace membership. A row is
        -- visible if (a) globally scoped, (b) user-owned, or (c) lives in a
        -- workspace the current user is a member of. The workspace_members
        -- subquery hits the service policy above (USING true) so it works
        -- under the normal user connection.
        DROP POLICY IF EXISTS memories_select ON memories;
        CREATE POLICY memories_select ON memories FOR SELECT
            USING (
                user_id IS NULL
                OR user_id = nullif(current_setting('app.user_id', true), '')
                OR (
                    workspace_id IS NOT NULL
                    AND EXISTS (
                        SELECT 1 FROM workspace_members wm
                        WHERE wm.workspace_id = memories.workspace_id
                          AND wm.member_identity->>'user_id'
                              = nullif(current_setting('app.user_id', true), '')
                    )
                )
            );

        -- INSERT/UPDATE/DELETE policies stay strict: only the row owner can
        -- write. Workspace members can read but not directly mutate other
        -- members' rows. Cross-member writes happen via app-level tools
        -- that act as the row author.
        """,
    ),
    (
        36,
        "Schema v1: SYSTEM_GLOBAL sentinel + NOT NULL user_id + RLS rewrite",
        r"""
        -- Kill the implicit-global path. Before this migration, ``user_id IS
        -- NULL`` was the convention for "global / seed / system-owned" rows,
        -- which meant any agent that forgot to set ``app.user_id`` silently
        -- wrote a globally readable row. After: every row has a non-null
        -- ``user_id``; the literal string ``__system_global_zathras__`` is
        -- the sentinel for system-owned rows. Forgetting to set
        -- ``app.user_id`` becomes a NOT NULL constraint violation — fail
        -- loud, not silent leak. Named-string (not UUID) so an agent cannot
        -- accidentally land on it via "just generate a UUID."

        -- Step 1: backfill every NULL across all user-scoped tables.
        UPDATE memories                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE memory_relationships      SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE behaviors                 SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE entities                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE entity_mentions           SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE episodes                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE episode_memories          SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE modes                     SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE alerts                    SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE check_ins                 SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE autonomy_policies         SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE policy_calibration_events SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE cost_entries              SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE triggers                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE calibration_records       SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
        UPDATE degradation_policies      SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;

        -- Step 2: NOT NULL on every user-scoped table, plus a column DEFAULT
        -- so INSERTs that don't explicitly specify user_id pick up the
        -- session's app.user_id. (When app.user_id is unset, the default
        -- evaluates to NULL → NOT NULL violation → fail loud.) This keeps
        -- the safety property while removing boilerplate from the call sites.
        ALTER TABLE memories                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE memory_relationships      ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE behaviors                 ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE entities                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE entity_mentions           ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE episodes                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE episode_memories          ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE modes                     ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE alerts                    ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE check_ins                 ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE autonomy_policies         ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE policy_calibration_events ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE cost_entries              ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE triggers                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE calibration_records       ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
        ALTER TABLE degradation_policies      ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;

        -- Step 3: rewrite RLS policies. Old form was
        --   user_id IS NULL OR user_id = current_setting
        -- which silently passed for unauthenticated writes. New form is
        --   user_id = current_setting OR user_id = SYSTEM_GLOBAL
        -- where SYSTEM_GLOBAL is a literal string only writable when an
        -- operator explicitly sets ``app.user_id`` to it. (current_setting
        -- returns the empty string when unset, which matches nothing.)

        -- memories: SELECT keeps the workspace-membership branch from migration 35.
        DROP POLICY IF EXISTS memories_select ON memories;
        CREATE POLICY memories_select ON memories FOR SELECT
            USING (
                user_id = '__system_global_zathras__'
                OR user_id = nullif(current_setting('app.user_id', true), '')
                OR (
                    workspace_id IS NOT NULL
                    AND EXISTS (
                        SELECT 1 FROM workspace_members wm
                        WHERE wm.workspace_id = memories.workspace_id
                          AND wm.member_identity->>'user_id'
                              = nullif(current_setting('app.user_id', true), '')
                    )
                )
            );

        DROP POLICY IF EXISTS memories_insert ON memories;
        CREATE POLICY memories_insert ON memories FOR INSERT
            WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));
        DROP POLICY IF EXISTS memories_update ON memories;
        CREATE POLICY memories_update ON memories FOR UPDATE
            USING (user_id = nullif(current_setting('app.user_id', true), ''));
        DROP POLICY IF EXISTS memories_delete ON memories;
        CREATE POLICY memories_delete ON memories FOR DELETE
            USING (user_id = nullif(current_setting('app.user_id', true), ''));

        -- All other tables: simple sentinel-or-self policy. Generated via
        -- a DO block to keep the migration short.
        DO $rls$
        DECLARE
            t TEXT;
            tables TEXT[] := ARRAY[
                'memory_relationships',
                'behaviors',
                'entities',
                'entity_mentions',
                'episodes',
                'episode_memories',
                'modes',
                'alerts',
                'check_ins',
                'autonomy_policies',
                'policy_calibration_events',
                'cost_entries',
                'triggers',
                'calibration_records',
                'degradation_policies'
            ];
        BEGIN
            FOREACH t IN ARRAY tables LOOP
                EXECUTE format('DROP POLICY IF EXISTS %I_select ON %I', t, t);
                EXECUTE format('DROP POLICY IF EXISTS %I_insert ON %I', t, t);
                EXECUTE format('DROP POLICY IF EXISTS %I_update ON %I', t, t);
                EXECUTE format('DROP POLICY IF EXISTS %I_delete ON %I', t, t);
                -- Some legacy migrations used different policy names.
                EXECUTE format('DROP POLICY IF EXISTS calibration_events_select ON %I', t);
                EXECUTE format('DROP POLICY IF EXISTS calibration_events_insert ON %I', t);
                EXECUTE format('DROP POLICY IF EXISTS calibration_events_update ON %I', t);
                EXECUTE format('DROP POLICY IF EXISTS calibration_events_delete ON %I', t);

                EXECUTE format($p$
                    CREATE POLICY %I_select ON %I FOR SELECT
                    USING (user_id = '__system_global_zathras__'
                           OR user_id = nullif(current_setting('app.user_id', true), ''))
                $p$, t, t);
                EXECUTE format($p$
                    CREATE POLICY %I_insert ON %I FOR INSERT
                    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''))
                $p$, t, t);
                EXECUTE format($p$
                    CREATE POLICY %I_update ON %I FOR UPDATE
                    USING (user_id = nullif(current_setting('app.user_id', true), ''))
                $p$, t, t);
                EXECUTE format($p$
                    CREATE POLICY %I_delete ON %I FOR DELETE
                    USING (user_id = nullif(current_setting('app.user_id', true), ''))
                $p$, t, t);
            END LOOP;
        END
        $rls$;

        -- Drop the now-stale partial unique index on modes (was scoped to
        -- ``WHERE user_id IS NULL``; after backfill nothing matches).
        -- Replace with one keyed to the sentinel.
        DROP INDEX IF EXISTS uq_modes_null_user_name;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_modes_global_user_name
            ON modes (name) WHERE user_id = '__system_global_zathras__';
        """,
    ),
    (
        37,
        "Schema v1: smart default for memories.author_identity",
        """
        -- Migration 34 added author_identity with a placeholder default of
        -- ``{"kind":"unknown"}``. Now that user_id is NOT NULL with a
        -- session-derived default (migration 36), we can compute a real
        -- author_identity at INSERT time from the same session value.
        --
        -- Rows the SYSTEM_GLOBAL sentinel writes get ``{"kind":"system",...}``;
        -- everything else gets ``{"kind":"local_user","user_id":"..."}``.
        -- Call sites that want to override (e.g. to record an agent acting
        -- on a user's behalf, ``{"kind":"agent","on_behalf_of":"..."}``)
        -- can still pass author_identity explicitly.

        ALTER TABLE memories
            ALTER COLUMN author_identity SET DEFAULT
            CASE
                WHEN nullif(current_setting('app.user_id', true), '')
                     = '__system_global_zathras__'
                THEN '{"kind":"system","component":"runtime"}'::jsonb
                WHEN nullif(current_setting('app.user_id', true), '') IS NOT NULL
                THEN jsonb_build_object(
                    'kind', 'local_user',
                    'user_id', current_setting('app.user_id', true)
                )
                ELSE '{"kind":"unknown"}'::jsonb
            END;
        """,
    ),
    (
        38,
        "Trackers v1: open-loop primitive (Wick Phase 3)",
        r"""
        -- Trackers are the lifecycle-aware primitive that memories aren't.
        -- Where memories are blob-shaped facts ("I pitched Tory Burch"),
        -- trackers carry state that changes over time, can be nudged, can
        -- be snoozed, and surface in the daily brief as "open loops."
        --
        -- Kinds (per weft_v2_spec.md §3, §4):
        --   outreach       — contact made, awaiting reply
        --   task           — work item not yet shaped as a Loom task
        --   follow_up      — generic follow-up
        --   meal_plan      — weekly menu (context.items shape)
        --   shopping_list  — grocery list (context.items shape)
        --   pantry         — what's in the pantry (context.items shape)
        --   watch          — passive monitor (e.g. "watch the AAPL earnings call")
        --   list           — generic list (sugar tools weft_list_*)
        --   trace          — Orchestrator long-running trace promotion
        --
        -- States: open ∈ {in_progress, awaiting_reply, blocked};
        --         terminal ∈ {done, abandoned}.
        -- state_history is an append-only JSONB array of
        --   {"from": "...", "to": "...", "at": "...", "note": "..."}.
        --
        -- Nudge modes (V1): none (query-only), once (fire at nudge_after,
        -- then silent), recur (fire every nudge_interval until closed).
        -- snooze_until temporarily suppresses due-results without changing
        -- mode. dismiss = bump last_touch (silence until next interval);
        -- close = terminal state.
        --
        -- provenance: 'supervisor' | 'agent'. Schema-only for V1; Phase 2
        -- will wire app-layer enforcement (Layer 1 of poisoning defense).

        CREATE TABLE IF NOT EXISTS trackers (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT
                            nullif(current_setting('app.user_id', true), ''),
            project_id      TEXT,
            entity_id       TEXT REFERENCES entities(id) ON DELETE SET NULL,
            kind            TEXT NOT NULL,
            title           TEXT NOT NULL,
            state           TEXT NOT NULL DEFAULT 'in_progress',
            state_history   JSONB NOT NULL DEFAULT '[]'::jsonb,
            context         JSONB NOT NULL DEFAULT '{}'::jsonb,
            last_touch      TIMESTAMPTZ NOT NULL DEFAULT now(),
            nudge_mode      TEXT NOT NULL DEFAULT 'none',
            nudge_after     TIMESTAMPTZ,
            nudge_interval  INTERVAL,
            snooze_until    TIMESTAMPTZ,
            trigger_ids     TEXT[] NOT NULL DEFAULT '{}',
            provenance      TEXT NOT NULL DEFAULT 'supervisor',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT trackers_kind_check CHECK (kind IN (
                'outreach', 'task', 'follow_up', 'meal_plan',
                'shopping_list', 'pantry', 'watch', 'list', 'trace'
            )),
            CONSTRAINT trackers_state_check CHECK (state IN (
                'in_progress', 'awaiting_reply', 'blocked', 'done', 'abandoned'
            )),
            CONSTRAINT trackers_nudge_mode_check CHECK (nudge_mode IN (
                'none', 'once', 'recur'
            )),
            CONSTRAINT trackers_provenance_check CHECK (provenance IN (
                'supervisor', 'agent'
            ))
        );

        -- Hot-path index for ``weft_tracker_due``: open-state rows with a
        -- due nudge time, optionally snoozed. Filtered partial index keeps
        -- it tiny — terminal states never appear in the due query.
        CREATE INDEX IF NOT EXISTS idx_trackers_due
            ON trackers (user_id, nudge_after)
            WHERE state IN ('in_progress', 'awaiting_reply', 'blocked')
              AND nudge_mode <> 'none';

        CREATE INDEX IF NOT EXISTS idx_trackers_user_kind
            ON trackers (user_id, kind);
        CREATE INDEX IF NOT EXISTS idx_trackers_user_state
            ON trackers (user_id, state);
        CREATE INDEX IF NOT EXISTS idx_trackers_project
            ON trackers (project_id) WHERE project_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_trackers_entity
            ON trackers (entity_id) WHERE entity_id IS NOT NULL;

        -- RLS: sentinel-or-self, same pattern as the 16 other user-scoped
        -- tables (see migration 36's DO block). No workspace branch yet —
        -- v1 trackers are user-private. If shared trackers become a need,
        -- mirror memories' workspace_id branch.
        ALTER TABLE trackers ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS trackers_select ON trackers;
        CREATE POLICY trackers_select ON trackers FOR SELECT
            USING (user_id = '__system_global_zathras__'
                   OR user_id = nullif(current_setting('app.user_id', true), ''));
        DROP POLICY IF EXISTS trackers_insert ON trackers;
        CREATE POLICY trackers_insert ON trackers FOR INSERT
            WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));
        DROP POLICY IF EXISTS trackers_update ON trackers;
        CREATE POLICY trackers_update ON trackers FOR UPDATE
            USING (user_id = nullif(current_setting('app.user_id', true), ''));
        DROP POLICY IF EXISTS trackers_delete ON trackers;
        CREATE POLICY trackers_delete ON trackers FOR DELETE
            USING (user_id = nullif(current_setting('app.user_id', true), ''));
        """,
    ),
]


async def _get_applied_versions(pool: asyncpg.Pool) -> set[int]:
    """Get set of already-applied migration versions."""
    # Supabase ships its own ``auth.schema_migrations`` and
    # ``storage.schema_migrations`` tables, so filter by current schema —
    # otherwise a fresh Supabase DB reports the table as existing and the
    # SELECT below blows up with UndefinedTableError on public.schema_migrations.
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'schema_migrations'
              AND table_schema = current_schema()
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
