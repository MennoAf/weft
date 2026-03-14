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
