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
            # Run all migrations (idempotent DDL — CREATE IF NOT EXISTS, etc.)
            for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
                async with lock_conn.transaction():
                    await lock_conn.execute(sql)

            # Track which ones were newly applied
            existing = await _get_applied_versions(pool)
            for version, description, _ in MIGRATIONS:
                if version not in existing:
                    await lock_conn.execute(
                        "INSERT INTO schema_migrations (version, description) "
                        "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        version,
                        description,
                    )
                    applied.append(version)
                    logger.info("Applied migration %d: %s", version, description)
        finally:
            await lock_conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID,
            )

    return applied
