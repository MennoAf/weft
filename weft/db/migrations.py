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


async def run_migrations(pool: asyncpg.Pool) -> list[int]:
    """Run all pending migrations in order. Returns list of applied versions."""
    applied: list[int] = []

    # Migration 3 (schema_migrations table) must run first if not present
    # but we need to handle bootstrap: run all migrations, then track them
    for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
        # Execute each migration in a transaction
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)

    # Now track which ones were applied
    existing = await _get_applied_versions(pool)
    for version, description, _ in MIGRATIONS:
        if version not in existing:
            await pool.execute(
                "INSERT INTO schema_migrations (version, description) VALUES ($1, $2)",
                version,
                description,
            )
            applied.append(version)
            logger.info("Applied migration %d: %s", version, description)

    return applied
