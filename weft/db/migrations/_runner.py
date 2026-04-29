"""Migration runner: applies pending migrations under an advisory lock.

Migration tuples themselves live in sibling ``vNN_*.py`` modules and are
aggregated by ``__init__.py``. This module is the only place that
executes SQL.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)

# Advisory lock ID for serializing migrations across processes
_MIGRATION_LOCK_ID = 839271  # arbitrary unique int


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


async def run_migrations(pool: asyncpg.Pool) -> list[int]:
    """Run all pending migrations in order. Returns list of applied versions.

    Uses a Postgres advisory lock to serialize concurrent migration runs.
    """
    # Lazy import: this module is imported by the package __init__ that
    # builds MIGRATIONS, so a top-level import would re-enter the package
    # mid-discovery.
    from weft.db.migrations import MIGRATIONS

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
