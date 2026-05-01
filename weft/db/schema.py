"""pgvector schema introspection and dimension self-healing.

Discovers actual vector column dimensions from PostgreSQL catalog tables
and validates them against the configured embedding dimensions. When a
mismatch is found, auto-migrates the column (drop index → alter type →
null stale embeddings → recreate index) so the system self-heals without
user intervention.

pgvector stores the dimension directly in pg_attribute.atttypmod (no
bit-shift encoding like numeric precision/scale).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

logger = logging.getLogger(__name__)

# Tables that carry a vector 'embedding' column.
VECTOR_TABLES: tuple[str, ...] = ("memories", "behaviors", "entities", "episode_turns")

# HNSW index parameters — must match what migrations use.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

_DISCOVER_SQL = """
SELECT c.relname AS table_name, a.atttypmod AS dimension
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
JOIN pg_type t ON a.atttypid = t.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE t.typname = 'vector'
  AND a.attname = 'embedding'
  AND a.atttypmod > 0
  AND n.nspname = 'public'
  AND c.relname = ANY($1::text[])
"""


@dataclass(frozen=True)
class DimensionMismatch:
    """A single table whose DB vector dimension differs from config."""

    table: str
    current_dim: int
    expected_dim: int

    def __str__(self) -> str:
        return (
            f"{self.table}: DB has vector({self.current_dim}), "
            f"config expects vector({self.expected_dim})"
        )


async def discover_vector_dimensions(
    conn: asyncpg.Connection,
) -> dict[str, int]:
    """Query pg_attribute to find actual vector(N) dimensions per table.

    Returns a dict mapping table name to dimension for each table in
    VECTOR_TABLES that has a typed vector column. Tables without a vector
    column (or where the column is untyped) are simply absent from the result.
    """
    rows = await conn.fetch(_DISCOVER_SQL, list(VECTOR_TABLES))
    result: dict[str, int] = {}
    for row in rows:
        table = row["table_name"]
        dim = row["dimension"]
        if dim <= 0:
            logger.warning("unexpected_atttypmod", extra={"table": table, "atttypmod": dim})
            continue
        result[table] = dim

    logger.info(
        "vector_dimension_discovery_complete",
        extra={"tables_found": list(result.keys()), "dimensions": result},
    )
    return result


def validate_dimensions(
    discovered: dict[str, int],
    config_dim: int,
) -> list[DimensionMismatch]:
    """Compare discovered DB dimensions against the configured dimension.

    Returns a list of mismatches (empty when everything matches).
    Tables in VECTOR_TABLES but absent from *discovered* are logged as
    warnings but not treated as mismatches — they may not exist yet.
    """
    if config_dim <= 0:
        raise ValueError(f"config embedding dimension must be positive, got {config_dim}")

    mismatches: list[DimensionMismatch] = []
    for table in VECTOR_TABLES:
        if table not in discovered:
            logger.warning(
                "vector_table_not_discovered",
                extra={"table": table},
            )
            continue
        db_dim = discovered[table]
        if db_dim != config_dim:
            m = DimensionMismatch(table=table, current_dim=db_dim, expected_dim=config_dim)
            logger.warning("dimension_mismatch_detected", extra={"mismatch": str(m)})
            mismatches.append(m)

    if not mismatches:
        logger.info(
            "dimension_validation_passed",
            extra={"config_dim": config_dim, "tables": list(discovered.keys())},
        )
    return mismatches


async def get_dimension_status(
    conn: asyncpg.Connection,
    config_dim: int,
) -> tuple[dict[str, int], list[DimensionMismatch]]:
    """Convenience: discover dimensions and validate in one call."""
    discovered = await discover_vector_dimensions(conn)
    mismatches = validate_dimensions(discovered, config_dim)
    return discovered, mismatches


async def auto_migrate_dimensions(
    conn: asyncpg.Connection,
    mismatches: list[DimensionMismatch],
) -> list[str]:
    """Auto-migrate vector columns to the expected dimension.

    For each mismatch: drops the HNSW index, alters the column type,
    nulls stale embeddings, and recreates the index — all inside a
    single transaction.

    Returns the list of table names that were migrated.
    """
    if not mismatches:
        return []

    migrated: list[str] = []
    target_dim = mismatches[0].expected_dim

    async with conn.transaction():
        for m in mismatches:
            index_name = f"idx_{m.table}_embedding_hnsw"
            logger.info(
                "auto_migrating_vector_dimension",
                extra={
                    "table": m.table,
                    "from_dim": m.current_dim,
                    "to_dim": m.expected_dim,
                },
            )

            # Drop HNSW index (cannot ALTER type with index present)
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")

            # Null out stale embeddings (old dims can't cast to new dims)
            await conn.execute(
                f"UPDATE {m.table} SET embedding = NULL WHERE embedding IS NOT NULL"
            )

            # Alter column type
            await conn.execute(
                f"ALTER TABLE {m.table} ALTER COLUMN embedding TYPE vector({target_dim})"
            )

            # Recreate HNSW index
            await conn.execute(
                f"CREATE INDEX {index_name} "
                f"ON {m.table} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )

            migrated.append(m.table)
            logger.info(
                "auto_migration_complete",
                extra={"table": m.table, "new_dim": target_dim},
            )

    logger.info(
        "all_dimension_migrations_complete",
        extra={"tables_migrated": migrated, "target_dim": target_dim},
    )
    return migrated


async def ensure_vector_dimensions(
    pool: asyncpg.Pool,
    config_dim: int,
) -> list[str]:
    """Top-level orchestrator: discover → validate → auto-migrate.

    Call after migrations have run and the pool is ready. If all dimensions
    match config, this is a no-op. If mismatches exist, auto-migrates the
    columns and returns the list of migrated table names.

    Returns an empty list when no migration was needed.
    """
    async with pool.acquire() as conn:
        discovered, mismatches = await get_dimension_status(conn, config_dim)

        if not discovered:
            logger.info("no_vector_tables_found_skipping_dimension_check")
            return []

        if not mismatches:
            return []

        migrated = await auto_migrate_dimensions(conn, mismatches)
        return migrated
