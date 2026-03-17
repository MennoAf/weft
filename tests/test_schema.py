"""Tests for pgvector schema introspection and dimension self-healing."""

from __future__ import annotations

import asyncpg
import pytest

from weft.db.connection import _pgvector_codec_init
from weft.db.migrations import run_migrations
from weft.db.schema import (
    VECTOR_TABLES,
    DimensionMismatch,
    auto_migrate_dimensions,
    discover_vector_dimensions,
    ensure_vector_dimensions,
    get_dimension_status,
    validate_dimensions,
)

# Tests that mutate schema (DROP TABLE, ALTER COLUMN) use this fixture
# instead of the shared `pool` to avoid polluting state for other tests.
# It creates a fresh pool, runs migrations, and drops ALL tables on teardown
# so the shared conftest pool fixture can recreate cleanly.
_pg_dsn: str | None = None


@pytest.fixture
async def schema_pool():
    """Isolated pool for schema-destructive tests. Re-runs migrations on teardown."""
    from tests.conftest import _pg_container

    dsn = _pg_container.get_connection_url().replace("+psycopg2", "")
    p = await asyncpg.create_pool(dsn, min_size=2, max_size=5, init=_pgvector_codec_init)
    await run_migrations(p)
    # TRUNCATE to start clean
    await p.execute(
        "TRUNCATE memory_access_log, entity_mentions, episode_memories, "
        "memory_relationships, entities, episodes, memories, behaviors, "
        "weft_metadata CASCADE"
    )
    yield p
    # Restore schema by re-running migrations (idempotent CREATE IF NOT EXISTS
    # handles tables; but ALTER COLUMN migrations need the tables to exist first).
    # Easiest: drop everything and recreate.
    await p.execute("DROP SCHEMA public CASCADE")
    await p.execute("CREATE SCHEMA public")
    await run_migrations(p)
    await p.close()


# ---------------------------------------------------------------------------
# DimensionMismatch dataclass
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    def test_frozen(self):
        m = DimensionMismatch(table="memories", current_dim=384, expected_dim=768)
        with pytest.raises(AttributeError):
            m.table = "behaviors"  # type: ignore[misc]

    def test_str(self):
        m = DimensionMismatch(table="memories", current_dim=384, expected_dim=768)
        assert str(m) == "memories: DB has vector(384), config expects vector(768)"

    def test_hashable(self):
        m = DimensionMismatch(table="memories", current_dim=384, expected_dim=768)
        assert hash(m) is not None
        assert m in {m}

    def test_equality(self):
        a = DimensionMismatch(table="memories", current_dim=384, expected_dim=768)
        b = DimensionMismatch(table="memories", current_dim=384, expected_dim=768)
        assert a == b


# ---------------------------------------------------------------------------
# validate_dimensions (pure function — no DB needed)
# ---------------------------------------------------------------------------


class TestValidateDimensions:
    def test_all_match(self):
        discovered = {"memories": 768, "behaviors": 768, "entities": 768}
        assert validate_dimensions(discovered, 768) == []

    def test_single_mismatch(self):
        discovered = {"memories": 384, "behaviors": 768, "entities": 768}
        mismatches = validate_dimensions(discovered, 768)
        assert len(mismatches) == 1
        assert mismatches[0] == DimensionMismatch("memories", 384, 768)

    def test_all_mismatch(self):
        discovered = {"memories": 384, "behaviors": 384, "entities": 384}
        mismatches = validate_dimensions(discovered, 768)
        assert len(mismatches) == 3
        tables = {m.table for m in mismatches}
        assert tables == {"memories", "behaviors", "entities"}

    def test_empty_discovered(self):
        assert validate_dimensions({}, 768) == []

    def test_partial_tables(self):
        """Only memories exists — behaviors and entities are missing (not a mismatch)."""
        discovered = {"memories": 768}
        assert validate_dimensions(discovered, 768) == []

    def test_negative_config_dim_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_dimensions({"memories": 768}, -1)

    def test_zero_config_dim_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_dimensions({"memories": 768}, 0)


# ---------------------------------------------------------------------------
# discover_vector_dimensions (requires DB)
# ---------------------------------------------------------------------------


class TestDiscoverVectorDimensions:
    @pytest.mark.asyncio
    async def test_returns_correct_dims(self, pool):
        """After migrations, all three tables should have vector(768)."""
        async with pool.acquire() as conn:
            result = await discover_vector_dimensions(conn)
        assert result == {"memories": 768, "behaviors": 768, "entities": 768}

    @pytest.mark.asyncio
    async def test_partial_tables(self, schema_pool):
        """Drop behaviors and entities — only memories should be discovered."""
        async with schema_pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS entity_mentions CASCADE")
            await conn.execute("DROP TABLE IF EXISTS episode_memories CASCADE")
            await conn.execute("DROP TABLE IF EXISTS entities CASCADE")
            await conn.execute("DROP TABLE IF EXISTS behaviors CASCADE")
            result = await discover_vector_dimensions(conn)
        assert list(result.keys()) == ["memories"]
        assert result["memories"] == 768

    @pytest.mark.asyncio
    async def test_no_vector_tables(self, schema_pool):
        """Drop all vector tables — should return empty dict."""
        async with schema_pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS entity_mentions CASCADE")
            await conn.execute("DROP TABLE IF EXISTS episode_memories CASCADE")
            await conn.execute("DROP TABLE IF EXISTS entities CASCADE")
            await conn.execute("DROP TABLE IF EXISTS behaviors CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memory_relationships CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memory_access_log CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memories CASCADE")
            result = await discover_vector_dimensions(conn)
        assert result == {}

    @pytest.mark.asyncio
    async def test_different_dims_per_table(self, schema_pool):
        """Alter one table to a different dimension — should discover both."""
        async with schema_pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_memories_embedding_hnsw")
            await conn.execute(
                "ALTER TABLE memories ALTER COLUMN embedding TYPE vector(384)"
            )
            result = await discover_vector_dimensions(conn)
        assert result["memories"] == 384
        assert result["behaviors"] == 768


# ---------------------------------------------------------------------------
# get_dimension_status (convenience wrapper)
# ---------------------------------------------------------------------------


class TestGetDimensionStatus:
    @pytest.mark.asyncio
    async def test_matching(self, pool):
        async with pool.acquire() as conn:
            discovered, mismatches = await get_dimension_status(conn, 768)
        assert len(discovered) == 3
        assert mismatches == []

    @pytest.mark.asyncio
    async def test_mismatch(self, pool):
        async with pool.acquire() as conn:
            discovered, mismatches = await get_dimension_status(conn, 384)
        assert len(mismatches) == 3
        assert all(m.expected_dim == 384 for m in mismatches)


# ---------------------------------------------------------------------------
# auto_migrate_dimensions
# ---------------------------------------------------------------------------


class TestAutoMigrateDimensions:
    @pytest.mark.asyncio
    async def test_no_mismatches_noop(self, pool):
        async with pool.acquire() as conn:
            migrated = await auto_migrate_dimensions(conn, [])
        assert migrated == []

    @pytest.mark.asyncio
    async def test_migrate_single_table(self, schema_pool):
        """Migrate memories from 768 → 1536."""
        mismatch = DimensionMismatch(table="memories", current_dim=768, expected_dim=1536)
        async with schema_pool.acquire() as conn:
            migrated = await auto_migrate_dimensions(conn, [mismatch])
            discovered = await discover_vector_dimensions(conn)
        assert migrated == ["memories"]
        assert discovered["memories"] == 1536
        assert discovered["behaviors"] == 768

    @pytest.mark.asyncio
    async def test_migrate_all_tables(self, schema_pool):
        """Migrate all tables from 768 → 1536."""
        mismatches = [
            DimensionMismatch(table=t, current_dim=768, expected_dim=1536)
            for t in VECTOR_TABLES
        ]
        async with schema_pool.acquire() as conn:
            migrated = await auto_migrate_dimensions(conn, mismatches)
            discovered = await discover_vector_dimensions(conn)
        assert set(migrated) == set(VECTOR_TABLES)
        assert all(dim == 1536 for dim in discovered.values())

    @pytest.mark.asyncio
    async def test_migration_nulls_embeddings(self, schema_pool):
        """After migration, existing embeddings should be NULL."""
        async with schema_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memories (id, type, content, embedding)
                   VALUES ('test-1', 'fact', 'test content', $1::vector)""",
                [0.1] * 768,
            )
            row = await conn.fetchrow("SELECT embedding FROM memories WHERE id = 'test-1'")
            assert row["embedding"] is not None

            mismatch = DimensionMismatch(table="memories", current_dim=768, expected_dim=1536)
            await auto_migrate_dimensions(conn, [mismatch])

            row = await conn.fetchrow("SELECT embedding FROM memories WHERE id = 'test-1'")
            assert row["embedding"] is None

    @pytest.mark.asyncio
    async def test_migration_recreates_hnsw_index(self, schema_pool):
        """After migration, HNSW index should exist at the new dimension."""
        mismatch = DimensionMismatch(table="memories", current_dim=768, expected_dim=1536)
        async with schema_pool.acquire() as conn:
            await auto_migrate_dimensions(conn, [mismatch])
            row = await conn.fetchrow(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'memories' AND indexname = 'idx_memories_embedding_hnsw'"
            )
        assert row is not None

    @pytest.mark.asyncio
    async def test_migrate_then_insert_new_dim(self, schema_pool):
        """After migration, inserting a vector at the new dimension should work."""
        mismatch = DimensionMismatch(table="memories", current_dim=768, expected_dim=1536)
        async with schema_pool.acquire() as conn:
            await auto_migrate_dimensions(conn, [mismatch])
            await conn.execute(
                """INSERT INTO memories (id, type, content, embedding)
                   VALUES ('test-new', 'fact', 'test', $1::vector)""",
                [0.1] * 1536,
            )
            row = await conn.fetchrow("SELECT embedding FROM memories WHERE id = 'test-new'")
        assert row["embedding"] is not None
        assert len(row["embedding"]) == 1536


# ---------------------------------------------------------------------------
# ensure_vector_dimensions (pool-level orchestrator)
# ---------------------------------------------------------------------------


class TestEnsureVectorDimensions:
    @pytest.mark.asyncio
    async def test_matching_dims_noop(self, pool):
        """When config matches DB, no migration happens."""
        migrated = await ensure_vector_dimensions(pool, 768)
        assert migrated == []

    @pytest.mark.asyncio
    async def test_mismatch_auto_migrates(self, schema_pool):
        """When config differs from DB, auto-migrates all tables."""
        migrated = await ensure_vector_dimensions(schema_pool, 1536)
        assert set(migrated) == set(VECTOR_TABLES)
        # Verify the migration actually happened
        async with schema_pool.acquire() as conn:
            discovered = await discover_vector_dimensions(conn)
        assert all(dim == 1536 for dim in discovered.values())

    @pytest.mark.asyncio
    async def test_no_tables_returns_empty(self, schema_pool):
        """When no vector tables exist, returns empty list gracefully."""
        async with schema_pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS entity_mentions CASCADE")
            await conn.execute("DROP TABLE IF EXISTS episode_memories CASCADE")
            await conn.execute("DROP TABLE IF EXISTS entities CASCADE")
            await conn.execute("DROP TABLE IF EXISTS behaviors CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memory_relationships CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memory_access_log CASCADE")
            await conn.execute("DROP TABLE IF EXISTS memories CASCADE")
        migrated = await ensure_vector_dimensions(schema_pool, 768)
        assert migrated == []

    @pytest.mark.asyncio
    async def test_partial_mismatch(self, schema_pool):
        """Only migrates tables that actually mismatch."""
        # Change just memories to 384
        async with schema_pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_memories_embedding_hnsw")
            await conn.execute("UPDATE memories SET embedding = NULL")
            await conn.execute(
                "ALTER TABLE memories ALTER COLUMN embedding TYPE vector(384)"
            )
            await conn.execute(
                "CREATE INDEX idx_memories_embedding_hnsw "
                "ON memories USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        migrated = await ensure_vector_dimensions(schema_pool, 768)
        assert migrated == ["memories"]
        # behaviors and entities should still be 768 (untouched)
        async with schema_pool.acquire() as conn:
            discovered = await discover_vector_dimensions(conn)
        assert discovered == {"memories": 768, "behaviors": 768, "entities": 768}
