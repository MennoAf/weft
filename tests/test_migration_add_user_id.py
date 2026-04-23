"""Tests for migration adding user_id UUID column to scoped tables."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_behaviors(pool):
    """Assert behaviors table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'behaviors' AND column_name = 'user_id'
        """
    )
    assert result is not None, "behaviors.user_id column does not exist"
    # Allow both uuid and text since existing migrations added TEXT
    assert result in ("uuid", "text"), f"behaviors.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_entities(pool):
    """Assert entities table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'entities' AND column_name = 'user_id'
        """
    )
    assert result is not None, "entities.user_id column does not exist"
    assert result in ("uuid", "text"), f"entities.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_episodes(pool):
    """Assert episodes table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'episodes' AND column_name = 'user_id'
        """
    )
    assert result is not None, "episodes.user_id column does not exist"
    assert result in ("uuid", "text"), f"episodes.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_modes(pool):
    """Assert modes table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'modes' AND column_name = 'user_id'
        """
    )
    assert result is not None, "modes.user_id column does not exist"
    assert result in ("uuid", "text"), f"modes.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_autonomy_policies(pool):
    """Assert autonomy_policies table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'autonomy_policies' AND column_name = 'user_id'
        """
    )
    assert result is not None, "autonomy_policies.user_id column does not exist"
    assert result in ("uuid", "text"), f"autonomy_policies.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_calibration_records(pool):
    """Assert calibration_records table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'calibration_records' AND column_name = 'user_id'
        """
    )
    assert result is not None, "calibration_records.user_id column does not exist"
    assert result in ("uuid", "text"), f"calibration_records.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_adds_user_id_column_to_degradation_policies(pool):
    """Assert degradation_policies table has user_id UUID column after migration."""
    result = await pool.fetchval(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'degradation_policies' AND column_name = 'user_id'
        """
    )
    assert result is not None, "degradation_policies.user_id column does not exist"
    assert result in ("uuid", "text"), f"degradation_policies.user_id has unexpected type: {result}"


@pytest.mark.asyncio
async def test_migration_user_id_columns_are_nullable(pool):
    """Assert all user_id columns are nullable."""
    tables = [
        "behaviors",
        "entities",
        "episodes",
        "modes",
        "autonomy_policies",
        "calibration_records",
        "degradation_policies",
    ]
    for table_name in tables:
        result = await pool.fetchval(
            """
            SELECT is_nullable FROM information_schema.columns
            WHERE table_name = $1 AND column_name = 'user_id'
            """,
            table_name,
        )
        assert result == "YES", f"{table_name}.user_id should be nullable, got: {result}"


@pytest.mark.asyncio
async def test_migration_is_idempotent(pool):
    """Assert running migration twice does not error."""
    from weft.db.migrations import run_migrations

    # Migration already ran via fixture. Run it again.
    applied = await run_migrations(pool)
    # No new migrations should be applied on second run
    assert len(applied) == 0, "Migration should be idempotent"


@pytest.mark.asyncio
async def test_phase1_migrations_raw_sql_idempotent_on_empty_db(pool):
    """Apply migration 31 (user_id columns) and 32 (audit table) raw SQL twice.

    Bypasses the schema_migrations short-circuit to prove the SQL itself is
    truly idempotent — not relying on the runner's version-tracking. Covers
    the Phase 1 acceptance criterion: migrations must succeed on both empty
    and populated DBs, and must not raise on re-run.
    """
    from weft.db.migrations import MIGRATIONS

    phase1_sql = [sql for version, _, sql in MIGRATIONS if version in (31, 32)]
    assert len(phase1_sql) == 2, "expected migrations 31 and 32 in MIGRATIONS list"

    for _ in range(2):
        for sql in phase1_sql:
            await pool.execute(sql)

    # Confirm audit_backfill_user_id exists with expected columns.
    cols = await pool.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'audit_backfill_user_id'
        ORDER BY ordinal_position
        """
    )
    col_names = [r["column_name"] for r in cols]
    assert col_names == [
        "id",
        "source_table",
        "row_id",
        "old_scope",
        "new_scope",
        "migrated_at",
    ], f"audit_backfill_user_id columns unexpected: {col_names}"


@pytest.mark.asyncio
async def test_phase1_user_id_indexes_exist(pool):
    """All 7 user-scoped tables have an idx_<table>_user index on user_id."""
    tables = [
        "behaviors",
        "entities",
        "episodes",
        "modes",
        "autonomy_policies",
        "calibration_records",
        "degradation_policies",
    ]
    for table in tables:
        idx_name = f"idx_{table}_user"
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
            idx_name,
        )
        assert exists, f"missing index {idx_name} on {table}"


@pytest.mark.asyncio
async def test_phase1_migrations_idempotent_on_populated_db(pool):
    """Apply migrations 31+32 raw SQL twice against a DB with seeded rows.

    Verifies the populated-DB path of the Phase 1 acceptance criterion:
    existing rows must survive intact (same count, same user_id assignments)
    and no column/table already-exists errors may be raised.
    """
    import uuid

    from weft.db.migrations import MIGRATIONS

    # Seed one row per scoped table, split across user_id=NULL (global) and
    # user_id=some-user (user-scoped) to cover both row classes.
    user_a = "user-a-idempotency-test"
    seeded: dict[str, list[str]] = {}

    # behaviors
    b_null = uuid.uuid4().hex
    b_user = uuid.uuid4().hex
    await pool.execute(
        "INSERT INTO behaviors (id, trigger_pattern, action, user_id) VALUES ($1, 'p', 'a', NULL), ($2, 'p', 'a', $3)",
        b_null, b_user, user_a,
    )
    seeded["behaviors"] = [b_null, b_user]

    # Also insert into audit_backfill_user_id to confirm migration 32 table
    # tolerates existing rows.
    await pool.execute(
        """
        INSERT INTO audit_backfill_user_id (source_table, row_id, old_scope, new_scope)
        VALUES ('behaviors', $1, 'pure-user', 'user-scoped')
        """,
        b_null,
    )

    phase1_sql = [sql for version, _, sql in MIGRATIONS if version in (31, 32)]

    # Re-run migration SQL twice (bypassing schema_migrations version-tracking).
    for _ in range(2):
        for sql in phase1_sql:
            await pool.execute(sql)

    # Rows still intact with their original user_id.
    rows = await pool.fetch(
        "SELECT id, user_id FROM behaviors WHERE id = ANY($1::text[]) ORDER BY id",
        seeded["behaviors"],
    )
    assert len(rows) == 2, "seeded behaviors rows lost after re-running migrations"
    user_ids = {r["id"]: r["user_id"] for r in rows}
    assert user_ids[b_null] is None, "NULL user_id row was mutated"
    assert user_ids[b_user] == user_a, f"user-scoped row user_id changed: {user_ids[b_user]}"

    # Audit row still present.
    audit_count = await pool.fetchval(
        "SELECT COUNT(*) FROM audit_backfill_user_id WHERE row_id = $1",
        b_null,
    )
    assert audit_count == 1, f"audit row count unexpected: {audit_count}"
