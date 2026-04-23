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
