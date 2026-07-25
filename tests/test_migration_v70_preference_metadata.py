from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_v70_preference_metadata_migration_is_registered_and_idempotent(pool):
    from weft.db.migrations import MIGRATIONS

    sql = [sql for version, _, sql in MIGRATIONS if version == 70]
    assert len(sql) == 1
    await pool.execute(sql[0])
    await pool.execute(sql[0])

    column = await pool.fetchrow(
        """
        SELECT is_nullable, column_default, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'memories'
          AND column_name = 'preference_metadata'
        """
    )
    assert column is not None
    assert column["is_nullable"] == "YES"
    assert column["column_default"] is None
    assert column["data_type"] == "jsonb"
