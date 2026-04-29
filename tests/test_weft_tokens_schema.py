"""Tests for migration 40 — weft_tokens table schema.

L1 of Phase 2.5 (credential-bound caller mode). The table itself; the
issuance / lookup module is L2. These tests assert the schema exists
with the columns, types, constraints, and indexes the spec calls for —
nothing here exercises the application layer.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_weft_tokens_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'weft_tokens'
              AND table_schema = current_schema()
        )
        """
    )
    assert exists, "weft_tokens table missing after migrations"


@pytest.mark.asyncio
async def test_weft_tokens_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'weft_tokens'
        ORDER BY ordinal_position
        """
    )
    cols = {r["column_name"]: r for r in rows}

    expected = {
        "token_hash":   ("text", "NO"),
        "user_id":      ("text", "NO"),
        "caller_mode":  ("text", "NO"),
        "label":        ("text", "YES"),
        "created_at":   ("timestamp with time zone", "NO"),
        "last_used_at": ("timestamp with time zone", "YES"),
        "expires_at":   ("timestamp with time zone", "YES"),
        "revoked_at":   ("timestamp with time zone", "YES"),
    }
    assert set(cols) == set(expected), f"column set mismatch: {set(cols) ^ set(expected)}"
    for name, (dtype, nullable) in expected.items():
        assert cols[name]["data_type"] == dtype, (
            f"{name} type {cols[name]['data_type']!r} != expected {dtype!r}"
        )
        assert cols[name]["is_nullable"] == nullable, (
            f"{name} nullable {cols[name]['is_nullable']!r} != expected {nullable!r}"
        )

    assert cols["created_at"]["column_default"], "created_at missing default"


@pytest.mark.asyncio
async def test_weft_tokens_primary_key_on_token_hash(pool):
    row = await pool.fetchrow(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'weft_tokens'::regclass AND i.indisprimary
        """
    )
    assert row is not None, "weft_tokens has no primary key"
    assert row["column_name"] == "token_hash"


@pytest.mark.asyncio
async def test_weft_tokens_caller_mode_check_rejects_invalid(pool):
    await pool.execute(
        """
        INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
        VALUES ('h-supervisor', 'u-1', 'supervisor')
        """
    )
    await pool.execute(
        """
        INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
        VALUES ('h-agent', 'u-1', 'agent')
        """
    )

    import asyncpg

    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            """
            INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
            VALUES ('h-bogus', 'u-1', 'admin')
            """
        )


@pytest.mark.asyncio
async def test_weft_tokens_indexes_present(pool):
    rows = await pool.fetch(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = 'weft_tokens' AND schemaname = current_schema()
        """
    )
    names = {r["indexname"] for r in rows}
    assert "idx_weft_tokens_user_active" in names, names
    assert "idx_weft_tokens_expires" in names, names


@pytest.mark.asyncio
async def test_weft_tokens_user_active_index_is_partial(pool):
    """The user_active index filters revoked_at IS NULL, so revoked rows
    don't bloat lookups for a user's live tokens."""
    pred = await pool.fetchval(
        """
        SELECT pg_get_expr(indpred, indrelid)
        FROM pg_index
        WHERE indexrelid = 'idx_weft_tokens_user_active'::regclass
        """
    )
    assert pred is not None, "idx_weft_tokens_user_active is not a partial index"
    assert "revoked_at" in pred.lower() and "null" in pred.lower(), pred


@pytest.mark.asyncio
async def test_weft_tokens_expires_index_is_partial(pool):
    pred = await pool.fetchval(
        """
        SELECT pg_get_expr(indpred, indrelid)
        FROM pg_index
        WHERE indexrelid = 'idx_weft_tokens_expires'::regclass
        """
    )
    assert pred is not None, "idx_weft_tokens_expires is not a partial index"
    assert "expires_at" in pred.lower() and "not null" in pred.lower(), pred


@pytest.mark.asyncio
async def test_migration_40_idempotent_on_populated_db(pool):
    """Re-running migration 40's SQL on a DB that already has the table
    must not error. Belt-and-suspenders alongside the schema_migrations
    short-circuit, since the runner serializes via advisory lock but
    individual statements still need to be idempotent on their own."""
    from weft.db.migrations import MIGRATIONS

    sql = next(s for v, _, s in MIGRATIONS if v == 40)
    await pool.execute(sql)
    await pool.execute(sql)

    cols = await pool.fetchval(
        """
        SELECT count(*) FROM information_schema.columns
        WHERE table_name = 'weft_tokens'
        """
    )
    assert cols == 8


@pytest.mark.asyncio
async def test_token_hash_is_unique(pool):
    """PK on token_hash means duplicate inserts blow up — this is the
    invariant lookup_token() will rely on at L2."""
    await pool.execute(
        """
        INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
        VALUES ('h-dup', 'u-1', 'supervisor')
        """
    )
    import asyncpg

    with pytest.raises(asyncpg.UniqueViolationError):
        await pool.execute(
            """
            INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
            VALUES ('h-dup', 'u-2', 'agent')
            """
        )
