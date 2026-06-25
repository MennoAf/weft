"""Tests for migration 56: GIN index on memories(topic[]) + topic_resolution_aliases.

done_when assertions:
  (1) A GIN index on memories(topic) exists in pg_indexes.
  (2) Table topic_resolution_aliases exists with columns:
        user_id, topic_token, resolved_tags(text[]), hit_count(int default 0),
        source(check in 'learned','manual'), updated_at,
        PK(user_id, topic_token).
  (3) RLS is enabled on topic_resolution_aliases with a SELECT policy
        admitting user_id = current app.user_id.
"""

from __future__ import annotations

import asyncpg
import pytest


# ---------------------------------------------------------------------------
# (1) GIN index on memories(topic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memories_topic_gin_index_exists(pool):
    """A GIN index named idx_memories_topic_gin exists on memories(topic)."""
    row = await pool.fetchrow(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'memories'
          AND indexname = 'idx_memories_topic_gin'
        """
    )
    assert row is not None, (
        "GIN index idx_memories_topic_gin is missing from pg_indexes"
    )
    # Confirm it is actually a GIN index on the topic column.
    assert "gin" in row["indexdef"].lower(), (
        f"Expected GIN index but got: {row['indexdef']}"
    )
    assert "topic" in row["indexdef"], (
        f"GIN index does not reference 'topic' column: {row['indexdef']}"
    )


# ---------------------------------------------------------------------------
# (2) topic_resolution_aliases table schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_resolution_aliases_columns(pool):
    """topic_resolution_aliases has the required columns with correct types."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'topic_resolution_aliases'
        ORDER BY ordinal_position
        """
    )
    assert cols, "table topic_resolution_aliases does not exist or has no columns"

    by_name = {r["column_name"]: r for r in cols}

    required = {
        "user_id", "topic_token", "resolved_tags",
        "hit_count", "source", "updated_at",
    }
    assert required.issubset(by_name), (
        f"Missing columns: {required - set(by_name)}"
    )

    # user_id
    assert by_name["user_id"]["data_type"] == "text"
    assert by_name["user_id"]["is_nullable"] == "NO"

    # topic_token
    assert by_name["topic_token"]["data_type"] == "text"
    assert by_name["topic_token"]["is_nullable"] == "NO"

    # resolved_tags — Postgres reports array types as "ARRAY"
    assert by_name["resolved_tags"]["data_type"] in ("ARRAY", "text[]"), (
        f"resolved_tags data_type should be ARRAY, got: {by_name['resolved_tags']['data_type']}"
    )
    assert by_name["resolved_tags"]["is_nullable"] == "NO"

    # hit_count — integer, default 0
    assert by_name["hit_count"]["data_type"] == "integer"
    assert by_name["hit_count"]["is_nullable"] == "NO"
    assert by_name["hit_count"]["column_default"] is not None, (
        "hit_count must have a DEFAULT value"
    )
    assert "0" in str(by_name["hit_count"]["column_default"]), (
        f"hit_count default should be 0, got: {by_name['hit_count']['column_default']}"
    )

    # source
    assert by_name["source"]["data_type"] == "text"
    assert by_name["source"]["is_nullable"] == "NO"

    # updated_at
    assert by_name["updated_at"]["data_type"] == "timestamp with time zone"
    assert by_name["updated_at"]["is_nullable"] == "NO"


@pytest.mark.asyncio
async def test_topic_resolution_aliases_primary_key(pool):
    """Primary key is (user_id, topic_token)."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'topic_resolution_aliases'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    pk_cols = [r["column_name"] for r in rows]
    assert pk_cols == ["user_id", "topic_token"], (
        f"Expected PK (user_id, topic_token), got: {pk_cols}"
    )


@pytest.mark.asyncio
async def test_topic_resolution_aliases_source_check(pool):
    """Inserting source='bogus' raises CheckViolationError."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL app.user_id = 'test-user-default'")
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO topic_resolution_aliases
                        (user_id, topic_token, resolved_tags, source)
                    VALUES ('test-user-default', 'weft', '{weft}', 'bogus')
                    """
                )


@pytest.mark.asyncio
async def test_topic_resolution_aliases_source_values_valid(pool):
    """Inserting source='learned' and source='manual' both succeed."""
    user_id = "test-user-default"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_resolution_aliases
                    (user_id, topic_token, resolved_tags, source)
                VALUES ($1, 'test-token-learned', '{tag-a}', 'learned')
                """,
                user_id,
            )
            await conn.execute(
                """
                INSERT INTO topic_resolution_aliases
                    (user_id, topic_token, resolved_tags, source)
                VALUES ($1, 'test-token-manual', '{tag-b}', 'manual')
                """,
                user_id,
            )
    # Verify both rows landed
    count = await pool.fetchval(
        """
        SELECT count(*) FROM topic_resolution_aliases
        WHERE user_id = $1
          AND topic_token IN ('test-token-learned', 'test-token-manual')
        """,
        user_id,
    )
    assert count == 2


@pytest.mark.asyncio
async def test_topic_resolution_aliases_hit_count_default(pool):
    """hit_count defaults to 0 when not supplied."""
    user_id = "test-user-default"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_resolution_aliases
                    (user_id, topic_token, resolved_tags, source)
                VALUES ($1, 'hc-default-token', '{x}', 'manual')
                """,
                user_id,
            )

    hit_count = await pool.fetchval(
        "SELECT hit_count FROM topic_resolution_aliases WHERE topic_token = 'hc-default-token'"
    )
    assert hit_count == 0


# ---------------------------------------------------------------------------
# (3) RLS on topic_resolution_aliases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_resolution_aliases_rls_enabled(pool):
    """RLS is enabled on topic_resolution_aliases."""
    enabled = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'topic_resolution_aliases'"
    )
    assert enabled is True, "RLS is not enabled on topic_resolution_aliases"


@pytest.mark.asyncio
async def test_topic_resolution_aliases_select_policy_exists(pool):
    """A SELECT policy exists on topic_resolution_aliases."""
    row = await pool.fetchrow(
        """
        SELECT polname, polcmd, polqual
        FROM pg_policy
        WHERE polrelid = 'topic_resolution_aliases'::regclass
          AND polcmd = 'r'  -- 'r' = SELECT
        """
    )
    assert row is not None, (
        "No SELECT policy found on topic_resolution_aliases"
    )
    assert row["polname"] == "topic_resolution_aliases_select"


@pytest.mark.asyncio
async def test_topic_resolution_aliases_select_policy_using_clause(pool):
    """SELECT policy USING clause references app.user_id (the admission predicate).

    The test pool connects as a superuser which bypasses RLS, so row-level
    isolation cannot be demonstrated here without a restricted role (see
    test_rls_pentest.py for the restricted-role pattern). Instead this test
    asserts the structural contract: the SELECT policy's USING expression
    references nullif(current_setting('app.user_id', ...), '') so the
    runtime predicate is correct.
    """
    row = await pool.fetchrow(
        """
        SELECT polname, pg_get_expr(polqual, polrelid) AS using_expr
        FROM pg_policy
        WHERE polrelid = 'topic_resolution_aliases'::regclass
          AND polcmd = 'r'
        """
    )
    assert row is not None, "SELECT policy missing"
    using_expr = row["using_expr"] or ""
    # The policy must reference app.user_id via current_setting
    assert "app.user_id" in using_expr, (
        f"SELECT policy USING clause does not reference app.user_id: {using_expr}"
    )


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v56_migration_idempotent(pool):
    """Running migration 56 SQL twice does not raise an error."""
    from weft.db.migrations import MIGRATIONS

    v56_sql = [sql for version, _, sql in MIGRATIONS if version == 56]
    assert len(v56_sql) == 1, "expected migration 56 in MIGRATIONS list"

    # Migration already ran via the pool fixture. Re-running must be a no-op.
    await pool.execute(v56_sql[0])
