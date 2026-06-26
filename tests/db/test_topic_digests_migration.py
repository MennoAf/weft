"""Tests for migration 57: topic_digests cache table.

done_when assertions:
  (1) Table topic_digests exists with columns:
        digest_id (TEXT, PK), user_id (NOT NULL), topic, scope (default 'global'),
        content, provenance (jsonb), generated_at (timestamptz),
        stale (bool, default false), detector_version.
  (2) A UNIQUE index on (user_id, topic, scope) exists.
  (3) RLS is enabled on topic_digests with a SELECT policy admitting
        user_id = current app.user_id.
"""

from __future__ import annotations

import asyncpg
import pytest


# ---------------------------------------------------------------------------
# (1) topic_digests table schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_digests_table_exists(pool):
    """Table topic_digests exists and has columns."""
    cols = await pool.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'topic_digests'
        ORDER BY ordinal_position
        """
    )
    assert cols, "table topic_digests does not exist or has no columns"


@pytest.mark.asyncio
async def test_topic_digests_columns(pool):
    """topic_digests has the required columns with correct types and constraints."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'topic_digests'
        ORDER BY ordinal_position
        """
    )
    assert cols, "table topic_digests does not exist or has no columns"

    by_name = {r["column_name"]: r for r in cols}

    required = {
        "digest_id", "user_id", "topic", "scope",
        "content", "provenance", "generated_at", "stale", "detector_version",
    }
    assert required.issubset(by_name), (
        f"Missing columns: {required - set(by_name)}"
    )

    # digest_id — TEXT, PK (not null as PK, checked separately)
    assert by_name["digest_id"]["data_type"] == "text"
    assert by_name["digest_id"]["is_nullable"] == "NO"

    # user_id — TEXT, NOT NULL
    assert by_name["user_id"]["data_type"] == "text"
    assert by_name["user_id"]["is_nullable"] == "NO"

    # topic — TEXT
    assert by_name["topic"]["data_type"] == "text"

    # scope — TEXT, default 'global'
    assert by_name["scope"]["data_type"] == "text"
    assert by_name["scope"]["is_nullable"] == "NO"
    assert by_name["scope"]["column_default"] is not None, (
        "scope must have a DEFAULT value"
    )
    assert "global" in str(by_name["scope"]["column_default"]), (
        f"scope default should be 'global', got: {by_name['scope']['column_default']}"
    )

    # content — TEXT (nullable — digest may not yet be synthesized)
    assert by_name["content"]["data_type"] == "text"

    # provenance — JSONB (nullable — not yet synthesized)
    assert by_name["provenance"]["data_type"] == "jsonb", (
        f"provenance must be jsonb, got: {by_name['provenance']['data_type']}"
    )

    # generated_at — TIMESTAMPTZ, NOT NULL
    assert by_name["generated_at"]["data_type"] == "timestamp with time zone"
    assert by_name["generated_at"]["is_nullable"] == "NO"

    # stale — boolean, NOT NULL, default false
    assert by_name["stale"]["data_type"] == "boolean"
    assert by_name["stale"]["is_nullable"] == "NO"
    assert by_name["stale"]["column_default"] is not None, (
        "stale must have a DEFAULT value"
    )
    assert "false" in str(by_name["stale"]["column_default"]).lower(), (
        f"stale default should be false, got: {by_name['stale']['column_default']}"
    )

    # detector_version — TEXT
    assert by_name["detector_version"]["data_type"] == "text"


@pytest.mark.asyncio
async def test_topic_digests_primary_key(pool):
    """digest_id is the primary key."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'topic_digests'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    pk_cols = [r["column_name"] for r in rows]
    assert pk_cols == ["digest_id"], (
        f"Expected PK (digest_id), got: {pk_cols}"
    )


# ---------------------------------------------------------------------------
# (2) UNIQUE index on (user_id, topic, scope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_digests_unique_index_exists(pool):
    """A UNIQUE index on (user_id, topic, scope) exists."""
    row = await pool.fetchrow(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'topic_digests'
          AND indexname = 'idx_topic_digests_user_topic_scope'
        """
    )
    assert row is not None, (
        "UNIQUE index idx_topic_digests_user_topic_scope is missing from pg_indexes"
    )
    assert "unique" in row["indexdef"].lower(), (
        f"Expected UNIQUE index but got: {row['indexdef']}"
    )


@pytest.mark.asyncio
async def test_topic_digests_unique_index_enforced(pool):
    """Inserting a duplicate (user_id, topic, scope) raises UniqueViolationError."""
    user_id = "test-user-default"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_digests
                    (digest_id, user_id, topic, scope, detector_version)
                VALUES ('td-first', $1, 'weft', 'global', 'v1')
                """,
                user_id,
            )
            with pytest.raises(asyncpg.UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO topic_digests
                        (digest_id, user_id, topic, scope, detector_version)
                    VALUES ('td-second', $1, 'weft', 'global', 'v1')
                    """,
                    user_id,
                )


@pytest.mark.asyncio
async def test_topic_digests_scope_default(pool):
    """scope defaults to 'global' when not supplied."""
    user_id = "test-user-default"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_digests
                    (digest_id, user_id, topic, detector_version)
                VALUES ('td-scope-default', $1, 'loom', 'v1')
                """,
                user_id,
            )

    scope = await pool.fetchval(
        "SELECT scope FROM topic_digests WHERE digest_id = 'td-scope-default'"
    )
    assert scope == "global", f"scope default should be 'global', got: {scope}"


@pytest.mark.asyncio
async def test_topic_digests_stale_default(pool):
    """stale defaults to false when not supplied."""
    user_id = "test-user-default"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_digests
                    (digest_id, user_id, topic, detector_version)
                VALUES ('td-stale-default', $1, 'muttr', 'v1')
                """,
                user_id,
            )

    stale = await pool.fetchval(
        "SELECT stale FROM topic_digests WHERE digest_id = 'td-stale-default'"
    )
    assert stale is False, f"stale default should be false, got: {stale}"


@pytest.mark.asyncio
async def test_topic_digests_provenance_jsonb(pool):
    """provenance column stores and retrieves JSONB correctly."""
    user_id = "test-user-default"
    provenance = {"mem-abc": ["span1", "span2"], "mem-xyz": ["span3"]}

    async with pool.acquire() as conn:
        import json
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO topic_digests
                    (digest_id, user_id, topic, provenance, detector_version)
                VALUES ('td-provenance', $1, 'delphi', $2::jsonb, 'v1')
                """,
                user_id,
                json.dumps(provenance),
            )

    stored = await pool.fetchval(
        "SELECT provenance FROM topic_digests WHERE digest_id = 'td-provenance'"
    )
    assert stored is not None, "provenance should not be None"
    assert "mem-abc" in stored, f"Expected mem-abc in provenance, got: {stored}"


# ---------------------------------------------------------------------------
# (3) RLS on topic_digests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_digests_rls_enabled(pool):
    """RLS is enabled on topic_digests."""
    enabled = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'topic_digests'"
    )
    assert enabled is True, "RLS is not enabled on topic_digests"


@pytest.mark.asyncio
async def test_topic_digests_select_policy_exists(pool):
    """A SELECT policy exists on topic_digests."""
    row = await pool.fetchrow(
        """
        SELECT polname, polcmd, polqual
        FROM pg_policy
        WHERE polrelid = 'topic_digests'::regclass
          AND polcmd = 'r'  -- 'r' = SELECT
        """
    )
    assert row is not None, "No SELECT policy found on topic_digests"
    assert row["polname"] == "topic_digests_select"


@pytest.mark.asyncio
async def test_topic_digests_select_policy_using_clause(pool):
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
        WHERE polrelid = 'topic_digests'::regclass
          AND polcmd = 'r'
        """
    )
    assert row is not None, "SELECT policy missing"
    using_expr = row["using_expr"] or ""
    assert "app.user_id" in using_expr, (
        f"SELECT policy USING clause does not reference app.user_id: {using_expr}"
    )


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v57_migration_idempotent(pool):
    """Running migration 57 SQL twice does not raise an error."""
    from weft.db.migrations import MIGRATIONS

    v57_sql = [sql for version, _, sql in MIGRATIONS if version == 57]
    assert len(v57_sql) == 1, "expected migration 57 in MIGRATIONS list"

    # Migration already ran via the pool fixture. Re-running must be a no-op.
    await pool.execute(v57_sql[0])
