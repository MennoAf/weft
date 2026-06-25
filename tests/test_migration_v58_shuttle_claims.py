"""Tests for migration 58: shuttle_claims table.

Schema + invariant tests mirror the v48 (belief_claims) pattern. The
load-bearing invariants for Shuttle are: (1) single-active-claim-per-attribute
(the blackboard's "current value" guarantee) and (2) supersession lineage
(so a fresh observer run cleanly replaces the prior value). Both are exercised
here against a real Postgres via the testcontainers ``pool`` fixture.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

_USER = "test-user-default"


def _claim_id() -> str:
    return f"sc-{uuid.uuid4().hex[:10]}"


async def _insert_claim(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    attribute: str,
    loop_id: str = "office.today",
    status: str = "active",
    occurred_at: datetime | None = None,
    scope: str = "global",
    value: str = '{"v": 1}',
    error_at: datetime | None = None,
    inputs_hash: str | None = None,
) -> str:
    """Low-level INSERT into shuttle_claims inside the current transaction.

    Caller must have issued ``SET LOCAL app.user_id = ...`` so both the GUC
    default and the RLS INSERT policy are satisfied.
    """
    claim_id = _claim_id()
    if occurred_at is None:
        occurred_at = datetime.now(timezone.utc)
    await conn.execute(
        """
        INSERT INTO shuttle_claims (
            claim_id, user_id, loop_id, attribute, value, scope,
            status, occurred_at, error_at, inputs_hash
        ) VALUES (
            $1, $2, $3, $4, $5::jsonb, $6,
            $7, $8, $9, $10
        )
        """,
        claim_id, user_id, loop_id, attribute, value, scope,
        status, occurred_at, error_at, inputs_hash,
    )
    return claim_id


@pytest.mark.asyncio
async def test_shuttle_claims_table_exists(pool):
    """shuttle_claims table is created with exactly the expected columns."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'shuttle_claims'
        ORDER BY ordinal_position
        """
    )
    by_name = {r["column_name"]: r for r in cols}
    expected = {
        "claim_id", "user_id", "loop_id", "attribute", "value", "scope",
        "superseded_by", "status", "created_at", "occurred_at",
        "error_at", "inputs_hash",
    }
    assert set(by_name) == expected

    assert by_name["claim_id"]["data_type"] == "text"
    assert by_name["claim_id"]["is_nullable"] == "NO"
    assert by_name["user_id"]["is_nullable"] == "NO"
    assert by_name["loop_id"]["is_nullable"] == "NO"
    assert by_name["attribute"]["is_nullable"] == "NO"
    assert by_name["value"]["data_type"] == "jsonb"
    assert by_name["value"]["is_nullable"] == "NO"
    assert by_name["status"]["is_nullable"] == "NO"
    assert by_name["occurred_at"]["data_type"] == "timestamp with time zone"
    assert by_name["occurred_at"]["is_nullable"] == "NO"
    # NET-NEW vs belief_claims, both nullable.
    assert by_name["error_at"]["data_type"] == "timestamp with time zone"
    assert by_name["error_at"]["is_nullable"] == "YES"
    assert by_name["inputs_hash"]["data_type"] == "text"
    assert by_name["inputs_hash"]["is_nullable"] == "YES"


@pytest.mark.asyncio
async def test_shuttle_claims_primary_key(pool):
    """PK is claim_id."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'shuttle_claims'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    assert [r["column_name"] for r in rows] == ["claim_id"]


@pytest.mark.asyncio
async def test_shuttle_claims_indexes_exist(pool):
    """All three named indexes are present."""
    for idx_name in (
        "idx_shuttle_claims_current",
        "idx_shuttle_claims_chain",
        "idx_shuttle_claims_loop",
    ):
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
            idx_name,
        )
        assert exists, f"missing index {idx_name}"


@pytest.mark.asyncio
async def test_shuttle_claims_partial_unique_active(pool):
    """Two 'active' rows for the same (user, attribute, scope) raise UniqueViolation.

    Superseding the first lets the second succeed — the partial unique index
    only covers status = 'active'. This is the single-active blackboard invariant.
    """
    now = datetime.now(timezone.utc)

    cid1 = ""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            cid1 = await _insert_claim(
                conn, user_id=_USER, attribute="office.today",
                occurred_at=now - timedelta(hours=2),
            )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            with pytest.raises(asyncpg.UniqueViolationError):
                await _insert_claim(
                    conn, user_id=_USER, attribute="office.today",
                    occurred_at=now,
                )

    # Supersede the first, then the second active insert must succeed.
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            await conn.execute(
                "UPDATE shuttle_claims SET status = 'superseded' WHERE claim_id = $1",
                cid1,
            )
            await _insert_claim(
                conn, user_id=_USER, attribute="office.today",
                value='{"v": 2}', occurred_at=now,
            )

    count = await pool.fetchval(
        "SELECT count(*) FROM shuttle_claims WHERE attribute = 'office.today'",
    )
    assert count == 2


@pytest.mark.asyncio
async def test_shuttle_claims_supersession_lineage(pool):
    """superseded_by forms a valid FK chain old -> new; exactly one stays active."""
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            old = await _insert_claim(
                conn, user_id=_USER, attribute="weather.today",
                occurred_at=now - timedelta(hours=1),
            )
            await conn.execute(
                "UPDATE shuttle_claims SET status = 'superseded' WHERE claim_id = $1",
                old,
            )
            new = await _insert_claim(
                conn, user_id=_USER, attribute="weather.today",
                value='{"high_c": 24}', occurred_at=now,
            )
            await conn.execute(
                "UPDATE shuttle_claims SET superseded_by = $1 WHERE claim_id = $2",
                new, old,
            )

    # The forward pointer resolves, and exactly one active claim remains.
    pointer = await pool.fetchval(
        "SELECT superseded_by FROM shuttle_claims WHERE claim_id = $1", old,
    )
    assert pointer == new
    active = await pool.fetch(
        "SELECT claim_id FROM shuttle_claims "
        "WHERE attribute = 'weather.today' AND status = 'active'",
    )
    assert [r["claim_id"] for r in active] == [new]


@pytest.mark.asyncio
async def test_shuttle_claims_status_check(pool):
    """Inserting with an invalid status raises CheckViolationError."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert_claim(
                    conn, user_id=_USER, attribute="office.today", status="bogus",
                )


@pytest.mark.asyncio
async def test_shuttle_claims_superseded_by_fk(pool):
    """superseded_by referencing a non-existent claim raises ForeignKeyViolation."""
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{_USER}'")
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await conn.execute(
                    """
                    INSERT INTO shuttle_claims (
                        claim_id, user_id, loop_id, attribute, value, scope,
                        status, occurred_at, superseded_by
                    ) VALUES (
                        $1, $2, 'office.today', 'fk.sentinel', '{"v":1}'::jsonb,
                        'global', 'active', $3, 'sc-does-not-exist'
                    )
                    """,
                    _claim_id(), _USER, now,
                )


@pytest.mark.asyncio
async def test_shuttle_claims_rls_enabled(pool):
    """RLS is enabled on shuttle_claims."""
    enabled = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'shuttle_claims'"
    )
    assert enabled is True


@pytest.mark.asyncio
async def test_v58_migration_idempotent(pool):
    """Running migration 58 SQL twice does not error."""
    from weft.db.migrations import MIGRATIONS

    v58_sql = [sql for version, _, sql in MIGRATIONS if version == 58]
    assert len(v58_sql) == 1, "expected migration 58 in MIGRATIONS list"
    await pool.execute(v58_sql[0])
