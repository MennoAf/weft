"""RC-FL-11: durable single-worker ownership and fencing acceptance tests."""

from __future__ import annotations

import asyncio
import uuid
from urllib.parse import urlsplit

import asyncpg
import pytest

from weft.db.migrations import MIGRATIONS, verify_migration_ledger
from weft.worker_lease import (
    DEFAULT_RENEWAL_INTERVAL_SECONDS,
    LeaseLostError,
    WorkerLease,
)


@pytest.mark.asyncio
async def test_v76_is_discovered_ledger_matches_and_schema_is_named(pool):
    versions = [version for version, _, _ in MIGRATIONS]
    assert versions[-1] == 76
    assert len(versions) == len(set(versions))
    await verify_migration_ledger(pool)

    columns = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'worker_leases'
        ORDER BY ordinal_position
        """
    )
    assert [(r["column_name"], r["data_type"], r["is_nullable"]) for r in columns] == [
        ("lease_key", "text", "NO"),
        ("owner_token", "text", "YES"),
        ("generation", "bigint", "NO"),
        ("acquired_at", "timestamp with time zone", "YES"),
        ("renewed_at", "timestamp with time zone", "YES"),
        ("expires_at", "timestamp with time zone", "YES"),
    ]
    assert await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE oid = 'public.worker_leases'::regclass"
    ) is False


@pytest.mark.asyncio
async def test_acquisition_renewal_takeover_and_expiry_use_database_clock(pool):
    key = f"rc-{uuid.uuid4()}"
    first = WorkerLease(pool, key, owner_token="owner-a", lease_duration=30.0)
    second = WorkerLease(pool, key, owner_token="owner-b", lease_duration=30.0)

    acquired = await first.acquire()
    assert acquired is not None
    assert acquired.lease_key == key
    assert acquired.owner_token == "owner-a"
    assert acquired.generation == 1
    assert acquired.acquired_at.tzinfo is not None
    assert acquired.renewed_at.tzinfo is not None
    assert acquired.expires_at > acquired.renewed_at
    assert first.renewal_interval == DEFAULT_RENEWAL_INTERVAL_SECONDS
    assert await second.acquire() is None
    assert await first.is_owner()

    renewed = await first.renew()
    assert renewed is not None
    assert renewed.generation == 1
    assert renewed.expires_at > renewed.renewed_at

    await pool.execute(
        "UPDATE worker_leases SET expires_at = clock_timestamp() - interval '1 second' WHERE lease_key = $1",
        key,
    )
    taken = await second.acquire()
    assert taken is not None
    assert taken.owner_token == "owner-b"
    assert taken.generation == 2
    assert not await first.is_owner()
    assert await second.is_owner()


@pytest.mark.asyncio
async def test_atomic_acquisition_suppresses_duplicate_workers(pool):
    key = f"rc-{uuid.uuid4()}"
    leases = [WorkerLease(pool, key, owner_token=f"owner-{i}") for i in range(12)]
    results = await asyncio.gather(*(lease.acquire() for lease in leases))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].generation == 1
    assert await pool.fetchval("SELECT generation FROM worker_leases WHERE lease_key = $1", key) == 1


@pytest.mark.asyncio
async def test_fencing_refuses_side_effect_after_takeover(pool):
    key = f"rc-{uuid.uuid4()}"
    old = WorkerLease(pool, key, owner_token="old-owner", lease_duration=30.0)
    new = WorkerLease(pool, key, owner_token="new-owner", lease_duration=30.0)
    await old.acquire()
    calls: list[str] = []

    async def side_effect():
        calls.append("old")

    await old.run_fenced(side_effect)
    assert calls == ["old"]

    await pool.execute(
        "UPDATE worker_leases SET expires_at = clock_timestamp() - interval '1 second' WHERE lease_key = $1",
        key,
    )
    await new.acquire()
    with pytest.raises(LeaseLostError):
        await old.run_fenced(side_effect)
    assert calls == ["old"]
    assert not await old.release()
    assert await new.release()


@pytest.mark.asyncio
async def test_restricted_role_has_no_implicit_lease_or_ddl_privilege(pool, pg_dsn):
    role = f"rc_lease_{uuid.uuid4().hex[:12]}"
    password = uuid.uuid4().hex
    db_name = await pool.fetchval("SELECT current_database()")
    await pool.execute(
        f'CREATE ROLE "{role}" LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB '
        f'NOCREATEROLE NOINHERIT PASSWORD $${password}$$'
    )
    await pool.execute(f'GRANT CONNECT ON DATABASE "{db_name}" TO "{role}"')
    await pool.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
    parsed = urlsplit(pg_dsn)
    try:
        restricted = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path.lstrip("/"),
            user=role,
            password=password,
        )
        try:
            assert await restricted.fetchval("SELECT has_table_privilege(current_user, 'public.worker_leases', 'SELECT')") is False
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await WorkerLease(restricted, "restricted", owner_token="nope").acquire()
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await restricted.execute("CREATE TABLE public.rc_lease_ddl_probe (id integer)")
        finally:
            await restricted.close()
    finally:
        await pool.execute(f'REVOKE ALL ON SCHEMA public FROM "{role}"')
        await pool.execute(f'REVOKE CONNECT ON DATABASE "{db_name}" FROM "{role}"')
        await pool.execute(f'DROP ROLE IF EXISTS "{role}"')


@pytest.mark.asyncio
async def test_release_is_atomic_and_does_not_release_takeover(pool):
    key = f"rc-{uuid.uuid4()}"
    old = WorkerLease(pool, key, owner_token="old")
    new = WorkerLease(pool, key, owner_token="new")
    await old.acquire()
    await pool.execute(
        "UPDATE worker_leases SET expires_at = clock_timestamp() - interval '1 second' WHERE lease_key = $1",
        key,
    )
    await new.acquire()
    assert not await old.release()
    row = await pool.fetchrow("SELECT owner_token, generation FROM worker_leases WHERE lease_key = $1", key)
    assert row["owner_token"] == "new"
    assert row["generation"] == 2
