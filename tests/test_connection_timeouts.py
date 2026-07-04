"""Connection-layer timeout hardening (intermittent-hang fixes).

These validate that the pool built via ``weft.db.connection.create_pool`` fails
fast instead of hanging when a query stalls (command_timeout) or the pool is
drained (acquire_timeout). Both were unbounded before and were the root cause of
intermittent ``prime``/``recall`` hangs under concurrency.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from weft.config import DatabaseConfig, WeftConfig
from weft.db.connection import acquire, create_pool


def _config(dsn: str, **db_overrides) -> WeftConfig:
    return WeftConfig(database=DatabaseConfig(url=dsn, **db_overrides))


class TestCommandTimeout:
    async def test_slow_query_is_cancelled_not_hung(self, pg_dsn):
        """A query exceeding command_timeout raises instead of blocking forever."""
        pool = await create_pool(
            _config(pg_dsn, command_timeout=0.5, pool_min_size=1, pool_max_size=2)
        )
        try:
            with pytest.raises(
                (asyncpg.QueryCanceledError, asyncio.TimeoutError)
            ):
                # pg_sleep(3) >> 0.5s ceiling — must be cancelled quickly.
                await asyncio.wait_for(
                    pool.fetchval("SELECT pg_sleep(3)"), timeout=2.0
                )
        finally:
            await pool.close()

    async def test_fast_query_unaffected(self, pg_dsn):
        pool = await create_pool(
            _config(pg_dsn, command_timeout=5.0, pool_min_size=1, pool_max_size=2)
        )
        try:
            assert await pool.fetchval("SELECT 1") == 1
        finally:
            await pool.close()

    async def test_per_call_timeout_overrides_pool_ceiling(self, pg_dsn):
        """Migrations rely on this: a positive per-call timeout must override the
        pool's short command_timeout so slow DDL (HNSW index builds) isn't
        cancelled at the request-path ceiling."""
        pool = await create_pool(
            _config(pg_dsn, command_timeout=0.5, pool_min_size=1, pool_max_size=2)
        )
        try:
            async with pool.acquire() as conn:
                # Pool default (0.5s) would cancel this; the explicit 5s wins.
                await conn.execute("SELECT pg_sleep(1)", timeout=5.0)
        finally:
            await pool.close()


class TestAcquireTimeout:
    async def test_drained_pool_fails_fast(self, pg_dsn):
        """When every connection is checked out, acquire() raises promptly
        instead of blocking indefinitely."""
        pool = await create_pool(
            _config(
                pg_dsn,
                acquire_timeout=0.5,
                pool_min_size=1,
                pool_max_size=1,
            )
        )
        try:
            # Hold the sole connection via the raw pool so the _current_conn
            # contextvar stays unset — otherwise acquire()'s idempotent reuse
            # would just hand back the same connection instead of contending.
            async with pool.acquire():
                start = asyncio.get_event_loop().time()
                with pytest.raises(asyncio.TimeoutError):
                    async with acquire(pool):
                        pass
                elapsed = asyncio.get_event_loop().time() - start
                # Bounded by acquire_timeout (0.5s), nowhere near a hang.
                assert elapsed < 2.0
        finally:
            await pool.close()

    async def test_available_connection_acquired_immediately(self, pg_dsn):
        pool = await create_pool(
            _config(pg_dsn, acquire_timeout=0.5, pool_min_size=1, pool_max_size=3)
        )
        try:
            async with acquire(pool) as conn:
                assert await conn.fetchval("SELECT 1") == 1
        finally:
            await pool.close()
