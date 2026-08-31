"""Connection-layer timeout hardening (intermittent-hang fixes).

These validate that the pool built via ``weft.db.connection.create_pool`` fails
fast instead of hanging when a query stalls (command_timeout) or the pool is
drained (acquire_timeout). Both were unbounded before and were the root cause of
intermittent ``prime``/``recall`` hangs under concurrency.
"""

from __future__ import annotations

import asyncio
import ssl
from unittest.mock import AsyncMock, call, patch

import asyncpg
import pytest

from weft.config import DatabaseConfig, WeftConfig
from weft.db.connection import (
    _pgvector_codec_init,
    acquire,
    create_pool,
    register_pgvector_codec,
)


_SUPABASE_DSN = "postgresql://weft_app:secret@db.example.supabase.co:5432/postgres"
_CA_PLACEHOLDER = "configured-ca-material"


def _config(dsn: str, **db_overrides) -> WeftConfig:
    return WeftConfig(database=DatabaseConfig(url=dsn, **db_overrides))


class TestSupabaseTLS:
    async def test_configured_ca_file_is_passed_as_cadata(self, tmp_path):
        pool = object()
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ca_file = tmp_path / "supabase-ca.crt"
        ca_file.write_text(_CA_PLACEHOLDER, encoding="utf-8")
        with (
            patch(
                "weft.db.connection.ssl.create_default_context",
                return_value=ssl_context,
            ) as create_context,
            patch(
                "weft.db.connection.asyncpg.create_pool",
                new=AsyncMock(return_value=pool),
            ),
        ):
            result = await create_pool(
                _config(_SUPABASE_DSN, ca_cert_file=ca_file)
            )

        assert result is pool
        create_context.assert_called_once_with(cadata=_CA_PLACEHOLDER)

    async def test_configured_ca_is_passed_as_cadata(self):
        pool = object()
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ssl_context.verify_flags |= ssl.VERIFY_X509_STRICT
        with (
            patch(
                "weft.db.connection.ssl.create_default_context",
                return_value=ssl_context,
            ) as create_context,
            patch(
                "weft.db.connection.asyncpg.create_pool",
                new=AsyncMock(return_value=pool),
            ) as create_pool_mock,
        ):
            result = await create_pool(
                _config(_SUPABASE_DSN, ca_cert=_CA_PLACEHOLDER)
            )

        assert result is pool
        create_context.assert_called_once_with(cadata=_CA_PLACEHOLDER)
        assert create_pool_mock.await_args.kwargs["ssl"] is ssl_context
        assert ssl_context.verify_mode is ssl.CERT_REQUIRED
        assert ssl_context.check_hostname is True
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            assert not ssl_context.verify_flags & ssl.VERIFY_X509_STRICT

    async def test_unconfigured_ca_uses_system_trust_store(self):
        pool = object()
        ssl_context = object()
        with (
            patch(
                "weft.db.connection.ssl.create_default_context",
                return_value=ssl_context,
            ) as create_context,
            patch(
                "weft.db.connection.asyncpg.create_pool",
                new=AsyncMock(return_value=pool),
            ) as create_pool_mock,
        ):
            result = await create_pool(_config(_SUPABASE_DSN))

        assert result is pool
        create_context.assert_called_once_with()
        assert create_pool_mock.await_args.kwargs["ssl"] is ssl_context

    async def test_malformed_ca_raises_clear_error(self):
        with patch(
            "weft.db.connection.ssl.create_default_context",
            side_effect=ssl.SSLError("malformed certificate"),
        ):
            with pytest.raises(
                ValueError,
                match="WEFT_DATABASE_CA_CERT is not a valid PEM certificate bundle",
            ):
                await create_pool(
                    _config(_SUPABASE_DSN, ca_cert=_CA_PLACEHOLDER)
                )


async def test_pool_init_sets_extensions_search_path_before_vector_codec():
    """Supabase pooler connections must resolve unqualified ``::vector`` casts."""
    conn = AsyncMock()

    await _pgvector_codec_init(conn)

    assert conn.mock_calls[:2] == [
        call.execute("SET search_path TO public, extensions"),
        call.set_type_codec(
            "vector",
            encoder=conn.set_type_codec.await_args.kwargs["encoder"],
            decoder=conn.set_type_codec.await_args.kwargs["decoder"],
            schema="public",
            format="text",
        ),
    ]


async def test_register_pgvector_codec_initializes_all_existing_connections(monkeypatch):
    """Every warm connection is initialized before any one is released."""
    connections = [object(), object(), object()]
    events = []

    class Pool:
        def get_size(self):
            return len(connections)

        async def acquire(self):
            connection = connections[len([event for event in events if event[0] == "acquire"])]
            events.append(("acquire", connection))
            return connection

        async def release(self, connection):
            events.append(("release", connection))

    async def init(connection):
        events.append(("init", connection))

    monkeypatch.setattr("weft.db.connection._pgvector_codec_init", init)

    await register_pgvector_codec(Pool())

    assert [event[0] for event in events] == [
        "acquire", "acquire", "acquire", "init", "init", "init",
        "release", "release", "release",
    ]
    assert [event[1] for event in events if event[0] == "init"] == connections
    assert [event[1] for event in events if event[0] == "release"] == connections


async def test_register_pgvector_codec_empty_pool_is_noop(monkeypatch):
    """A pool with no currently created connections needs no acquire/release."""
    init = AsyncMock()
    monkeypatch.setattr("weft.db.connection._pgvector_codec_init", init)

    class Pool:
        def get_size(self):
            return 0

        async def acquire(self):
            raise AssertionError("empty pool must not acquire")

        async def release(self, connection):
            raise AssertionError("empty pool must not release")

    await register_pgvector_codec(Pool())

    init.assert_not_awaited()


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
