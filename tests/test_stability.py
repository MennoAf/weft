"""Tests for stability improvements: retry, keepalive, fallback refresh, migration locking."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from weft.mcp.server import _connect_with_retry


# --- Startup retry ---


class TestConnectWithRetry:
    """Tests for _connect_with_retry exponential backoff."""

    async def test_succeeds_first_try(self):
        fn = AsyncMock(return_value="ok")
        result = await _connect_with_retry(fn, "Test", max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    async def test_succeeds_after_failures(self):
        fn = AsyncMock(side_effect=[ConnectionError("down"), ConnectionError("down"), "ok"])
        result = await _connect_with_retry(fn, "Test", max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 3

    async def test_raises_after_max_retries(self):
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError, match="down"):
            await _connect_with_retry(fn, "Test", max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # initial + 2 retries

    async def test_zero_retries_raises_immediately(self):
        fn = AsyncMock(side_effect=ConnectionError("nope"))
        with pytest.raises(ConnectionError):
            await _connect_with_retry(fn, "Test", max_retries=0, base_delay=0.01)
        assert fn.call_count == 1


# Content fallback is intentionally disabled in the multi-user MCP server.
# The standalone ``weft export`` command remains covered by tests/test_fallback.py.


# --- Migration advisory lock ---


class TestMigrationLocking:
    """Tests for advisory-lock-based migration serialization."""

    async def test_migrations_use_advisory_lock(self, pool):
        """Verify the advisory lock is acquired and released during migrations."""
        from weft.db.migrations import _MIGRATION_LOCK_ID, run_migrations

        # Run migrations (already applied, so no-op but lock should still work)
        applied = await run_migrations(pool)

        # Verify lock is released by trying to acquire it
        async with pool.acquire() as conn:
            locked = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", _MIGRATION_LOCK_ID,
            )
            assert locked, "Advisory lock should be available after migrations complete"
            # Release it
            await conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID,
            )

    async def test_concurrent_migrations_serialize(self, pool):
        """Two concurrent migration runs don't cause conflicts."""
        from weft.db.migrations import run_migrations

        # Run two migration calls concurrently — should not raise
        results = await asyncio.gather(
            run_migrations(pool),
            run_migrations(pool),
        )
        # Both should succeed (idempotent)
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)

    async def test_on_conflict_do_nothing(self, pool):
        """Tracking inserts use ON CONFLICT DO NOTHING."""
        from weft.db.migrations import run_migrations

        # Run twice — second run should not raise on duplicate PK
        await run_migrations(pool)
        applied = await run_migrations(pool)
        assert applied == []  # nothing new to apply
