"""Tests for connection-level auth context (SET LOCAL app.user_id)."""

from __future__ import annotations

import pytest

from weft.auth import current_user_id
from weft.db.connection import acquire, set_user_context


class TestSetUserContext:
    async def test_sets_user_id_when_present(self, pool):
        tok = current_user_id.set("user-abc-123")
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await set_user_context(conn)
                    val = await conn.fetchval(
                        "SELECT current_setting('app.user_id', true)"
                    )
                    assert val == "user-abc-123"
        finally:
            current_user_id.reset(tok)

    async def test_no_set_when_none(self, pool):
        # Ensure contextvar is None (default)
        assert current_user_id.get() is None
        async with pool.acquire() as conn:
            # Clear the test fixture's session default to simulate a real
            # unauthenticated connection.
            await conn.execute("RESET app.user_id")
            async with conn.transaction():
                await set_user_context(conn)
                val = await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
                # Should be NULL (empty string or None from asyncpg)
                assert val is None or val == ""

    async def test_no_pool_leakage(self, pool):
        """SET LOCAL is transaction-scoped — verify it doesn't leak."""
        # Set user_id in one transaction
        tok = current_user_id.set("user-leak-test")
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await set_user_context(conn)
                    val = await conn.fetchval(
                        "SELECT current_setting('app.user_id', true)"
                    )
                    assert val == "user-leak-test"
        finally:
            current_user_id.reset(tok)

        # Next acquisition should NOT have the old user_id (transaction-scoped)
        async with pool.acquire() as conn:
            await conn.execute("RESET app.user_id")
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val is None or val == ""


class TestAcquireContextManager:
    async def test_sets_user_id_in_transaction(self, pool):
        tok = current_user_id.set("user-acquire-test")
        try:
            async with acquire(pool) as conn:
                val = await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
                assert val == "user-acquire-test"
        finally:
            current_user_id.reset(tok)

    async def test_no_transaction_wrapper_when_no_user(self, pool):
        """When no user is set, acquire yields without extra transaction."""
        assert current_user_id.get() is None
        async with acquire(pool) as conn:
            await conn.execute("RESET app.user_id")
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val is None or val == ""

    async def test_sequential_users_no_leak(self, pool):
        """Two sequential acquire calls with different users don't leak."""
        tok1 = current_user_id.set("user-first")
        try:
            async with acquire(pool) as conn:
                val = await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
                assert val == "user-first"
        finally:
            current_user_id.reset(tok1)

        tok2 = current_user_id.set("user-second")
        try:
            async with acquire(pool) as conn:
                val = await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
                assert val == "user-second"
        finally:
            current_user_id.reset(tok2)

        # Unauthenticated after both
        async with acquire(pool) as conn:
            await conn.execute("RESET app.user_id")
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val is None or val == ""

    async def test_queries_work_inside_acquire(self, pool):
        """Verify normal DB operations work within acquire context."""
        tok = current_user_id.set("user-query-test")
        try:
            async with acquire(pool) as conn:
                # Should be able to query tables normally
                count = await conn.fetchval("SELECT count(*) FROM memories")
                assert count == 0  # table is truncated in fixture
        finally:
            current_user_id.reset(tok)
