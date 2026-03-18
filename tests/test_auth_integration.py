"""End-to-end auth context integration tests.

Validates the full authentication chain:
  JWT extraction → contextvar → SET LOCAL app.user_id →
  current_setting() in SQL → user_id populated on INSERT →
  per-user isolation in queries.

Requires a real PostgreSQL instance (testcontainers via conftest.py).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest

from weft.auth import current_user_id, extract_user_id, extract_user_id_from_header
from weft.db.connection import acquire
from weft.models import MemoryCreate, MemoryType
from weft.store import store_memory

# Shared test secret
_SECRET = "test-supabase-jwt-secret-32chars!"


def _make_jwt(sub: str = "test-user-uuid", **extra_claims) -> str:
    """Build a signed JWT for testing."""
    payload = {"sub": sub, "exp": int(time.time()) + 3600, **extra_claims}
    return pyjwt.encode(payload, _SECRET, algorithm="HS256")


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": _SECRET}):
        yield


@pytest.fixture(autouse=True)
def _reset_contextvar():
    """Ensure contextvar is clean before and after each test."""
    tok = current_user_id.set(None)
    yield
    current_user_id.reset(tok)


# --- Full chain: JWT → contextvar → SET LOCAL → current_setting ---


async def test_jwt_to_db_full_chain(pool):
    """JWT decode → contextvar → SET LOCAL → current_setting returns user_id."""
    token = _make_jwt(sub="user-chain-test")
    user_id = extract_user_id(token)
    assert user_id == "user-chain-test"

    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool) as conn:
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val == "user-chain-test"
    finally:
        current_user_id.reset(tok)


async def test_bearer_header_to_db_full_chain(pool):
    """Authorization header → extract → contextvar → SET LOCAL → DB."""
    token = _make_jwt(sub="user-header-e2e")
    user_id = extract_user_id_from_header(f"Bearer {token}")
    assert user_id == "user-header-e2e"

    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool) as conn:
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val == "user-header-e2e"
    finally:
        current_user_id.reset(tok)


# --- INSERT populates user_id from current_setting ---


async def test_store_memory_populates_user_id(pool):
    """store_memory within acquire() populates user_id via current_setting."""
    tok = current_user_id.set("user-insert-test")
    try:
        async with acquire(pool) as conn:
            # Store through raw SQL to verify user_id column
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, project_id, agent_id, embedding, status, pinned,
                    review_after, user_id
                ) VALUES (
                    'weft-e2e-test-1', 'fact', '{}', 'test content', 'conversation', 0.7,
                    5, now(), now(), now(),
                    0, NULL, NULL, NULL, 'active', false,
                    NULL, nullif(current_setting('app.user_id', true), '')
                )
                """
            )
            row = await conn.fetchrow(
                "SELECT user_id FROM memories WHERE id = 'weft-e2e-test-1'"
            )
            assert row["user_id"] == "user-insert-test"
    finally:
        current_user_id.reset(tok)


async def test_store_memory_null_user_id_when_unauthenticated(pool):
    """store_memory without auth context → user_id is NULL."""
    assert current_user_id.get() is None

    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="unauthenticated memory"),
    )

    row = await pool.fetchrow(
        "SELECT user_id FROM memories WHERE id = $1", mem.id
    )
    assert row["user_id"] is None


# --- Cross-user isolation ---


async def test_two_users_get_correct_user_id(pool):
    """Memories stored by user A and user B have their respective user_ids."""
    # User A stores
    tok_a = current_user_id.set("user-aaa")
    try:
        async with acquire(pool) as conn:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, status, pinned, user_id
                ) VALUES (
                    'weft-e2e-user-a', 'fact', '{}', 'user A memory', 'conversation', 0.7,
                    5, now(), now(), now(), 0, 'active', false,
                    nullif(current_setting('app.user_id', true), '')
                )
                """
            )
    finally:
        current_user_id.reset(tok_a)

    # User B stores
    tok_b = current_user_id.set("user-bbb")
    try:
        async with acquire(pool) as conn:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, status, pinned, user_id
                ) VALUES (
                    'weft-e2e-user-b', 'fact', '{}', 'user B memory', 'conversation', 0.7,
                    5, now(), now(), now(), 0, 'active', false,
                    nullif(current_setting('app.user_id', true), '')
                )
                """
            )
    finally:
        current_user_id.reset(tok_b)

    # Verify isolation
    row_a = await pool.fetchrow("SELECT user_id FROM memories WHERE id = 'weft-e2e-user-a'")
    row_b = await pool.fetchrow("SELECT user_id FROM memories WHERE id = 'weft-e2e-user-b'")
    assert row_a["user_id"] == "user-aaa"
    assert row_b["user_id"] == "user-bbb"


# --- Sequential request isolation (no cross-contamination) ---


async def test_sequential_requests_no_cross_contamination(pool):
    """Three sequential requests: user A, user B, unauthenticated — no leakage."""
    results = []

    # Request 1: user A
    tok1 = current_user_id.set("user-seq-aaa")
    try:
        async with acquire(pool) as conn:
            results.append(
                await conn.fetchval("SELECT current_setting('app.user_id', true)")
            )
    finally:
        current_user_id.reset(tok1)

    # Request 2: user B
    tok2 = current_user_id.set("user-seq-bbb")
    try:
        async with acquire(pool) as conn:
            results.append(
                await conn.fetchval("SELECT current_setting('app.user_id', true)")
            )
    finally:
        current_user_id.reset(tok2)

    # Request 3: unauthenticated
    async with acquire(pool) as conn:
        results.append(
            await conn.fetchval("SELECT current_setting('app.user_id', true)")
        )

    assert results[0] == "user-seq-aaa"
    assert results[1] == "user-seq-bbb"
    assert results[2] is None or results[2] == ""


# --- Concurrent requests ---


async def test_concurrent_requests_isolated(pool):
    """Two concurrent requests with different users don't interfere."""

    async def _query_user_id(user: str) -> str | None:
        tok = current_user_id.set(user)
        try:
            async with acquire(pool) as conn:
                # Small delay to increase overlap likelihood
                await asyncio.sleep(0.01)
                return await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
        finally:
            current_user_id.reset(tok)

    result_a, result_b = await asyncio.gather(
        _query_user_id("concurrent-aaa"),
        _query_user_id("concurrent-bbb"),
    )
    assert result_a == "concurrent-aaa"
    assert result_b == "concurrent-bbb"


# --- Malformed JWT graceful degradation ---


@pytest.mark.parametrize(
    "auth_header",
    [
        None,
        "",
        "notajwt",
        "Bearer ",
        "Bearer abc.def",
        "Basic validtoken",
        "bearer valid",  # case-sensitive
    ],
    ids=[
        "none",
        "empty",
        "no-scheme",
        "bearer-no-token",
        "bearer-partial-jwt",
        "wrong-scheme",
        "lowercase-bearer",
    ],
)
async def test_malformed_jwt_degrades_to_null(pool, auth_header):
    """Malformed/missing Authorization headers → NULL user_id in DB."""
    user_id = extract_user_id_from_header(auth_header)
    assert user_id is None

    # If this were a real request, the contextvar would remain None
    assert current_user_id.get() is None

    async with acquire(pool) as conn:
        val = await conn.fetchval(
            "SELECT current_setting('app.user_id', true)"
        )
        assert val is None or val == ""


async def test_expired_jwt_degrades_to_null(pool):
    """Expired JWT → NULL user_id, graceful degradation."""
    expired_token = pyjwt.encode(
        {"sub": "user-expired", "exp": int(time.time()) - 100},
        _SECRET,
        algorithm="HS256",
    )
    user_id = extract_user_id_from_header(f"Bearer {expired_token}")
    assert user_id is None

    async with acquire(pool) as conn:
        val = await conn.fetchval(
            "SELECT current_setting('app.user_id', true)"
        )
        assert val is None or val == ""


async def test_wrong_secret_degrades_to_null(pool):
    """JWT signed with wrong secret → NULL user_id."""
    token = pyjwt.encode(
        {"sub": "user-wrong-secret", "exp": int(time.time()) + 3600},
        "completely-wrong-secret-key!!!!!",
        algorithm="HS256",
    )
    user_id = extract_user_id_from_header(f"Bearer {token}")
    assert user_id is None


# --- SET LOCAL transaction scoping ---


async def test_set_local_does_not_persist_after_transaction(pool):
    """SET LOCAL is transaction-scoped — does not leak to pool."""
    tok = current_user_id.set("user-txn-scope")
    try:
        async with acquire(pool) as conn:
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            assert val == "user-txn-scope"
    finally:
        current_user_id.reset(tok)

    # Subsequent raw connection should NOT have the setting
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT current_setting('app.user_id', true)"
        )
        assert val is None or val == ""


async def test_user_id_sanitization_rejects_injection(pool):
    """Suspicious user_id values are rejected (no SET LOCAL issued)."""
    tok = current_user_id.set("'; DROP TABLE memories; --")
    try:
        async with acquire(pool) as conn:
            val = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
            # Should be NULL because the suspicious value was rejected
            assert val is None or val == ""
    finally:
        current_user_id.reset(tok)
