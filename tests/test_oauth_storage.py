"""Tests for PostgresKeyValueStore — OAuth state persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from weft.mcp.oauth_storage import PostgresKeyValueStore


@pytest.fixture
def store(pool):
    """Store backed by the test pool."""
    return PostgresKeyValueStore(lambda: pool)


# ------------------------------------------------------------------
# Basic CRUD
# ------------------------------------------------------------------


async def test_put_and_get(store):
    await store.put("key1", {"foo": "bar"})
    result = await store.get("key1")
    assert result == {"foo": "bar"}


async def test_get_missing_key(store):
    result = await store.get("nonexistent")
    assert result is None


async def test_put_overwrites(store):
    await store.put("key1", {"v": 1})
    await store.put("key1", {"v": 2})
    result = await store.get("key1")
    assert result == {"v": 2}


async def test_delete_existing(store):
    await store.put("key1", {"v": 1})
    deleted = await store.delete("key1")
    assert deleted is True
    assert await store.get("key1") is None


async def test_delete_missing(store):
    deleted = await store.delete("nonexistent")
    assert deleted is False


# ------------------------------------------------------------------
# Collections
# ------------------------------------------------------------------


async def test_collections_isolate_keys(store):
    await store.put("key1", {"v": "a"}, collection="col_a")
    await store.put("key1", {"v": "b"}, collection="col_b")
    assert (await store.get("key1", collection="col_a"))["v"] == "a"
    assert (await store.get("key1", collection="col_b"))["v"] == "b"
    # Default collection is empty string
    assert await store.get("key1") is None


async def test_delete_respects_collection(store):
    await store.put("key1", {"v": 1}, collection="col_a")
    await store.put("key1", {"v": 2}, collection="col_b")
    await store.delete("key1", collection="col_a")
    assert await store.get("key1", collection="col_a") is None
    assert await store.get("key1", collection="col_b") is not None


# ------------------------------------------------------------------
# TTL
# ------------------------------------------------------------------


async def test_ttl_expiry(store, pool):
    # Insert with a TTL that's already expired by manipulating the DB directly
    await store.put("key1", {"v": 1})
    await pool.execute(
        "UPDATE oauth_storage SET expires_at = $1 WHERE key = 'key1'",
        datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    result = await store.get("key1")
    assert result is None  # Expired, cleaned up on read


async def test_ttl_not_expired(store):
    await store.put("key1", {"v": 1}, ttl=3600)
    result = await store.get("key1")
    assert result == {"v": 1}


async def test_ttl_method_returns_remaining(store):
    await store.put("key1", {"v": 1}, ttl=3600)
    value, remaining = await store.ttl("key1")
    assert value == {"v": 1}
    assert remaining is not None
    assert 3500 < remaining <= 3600


async def test_ttl_method_no_expiry(store):
    await store.put("key1", {"v": 1})
    value, remaining = await store.ttl("key1")
    assert value == {"v": 1}
    assert remaining is None


async def test_ttl_method_expired(store, pool):
    await store.put("key1", {"v": 1})
    await pool.execute(
        "UPDATE oauth_storage SET expires_at = $1 WHERE key = 'key1'",
        datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    value, remaining = await store.ttl("key1")
    assert value is None
    assert remaining is None


async def test_ttl_method_missing(store):
    value, remaining = await store.ttl("nonexistent")
    assert value is None
    assert remaining is None


# ------------------------------------------------------------------
# Bulk operations
# ------------------------------------------------------------------


async def test_get_many(store):
    await store.put("a", {"v": 1})
    await store.put("b", {"v": 2})
    results = await store.get_many(["a", "b", "c"])
    assert results == [{"v": 1}, {"v": 2}, None]


async def test_get_many_filters_expired(store, pool):
    await store.put("a", {"v": 1})
    await store.put("b", {"v": 2})
    # Expire 'b'
    await pool.execute(
        "UPDATE oauth_storage SET expires_at = $1 WHERE key = 'b'",
        datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    results = await store.get_many(["a", "b"])
    assert results == [{"v": 1}, None]


async def test_put_many(store):
    await store.put_many(["a", "b"], [{"v": 1}, {"v": 2}])
    assert await store.get("a") == {"v": 1}
    assert await store.get("b") == {"v": 2}


async def test_delete_many(store):
    await store.put_many(["a", "b", "c"], [{"v": 1}, {"v": 2}, {"v": 3}])
    count = await store.delete_many(["a", "c"])
    assert count == 2
    assert await store.get("a") is None
    assert await store.get("b") == {"v": 2}
    assert await store.get("c") is None


async def test_ttl_many(store):
    await store.put("a", {"v": 1}, ttl=3600)
    await store.put("b", {"v": 2})  # no TTL
    results = await store.ttl_many(["a", "b", "c"])
    assert len(results) == 3
    # 'a' has TTL
    assert results[0][0] == {"v": 1}
    assert results[0][1] is not None and results[0][1] > 0
    # 'b' has no TTL
    assert results[1][0] == {"v": 2}
    assert results[1][1] is None
    # 'c' missing
    assert results[2] == (None, None)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


async def test_complex_json_values(store):
    """Verify nested dicts, lists, and special types are round-tripped."""
    value = {
        "nested": {"deep": [1, 2, 3]},
        "list": ["a", "b"],
        "null": None,
        "number": 42.5,
    }
    await store.put("complex", value)
    result = await store.get("complex")
    assert result == value


async def test_empty_collection_default(store):
    """Default collection is empty string — verify it works."""
    await store.put("k", {"v": 1}, collection=None)
    await store.put("k", {"v": 2}, collection="")
    # Both should refer to the same row
    result = await store.get("k")
    assert result == {"v": 2}
