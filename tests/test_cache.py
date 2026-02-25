"""Tests for weft.cache — Redis caching layer."""

from __future__ import annotations

import pytest

from weft.cache import Cache
from weft.models import Memory, MemoryType, MemorySource, MemoryStatus


async def test_memory_cache_roundtrip(redis_conn):
    """Cache a memory and retrieve it."""
    cache = Cache(redis_conn)
    mem = Memory(
        type=MemoryType.fact,
        content="cached memory",
        topic=["test"],
    )

    # Miss first
    assert await cache.get_memory(mem.id) is None

    # Set and hit
    await cache.set_memory(mem)
    cached = await cache.get_memory(mem.id)
    assert cached is not None
    assert cached.id == mem.id
    assert cached.content == "cached memory"


async def test_memory_cache_invalidate(redis_conn):
    """Invalidating removes a memory from cache."""
    cache = Cache(redis_conn)
    mem = Memory(type=MemoryType.fact, content="to invalidate")

    await cache.set_memory(mem)
    assert await cache.get_memory(mem.id) is not None

    await cache.invalidate_memory(mem.id)
    assert await cache.get_memory(mem.id) is None


async def test_embedding_cache(redis_conn):
    """Cache an embedding and retrieve it."""
    cache = Cache(redis_conn)
    text = "hello world"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    assert await cache.get_embedding(text, "fastembed") is None

    await cache.set_embedding(text, "fastembed", embedding)
    cached = await cache.get_embedding(text, "fastembed")
    assert cached == embedding


async def test_embedding_cache_provider_isolation(redis_conn):
    """Different providers have separate cache entries."""
    cache = Cache(redis_conn)
    text = "same text"
    emb1 = [0.1, 0.2]
    emb2 = [0.3, 0.4]

    await cache.set_embedding(text, "fastembed", emb1)
    await cache.set_embedding(text, "openai", emb2)

    assert await cache.get_embedding(text, "fastembed") == emb1
    assert await cache.get_embedding(text, "openai") == emb2


async def test_stats_cache(redis_conn):
    """Cache and retrieve stats."""
    cache = Cache(redis_conn)

    assert await cache.get_stats() is None

    stats = {"total": 42, "by_type": {"fact": 20, "pattern": 22}}
    await cache.set_stats(stats)
    cached = await cache.get_stats()
    assert cached["total"] == 42

    await cache.invalidate_stats()
    assert await cache.get_stats() is None


async def test_flush_all(redis_conn):
    """Flush removes all Weft keys."""
    cache = Cache(redis_conn)
    mem = Memory(type=MemoryType.fact, content="will be flushed")

    await cache.set_memory(mem)
    await cache.set_embedding("text", "fastembed", [0.1])
    await cache.set_stats({"total": 1})

    await cache.flush_all()

    assert await cache.get_memory(mem.id) is None
    assert await cache.get_embedding("text", "fastembed") is None
    assert await cache.get_stats() is None
