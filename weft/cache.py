"""Redis caching layer — ONLY reader from Redis for memory data.

Falls back to store.py on cache miss. Degrades gracefully if Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import redis.asyncio as aioredis

from weft.models import Memory, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)

# Key prefixes to avoid collision with other Redis users
PREFIX = "weft:"
MEMORY_KEY = f"{PREFIX}mem:"
EMBEDDING_KEY = f"{PREFIX}emb:"
STATS_KEY = f"{PREFIX}stats"

# TTLs
MEMORY_TTL = timedelta(hours=1)
EMBEDDING_TTL = timedelta(hours=24)
STATS_TTL = timedelta(minutes=5)


class Cache:
    """Redis cache for Weft memory data."""

    def __init__(self, redis: aioredis.Redis):
        self._redis = redis

    # --- Memory cache ---

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a cached memory by ID. Returns None on miss or Redis error."""
        try:
            data = await self._redis.get(f"{MEMORY_KEY}{memory_id}")
            if data is None:
                return None
            return Memory.model_validate_json(data)
        except Exception:
            logger.debug("Cache miss/error for memory %s", memory_id, exc_info=True)
            return None

    async def set_memory(self, memory: Memory) -> None:
        """Cache a memory. Silently fails on Redis error."""
        try:
            key = f"{MEMORY_KEY}{memory.id}"
            data = memory.model_dump_json()
            await self._redis.set(key, data, ex=int(MEMORY_TTL.total_seconds()))
        except Exception:
            logger.debug("Failed to cache memory %s", memory.id, exc_info=True)

    async def invalidate_memory(self, memory_id: str) -> None:
        """Remove a memory from cache."""
        try:
            await self._redis.delete(f"{MEMORY_KEY}{memory_id}")
        except Exception:
            logger.debug("Failed to invalidate memory %s", memory_id, exc_info=True)

    # --- Embedding cache ---

    async def get_embedding(self, text: str, provider: str) -> list[float] | None:
        """Get a cached embedding for text+provider. Returns None on miss."""
        try:
            key = f"{EMBEDDING_KEY}{provider}:{_hash_text(text)}"
            data = await self._redis.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception:
            logger.debug("Embedding cache miss/error", exc_info=True)
            return None

    async def set_embedding(self, text: str, provider: str, embedding: list[float]) -> None:
        """Cache an embedding. Silently fails on Redis error."""
        try:
            key = f"{EMBEDDING_KEY}{provider}:{_hash_text(text)}"
            data = json.dumps(embedding)
            await self._redis.set(key, data, ex=int(EMBEDDING_TTL.total_seconds()))
        except Exception:
            logger.debug("Failed to cache embedding", exc_info=True)

    # --- Stats cache ---

    async def get_stats(self) -> dict | None:
        """Get cached stats. Returns None on miss."""
        try:
            data = await self._redis.get(STATS_KEY)
            if data is None:
                return None
            return json.loads(data)
        except Exception:
            return None

    async def set_stats(self, stats: dict) -> None:
        """Cache stats."""
        try:
            await self._redis.set(STATS_KEY, json.dumps(stats, default=str), ex=int(STATS_TTL.total_seconds()))
        except Exception as e:
            logger.warning("set_stats failed: %s", e, exc_info=True)

    async def invalidate_stats(self) -> None:
        """Invalidate stats cache (after writes)."""
        try:
            await self._redis.delete(STATS_KEY)
        except Exception as e:
            logger.warning("invalidate_stats failed: %s", e, exc_info=True)

    # --- Bulk operations ---

    async def flush_all(self) -> None:
        """Clear all Weft keys from Redis."""
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=f"{PREFIX}*", count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            logger.debug("Failed to flush cache", exc_info=True)


class NullCache:
    """No-op cache used when Redis is not available."""

    async def get_memory(self, memory_id: str) -> Memory | None:
        return None

    async def set_memory(self, memory: Memory) -> None:
        pass

    async def invalidate_memory(self, memory_id: str) -> None:
        pass

    async def get_embedding(self, text: str, provider: str) -> list[float] | None:
        return None

    async def set_embedding(self, text: str, provider: str, embedding: list[float]) -> None:
        pass

    async def get_stats(self) -> dict | None:
        return None

    async def set_stats(self, stats: dict) -> None:
        pass

    async def invalidate_stats(self) -> None:
        pass

    async def flush_all(self) -> None:
        pass


def _hash_text(text: str) -> str:
    """Simple hash for cache keys. Not cryptographic."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]
