"""PostgreSQL-backed key-value store for FastMCP OAuthProxy state.

Implements the ``key_value.aio.protocols.AsyncKeyValue`` protocol so that
OAuth client registrations, tokens, and authorization codes persist across
Fly.io deploys (the default file-based store is wiped on every deploy).

Uses a lazy pool accessor because the store is constructed at module load
time (before lifespan creates the pool) but only accessed during actual
OAuth operations (after startup).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence

import asyncpg

logger = logging.getLogger(__name__)


class PostgresKeyValueStore:
    """AsyncKeyValue backed by the ``oauth_storage`` table.

    Parameters
    ----------
    pool_factory:
        Callable that returns the current asyncpg pool.  Called on every
        operation (not cached) so the store tracks pool recreation.
    """

    def __init__(self, pool_factory: Callable[[], asyncpg.Pool]) -> None:
        self._pool = pool_factory

    def _col(self, collection: str | None) -> str:
        return collection or ""

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(
        self, key: str, *, collection: str | None = None,
    ) -> dict[str, object] | None:
        pool = self._pool()
        row = await pool.fetchrow(
            "SELECT value, expires_at FROM oauth_storage "
            "WHERE collection = $1 AND key = $2",
            self._col(collection), key,
        )
        if row is None:
            return None
        if row["expires_at"] is not None:
            if row["expires_at"].timestamp() < time.time():
                # Expired — delete and return None
                await pool.execute(
                    "DELETE FROM oauth_storage WHERE collection = $1 AND key = $2",
                    self._col(collection), key,
                )
                return None
        return json.loads(row["value"])

    async def put(
        self,
        key: str,
        value: Mapping[str, object],
        *,
        collection: str | None = None,
        ttl: float | None = None,
    ) -> None:
        pool = self._pool()
        expires_at = None
        if ttl is not None and ttl > 0:
            from datetime import datetime, timezone, timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        await pool.execute(
            "INSERT INTO oauth_storage (collection, key, value, expires_at, updated_at) "
            "VALUES ($1, $2, $3::jsonb, $4, now()) "
            "ON CONFLICT (collection, key) DO UPDATE "
            "SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at, updated_at = now()",
            self._col(collection), key, json.dumps(value, default=str), expires_at,
        )

    async def delete(
        self, key: str, *, collection: str | None = None,
    ) -> bool:
        pool = self._pool()
        result = await pool.execute(
            "DELETE FROM oauth_storage WHERE collection = $1 AND key = $2",
            self._col(collection), key,
        )
        return result == "DELETE 1"

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    async def get_many(
        self, keys: Sequence[str], *, collection: str | None = None,
    ) -> list[dict[str, object] | None]:
        pool = self._pool()
        col = self._col(collection)
        rows = await pool.fetch(
            "SELECT key, value, expires_at FROM oauth_storage "
            "WHERE collection = $1 AND key = ANY($2::text[])",
            col, list(keys),
        )
        now = time.time()
        row_map: dict[str, dict[str, object] | None] = {}
        expired_keys: list[str] = []
        for r in rows:
            if r["expires_at"] is not None and r["expires_at"].timestamp() < now:
                expired_keys.append(r["key"])
                row_map[r["key"]] = None
            else:
                row_map[r["key"]] = json.loads(r["value"])
        # Clean up expired
        if expired_keys:
            await pool.execute(
                "DELETE FROM oauth_storage WHERE collection = $1 AND key = ANY($2::text[])",
                col, expired_keys,
            )
        return [row_map.get(k) for k in keys]

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, object]],
        *,
        collection: str | None = None,
        ttl: float | None = None,
    ) -> None:
        pool = self._pool()
        col = self._col(collection)
        expires_at = None
        if ttl is not None and ttl > 0:
            from datetime import datetime, timezone, timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        async with pool.acquire() as conn:
            async with conn.transaction():
                for k, v in zip(keys, values):
                    await conn.execute(
                        "INSERT INTO oauth_storage (collection, key, value, expires_at, updated_at) "
                        "VALUES ($1, $2, $3::jsonb, $4, now()) "
                        "ON CONFLICT (collection, key) DO UPDATE "
                        "SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at, updated_at = now()",
                        col, k, json.dumps(v, default=str), expires_at,
                    )

    async def delete_many(
        self, keys: Sequence[str], *, collection: str | None = None,
    ) -> int:
        pool = self._pool()
        result = await pool.execute(
            "DELETE FROM oauth_storage WHERE collection = $1 AND key = ANY($2::text[])",
            self._col(collection), list(keys),
        )
        # Result is "DELETE N"
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # TTL operations
    # ------------------------------------------------------------------

    async def ttl(
        self, key: str, *, collection: str | None = None,
    ) -> tuple[dict[str, object] | None, float | None]:
        pool = self._pool()
        row = await pool.fetchrow(
            "SELECT value, expires_at FROM oauth_storage "
            "WHERE collection = $1 AND key = $2",
            self._col(collection), key,
        )
        if row is None:
            return (None, None)
        now = time.time()
        if row["expires_at"] is not None:
            remaining = row["expires_at"].timestamp() - now
            if remaining <= 0:
                await pool.execute(
                    "DELETE FROM oauth_storage WHERE collection = $1 AND key = $2",
                    self._col(collection), key,
                )
                return (None, None)
            return (json.loads(row["value"]), remaining)
        return (json.loads(row["value"]), None)

    async def ttl_many(
        self, keys: Sequence[str], *, collection: str | None = None,
    ) -> list[tuple[dict[str, object] | None, float | None]]:
        pool = self._pool()
        col = self._col(collection)
        rows = await pool.fetch(
            "SELECT key, value, expires_at FROM oauth_storage "
            "WHERE collection = $1 AND key = ANY($2::text[])",
            col, list(keys),
        )
        now = time.time()
        row_map: dict[str, tuple[dict[str, object] | None, float | None]] = {}
        expired_keys: list[str] = []
        for r in rows:
            if r["expires_at"] is not None:
                remaining = r["expires_at"].timestamp() - now
                if remaining <= 0:
                    expired_keys.append(r["key"])
                    row_map[r["key"]] = (None, None)
                else:
                    row_map[r["key"]] = (json.loads(r["value"]), remaining)
            else:
                row_map[r["key"]] = (json.loads(r["value"]), None)
        if expired_keys:
            await pool.execute(
                "DELETE FROM oauth_storage WHERE collection = $1 AND key = ANY($2::text[])",
                col, expired_keys,
            )
        return [row_map.get(k, (None, None)) for k in keys]
