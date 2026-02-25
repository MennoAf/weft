"""Database connection management."""

from __future__ import annotations

import asyncpg

from weft.config import WeftConfig


async def create_pool(config: WeftConfig) -> asyncpg.Pool:
    """Create an asyncpg connection pool from config."""
    dsn = config.database.url
    # asyncpg doesn't accept psycopg2 scheme from testcontainers
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")
    return await asyncpg.create_pool(
        dsn,
        min_size=config.database.pool_min_size,
        max_size=config.database.pool_max_size,
    )
