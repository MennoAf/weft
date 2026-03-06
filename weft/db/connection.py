"""Database connection management."""

from __future__ import annotations

import ssl

import asyncpg

from weft.config import WeftConfig


async def create_pool(config: WeftConfig) -> asyncpg.Pool:
    """Create an asyncpg connection pool from config."""
    dsn = config.database.url
    # asyncpg doesn't accept psycopg2 scheme from testcontainers
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")
    kwargs: dict = {
        "min_size": config.database.pool_min_size,
        "max_size": config.database.pool_max_size,
    }
    if config.database.statement_cache_size is not None:
        kwargs["statement_cache_size"] = config.database.statement_cache_size
    # Enable SSL for Supabase and other cloud Postgres providers
    if "supabase.co" in dsn or "sslmode=require" in dsn:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = ctx
    return await asyncpg.create_pool(dsn, **kwargs)
