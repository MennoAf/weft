"""Database connection management."""

from __future__ import annotations

import logging
import ssl
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from weft.auth import current_user_id
from weft.config import WeftConfig

logger = logging.getLogger(__name__)


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
    # Supabase pooler (port 6543) requires statement_cache_size=0 for pgBouncer
    elif ":6543/" in dsn:
        kwargs["statement_cache_size"] = 0
    # Enable SSL for Supabase and other cloud Postgres providers
    if "supabase.co" in dsn or "sslmode=require" in dsn:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = ctx
    return await asyncpg.create_pool(dsn, **kwargs)


async def set_user_context(conn: asyncpg.Connection) -> None:
    """Set ``app.user_id`` on a connection from the current contextvar.

    Issues ``SET LOCAL app.user_id = ...`` when a user_id is present.
    SET LOCAL is transaction-scoped, so the setting is automatically
    cleared when the transaction ends — no pool leakage.

    When user_id is None (unauthenticated), no SET LOCAL is issued,
    meaning ``current_setting('app.user_id', true)`` returns NULL.
    RLS policies then show only global (user_id IS NULL) rows.
    """
    user_id = current_user_id.get()
    if user_id is not None:
        # SET is a utility command — doesn't support $1 parameterization.
        # Sanitize by rejecting non-UUID-safe characters.
        if not user_id.replace("-", "").isalnum():
            logger.warning("Rejecting suspicious user_id: %r", user_id)
            return
        await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")


@asynccontextmanager
async def acquire(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection with user identity context.

    Wraps ``pool.acquire()`` and issues ``SET LOCAL app.user_id``
    inside a transaction when a user is authenticated. This is the
    preferred way to get a connection for user-scoped operations.

    Usage::

        async with acquire(pool) as conn:
            rows = await conn.fetch("SELECT * FROM memories")
            # RLS automatically filters by the current user
    """
    async with pool.acquire() as conn:
        user_id = current_user_id.get()
        if user_id is not None:
            # SET LOCAL requires a transaction context.
            async with conn.transaction():
                # SET is a utility command — doesn't support $1 parameterization.
                # Sanitize by rejecting non-UUID-safe characters.
                if not user_id.replace("-", "").isalnum():
                    logger.warning("Rejecting suspicious user_id: %r", user_id)
                    yield conn
                else:
                    await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
                    yield conn
        else:
            yield conn
