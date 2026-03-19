"""Database connection management."""

from __future__ import annotations

import logging
import ssl
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Union

import asyncpg

from weft.auth import current_user_id
from weft.config import WeftConfig

logger = logging.getLogger(__name__)

# Connection contextvar: when set (inside acquire()), store functions use
# this connection instead of the pool.  This ensures all DB operations
# within an MCP tool call share a single connection with SET LOCAL
# app.user_id active, making RLS policies effective.
_current_conn: ContextVar[asyncpg.Connection | None] = ContextVar(
    "_current_conn", default=None,
)


def get_db(pool: asyncpg.Pool) -> Union[asyncpg.Pool, asyncpg.Connection]:
    """Return the RLS-scoped connection if inside acquire(), else the pool.

    Both asyncpg.Pool and asyncpg.Connection expose the same query
    interface (execute, fetch, fetchrow, fetchval), so callers work
    identically regardless of which is returned.

    When inside an acquire() context, the returned connection has
    SET LOCAL app.user_id active, so RLS policies filter correctly.
    When outside (tests, CLI, background tasks), the pool is returned
    and queries run without user scoping — seeing only global rows.
    """
    conn = _current_conn.get(None)
    return conn if conn is not None else pool


async def _pgvector_codec_init(conn: asyncpg.Connection) -> None:
    """Pool init callback: register pgvector type codec on each new connection.

    Encodes list[float] → pgvector text format and decodes back automatically,
    eliminating manual string manipulation.

    Tries multiple schemas since managed Postgres providers may install
    pgvector in different schemas (public, extensions, pg_catalog).
    """
    for schema in ("public", "extensions", "pg_catalog"):
        try:
            await conn.set_type_codec(
                "vector",
                encoder=lambda v: "[" + ",".join(str(x) for x in v) + "]",
                decoder=lambda s: [float(x) for x in s.strip("[]").split(",")],
                schema=schema,
                format="text",
            )
            logger.debug("pgvector codec registered (schema=%s)", schema)
            return
        except Exception:
            continue
    logger.warning("pgvector codec registration failed: vector type not found in any schema")


async def register_pgvector_codec(pool: asyncpg.Pool) -> None:
    """Register the pgvector codec on all existing pool connections.

    Call this AFTER migrations have run (which CREATE EXTENSION vector).
    The pool's init callback handles future connections automatically.
    """
    async with pool.acquire() as conn:
        await _pgvector_codec_init(conn)


async def create_pool(config: WeftConfig) -> asyncpg.Pool:
    """Create an asyncpg connection pool from config."""
    dsn = config.database.url
    # asyncpg doesn't accept psycopg2 scheme from testcontainers
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")
    kwargs: dict = {
        "min_size": config.database.pool_min_size,
        "max_size": config.database.pool_max_size,
        "init": _pgvector_codec_init,
    }
    if config.database.statement_cache_size is not None:
        kwargs["statement_cache_size"] = config.database.statement_cache_size
    # Supabase pooler requires statement_cache_size=0 (no prepared statements)
    elif ":6543/" in dsn or "pooler.supabase.com" in dsn:
        kwargs["statement_cache_size"] = 0
    # Enable SSL for Supabase and other cloud Postgres providers
    if "supabase.co" in dsn or "supabase.com" in dsn or "sslmode=require" in dsn:
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
    inside a transaction when a user is authenticated.  Also sets
    the ``_current_conn`` contextvar so downstream code can call
    ``get_db(pool)`` to reuse this RLS-scoped connection.

    **Idempotent:** if already inside an ``acquire()`` context,
    yields the existing connection without re-acquiring.  This
    makes it safe for store functions that call ``acquire()``
    internally (e.g. ``revise_memory``, ``record_feedback``) —
    when called from a tool that already holds the scope, they
    just reuse it.

    Usage::

        async with acquire(pool) as conn:
            rows = await conn.fetch("SELECT * FROM memories")
            # RLS automatically filters by the current user
    """
    existing = _current_conn.get(None)
    if existing is not None:
        # Already inside an acquire() context — reuse the connection.
        yield existing
        return

    async with pool.acquire() as conn:
        user_id = current_user_id.get()
        if user_id is not None:
            # SET LOCAL requires a transaction context.
            async with conn.transaction():
                # SET is a utility command — doesn't support $1 parameterization.
                # Sanitize by rejecting non-UUID-safe characters.
                if not user_id.replace("-", "").isalnum():
                    logger.warning("Rejecting suspicious user_id: %r", user_id)
                    token = _current_conn.set(conn)
                    try:
                        yield conn
                    finally:
                        _current_conn.reset(token)
                else:
                    await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
                    token = _current_conn.set(conn)
                    try:
                        yield conn
                    finally:
                        _current_conn.reset(token)
        else:
            token = _current_conn.set(conn)
            try:
                yield conn
            finally:
                _current_conn.reset(token)
