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


def _resolve_user_id() -> str | None:
    """Resolve the current user ID from the contextvar set by middleware."""
    return current_user_id.get()

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
    """Create an asyncpg connection pool from config.

    If the initial connection fails and the DSN points to Supabase, this
    attempts to restore (unpause) the project via the Management API before
    retrying — but ONLY when ``config.supabase_access_token`` is set. Supabase
    free-tier projects auto-pause after ~1 week idle; without this the first
    connection after a pause dies with a raw socket error. When no token is
    configured, a paused project yields a clear actionable error (dashboard
    link) instead of auto-restoring. See ``weft.supabase``.
    """
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

    try:
        return await asyncpg.create_pool(dsn, **kwargs)
    except (ConnectionRefusedError, OSError) as exc:
        # A paused Supabase project refuses connections at the socket layer,
        # which asyncpg propagates as ConnectionRefusedError / TimeoutError /
        # socket.gaierror — all OSError subclasses. Only intercept when the DSN
        # is Supabase; everything else re-raises unchanged.
        from weft.supabase import (
            extract_project_ref,
            is_supabase_dsn,
            restore_project,
            wait_for_restore,
        )

        if not is_supabase_dsn(dsn):
            raise

        project_ref = extract_project_ref(dsn)
        if not project_ref:
            raise ConnectionError(
                f"Connection to Supabase failed: {exc}\n"
                "Your Supabase project may be paused. Unpause it at "
                "https://supabase.com/dashboard"
            ) from exc

        token = config.supabase_access_token
        if not token:
            # No token → don't auto-restore (account-scoped token, opt-in only).
            # Still upgrade the cryptic socket error into an actionable message.
            raise ConnectionError(
                f"Connection to Supabase failed (project likely paused): {exc}\n\n"
                "To auto-restore, set SUPABASE_ACCESS_TOKEN in your environment.\n"
                "Generate one at: https://supabase.com/dashboard/account/tokens\n\n"
                f"Or manually unpause project '{project_ref}' at:\n"
                f"https://supabase.com/dashboard/project/{project_ref}"
            ) from exc

        logger.warning(
            "Connection to Supabase failed — attempting to restore paused "
            "project %s",
            project_ref,
        )
        restored = await restore_project(project_ref, token)
        if not restored:
            raise ConnectionError(
                f"Failed to restore Supabase project '{project_ref}'. "
                "Check your SUPABASE_ACCESS_TOKEN or restore manually at:\n"
                f"https://supabase.com/dashboard/project/{project_ref}"
            ) from exc

        logger.info(
            "Restore request accepted — waiting for project %s to come online…",
            project_ref,
        )
        ready = await wait_for_restore(project_ref, token, timeout=120, poll_interval=5)
        if not ready:
            raise ConnectionError(
                f"Supabase project '{project_ref}' restore was accepted but the "
                "database did not become available within 120s. Check status at:\n"
                f"https://supabase.com/dashboard/project/{project_ref}"
            ) from exc

        logger.info(
            "Supabase project %s is back online — retrying connection",
            project_ref,
        )
        return await asyncpg.create_pool(dsn, **kwargs)


async def set_user_context(conn: asyncpg.Connection) -> None:
    """Set ``app.user_id`` on a connection from the current contextvar.

    Issues ``SET LOCAL app.user_id = ...`` when a user_id is present.
    SET LOCAL is transaction-scoped, so the setting is automatically
    cleared when the transaction ends — no pool leakage.

    When user_id is None (unauthenticated), no SET LOCAL is issued.
    ``current_setting('app.user_id', true)`` then returns the empty
    string, which matches no row's user_id, so RLS policies expose
    only the SYSTEM_GLOBAL sentinel rows.
    """
    user_id = _resolve_user_id()
    if user_id is not None:
        # SET is a utility command — doesn't support $1 parameterization.
        # Sanitize by rejecting characters outside the safe set: alphanumeric,
        # dashes (UUIDs), and underscores (for the SYSTEM_GLOBAL sentinel
        # ``__system_global_zathras__`` and similar named identities).
        if not user_id.replace("-", "").replace("_", "").isalnum():
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
        user_id = _resolve_user_id()
        if user_id is not None:
            # SET LOCAL requires a transaction context.
            async with conn.transaction():
                # SET is a utility command — doesn't support $1 parameterization.
                # Allow alphanumerics, dashes (UUIDs), and underscores
                # (sentinels like ``__system_global_zathras__``).
                if not user_id.replace("-", "").replace("_", "").isalnum():
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
