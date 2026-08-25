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

# Fallback used only for pools not created via create_pool (e.g. tests that
# construct a raw asyncpg pool). Bounds acquire() so a drained pool fails fast
# instead of blocking forever.
_DEFAULT_ACQUIRE_TIMEOUT = 10.0

# Per-pool acquire timeout, recorded at create_pool time and consumed by
# acquire(). Keyed by id(pool) because asyncpg.Pool uses __slots__ with no
# __dict__ or __weakref__ — it can neither carry a custom attribute nor be a
# WeakKeyDictionary key. create_pool always writes the entry before the pool is
# used, so a pool built here always reads its own configured value; the only
# stale case is a raw pool whose id collides with a closed pool's, which merely
# yields a different *bounded* timeout — never a hang. Entries are tiny
# (int -> float); a leak here is negligible for the once-per-process pool.
_ACQUIRE_TIMEOUTS: dict[int, float | None] = {}


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
    """Initialize pgvector resolution and its codec on each connection.

    Supabase's pooler resets ``search_path`` to ``"$user", public`` even when
    the login role has a role-level setting. Weft contains unqualified
    ``::vector`` casts, while Supabase installs pgvector in ``extensions``.
    Set the application search path explicitly before registering the codec so
    both SQL type resolution and asyncpg's Python codec work consistently.

    Tries multiple schemas since managed Postgres providers may install
    pgvector in different schemas (public, extensions, pg_catalog).
    """
    await conn.execute("SET search_path TO public, extensions")
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
    # Per-query ceiling: a stuck query is cancelled instead of hanging forever
    # and holding its connection out of the pool.
    if config.database.command_timeout is not None:
        kwargs["command_timeout"] = config.database.command_timeout
    if config.database.statement_cache_size is not None:
        kwargs["statement_cache_size"] = config.database.statement_cache_size
    # Supabase pooler requires statement_cache_size=0 (no prepared statements)
    elif ":6543/" in dsn or "pooler.supabase.com" in dsn:
        kwargs["statement_cache_size"] = 0
    # Enable certificate- and hostname-verified SSL for Supabase and other
    # cloud Postgres providers. A deployment may provide the provider's CA
    # bundle when it is not present in the runtime image trust store.
    if "supabase.co" in dsn or "supabase.com" in dsn or "sslmode=require" in dsn:
        if config.database.ca_cert:
            try:
                kwargs["ssl"] = ssl.create_default_context(
                    cadata=config.database.ca_cert,
                )
            except ssl.SSLError as exc:
                raise ValueError(
                    "WEFT_DATABASE_CA_CERT is not a valid PEM certificate bundle"
                ) from exc
        else:
            kwargs["ssl"] = ssl.create_default_context()

    try:
        pool = await asyncpg.create_pool(dsn, **kwargs)
        _ACQUIRE_TIMEOUTS[id(pool)] = config.database.acquire_timeout
        return pool
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
        pool = await asyncpg.create_pool(dsn, **kwargs)
        _ACQUIRE_TIMEOUTS[id(pool)] = config.database.acquire_timeout
        return pool


def _validate_user_id(user_id: str) -> bool:
    """Validate a user_id string for safe interpolation into SET LOCAL.

    SET is a utility command — doesn't support $1 parameterization.
    Allow alphanumerics, dashes (UUIDs), and underscores
    (sentinels like ``__system_global_zathras__``).
    """
    return user_id.replace("-", "").replace("_", "").isalnum()


async def set_user_context_value(conn: asyncpg.Connection, user_id: str) -> bool:
    """Set ``app.user_id`` on a connection from an explicit user_id string.

    Issues ``SET LOCAL app.user_id = ...`` inside the current transaction.
    SET LOCAL is transaction-scoped, so the setting is automatically
    cleared when the transaction ends — no pool leakage.

    Returns True if the GUC was set, False if the user_id was rejected
    as suspicious (non-alphanumeric characters outside dashes/underscores).

    This is the explicit-identity variant of :func:`set_user_context`,
    which resolves identity from the ``current_user_id`` ContextVar.
    Both share the same validation and SET LOCAL logic.
    """
    if not _validate_user_id(user_id):
        logger.warning("Rejecting suspicious user_id: %r", user_id)
        return False
    await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
    return True


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
        await set_user_context_value(conn, user_id)


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

    # Bound the wait for a free connection: when the pool is drained, block only
    # up to acquire_timeout, then raise asyncio.TimeoutError instead of hanging
    # indefinitely. Falls back to a safe default for pools not built via
    # create_pool (e.g. raw test pools).
    acquire_timeout = _ACQUIRE_TIMEOUTS.get(id(pool), _DEFAULT_ACQUIRE_TIMEOUT)
    async with pool.acquire(timeout=acquire_timeout) as conn:
        user_id = _resolve_user_id()
        if user_id is not None:
            # SET LOCAL requires a transaction context.
            async with conn.transaction():
                # Use the shared validator — same logic as
                # set_user_context_value, just inlined here because
                # we need to manage the contextvar token within the
                # transaction scope.
                if not _validate_user_id(user_id):
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
