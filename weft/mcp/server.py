"""Weft MCP server — FastMCP with stdio/HTTP transport."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from weft.auth import current_user_id, extract_user_id_from_header
from weft.cache import Cache, NullCache
from weft.config import WeftConfig, load_config
from weft.db.connection import create_pool
from weft.db.migrations import run_migrations
from weft.embeddings import get_provider
from weft.embeddings.base import EmbeddingProvider
from weft.mcp.auth import get_auth_provider
from weft.seed import seed_memories

logger = logging.getLogger(__name__)

FALLBACK_PATH = Path.home() / ".weft" / "fallback.md"

# Pool health check interval in seconds
_KEEPALIVE_INTERVAL = 300  # 5 minutes
# Fallback snapshot refresh interval in seconds
_FALLBACK_REFRESH_INTERVAL = 1800  # 30 minutes
# Startup retry config
_STARTUP_MAX_RETRIES = 5
_STARTUP_BASE_DELAY = 1.0  # seconds, doubles each retry


class UserIdentityMiddleware(BaseHTTPMiddleware):
    """Extract user identity from Authorization header and set contextvar.

    Graceful degradation: missing, invalid, or expired tokens are silently
    ignored — the request proceeds with current_user_id=None (global-only
    visibility under RLS).
    """

    async def dispatch(self, request: Request, call_next):
        auth_header = request.REDACTEDget("authorization")
        user_id = extract_user_id_from_header(auth_header)
        token = current_user_id.set(user_id)
        try:
            return await call_next(request)
        finally:
            current_user_id.reset(token)


# Starlette Middleware descriptor for passing to FastMCP http_app / run()
user_identity_middleware = Middleware(UserIdentityMiddleware)


@dataclass
class AppContext:
    pool: asyncpg.Pool
    cache: Cache | NullCache
    embedding: EmbeddingProvider
    config: WeftConfig
    _keepalive_task: asyncio.Task | None = field(default=None, repr=False)
    _fallback_task: asyncio.Task | None = field(default=None, repr=False)


async def _connect_with_retry(
    connect_fn,
    label: str,
    max_retries: int = _STARTUP_MAX_RETRIES,
    base_delay: float = _STARTUP_BASE_DELAY,
):
    """Call *connect_fn* with exponential backoff on failure.

    Returns the result of *connect_fn()* on success.
    Raises the last exception after exhausting retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await connect_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s connection failed (attempt %d/%d): %s — retrying in %.1fs",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s connection failed after %d attempts: %s",
                    label, max_retries + 1, exc,
                )
    raise last_exc  # type: ignore[misc]


async def _pool_keepalive(ctx: AppContext) -> None:
    """Periodically ping the pool; recreate it if connections are stale."""
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        try:
            async with ctx.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            logger.warning("Pool health check failed: %s — recreating pool", exc)
            old_pool = ctx.pool
            try:
                new_pool = await create_pool(ctx.config)
                ctx.pool = new_pool
                logger.info("Pool recreated successfully")
            except Exception as create_exc:
                logger.error("Failed to recreate pool: %s", create_exc)
                continue
            # Best-effort close of the old pool
            try:
                await old_pool.close()
            except Exception:
                pass


async def _redis_keepalive(ctx: AppContext) -> None:
    """Periodically ping Redis; recreate the client if the connection is stale."""
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        try:
            await ctx.cache._redis.ping()
        except Exception as exc:
            logger.warning("Redis health check failed: %s — reconnecting", exc)
            old_redis = ctx.cache._redis
            try:
                new_redis = aioredis.from_url(
                    ctx.config.redis.url, decode_responses=True,
                )
                await new_redis.ping()
                ctx.cache._redis = new_redis
                logger.info("Redis reconnected successfully")
            except Exception as create_exc:
                logger.error("Failed to reconnect Redis: %s", create_exc)
                continue
            try:
                await old_redis.aclose()
            except Exception:
                pass


async def _refresh_fallback(ctx: AppContext) -> None:
    """Periodically re-export the fallback snapshot."""
    while True:
        await asyncio.sleep(_FALLBACK_REFRESH_INTERVAL)
        await _write_fallback_snapshot(ctx.pool)


async def _write_fallback_snapshot(pool: asyncpg.Pool) -> None:
    """Export active memories to the fallback markdown file."""
    try:
        from weft.exporter import export_memories

        content = await export_memories(pool, format="md")
        FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_PATH.write_text(content, encoding="utf-8")
        logger.info("Fallback snapshot written to %s", FALLBACK_PATH)
    except Exception as e:
        logger.warning("Failed to write fallback snapshot: %s", e)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize database, Redis, and embedding provider."""
    config = load_config()
    logging.basicConfig(level=getattr(logging, config.log_level))

    # Database (with retry)
    pool = await _connect_with_retry(
        lambda: create_pool(config),
        "Postgres",
    )
    await run_migrations(pool)

    # Export fallback snapshot
    await _write_fallback_snapshot(pool)

    # Redis (optional — use NullCache if not configured)
    r: aioredis.Redis | None = None
    cache: Cache | NullCache
    _redis_url = config.redis.url
    if _redis_url:
        try:
            async def _connect_redis():
                client = aioredis.from_url(_redis_url, decode_responses=True)
                await client.ping()
                return client

            r = await _connect_with_retry(_connect_redis, "Redis")
            cache = Cache(r)
        except Exception as exc:
            logger.warning("Redis unavailable, using NullCache: %s", exc)
            r = None
            cache = NullCache()
    else:
        logger.info("No Redis URL configured, using NullCache")
        cache = NullCache()

    # Embedding provider
    embedding = get_provider(config.embedding.provider, model_name=config.embedding.model)

    # Seed memories on fresh installs (best-effort, never blocks startup)
    try:
        seeded = await seed_memories(pool, embedding)
        if seeded:
            logger.info("Seeded %d starter memories", seeded)
    except Exception as exc:
        logger.warning("Seed bootstrapping failed (non-fatal): %s", exc)

    ctx = AppContext(pool=pool, cache=cache, embedding=embedding, config=config)
    _app_ctx_ref.ctx = ctx

    # Start background tasks
    ctx._keepalive_task = asyncio.create_task(_pool_keepalive(ctx))
    _redis_task = asyncio.create_task(_redis_keepalive(ctx)) if r else None
    ctx._fallback_task = asyncio.create_task(_refresh_fallback(ctx))

    try:
        yield ctx
    finally:
        _app_ctx_ref.ctx = None
        for task in (ctx._keepalive_task, _redis_task, ctx._fallback_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await pool.close()
        if r is not None:
            await r.aclose()


class _AppCtxRef:
    """Module-level holder for the AppContext, set during lifespan."""
    ctx: AppContext | None = None


_app_ctx_ref = _AppCtxRef()

_config = load_config()
_auth = get_auth_provider(_config.api_key, _config.is_production)
mcp = FastMCP("weft", lifespan=lifespan, auth=_auth)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    """Health check endpoint — unauthenticated, used by Fly.io."""
    ctx = _app_ctx_ref.ctx
    if ctx is None:
        return JSONResponse(
            {"status": "unhealthy", "error": "server starting up"},
            status_code=503,
        )
    try:
        async with ctx.pool.acquire(timeout=3.0) as conn:
            await conn.fetchval("SELECT 1")
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return JSONResponse({"status": "unhealthy", "error": str(exc)}, status_code=503)
