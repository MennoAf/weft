"""Weft MCP server — FastMCP with stdio transport."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
from fastmcp import FastMCP

from weft.cache import Cache
from weft.config import WeftConfig, load_config
from weft.db.migrations import run_migrations
from weft.embeddings import get_provider
from weft.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

FALLBACK_PATH = Path.home() / ".weft" / "fallback.md"

# Pool health check interval in seconds
_KEEPALIVE_INTERVAL = 300  # 5 minutes


@dataclass
class AppContext:
    pool: asyncpg.Pool
    cache: Cache
    embedding: EmbeddingProvider
    config: WeftConfig
    _keepalive_task: asyncio.Task | None = field(default=None, repr=False)


async def _pool_keepalive(ctx: AppContext) -> None:
    """Periodically ping the pool; recreate it if connections are stale."""
    dsn = ctx.config.database.url
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        try:
            async with ctx.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            logger.warning("Pool health check failed: %s — recreating pool", exc)
            old_pool = ctx.pool
            try:
                new_pool = await asyncpg.create_pool(
                    dsn,
                    min_size=ctx.config.database.pool_min_size,
                    max_size=ctx.config.database.pool_max_size,
                )
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


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize database, Redis, and embedding provider."""
    config = load_config()
    logging.basicConfig(level=getattr(logging, config.log_level))

    # Database
    dsn = config.database.url
    pool = await asyncpg.create_pool(dsn, min_size=config.database.pool_min_size, max_size=config.database.pool_max_size)
    await run_migrations(pool)

    # Export fallback snapshot
    try:
        from weft.exporter import export_memories

        content = await export_memories(pool, format="md")
        FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_PATH.write_text(content, encoding="utf-8")
        logger.info("Fallback snapshot written to %s", FALLBACK_PATH)
    except Exception as e:
        logger.warning("Failed to write fallback snapshot: %s", e)

    # Redis
    r = aioredis.from_url(config.redis.url, decode_responses=True)
    cache = Cache(r)

    # Embedding provider
    embedding = get_provider(config.embedding.provider, model_name=config.embedding.model)

    ctx = AppContext(pool=pool, cache=cache, embedding=embedding, config=config)

    # Start background keepalive
    ctx._keepalive_task = asyncio.create_task(_pool_keepalive(ctx))

    try:
        yield ctx
    finally:
        ctx._keepalive_task.cancel()
        try:
            await ctx._keepalive_task
        except asyncio.CancelledError:
            pass
        await pool.close()
        await r.aclose()


mcp = FastMCP("weft", lifespan=lifespan)
