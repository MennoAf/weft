"""Weft MCP server — FastMCP with stdio transport."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
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


@dataclass
class AppContext:
    pool: asyncpg.Pool
    cache: Cache
    embedding: EmbeddingProvider
    config: WeftConfig


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
    try:
        yield ctx
    finally:
        await pool.close()
        await r.aclose()


mcp = FastMCP("weft", lifespan=lifespan)
