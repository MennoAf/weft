"""Shared test fixtures using testcontainers for real Postgres (pgvector) + Redis."""

from __future__ import annotations

import os
import subprocess

# Prevent load_dotenv from polluting test environment
os.environ["WEFT_TESTING"] = "1"

import asyncpg
import pytest
import redis.asyncio as aioredis
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
from weft.db.migrations import run_migrations

# Module-level containers — started once, shared across all tests
_pg_container: PostgresContainer | None = None
_redis_container: RedisContainer | None = None


def _cleanup_stale_reaper():
    """Remove stale Ryuk reaper containers from previous test runs."""
    try:
        result = subprocess.run(
            [
                "docker", "ps", "-a",
                "--filter", "ancestor=testcontainers/ryuk",
                "--format", "{{.ID}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split("\n")
        for cid in container_ids:
            if cid:
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    capture_output=True,
                    timeout=10,
                )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def pytest_configure(config):
    """Start containers once for the entire test session."""
    global _pg_container, _redis_container
    _cleanup_stale_reaper()

    # pgvector image instead of plain postgres
    _pg_container = PostgresContainer("pgvector/pgvector:pg16")
    _pg_container.start()

    _redis_container = RedisContainer("redis:7-alpine")
    _redis_container.start()


def pytest_unconfigure(config):
    """Stop containers at end of session."""
    global _pg_container, _redis_container
    if _pg_container:
        _pg_container.stop()
    if _redis_container:
        _redis_container.stop()


@pytest.fixture
async def pool():
    """Function-scoped asyncpg pool — migrations + clean slate each test."""
    dsn = _pg_container.get_connection_url().replace("+psycopg2", "")
    p = await asyncpg.create_pool(dsn, min_size=2, max_size=5, init=_pgvector_codec_init)
    await run_migrations(p)
    await register_pgvector_codec(p)
    # TRUNCATE resets tables and HNSW index state cleanly (DELETE leaves
    # dead tuples in the index which can cause approximate search to miss rows)
    await p.execute("TRUNCATE memory_access_log, entity_mentions, episode_memories, memory_relationships, entities, episodes, memories, behaviors, weft_metadata, modes, alerts, check_ins CASCADE")
    yield p
    await p.close()


@pytest.fixture
async def redis_conn():
    """Function-scoped Redis connection — flushed each test."""
    host = _redis_container.get_container_host_ip()
    port = _redis_container.get_exposed_port(6379)
    r = aioredis.Redis(host=host, port=int(port), decode_responses=True)
    await r.flushdb()
    yield r
    await r.aclose()
