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

# Default test user. After migration 36 (NOT NULL user_id), tests cannot
# rely on "no auth = global" — every write needs an explicit user_id. The
# pool's ``setup`` callback issues ``SET app.user_id = ...`` on every pool
# acquire so direct ``pool.execute()`` writes (which don't go through
# ``acquire()``) satisfy the NOT NULL + RLS contract. Tests that exercise
# the auth chain (or specific users) RESET this and set their own via
# ``current_user_id.set(...)``.
DEFAULT_TEST_USER_ID = "test-user-default"

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


async def _test_init(conn):
    """Connection init — pgvector codec only."""
    await _pgvector_codec_init(conn)


async def _test_setup(conn):
    """Connection setup — runs on every acquire (after asyncpg's DISCARD ALL).

    Sets a session-level ``app.user_id`` so direct ``pool.execute()`` calls
    (which don't go through ``acquire()``) still satisfy the migration-34
    NOT NULL + RLS WITH CHECK contract. Tests that exercise the auth chain
    explicitly RESET this when verifying unauthenticated semantics.
    """
    await conn.execute(f"SET app.user_id = '{DEFAULT_TEST_USER_ID}'")


@pytest.fixture
async def pool():
    """Function-scoped asyncpg pool — migrations + clean slate each test."""
    dsn = _pg_container.get_connection_url().replace("+psycopg2", "")
    p = await asyncpg.create_pool(
        dsn, min_size=2, max_size=5, init=_test_init, setup=_test_setup,
    )
    await run_migrations(p)
    await register_pgvector_codec(p)
    # TRUNCATE resets tables and HNSW index state cleanly (DELETE leaves
    # dead tuples in the index which can cause approximate search to miss rows)
    await p.execute("TRUNCATE memory_access_log, turn_access_log, entity_mentions, belief_claims, shuttle_claims, episode_turns, episode_memories, memory_relationships, entities, episodes, memories, behaviors, weft_metadata, modes, alerts, alert_state, check_ins, autonomy_policies, policy_calibration_events, autonomy_overrides, cost_entries, cost_enforcement_state, triggers, calibration_records, degradation_policies, audit_backfill_user_id, workspace_members, workspaces, trackers, weft_tokens, weft_recall_queries, replay_queue, weft_counters, topic_digests, topic_resolution_aliases, recall_canary_audit, recall_canary CASCADE")
    yield p
    await p.close()


@pytest.fixture
def pg_dsn():
    """Raw DSN for the shared test Postgres container.

    Lets a test open a *codec-less* asyncpg pool — mirroring the bare
    connection the backup workflow and `weft backup` CLI use (no pgvector
    codec registered) — to exercise serialization paths the codec-equipped
    `pool` fixture hides.
    """
    return _pg_container.get_connection_url().replace("+psycopg2", "")


@pytest.fixture
async def redis_conn():
    """Function-scoped Redis connection — flushed each test."""
    host = _redis_container.get_container_host_ip()
    port = _redis_container.get_exposed_port(6379)
    r = aioredis.Redis(host=host, port=int(port), decode_responses=True)
    await r.flushdb()
    yield r
    await r.aclose()
