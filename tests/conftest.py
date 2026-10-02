"""Shared test fixtures using testcontainers for real Postgres (pgvector) + Redis."""

from __future__ import annotations

import os

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

@pytest.fixture(scope="session")
def pg_container():
    """Start Postgres only when a test requests a database fixture."""
    try:
        container = PostgresContainer("pgvector/pgvector:pg16")
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker unavailable: skipping DB-backed tests ({exc})")
    yield container
    container.stop()


@pytest.fixture(scope="session")
def redis_container():
    """Start Redis only when a test requests a Redis fixture."""
    try:
        container = RedisContainer("redis:7-alpine")
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker unavailable: skipping Redis-backed tests ({exc})")
    yield container
    container.stop()


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
async def pool(pg_container):
    """Function-scoped asyncpg pool — migrations + clean slate each test."""
    dsn = pg_container.get_connection_url().replace("+psycopg2", "")
    p = await asyncpg.create_pool(
        dsn, min_size=2, max_size=5, init=_test_init, setup=_test_setup,
    )
    await run_migrations(p)
    await register_pgvector_codec(p)
    # TRUNCATE resets tables and HNSW index state cleanly (DELETE leaves
    # dead tuples in the index which can cause approximate search to miss rows)
    await p.execute("TRUNCATE memory_access_log, turn_access_log, entity_mentions, belief_claims, shuttle_claims, episode_turns, episode_memories, memory_relationships, entities, episodes, memories, behaviors, weft_metadata, modes, alerts, alert_state, check_ins, autonomy_policies, policy_calibration_events, autonomy_overrides, cost_entries, cost_enforcement_state, triggers, calibration_records, degradation_policies, audit_backfill_user_id, workspace_members, workspaces, trackers, weft_tokens, weft_recall_queries, weft_recovery_attempts, replay_queue, weft_counters, weft_tool_usage_daily, weft_tool_usage_coverage, topic_digests, topic_resolution_aliases, recall_canary_audit, recall_canary, board_triage_events, board_feedback_proposals CASCADE")
    yield p
    await p.close()


@pytest.fixture
def pg_dsn(pg_container):
    """Raw DSN for the shared test Postgres container.

    Lets a test open a *codec-less* asyncpg pool — mirroring the bare
    connection the backup workflow and `weft backup` CLI use (no pgvector
    codec registered) — to exercise serialization paths the codec-equipped
    `pool` fixture hides.
    """
    return pg_container.get_connection_url().replace("+psycopg2", "")


@pytest.fixture
async def redis_conn(redis_container):
    """Function-scoped Redis connection — flushed each test."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    r = aioredis.Redis(host=host, port=int(port), decode_responses=True)
    await r.flushdb()
    yield r
    await r.aclose()
