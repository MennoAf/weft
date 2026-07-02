"""Tests for migration 53: replay_queue table + RLS."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg
import pytest


def _replay_queue_id() -> str:
    """Generate a replay queue ID with rq- prefix."""
    return f"rq-{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_replay_queue_table_exists(pool):
    """replay_queue table is created with the expected columns."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'replay_queue'
        ORDER BY ordinal_position
        """
    )
    by_name = {r["column_name"]: r for r in cols}
    expected = {
        "id", "episode_id", "turn_ids", "reason", "status",
        "created_at", "user_id",
    }
    assert set(by_name) == expected

    assert by_name["id"]["data_type"] == "text"
    assert by_name["id"]["is_nullable"] == "NO"
    assert by_name["episode_id"]["data_type"] == "text"
    assert by_name["episode_id"]["is_nullable"] == "NO"
    assert by_name["turn_ids"]["data_type"] == "ARRAY"
    assert by_name["turn_ids"]["is_nullable"] == "NO"
    assert by_name["reason"]["data_type"] == "text"
    assert by_name["reason"]["is_nullable"] == "NO"
    assert by_name["status"]["data_type"] == "text"
    assert by_name["status"]["is_nullable"] == "NO"
    assert by_name["created_at"]["data_type"] == "timestamp with time zone"
    assert by_name["created_at"]["is_nullable"] == "NO"
    assert by_name["user_id"]["data_type"] == "text"
    assert by_name["user_id"]["is_nullable"] == "NO"


@pytest.mark.asyncio
async def test_replay_queue_primary_key(pool):
    """PK is id."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'replay_queue'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    assert [r["column_name"] for r in rows] == ["id"]


@pytest.mark.asyncio
async def test_replay_queue_indexes_exist(pool):
    """Status and episode_id indexes are present."""
    for idx_name in (
        "idx_replay_queue_status",
        "idx_replay_queue_episode",
    ):
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
            idx_name,
        )
        assert exists, f"missing index {idx_name}"


@pytest.mark.asyncio
async def test_replay_queue_status_check(pool):
    """Inserting with invalid status raises CheckViolationError."""
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate

    user_id = "test-user-default"
    ep = await create_episode(pool, EpisodeCreate(title="test-episode"))

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO replay_queue (
                        id, episode_id, turn_ids, reason, status, user_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6
                    )
                    """,
                    _replay_queue_id(),
                    ep.id,
                    ["et-aaa", "et-bbb"],
                    "test",
                    "invalid_status",
                    user_id,
                )


@pytest.mark.asyncio
async def test_replay_queue_rls_enabled(pool):
    """RLS is enabled on replay_queue."""
    enabled = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'replay_queue'"
    )
    assert enabled is True


@pytest.mark.asyncio
async def test_replay_queue_insert_select_under_rls(pool):
    """Insert one row under app.user_id GUC, then select it back via RLS."""
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate

    user_id = "test-user-replay-001"
    ep = await create_episode(pool, EpisodeCreate(title="replay-test-episode"))

    # Insert under app.user_id GUC
    rq_id = _replay_queue_id()
    turn_ids = ["et-turn-1", "et-turn-2"]
    reason = "correction"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO replay_queue (
                    id, episode_id, turn_ids, reason, status, user_id
                ) VALUES (
                    $1, $2, $3, $4, 'pending', $5
                )
                """,
                rq_id,
                ep.id,
                turn_ids,
                reason,
                user_id,
            )

    # Select back via RLS under the same user context
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            row = await conn.fetchrow(
                "SELECT id, episode_id, turn_ids, reason, status FROM replay_queue WHERE id = $1",
                rq_id,
            )

    assert row is not None
    assert row["id"] == rq_id
    assert row["episode_id"] == ep.id
    assert row["turn_ids"] == turn_ids
    assert row["reason"] == reason
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_replay_queue_with_multiple_users(pool):
    """Multiple users can insert and retrieve their own rows independently."""
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate

    user_a = "test-user-a"
    user_b = "test-user-b"

    ep_a = await create_episode(pool, EpisodeCreate(title="episode-a"))
    ep_b = await create_episode(pool, EpisodeCreate(title="episode-b"))

    # User A inserts a row in transaction
    rq_a_id = _replay_queue_id()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_a}'")
            await conn.execute(
                """
                INSERT INTO replay_queue (
                    id, episode_id, turn_ids, reason, status, user_id
                ) VALUES (
                    $1, $2, $3, $4, 'pending', $5
                )
                """,
                rq_a_id,
                ep_a.id,
                ["et-a"],
                "test-a",
                user_a,
            )

    # User B inserts a row in transaction
    rq_b_id = _replay_queue_id()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_b}'")
            await conn.execute(
                """
                INSERT INTO replay_queue (
                    id, episode_id, turn_ids, reason, status, user_id
                ) VALUES (
                    $1, $2, $3, $4, 'pending', $5
                )
                """,
                rq_b_id,
                ep_b.id,
                ["et-b"],
                "test-b",
                user_b,
            )

    # User A can retrieve their own row
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_a}'")
            row = await conn.fetchrow(
                "SELECT id, episode_id FROM replay_queue WHERE id = $1", rq_a_id
            )
    assert row is not None
    assert row["id"] == rq_a_id

    # User B can retrieve their own row
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_b}'")
            row = await conn.fetchrow(
                "SELECT id, episode_id FROM replay_queue WHERE id = $1", rq_b_id
            )
    assert row is not None
    assert row["id"] == rq_b_id


@pytest.mark.asyncio
async def test_v53_migration_idempotent(pool):
    """Running migration 53 SQL twice does not error."""
    from weft.db.migrations import MIGRATIONS

    v53_sql = [sql for version, _, sql in MIGRATIONS if version == 53]
    assert len(v53_sql) == 1, "expected migration 53 in MIGRATIONS list"

    # Migration already ran via fixture. Run the raw SQL again — must not raise.
    await pool.execute(v53_sql[0])


@pytest.mark.asyncio
async def test_count_stale_pending_replays(pool):
    """count_stale_pending_replays counts only pending rows past the staleness window."""
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate
    from weft.replay import (
        REPLAY_QUEUE_STALENESS_DAYS,
        count_stale_pending_replays,
    )
    from tests.conftest import DEFAULT_TEST_USER_ID

    ep = await create_episode(pool, EpisodeCreate(title="stale-count-episode"))
    stale_age = REPLAY_QUEUE_STALENESS_DAYS + 6  # comfortably past the window

    async def _insert(rq_id, *, status, age_days):
        await pool.execute(
            """
            INSERT INTO replay_queue (id, episode_id, turn_ids, reason, status, user_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, now() - ($7 || ' days')::interval)
            """,
            rq_id, ep.id, ["et-x"], "stale-count", status, DEFAULT_TEST_USER_ID, str(age_days),
        )

    baseline = await count_stale_pending_replays(pool)
    await _insert(_replay_queue_id(), status="pending", age_days=stale_age)   # counts
    await _insert(_replay_queue_id(), status="pending", age_days=1)           # fresh — no
    await _insert(_replay_queue_id(), status="done", age_days=stale_age)      # terminal — no
    await _insert(_replay_queue_id(), status="failed", age_days=stale_age)    # terminal — no

    assert await count_stale_pending_replays(pool) == baseline + 1


@pytest.mark.asyncio
async def test_count_stale_pending_replays_respects_window(pool):
    """A row just inside the window is not stale; the same row just outside is."""
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate
    from weft.replay import (
        REPLAY_QUEUE_STALENESS_DAYS,
        count_stale_pending_replays,
    )
    from tests.conftest import DEFAULT_TEST_USER_ID

    ep = await create_episode(pool, EpisodeCreate(title="stale-window-episode"))
    rq_id = _replay_queue_id()
    baseline = await count_stale_pending_replays(pool)

    # Inside the window (age < staleness): not counted.
    await pool.execute(
        """
        INSERT INTO replay_queue (id, episode_id, turn_ids, reason, status, user_id, created_at)
        VALUES ($1, $2, $3, 'w', 'pending', $4, now() - ($5 || ' days')::interval)
        """,
        rq_id, ep.id, ["et-y"], DEFAULT_TEST_USER_ID, str(REPLAY_QUEUE_STALENESS_DAYS - 2),
    )
    assert await count_stale_pending_replays(pool) == baseline

    # Age it past the window: now counted.
    await pool.execute(
        "UPDATE replay_queue SET created_at = now() - ($2 || ' days')::interval WHERE id = $1",
        rq_id, str(REPLAY_QUEUE_STALENESS_DAYS + 2),
    )
    assert await count_stale_pending_replays(pool) == baseline + 1
