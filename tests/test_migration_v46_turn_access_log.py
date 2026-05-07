"""Tests for migration 46: turn_access_log table + episode_turns usefulness columns."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_turn_access_log_table_exists(pool):
    """turn_access_log table is created with expected columns."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'turn_access_log'
        ORDER BY ordinal_position
        """
    )
    by_name = {r["column_name"]: r for r in cols}
    assert set(by_name) == {"session_id", "turn_id", "tool_name", "accessed_at"}
    assert by_name["session_id"]["data_type"] == "text"
    assert by_name["turn_id"]["data_type"] == "text"
    assert by_name["turn_id"]["is_nullable"] == "NO"
    assert by_name["tool_name"]["is_nullable"] == "NO"
    assert by_name["accessed_at"]["data_type"] == "timestamp with time zone"


@pytest.mark.asyncio
async def test_turn_access_log_primary_key(pool):
    """Composite PK on (session_id, turn_id)."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'turn_access_log'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    assert [r["column_name"] for r in rows] == ["session_id", "turn_id"]


@pytest.mark.asyncio
async def test_turn_access_log_indexes_exist(pool):
    """Session and accessed_at indexes are present."""
    for idx_name in ("idx_turn_access_log_session", "idx_turn_access_log_accessed_at"):
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
            idx_name,
        )
        assert exists, f"missing index {idx_name}"


@pytest.mark.asyncio
async def test_turn_access_log_rls_enabled(pool):
    """RLS is enabled on turn_access_log (mirrors memory_access_log)."""
    enabled = await pool.fetchval(
        """
        SELECT relrowsecurity FROM pg_class
        WHERE relname = 'turn_access_log'
        """
    )
    assert enabled is True


@pytest.mark.asyncio
async def test_turn_access_log_fk_cascade_on_turn_delete(pool):
    """Deleting an episode_turns row cascades to turn_access_log."""
    from weft.episode_turns import append_turn
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole

    ep = await create_episode(pool, EpisodeCreate(title="cascade-test"))
    turn = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="hi"),
    )
    await pool.execute(
        "INSERT INTO turn_access_log (session_id, turn_id, tool_name) "
        "VALUES ($1, $2, $3)",
        "sess-1", turn.id, "weft_recall",
    )
    count_before = await pool.fetchval(
        "SELECT count(*) FROM turn_access_log WHERE turn_id = $1", turn.id,
    )
    assert count_before == 1

    await pool.execute("DELETE FROM episode_turns WHERE id = $1", turn.id)

    count_after = await pool.fetchval(
        "SELECT count(*) FROM turn_access_log WHERE turn_id = $1", turn.id,
    )
    assert count_after == 0


@pytest.mark.asyncio
async def test_episode_turns_usefulness_columns_exist(pool):
    """episode_turns has usefulness_score, usefulness_count, last_boosted_at."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, column_default
        FROM information_schema.columns
        WHERE table_name = 'episode_turns'
          AND column_name IN ('usefulness_score', 'usefulness_count', 'last_boosted_at')
        """
    )
    by_name = {r["column_name"]: r for r in cols}
    assert set(by_name) == {"usefulness_score", "usefulness_count", "last_boosted_at"}
    assert by_name["usefulness_score"]["data_type"] == "real"
    assert by_name["usefulness_count"]["data_type"] == "integer"
    assert by_name["last_boosted_at"]["data_type"] == "timestamp with time zone"
    # Defaults match the v17/v04 belief-tier pattern.
    assert by_name["usefulness_score"]["column_default"] is not None
    assert "0.7" in by_name["usefulness_score"]["column_default"]
    assert by_name["usefulness_count"]["column_default"] is not None
    assert "0" in by_name["usefulness_count"]["column_default"]
    # last_boosted_at is nullable with no default (matches v17 on memories).
    assert by_name["last_boosted_at"]["column_default"] in (None, "NULL::timestamp with time zone")


@pytest.mark.asyncio
async def test_episode_turns_usefulness_defaults_applied_on_insert(pool):
    """New episode_turns rows get default usefulness_score=0.7, count=0."""
    from weft.episode_turns import append_turn
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole

    ep = await create_episode(pool, EpisodeCreate(title="defaults-test"))
    turn = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="hi"),
    )
    row = await pool.fetchrow(
        "SELECT usefulness_score, usefulness_count, last_boosted_at "
        "FROM episode_turns WHERE id = $1",
        turn.id,
    )
    assert row["usefulness_score"] == pytest.approx(0.7)
    assert row["usefulness_count"] == 0
    assert row["last_boosted_at"] is None


@pytest.mark.asyncio
async def test_v46_migration_idempotent(pool):
    """Running migration 46 SQL twice does not error."""
    from weft.db.migrations import MIGRATIONS

    v46_sql = [sql for version, _, sql in MIGRATIONS if version == 46]
    assert len(v46_sql) == 1, "expected migration 46 in MIGRATIONS list"

    # Migration already ran via fixture. Run the raw SQL again — must not raise.
    await pool.execute(v46_sql[0])
