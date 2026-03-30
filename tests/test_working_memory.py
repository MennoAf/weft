"""Tests for working memory — TTL episodes, get_working_memory, expire_stale_episodes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.episodes import (
    create_episode,
    expire_stale_episodes,
    get_episode,
    get_working_memory,
)
from weft.models import EpisodeCreate, EpisodeStatus


# --- Model tests ---


def test_episode_status_has_expired():
    assert EpisodeStatus.expired.value == "expired"


def test_episode_create_ttl_hours():
    ec = EpisodeCreate(title="TTL episode", ttl_hours=24.0)
    assert ec.ttl_hours == 24.0


def test_episode_create_ttl_hours_default_none():
    ec = EpisodeCreate(title="No TTL")
    assert ec.ttl_hours is None


def test_episode_is_expired_property():
    from weft.models import Episode

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    ep = Episode(title="Expired", expires_at=past)
    assert ep.is_expired is True


def test_episode_not_expired_property():
    from weft.models import Episode

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    ep = Episode(title="Still alive", expires_at=future)
    assert ep.is_expired is False


def test_episode_no_ttl_not_expired():
    from weft.models import Episode

    ep = Episode(title="No TTL")
    assert ep.is_expired is False


def test_episode_to_dict_includes_expires_at():
    from weft.models import Episode

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    ep = Episode(title="With TTL", expires_at=future)
    d = ep.to_dict()
    assert "expires_at" in d
    assert d["expires_at"] is not None


# --- Migration tests ---


async def test_episodes_has_expires_at_column(pool):
    rows = await pool.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'episodes'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"] for r in rows}
    assert "expires_at" in columns


async def test_episodes_expires_at_index(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'episodes'"
    )
    index_names = {r["indexname"] for r in rows}
    assert "idx_episodes_expires" in index_names


# --- Store tests: create with TTL ---


class TestCreateEpisodeWithTTL:
    @pytest.mark.asyncio
    async def test_create_with_ttl(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="TTL episode", ttl_hours=24.0,
        ))
        assert ep.expires_at is not None
        assert ep.expires_at > datetime.now(timezone.utc)
        # Should be roughly 24h from now
        delta = ep.expires_at - datetime.now(timezone.utc)
        assert 23.9 < delta.total_seconds() / 3600 < 24.1

    @pytest.mark.asyncio
    async def test_create_without_ttl(self, pool):
        ep = await create_episode(pool, EpisodeCreate(title="No TTL"))
        assert ep.expires_at is None

    @pytest.mark.asyncio
    async def test_ttl_persists_in_db(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="Persisted TTL", ttl_hours=1.0,
        ))
        fetched = await get_episode(pool, ep.id)
        assert fetched is not None
        assert fetched.expires_at is not None


# --- Store tests: get_working_memory ---


class TestGetWorkingMemory:
    @pytest.mark.asyncio
    async def test_returns_open_episodes(self, pool):
        await create_episode(pool, EpisodeCreate(title="Open ep"))
        wm = await get_working_memory(pool)
        titles = {ewm.episode.title for ewm in wm}
        assert "Open ep" in titles

    @pytest.mark.asyncio
    async def test_excludes_closed_episodes(self, pool):
        from weft.episodes import close_episode

        ep = await create_episode(pool, EpisodeCreate(title="Will close"))
        await close_episode(pool, ep.id)
        wm = await get_working_memory(pool)
        ids = {ewm.episode.id for ewm in wm}
        assert ep.id not in ids

    @pytest.mark.asyncio
    async def test_excludes_expired_ttl(self, pool):
        # Create episode with TTL in the past by inserting directly
        ep = await create_episode(pool, EpisodeCreate(
            title="Already expired", ttl_hours=0.001,
        ))
        # Force expires_at to the past
        await pool.execute(
            "UPDATE episodes SET expires_at = $1 WHERE id = $2",
            datetime.now(timezone.utc) - timedelta(hours=1),
            ep.id,
        )
        wm = await get_working_memory(pool)
        ids = {ewm.episode.id for ewm in wm}
        assert ep.id not in ids

    @pytest.mark.asyncio
    async def test_includes_no_ttl_episodes(self, pool):
        ep = await create_episode(pool, EpisodeCreate(title="No TTL ep"))
        wm = await get_working_memory(pool)
        ids = {ewm.episode.id for ewm in wm}
        assert ep.id in ids

    @pytest.mark.asyncio
    async def test_includes_future_ttl_episodes(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="Future TTL", ttl_hours=24.0,
        ))
        wm = await get_working_memory(pool)
        ids = {ewm.episode.id for ewm in wm}
        assert ep.id in ids


# --- Store tests: expire_stale_episodes ---


class TestExpireStaleEpisodes:
    @pytest.mark.asyncio
    async def test_expires_past_ttl_episodes(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="Stale ep", ttl_hours=0.001,
        ))
        # Force expires_at to the past
        await pool.execute(
            "UPDATE episodes SET expires_at = $1 WHERE id = $2",
            datetime.now(timezone.utc) - timedelta(hours=1),
            ep.id,
        )
        count = await expire_stale_episodes(pool)
        assert count >= 1

        fetched = await get_episode(pool, ep.id)
        assert fetched.status == EpisodeStatus.expired
        assert fetched.ended_at is not None

    @pytest.mark.asyncio
    async def test_does_not_expire_future_ttl(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="Future ep", ttl_hours=24.0,
        ))
        count = await expire_stale_episodes(pool)
        fetched = await get_episode(pool, ep.id)
        assert fetched.status == EpisodeStatus.open

    @pytest.mark.asyncio
    async def test_does_not_expire_no_ttl(self, pool):
        ep = await create_episode(pool, EpisodeCreate(title="No TTL"))
        await expire_stale_episodes(pool)
        fetched = await get_episode(pool, ep.id)
        assert fetched.status == EpisodeStatus.open

    @pytest.mark.asyncio
    async def test_does_not_re_expire_already_expired(self, pool):
        ep = await create_episode(pool, EpisodeCreate(
            title="Double expire", ttl_hours=0.001,
        ))
        await pool.execute(
            "UPDATE episodes SET expires_at = $1 WHERE id = $2",
            datetime.now(timezone.utc) - timedelta(hours=1),
            ep.id,
        )
        count1 = await expire_stale_episodes(pool)
        assert count1 >= 1
        count2 = await expire_stale_episodes(pool)
        # Should not re-expire since status is now 'expired', not 'open'
        fetched = await get_episode(pool, ep.id)
        assert fetched.status == EpisodeStatus.expired

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_expire(self, pool):
        count = await expire_stale_episodes(pool)
        assert count == 0
