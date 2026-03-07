"""Tests for episodes store layer (CRUD + timeline queries)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.episodes import (
    add_memory_to_episode,
    close_episode,
    create_episode,
    get_episode,
    get_episode_memories,
    get_episodes_for_memory,
    list_episodes,
    remove_memory_from_episode,
    timeline_query,
)
from weft.models import EpisodeCreate, EpisodeStatus, MemoryCreate, MemoryType
from weft.store import store_memory


# --- Helpers ---

async def _make_memory(pool, content="test memory"):
    return await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content=content,
    ))


async def _make_episode(pool, title="test episode", **kwargs):
    return await create_episode(pool, EpisodeCreate(title=title, **kwargs))


# --- create_episode ---


async def test_create_episode_minimal(pool):
    ep = await _make_episode(pool)
    assert ep.id.startswith("weft-")
    assert ep.title == "test episode"
    assert ep.status == EpisodeStatus.open
    assert ep.ended_at is None
    assert ep.summary is None
    assert ep.project_id is None


async def test_create_episode_full(pool):
    ep = await create_episode(pool, EpisodeCreate(
        title="Debug session",
        summary="Fixed auth bug",
        project_id="proj-1",
        agent_id="warp",
    ))
    assert ep.title == "Debug session"
    assert ep.summary == "Fixed auth bug"
    assert ep.project_id == "proj-1"
    assert ep.agent_id == "warp"


# --- get_episode ---


async def test_get_episode_found(pool):
    ep = await _make_episode(pool)
    found = await get_episode(pool, ep.id)
    assert found is not None
    assert found.id == ep.id


async def test_get_episode_not_found(pool):
    assert await get_episode(pool, "nonexistent") is None


# --- list_episodes ---


async def test_list_episodes_empty(pool):
    assert await list_episodes(pool) == []


async def test_list_episodes_filters_by_status(pool):
    ep1 = await _make_episode(pool, "open one")
    ep2 = await _make_episode(pool, "to close")
    await close_episode(pool, ep2.id)

    open_eps = await list_episodes(pool, status=EpisodeStatus.open)
    assert len(open_eps) == 1
    assert open_eps[0].title == "open one"

    closed_eps = await list_episodes(pool, status=EpisodeStatus.closed)
    assert len(closed_eps) == 1
    assert closed_eps[0].title == "to close"


async def test_list_episodes_or_null_scoping(pool):
    await _make_episode(pool, "global")
    await _make_episode(pool, "proj-specific", project_id="proj-1")
    await _make_episode(pool, "other-proj", project_id="proj-2")

    results = await list_episodes(pool, project_id="proj-1")
    titles = {ep.title for ep in results}
    assert "global" in titles
    assert "proj-specific" in titles
    assert "other-proj" not in titles


async def test_list_episodes_ordered_by_started_at_desc(pool):
    ep1 = await _make_episode(pool, "first")
    ep2 = await _make_episode(pool, "second")
    ep3 = await _make_episode(pool, "third")

    results = await list_episodes(pool)
    assert results[0].title == "third"
    assert results[-1].title == "first"


async def test_list_episodes_respects_limit(pool):
    for i in range(5):
        await _make_episode(pool, f"ep-{i}")

    results = await list_episodes(pool, limit=3)
    assert len(results) == 3


# --- close_episode ---


async def test_close_episode(pool):
    ep = await _make_episode(pool)
    assert ep.status == EpisodeStatus.open

    closed = await close_episode(pool, ep.id)
    assert closed is not None
    assert closed.status == EpisodeStatus.closed
    assert closed.ended_at is not None
    assert closed.updated_at > ep.updated_at


async def test_close_episode_with_summary(pool):
    ep = await _make_episode(pool)
    closed = await close_episode(pool, ep.id, summary="Wrapped up the session")
    assert closed.summary == "Wrapped up the session"


async def test_close_episode_not_found(pool):
    assert await close_episode(pool, "nonexistent") is None


# --- add_memory_to_episode ---


async def test_add_memory_to_episode(pool):
    ep = await _make_episode(pool)
    mem = await _make_memory(pool)

    created = await add_memory_to_episode(pool, ep.id, mem.id)
    assert created is True

    memories = await get_episode_memories(pool, ep.id)
    assert len(memories) == 1
    assert memories[0].id == mem.id


async def test_add_memory_auto_position(pool):
    ep = await _make_episode(pool)
    m1 = await _make_memory(pool, "first")
    m2 = await _make_memory(pool, "second")
    m3 = await _make_memory(pool, "third")

    await add_memory_to_episode(pool, ep.id, m1.id)
    await add_memory_to_episode(pool, ep.id, m2.id)
    await add_memory_to_episode(pool, ep.id, m3.id)

    memories = await get_episode_memories(pool, ep.id)
    assert len(memories) == 3
    assert memories[0].content == "first"
    assert memories[1].content == "second"
    assert memories[2].content == "third"


async def test_add_memory_explicit_position(pool):
    ep = await _make_episode(pool)
    m1 = await _make_memory(pool, "at-5")
    m2 = await _make_memory(pool, "at-0")

    await add_memory_to_episode(pool, ep.id, m1.id, position=5)
    await add_memory_to_episode(pool, ep.id, m2.id, position=0)

    memories = await get_episode_memories(pool, ep.id)
    assert memories[0].content == "at-0"
    assert memories[1].content == "at-5"


async def test_add_memory_idempotent(pool):
    ep = await _make_episode(pool)
    mem = await _make_memory(pool)

    assert await add_memory_to_episode(pool, ep.id, mem.id) is True
    assert await add_memory_to_episode(pool, ep.id, mem.id) is False

    memories = await get_episode_memories(pool, ep.id)
    assert len(memories) == 1


# --- remove_memory_from_episode ---


async def test_remove_memory_from_episode(pool):
    ep = await _make_episode(pool)
    mem = await _make_memory(pool)
    await add_memory_to_episode(pool, ep.id, mem.id)

    removed = await remove_memory_from_episode(pool, ep.id, mem.id)
    assert removed is True

    memories = await get_episode_memories(pool, ep.id)
    assert len(memories) == 0


async def test_remove_memory_not_linked(pool):
    ep = await _make_episode(pool)
    assert await remove_memory_from_episode(pool, ep.id, "nonexistent") is False


# --- get_episode_memories ---


async def test_get_episode_memories_excludes_archived(pool):
    ep = await _make_episode(pool)
    mem = await _make_memory(pool)
    await add_memory_to_episode(pool, ep.id, mem.id)

    # Archive the memory
    await pool.execute(
        "UPDATE memories SET status = 'archived' WHERE id = $1", mem.id,
    )

    memories = await get_episode_memories(pool, ep.id)
    assert len(memories) == 0


async def test_get_episode_memories_respects_limit(pool):
    ep = await _make_episode(pool)
    for i in range(10):
        mem = await _make_memory(pool, f"mem-{i}")
        await add_memory_to_episode(pool, ep.id, mem.id)

    memories = await get_episode_memories(pool, ep.id, limit=5)
    assert len(memories) == 5


# --- get_episodes_for_memory ---


async def test_get_episodes_for_memory(pool):
    ep1 = await _make_episode(pool, "episode one")
    ep2 = await _make_episode(pool, "episode two")
    mem = await _make_memory(pool)

    await add_memory_to_episode(pool, ep1.id, mem.id)
    await add_memory_to_episode(pool, ep2.id, mem.id)

    episodes = await get_episodes_for_memory(pool, mem.id)
    assert len(episodes) == 2
    titles = {ep.title for ep in episodes}
    assert "episode one" in titles
    assert "episode two" in titles


async def test_get_episodes_for_memory_none(pool):
    mem = await _make_memory(pool)
    assert await get_episodes_for_memory(pool, mem.id) == []


# --- timeline_query ---


async def test_timeline_query_overlapping(pool):
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    two_hours_ago = now - timedelta(hours=2)

    # Create and close an episode that spans [2h ago, 1h ago]
    ep = await _make_episode(pool, "past episode")
    await pool.execute(
        "UPDATE episodes SET started_at = $1, ended_at = $2 WHERE id = $3",
        two_hours_ago, hour_ago, ep.id,
    )

    # Query overlapping range
    results = await timeline_query(
        pool,
        start=two_hours_ago - timedelta(minutes=30),
        end=hour_ago + timedelta(minutes=30),
    )
    assert len(results) == 1
    assert results[0].id == ep.id


async def test_timeline_query_no_overlap(pool):
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    two_hours_ago = now - timedelta(hours=2)

    ep = await _make_episode(pool, "past episode")
    await pool.execute(
        "UPDATE episodes SET started_at = $1, ended_at = $2 WHERE id = $3",
        two_hours_ago, hour_ago, ep.id,
    )

    # Query a range that doesn't overlap
    results = await timeline_query(
        pool,
        start=now,
        end=now + timedelta(hours=1),
    )
    assert len(results) == 0


async def test_timeline_query_open_episodes(pool):
    """Open episodes (ended_at IS NULL) match any range after their start."""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    ep = await _make_episode(pool, "open episode")
    await pool.execute(
        "UPDATE episodes SET started_at = $1 WHERE id = $2",
        hour_ago, ep.id,
    )

    results = await timeline_query(pool, start=now, end=now + timedelta(hours=1))
    assert len(results) == 1
    assert results[0].title == "open episode"


async def test_timeline_query_project_scoped(pool):
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    await _make_episode(pool, "global ep")
    await _make_episode(pool, "proj ep", project_id="proj-1")
    await _make_episode(pool, "other ep", project_id="proj-2")

    results = await timeline_query(
        pool, start=hour_ago, end=now + timedelta(hours=1),
        project_id="proj-1",
    )
    titles = {ep.title for ep in results}
    assert "global ep" in titles
    assert "proj ep" in titles
    assert "other ep" not in titles


async def test_timeline_query_status_filter(pool):
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    ep1 = await _make_episode(pool, "open")
    ep2 = await _make_episode(pool, "closed")
    await close_episode(pool, ep2.id)

    results = await timeline_query(
        pool, start=hour_ago, end=now + timedelta(hours=1),
        status=EpisodeStatus.open,
    )
    assert len(results) == 1
    assert results[0].title == "open"
