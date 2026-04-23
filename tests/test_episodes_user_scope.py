"""Tests for user_id OR-NULL scoping in episodes list/search functions.

Verifies that user_id parameter correctly filters episodes:
- user_id=None: returns all rows (existing behavior)
- user_id="user-a": returns user-a rows + NULL rows, excludes other users
- user_id="user-a" with only NULL rows: returns them
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.episodes import (
    close_episode,
    create_episode,
    get_working_memory,
    list_episodes,
    timeline_query,
)
from weft.models import EpisodeCreate, EpisodeStatus


# --- Helpers ---


async def _make_episode(pool, title="test episode", user_id: str | None = None, **kwargs):
    """Create an episode. If user_id is provided, manually set it in DB."""
    ep = await create_episode(pool, EpisodeCreate(title=title, **kwargs))
    if user_id is not None:
        # Manually override user_id in the database
        await pool.execute(
            "UPDATE episodes SET user_id = $1 WHERE id = $2",
            user_id,
            ep.id,
        )
    return ep


# --- list_episodes with user_id ---


async def test_list_episodes_user_id_none_returns_all(pool):
    """user_id=None should return all episodes (existing behavior)."""
    await _make_episode(pool, "user-a ep", user_id="user-a")
    await _make_episode(pool, "user-b ep", user_id="user-b")
    await _make_episode(pool, "global ep", user_id=None)

    results = await list_episodes(pool, user_id=None)
    assert len(results) == 3
    titles = {ep.title for ep in results}
    assert "user-a ep" in titles
    assert "user-b ep" in titles
    assert "global ep" in titles


async def test_list_episodes_user_id_filters_to_user_and_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows, exclude others."""
    await _make_episode(pool, "user-a ep1", user_id="user-a")
    await _make_episode(pool, "user-a ep2", user_id="user-a")
    await _make_episode(pool, "user-b ep", user_id="user-b")
    await _make_episode(pool, "global ep", user_id=None)

    results = await list_episodes(pool, user_id="user-a")
    assert len(results) == 3
    titles = {ep.title for ep in results}
    assert "user-a ep1" in titles
    assert "user-a ep2" in titles
    assert "global ep" in titles
    assert "user-b ep" not in titles


async def test_list_episodes_user_id_with_only_null_rows(pool):
    """user_id='user-a' should return NULL rows even if no user-a rows exist."""
    await _make_episode(pool, "global ep1", user_id=None)
    await _make_episode(pool, "global ep2", user_id=None)
    await _make_episode(pool, "user-b ep", user_id="user-b")

    results = await list_episodes(pool, user_id="user-a")
    assert len(results) == 2
    titles = {ep.title for ep in results}
    assert "global ep1" in titles
    assert "global ep2" in titles
    assert "user-b ep" not in titles


async def test_list_episodes_user_id_combined_with_project_id(pool):
    """user_id filter should work with other filters like project_id."""
    await _make_episode(pool, "user-a proj-1", user_id="user-a", project_id="proj-1")
    await _make_episode(pool, "user-b proj-1", user_id="user-b", project_id="proj-1")
    await _make_episode(pool, "global proj-1", user_id=None, project_id="proj-1")
    await _make_episode(pool, "user-a proj-2", user_id="user-a", project_id="proj-2")

    results = await list_episodes(pool, project_id="proj-1", user_id="user-a")
    assert len(results) == 2
    titles = {ep.title for ep in results}
    assert "user-a proj-1" in titles
    assert "global proj-1" in titles
    assert "user-b proj-1" not in titles
    assert "user-a proj-2" not in titles


async def test_list_episodes_user_id_combined_with_status(pool):
    """user_id filter should work with status filter."""
    ep_a_open = await _make_episode(pool, "user-a open", user_id="user-a")
    ep_a_closed = await _make_episode(pool, "user-a closed", user_id="user-a")
    closed = await close_episode(pool, ep_a_closed.id)
    assert closed is not None

    ep_global = await _make_episode(pool, "global open", user_id=None)

    results = await list_episodes(pool, user_id="user-a", status=EpisodeStatus.open)
    assert len(results) == 2
    titles = {ep.title for ep in results}
    assert "user-a open" in titles
    assert "global open" in titles


# --- timeline_query with user_id ---


async def test_timeline_query_user_id_none_returns_all(pool):
    """timeline_query with user_id=None should return all episodes."""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    ep_a = await _make_episode(pool, "user-a ep", user_id="user-a")
    ep_b = await _make_episode(pool, "user-b ep", user_id="user-b")
    ep_g = await _make_episode(pool, "global ep", user_id=None)

    # Set started_at to within query range
    for ep in [ep_a, ep_b, ep_g]:
        await pool.execute(
            "UPDATE episodes SET started_at = $1 WHERE id = $2",
            hour_ago + timedelta(minutes=30),
            ep.id,
        )

    results = await timeline_query(pool, start=hour_ago, end=now, user_id=None)
    assert len(results) == 3


async def test_timeline_query_user_id_filters_to_user_and_null(pool):
    """timeline_query with user_id='user-a' should return user-a + NULL rows."""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    ep_a1 = await _make_episode(pool, "user-a ep1", user_id="user-a")
    ep_a2 = await _make_episode(pool, "user-a ep2", user_id="user-a")
    ep_b = await _make_episode(pool, "user-b ep", user_id="user-b")
    ep_g = await _make_episode(pool, "global ep", user_id=None)

    # Set started_at to within query range
    for ep in [ep_a1, ep_a2, ep_b, ep_g]:
        await pool.execute(
            "UPDATE episodes SET started_at = $1 WHERE id = $2",
            hour_ago + timedelta(minutes=30),
            ep.id,
        )

    results = await timeline_query(pool, start=hour_ago, end=now, user_id="user-a")
    assert len(results) == 3
    titles = {ep.title for ep in results}
    assert "user-a ep1" in titles
    assert "user-a ep2" in titles
    assert "global ep" in titles
    assert "user-b ep" not in titles


async def test_timeline_query_user_id_with_project_id(pool):
    """timeline_query user_id should work with project_id filter."""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    ep_a1 = await _make_episode(pool, "user-a proj-1", user_id="user-a", project_id="proj-1")
    ep_b1 = await _make_episode(pool, "user-b proj-1", user_id="user-b", project_id="proj-1")
    ep_g1 = await _make_episode(pool, "global proj-1", user_id=None, project_id="proj-1")
    ep_a2 = await _make_episode(pool, "user-a proj-2", user_id="user-a", project_id="proj-2")

    # Set started_at to within query range
    for ep in [ep_a1, ep_b1, ep_g1, ep_a2]:
        await pool.execute(
            "UPDATE episodes SET started_at = $1 WHERE id = $2",
            hour_ago + timedelta(minutes=30),
            ep.id,
        )

    results = await timeline_query(
        pool, start=hour_ago, end=now,
        project_id="proj-1", user_id="user-a"
    )
    assert len(results) == 2
    titles = {ep.title for ep in results}
    assert "user-a proj-1" in titles
    assert "global proj-1" in titles


# --- get_working_memory with user_id ---


async def test_get_working_memory_user_id_none_returns_all(pool):
    """get_working_memory with user_id=None should return all open episodes."""
    await _make_episode(pool, "user-a", user_id="user-a")
    await _make_episode(pool, "user-b", user_id="user-b")
    await _make_episode(pool, "global", user_id=None)

    results = await get_working_memory(pool, user_id=None)
    assert len(results) == 3


async def test_get_working_memory_user_id_filters_to_user_and_null(pool):
    """get_working_memory with user_id='user-a' should return user-a + NULL."""
    await _make_episode(pool, "user-a ep", user_id="user-a")
    await _make_episode(pool, "user-b ep", user_id="user-b")
    await _make_episode(pool, "global ep", user_id=None)

    results = await get_working_memory(pool, user_id="user-a")
    assert len(results) == 2
    titles = {ep.episode.title for ep in results}
    assert "user-a ep" in titles
    assert "global ep" in titles
    assert "user-b ep" not in titles


async def test_get_working_memory_user_id_with_project_id(pool):
    """get_working_memory user_id should work with project_id filter."""
    await _make_episode(pool, "user-a proj-1", user_id="user-a", project_id="proj-1")
    await _make_episode(pool, "user-b proj-1", user_id="user-b", project_id="proj-1")
    await _make_episode(pool, "global proj-1", user_id=None, project_id="proj-1")
    await _make_episode(pool, "user-a proj-2", user_id="user-a", project_id="proj-2")

    results = await get_working_memory(pool, project_id="proj-1", user_id="user-a")
    assert len(results) == 2
    titles = {ep.episode.title for ep in results}
    assert "user-a proj-1" in titles
    assert "global proj-1" in titles
