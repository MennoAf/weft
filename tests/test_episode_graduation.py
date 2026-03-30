"""Tests for episode graduation path (episode -> persistent memory)."""

from __future__ import annotations

import pytest

from weft.episodes import (
    add_memory_to_episode,
    close_episode,
    create_episode,
    get_episode,
    get_episode_memories,
    graduate_episode,
)
from weft.models import (
    EpisodeCreate,
    EpisodeStatus,
    MemoryCreate,
    MemoryType,
)
from weft.store import store_memory


# --- Helpers ---


async def _make_memory(pool, content="test memory"):
    return await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content=content,
    ))


async def _make_episode(pool, title="test episode", **kwargs):
    return await create_episode(pool, EpisodeCreate(title=title, **kwargs))


# --- graduate_episode ---


async def test_graduate_episode_basic(pool):
    """Graduate an open episode — creates memory, sets status and graduated_memory_id."""
    ep = await _make_episode(pool, title="Debug session", summary="Found the root cause")

    updated_ep, memory = await graduate_episode(pool, ep.id)

    assert updated_ep.status == EpisodeStatus.graduated
    assert updated_ep.graduated_memory_id == memory.id
    assert updated_ep.ended_at is not None
    assert memory.content == "Debug session\n\nFound the root cause"
    assert memory.type == MemoryType.fact
    assert memory.project_id == ep.project_id
    assert memory.agent_id == ep.agent_id


async def test_graduate_episode_custom_content(pool):
    """Graduate with custom content overrides episode title/summary."""
    ep = await _make_episode(pool, title="Session X")

    updated_ep, memory = await graduate_episode(
        pool, ep.id,
        content="Custom insight extracted from the session",
        memory_type=MemoryType.solution,
        confidence=0.9,
        topic=["debugging", "postgres"],
    )

    assert updated_ep.status == EpisodeStatus.graduated
    assert memory.content == "Custom insight extracted from the session"
    assert memory.type == MemoryType.solution
    assert memory.confidence == pytest.approx(0.9)
    assert set(memory.topic) == {"debugging", "postgres"}


async def test_graduate_episode_links_memory(pool):
    """Graduated memory is linked to the episode."""
    ep = await _make_episode(pool, title="Test episode")

    updated_ep, memory = await graduate_episode(pool, ep.id)

    linked = await get_episode_memories(pool, ep.id)
    assert any(m.id == memory.id for m in linked)


async def test_graduate_episode_preserves_existing_ended_at(pool):
    """If episode was already closed (has ended_at), graduation preserves it."""
    ep = await _make_episode(pool, title="Already closed")
    closed_ep = await close_episode(pool, ep.id, summary="Done")
    original_ended_at = closed_ep.ended_at

    updated_ep, memory = await graduate_episode(pool, ep.id)

    assert updated_ep.status == EpisodeStatus.graduated
    assert updated_ep.ended_at == original_ended_at


async def test_graduate_episode_not_found(pool):
    """Graduating a non-existent episode raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await graduate_episode(pool, "weft-nonexistent")


async def test_graduate_episode_already_graduated(pool):
    """Graduating an already-graduated episode raises ValueError."""
    ep = await _make_episode(pool, title="Will graduate twice")
    await graduate_episode(pool, ep.id)

    with pytest.raises(ValueError, match="already graduated"):
        await graduate_episode(pool, ep.id)


async def test_graduate_episode_title_only(pool):
    """Episode with no summary graduates using title alone."""
    ep = await _make_episode(pool, title="Just a title")

    updated_ep, memory = await graduate_episode(pool, ep.id)

    assert memory.content == "Just a title"


async def test_graduate_episode_inherits_project_agent(pool):
    """Graduated memory inherits project_id and agent_id from episode."""
    ep = await _make_episode(
        pool, title="Scoped episode",
        project_id="test-project", agent_id="test-agent",
    )

    updated_ep, memory = await graduate_episode(pool, ep.id)

    assert memory.project_id == "test-project"
    assert memory.agent_id == "test-agent"


async def test_graduate_episode_with_existing_memories(pool):
    """Episode with pre-linked memories graduates cleanly; new memory is appended."""
    ep = await _make_episode(pool, title="Rich episode")
    mem1 = await _make_memory(pool, content="First observation")
    mem2 = await _make_memory(pool, content="Second observation")
    await add_memory_to_episode(pool, ep.id, mem1.id)
    await add_memory_to_episode(pool, ep.id, mem2.id)

    updated_ep, grad_memory = await graduate_episode(pool, ep.id)

    linked = await get_episode_memories(pool, ep.id)
    assert len(linked) == 3
    assert linked[-1].id == grad_memory.id


async def test_graduate_episode_persists_in_db(pool):
    """Graduated state persists when re-fetched from database."""
    ep = await _make_episode(pool, title="Persistence test")
    updated_ep, memory = await graduate_episode(pool, ep.id)

    refetched = await get_episode(pool, ep.id)
    assert refetched.status == EpisodeStatus.graduated
    assert refetched.graduated_memory_id == memory.id
