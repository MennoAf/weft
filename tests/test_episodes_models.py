"""Tests for episode models and migration."""

from __future__ import annotations

import pytest

from weft.models import (
    Episode,
    EpisodeCreate,
    EpisodeStatus,
    EpisodeWithMemories,
    Memory,
    MemoryType,
)
from weft.store import store_memory
from weft.models import MemoryCreate


# --- EpisodeStatus enum ---


def test_episode_status_values():
    assert EpisodeStatus.open.value == "open"
    assert EpisodeStatus.closed.value == "closed"


# --- EpisodeCreate validation ---


def test_episode_create_minimal():
    ec = EpisodeCreate(title="Session work")
    assert ec.title == "Session work"
    assert ec.summary is None
    assert ec.project_id is None
    assert ec.agent_id is None


def test_episode_create_full():
    ec = EpisodeCreate(
        title="Debugging session",
        summary="Fixed the auth bug",
        project_id="proj-1",
        agent_id="warp",
    )
    assert ec.summary == "Fixed the auth bug"
    assert ec.project_id == "proj-1"


# --- Episode model ---


def test_episode_defaults():
    e = Episode(title="Test episode")
    assert e.id.startswith("weft-")
    assert e.status == EpisodeStatus.open
    assert e.ended_at is None
    assert e.token_count == 0
    assert e.summary is None


def test_episode_to_dict():
    e = Episode(
        title="Deploy session",
        summary="Deployed v2",
        project_id="proj-1",
        status=EpisodeStatus.closed,
    )
    d = e.to_dict()
    assert d["title"] == "Deploy session"
    assert d["status"] == "closed"
    assert d["project_id"] == "proj-1"
    assert "id" in d
    assert "started_at" in d


# --- EpisodeWithMemories ---


def test_episode_with_memories_to_dict():
    e = Episode(title="Test")
    m = Memory(type=MemoryType.fact, content="A fact")
    ewm = EpisodeWithMemories(episode=e, memories=[m])
    d = ewm.to_dict()
    assert d["title"] == "Test"
    assert d["memory_count"] == 1
    assert len(d["memories"]) == 1
    assert d["memories"][0]["content"] == "A fact"


def test_episode_with_memories_empty():
    e = Episode(title="Empty")
    ewm = EpisodeWithMemories(episode=e)
    d = ewm.to_dict()
    assert d["memory_count"] == 0
    assert d["memories"] == []


# --- Migration (episodes + episode_memories tables) ---


async def test_episodes_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'episodes'
        )
        """
    )
    assert exists is True


async def test_episode_memories_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'episode_memories'
        )
        """
    )
    assert exists is True


async def test_episodes_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'episodes'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "title" in columns
    assert "summary" in columns
    assert "project_id" in columns
    assert "agent_id" in columns
    assert "started_at" in columns
    assert "ended_at" in columns
    assert "status" in columns
    assert "token_count" in columns
    assert "created_at" in columns
    assert "updated_at" in columns

    assert columns["title"]["is_nullable"] == "NO"
    assert columns["started_at"]["is_nullable"] == "NO"
    assert columns["status"]["is_nullable"] == "NO"
    assert columns["ended_at"]["is_nullable"] == "YES"
    assert columns["summary"]["is_nullable"] == "YES"


async def test_episode_memories_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'episode_memories'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "episode_id" in columns
    assert "memory_id" in columns
    assert "position" in columns
    assert "added_at" in columns

    assert columns["episode_id"]["is_nullable"] == "NO"
    assert columns["memory_id"]["is_nullable"] == "NO"
    assert columns["position"]["is_nullable"] == "NO"


async def test_episodes_table_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'episodes'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "episodes_pkey" in index_names
    assert "idx_episodes_project" in index_names
    assert "idx_episodes_status" in index_names
    assert "idx_episodes_started" in index_names


async def test_episode_memories_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'episode_memories'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "episode_memories_pkey" in index_names
    assert "idx_epmem_memory" in index_names


async def test_episodes_insert_and_read(pool):
    await pool.execute(
        """
        INSERT INTO episodes (id, title, summary, project_id, status)
        VALUES ($1, $2, $3, $4, $5)
        """,
        "test-ep-1", "Test Episode", "A summary", "proj-1", "open",
    )

    row = await pool.fetchrow("SELECT * FROM episodes WHERE id = $1", "test-ep-1")
    assert row is not None
    assert row["title"] == "Test Episode"
    assert row["summary"] == "A summary"
    assert row["status"] == "open"
    assert row["ended_at"] is None


async def test_episodes_defaults(pool):
    await pool.execute(
        "INSERT INTO episodes (id, title) VALUES ($1, $2)",
        "test-ep-2", "Minimal",
    )

    row = await pool.fetchrow("SELECT * FROM episodes WHERE id = $1", "test-ep-2")
    assert row["status"] == "open"
    assert row["token_count"] == 0
    assert row["ended_at"] is None
    assert row["project_id"] is None


async def test_episode_memories_join(pool):
    """Can link a memory to an episode via join table."""
    await pool.execute(
        "INSERT INTO episodes (id, title) VALUES ($1, $2)",
        "test-ep-3", "Join test",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="A fact for the episode",
    ))

    await pool.execute(
        """
        INSERT INTO episode_memories (episode_id, memory_id, position)
        VALUES ($1, $2, $3)
        """,
        "test-ep-3", mem.id, 0,
    )

    rows = await pool.fetch(
        "SELECT * FROM episode_memories WHERE episode_id = $1 ORDER BY position",
        "test-ep-3",
    )
    assert len(rows) == 1
    assert rows[0]["memory_id"] == mem.id
    assert rows[0]["position"] == 0


async def test_episode_memories_cascade_on_episode_delete(pool):
    """Deleting an episode cascades to episode_memories."""
    await pool.execute(
        "INSERT INTO episodes (id, title) VALUES ($1, $2)",
        "test-ep-4", "Cascade test",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="linked memory",
    ))
    await pool.execute(
        "INSERT INTO episode_memories (episode_id, memory_id, position) VALUES ($1, $2, $3)",
        "test-ep-4", mem.id, 0,
    )

    await pool.execute("DELETE FROM episodes WHERE id = $1", "test-ep-4")

    links = await pool.fetch(
        "SELECT * FROM episode_memories WHERE episode_id = $1", "test-ep-4",
    )
    assert len(links) == 0
    # Memory itself should still exist
    row = await pool.fetchrow("SELECT * FROM memories WHERE id = $1", mem.id)
    assert row is not None


async def test_episode_memories_cascade_on_memory_delete(pool):
    """Hard-deleting a memory cascades to episode_memories."""
    await pool.execute(
        "INSERT INTO episodes (id, title) VALUES ($1, $2)",
        "test-ep-5", "Memory cascade",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="will be deleted",
    ))
    await pool.execute(
        "INSERT INTO episode_memories (episode_id, memory_id, position) VALUES ($1, $2, $3)",
        "test-ep-5", mem.id, 0,
    )

    await pool.execute("DELETE FROM memories WHERE id = $1", mem.id)

    links = await pool.fetch(
        "SELECT * FROM episode_memories WHERE memory_id = $1", mem.id,
    )
    assert len(links) == 0
    # Episode itself should still exist
    row = await pool.fetchrow("SELECT * FROM episodes WHERE id = $1", "test-ep-5")
    assert row is not None


async def test_episode_memories_idempotent(pool):
    """Inserting the same memory into an episode twice uses ON CONFLICT."""
    await pool.execute(
        "INSERT INTO episodes (id, title) VALUES ($1, $2)",
        "test-ep-6", "Idempotent test",
    )
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="unique link",
    ))

    for _ in range(2):
        await pool.execute(
            """
            INSERT INTO episode_memories (episode_id, memory_id, position)
            VALUES ($1, $2, $3)
            ON CONFLICT (episode_id, memory_id) DO NOTHING
            """,
            "test-ep-6", mem.id, 0,
        )

    rows = await pool.fetch(
        "SELECT * FROM episode_memories WHERE episode_id = $1", "test-ep-6",
    )
    assert len(rows) == 1
