"""Tests for weft.exporter — memory export as markdown and JSON."""

from __future__ import annotations

import json

import pytest

from weft.exporter import export_memories
from weft.models import MemoryCreate, MemoryType
from weft.store import store_memory


async def _seed(pool):
    """Seed a few memories for export tests."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Redis is used for caching",
        topic=["infrastructure"],
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Always use dark mode",
        topic=["preferences"],
        confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.solution,
        content="Testcontainers reaper fix",
        topic=["testing"],
        confidence=0.8,
    ))
    # A memory with no topics
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Orphan memory without topics",
        topic=[],
        confidence=0.5,
    ))


async def test_export_markdown_basic(pool):
    """Markdown export groups by topic and includes metadata."""
    await _seed(pool)
    result = await export_memories(pool, format="md")

    assert result.startswith("# Weft Memory Export")
    assert "## Topic: infrastructure" in result
    assert "## Topic: preferences" in result
    assert "## Topic: testing" in result
    assert "## Uncategorized" in result
    assert "[fact]" in result
    assert "[preference]" in result
    assert "[solution]" in result
    assert "confidence: 0.9" in result
    assert "Orphan memory without topics" in result


async def test_export_json_basic(pool):
    """JSON export produces valid JSON with correct structure."""
    await _seed(pool)
    result = await export_memories(pool, format="json")

    data = json.loads(result)
    assert data["count"] == 4
    assert "exported_at" in data
    assert len(data["memories"]) == 4

    # Each memory should have type, content, confidence
    for m in data["memories"]:
        assert "type" in m
        assert "content" in m
        assert "confidence" in m


async def test_export_filter_by_type(pool):
    """Export can filter by memory type."""
    await _seed(pool)
    result = await export_memories(pool, format="json", memory_type="fact")

    data = json.loads(result)
    assert data["count"] == 2
    for m in data["memories"]:
        assert m["type"] == "fact"


async def test_export_filter_by_topic(pool):
    """Export can filter by topic."""
    await _seed(pool)
    result = await export_memories(pool, format="json", topic="infrastructure")

    data = json.loads(result)
    assert data["count"] == 1
    assert data["memories"][0]["content"] == "Redis is used for caching"


async def test_export_filter_by_status(pool):
    """Export can filter by status; archived memories excluded by default."""
    await _seed(pool)

    # Archive one memory
    from weft.store import delete_memory, list_memories

    mems = await list_memories(pool)
    await delete_memory(pool, mems[0].id)  # soft-delete = archive

    # Default (active) should exclude the archived one
    result = await export_memories(pool, format="json", status="active")
    data = json.loads(result)
    assert data["count"] == 3

    # Explicitly requesting archived should return exactly 1
    result = await export_memories(pool, format="json", status="archived")
    data = json.loads(result)
    assert data["count"] == 1


async def test_export_empty(pool):
    """Export with no matching memories handles gracefully."""
    result_md = await export_memories(pool, format="md")
    assert "No memories found" in result_md

    result_json = await export_memories(pool, format="json")
    data = json.loads(result_json)
    assert data["count"] == 0
    assert data["memories"] == []


async def test_export_user_scope_excludes_other_owner(pool):
    """A caller-scoped export never includes another user's memory."""
    from weft.auth import current_user_id
    from weft.db.connection import acquire

    token = current_user_id.set("export-owner-a")
    try:
        async with acquire(pool):
            await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="Owner A secret"))
    finally:
        current_user_id.reset(token)
    token = current_user_id.set("export-owner-b")
    try:
        async with acquire(pool):
            await store_memory(pool, MemoryCreate(type=MemoryType.fact, content="Owner B secret"))
    finally:
        current_user_id.reset(token)

    result = await export_memories(pool, format="json", user_id="export-owner-a")
    data = json.loads(result)
    contents = {memory["content"] for memory in data["memories"]}
    assert contents == {"Owner A secret"}


async def test_export_all_scope_is_explicitly_available_to_operator(pool):
    """The low-level exporter can still perform an explicit all-user export."""
    from weft.auth import current_user_id
    from weft.db.connection import acquire

    for owner, content in (("export-owner-a", "Owner A secret"), ("export-owner-b", "Owner B secret")):
        token = current_user_id.set(owner)
        try:
            async with acquire(pool):
                await store_memory(pool, MemoryCreate(type=MemoryType.fact, content=content))
        finally:
            current_user_id.reset(token)

    result = await export_memories(pool, format="json", user_id=None)
    contents = {memory["content"] for memory in json.loads(result)["memories"]}
    assert contents == {"Owner A secret", "Owner B secret"}


async def test_export_markdown_topic_sorting(pool):
    """Markdown export renders topics in alphabetical order."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="z-topic memory", topic=["zebra"],
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="a-topic memory", topic=["alpha"],
    ))

    result = await export_memories(pool, format="md")

    alpha_pos = result.index("## Topic: alpha")
    zebra_pos = result.index("## Topic: zebra")
    assert alpha_pos < zebra_pos
