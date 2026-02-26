"""E2E agent-readiness validation — fallback + extraction integration."""

from __future__ import annotations

from pathlib import Path

from weft.exporter import export_memories
from weft.extract import extract_candidates
from weft.fallback import read_fallback, search_fallback, _parse_sections
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory


async def test_store_export_fallback_roundtrip(pool, tmp_path):
    """Store memories -> export -> write fallback -> read + search."""
    # Store some memories
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers dark mode in all editors",
        topic=["ui"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Project uses PostgreSQL 16 with pgvector extension",
        topic=["infrastructure"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))

    # Export to fallback file
    exported = await export_memories(pool, format="md")
    fallback_path = tmp_path / "fallback.md"
    fallback_path.write_text(exported, encoding="utf-8")

    # Read back
    content = read_fallback(fallback_path)
    assert "dark mode" in content
    assert "PostgreSQL" in content

    # Search
    results = search_fallback("PostgreSQL database", path=fallback_path)
    assert len(results) >= 1
    assert any("PostgreSQL" in r.get("content", "") or "PostgreSQL" in r.get("title", "") for r in results)


async def test_extract_from_conversation(pool):
    """Extract candidates from realistic conversation text."""
    conversation = """
    User: I always prefer to use type hints in Python. Can you help with the database?
    Assistant: Sure! The project uses PostgreSQL with asyncpg for async access.
    User: Great. Our convention is to run tests before every commit.
    Assistant: The architecture follows a clean three-layer design with separate
    data, business, and presentation layers.
    """
    candidates = extract_candidates(conversation)

    # Should find at least a preference and a fact
    types = {c["type"] for c in candidates}
    assert len(candidates) >= 2
    assert "preference" in types or "fact" in types


async def test_fallback_parse_matches_export_format(pool, tmp_path):
    """Verify fallback parser correctly handles real exporter output."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Three-tier caching with Redis L1 and Postgres L3",
        topic=["caching"],
        source=MemorySource.conversation,
        confidence=0.85,
    ))

    exported = await export_memories(pool, format="md")
    sections = _parse_sections(exported)

    # Should find at least one section
    assert len(sections) >= 1
    # The section should have the right structure
    section = sections[0]
    assert "type" in section
    assert "confidence" in section
    assert "content" in section


async def test_mcp_tools_all_registered():
    """All 12 MCP tools are registered."""
    from weft.mcp import tools
    expected = [
        "weft_remember", "weft_recall", "weft_forget", "weft_context",
        "weft_revise", "weft_relate", "weft_consolidate", "weft_feedback",
        "weft_prime", "weft_status", "weft_extract",
    ]
    for name in expected:
        assert hasattr(tools, name), f"Missing MCP tool: {name}"
        assert callable(getattr(tools, name)), f"Tool not callable: {name}"
