"""Phase 4 E2E validation — realistic scenarios for the full integration suite.

Covers:
1. Import → query → export round-trip
2. Cross-project sharing with isolation
3. Feedback loop affecting relevance
4. Session priming with preferences, recent work, budget
5. Full workflow: import → use → feedback → consolidate → prime → export
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from weft.embeddings import get_provider
from weft.exporter import export_memories
from weft.importer import import_memories, parse_memory_md_text
from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import build_primer
from weft.relevance import score_memory
from weft.store import (
    list_memories,
    record_feedback,
    search_by_vector,
    store_memory,
    touch_memory,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


async def test_import_query_export_roundtrip(pool, provider):
    """Import MEMORY.md → query imported memories → export → verify content preserved."""
    md_text = """\
## Database Preferences
- Always use PostgreSQL for persistent storage
- Use pgvector for semantic search

## Testing Patterns
- Use testcontainers for database isolation
- Run pytest with -v for verbose output
"""
    parsed = parse_memory_md_text(md_text)
    assert len(parsed.memories) >= 2

    report = await import_memories(pool, provider, parsed.memories)
    assert report.stored >= 2
    assert report.errors == []

    # Query imported memories
    query_emb = await provider.embed("database testing")
    results = await search_by_vector(pool, query_emb, limit=10)
    assert len(results) >= 2

    # Export and verify
    exported = await export_memories(pool, format="json")
    assert "PostgreSQL" in exported or "testcontainers" in exported


async def test_cross_project_sharing_isolation(pool, provider):
    """Cross-project: global memories visible everywhere, project-scoped memories isolated."""
    # Global preference
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Always format code with black",
        topic=["coding"],
        source=MemorySource.conversation,
        confidence=1.0,
        project_id=None,
    ))
    # Project-scoped facts
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Frontend uses React 18",
        topic=["frontend"],
        project_id="project-frontend",
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Backend uses FastAPI 0.100",
        topic=["backend"],
        project_id="project-backend",
    ))

    # Frontend project sees global + its own, not backend's
    frontend = await list_memories(pool, project_id="project-frontend")
    frontend_contents = {m.content for m in frontend}
    assert "Always format code with black" in frontend_contents
    assert "Frontend uses React 18" in frontend_contents
    assert "Backend uses FastAPI 0.100" not in frontend_contents


async def test_feedback_loop_affects_relevance(pool, provider):
    """Feedback: store memories, give feedback, verify relevance changes."""
    from weft.models import MemoryRecall

    emb = await provider.embed("project configuration settings")
    mem_good = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Configuration uses TOML format with layered precedence",
        topic=["config"],
    ), embedding=emb)
    mem_bad = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Configuration was stored in XML before migration",
        topic=["config"],
    ), embedding=emb)

    # Give positive feedback to good memory, negative to bad
    await record_feedback(pool, mem_good.id, helpful=True)
    await record_feedback(pool, mem_bad.id, helpful=False)
    await record_feedback(pool, mem_bad.id, helpful=False)

    # Refresh from DB
    from weft.store import get_memory
    mem_good_updated = await get_memory(pool, mem_good.id)
    mem_bad_updated = await get_memory(pool, mem_bad.id)

    # Score both
    recall_good = MemoryRecall(memory=mem_good_updated, similarity=0.8)
    recall_bad = MemoryRecall(memory=mem_bad_updated, similarity=0.8)

    scored_good = score_memory(recall_good)
    scored_bad = score_memory(recall_bad)

    # Good memory should score higher due to better usefulness_score
    assert scored_good.score > scored_bad.score
    assert scored_good.usefulness_factor > scored_bad.usefulness_factor


async def test_session_priming_workflow(pool):
    """Session priming: rules (pinned), decisions, budget respected."""
    # Store a pinned rule
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Always use weft_remember, never flat-file memory",
        topic=["workflow"],
        confidence=1.0,
        pinned=True,
    ))
    # Store a decision
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Don't suggest mocks — use real integration tests",
        topic=["testing"],
        confidence=0.9,
    ))
    # Store a regular fact (should NOT appear in primer)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Weft uses three-layer retrieval: vector, relevance, context",
        topic=["weft", "architecture"],
        confidence=0.85,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    # Pinned rule should be in rules
    assert len(result["rules"]) >= 1
    assert any("weft_remember" in r["content"] for r in result["rules"])

    # Decision should be in decisions
    assert len(result["decisions"]) >= 1

    # Architecture fact should NOT be anywhere
    all_contents = (
        [m["content"] for m in result["rules"]]
        + [m["content"] for m in result["handoff"]]
        + [m["content"] for m in result["issues"]["items"]]
        + [m["content"] for m in result["decisions"]]
    )
    assert not any("three-layer" in c for c in all_contents)

    # Budget should be respected
    assert result["total_tokens"] <= result["budget_tokens"]
    assert result["budget_remaining"] >= 0
    assert result["total_tokens"] + result["budget_remaining"] == result["budget_tokens"]

    # All expected keys present
    assert {
        "grounding", "rules", "behaviors", "handoff", "recent_work", "issues", "decisions",
        "entities", "changes_since", "total_tokens", "budget_tokens", "budget_remaining",
        "excluded", "freshness_hours", "section_tokens", "hints", "onboarding",
    }.issubset(set(result.keys()))


async def test_full_phase4_workflow(pool, provider):
    """Full workflow: import → use → feedback → prime → export."""
    # 1. Import memories
    md = """\
## Architecture Decisions
- Weft uses PostgreSQL with pgvector for semantic storage
- Three-tier caching: Redis L1, in-memory L2, DB L3

## User Preferences
- Always use dark mode in IDE
- Prefer verbose test output
"""
    parsed = parse_memory_md_text(md)
    report = await import_memories(pool, provider, parsed.memories)
    assert report.stored >= 2

    # 2. Use memories (query + touch)
    emb = await provider.embed("architecture and storage")
    results = await search_by_vector(pool, emb, limit=5)
    assert len(results) >= 1
    for r in results:
        await touch_memory(pool, r.memory.id)

    # 3. Feedback
    if results:
        await record_feedback(pool, results[0].memory.id, helpful=True)

    # 4. Prime session
    primer = await build_primer(pool, budget_tokens=2000)
    assert primer["total_tokens"] <= 2000
    # Budget invariant
    assert primer["total_tokens"] + primer["budget_remaining"] == primer["budget_tokens"]

    # 5. Export
    exported_md = await export_memories(pool, format="md")
    assert len(exported_md) > 0

    exported_json = await export_memories(pool, format="json")
    import json
    parsed_export = json.loads(exported_json)
    assert len(parsed_export) >= 2


async def test_weft_prime_mcp_tool_exists():
    """Verify weft_prime MCP tool is registered."""
    from weft.mcp import tools
    assert hasattr(tools, "weft_prime")
    assert callable(tools.weft_prime)
