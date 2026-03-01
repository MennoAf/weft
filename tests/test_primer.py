"""Tests for weft.primer — session primer context assembly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import build_primer
from weft.store import store_memory
from weft.tokens import estimate_tokens


async def test_primer_empty_db(pool):
    """No memories -> all sections empty, budget_remaining = budget_tokens."""
    result = await build_primer(pool, budget_tokens=4000)

    assert result["preferences"] == []
    assert result["recent_work"] == []
    assert result["ideas"] == []
    assert result["relevant"] == []
    assert result["total_tokens"] == 0
    assert result["budget_tokens"] == 4000
    assert result["budget_remaining"] == 4000


async def test_primer_preferences_first(pool):
    """Store a preference and a fact. Primer should include preference in
    preferences section, and the fact should appear in recent_work."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers dark mode",
        topic=["ui"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="The project uses PostgreSQL",
        topic=["architecture"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Preference should be in preferences section
    assert len(result["preferences"]) == 1
    assert result["preferences"][0]["type"] == "preference"
    assert "dark mode" in result["preferences"][0]["content"]

    # Fact should be in recent_work section (recently created)
    assert len(result["recent_work"]) == 1
    assert result["recent_work"][0]["type"] == "fact"


async def test_primer_user_model_in_preferences(pool):
    """user_model type memories appear in preferences section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.user_model,
        content="User is a senior Python developer",
        topic=["user"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    assert len(result["preferences"]) == 1
    assert result["preferences"][0]["type"] == "user_model"
    assert "Python developer" in result["preferences"][0]["content"]


async def test_primer_recent_work_section(pool):
    """Store a recently-accessed fact. Should appear in recent_work section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Implemented caching layer with Redis",
        topic=["implementation"],
        source=MemorySource.conversation,
        confidence=0.85,
    ))

    result = await build_primer(pool, budget_tokens=4000, recent_days=7)

    # Fact was just created (accessed_at = now), so it's recent
    assert len(result["recent_work"]) == 1
    assert "caching layer" in result["recent_work"][0]["content"]


async def test_primer_budget_enforcement(pool):
    """Store many memories exceeding budget. Total tokens should not exceed budget."""
    # Create many memories with substantial content
    for i in range(30):
        content = f"Memory number {i}: " + "x" * 200  # ~50+ tokens each
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=content,
            topic=["bulk"],
            source=MemorySource.conversation,
            confidence=0.7,
        ))

    small_budget = 100
    result = await build_primer(pool, budget_tokens=small_budget)

    assert result["total_tokens"] <= small_budget
    assert result["budget_remaining"] >= 0
    assert result["budget_remaining"] == small_budget - result["total_tokens"]
    # Not all 30 memories should fit in 100 tokens
    total_memories = (
        len(result["preferences"])
        + len(result["recent_work"])
        + len(result["ideas"])
        + len(result["relevant"])
    )
    assert total_memories < 30


async def test_primer_excludes_old_from_recent(pool):
    """Memory accessed > recent_days ago should NOT appear in recent_work."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="An old fact from long ago",
        topic=["history"],
        source=MemorySource.conversation,
        confidence=0.7,
    ))

    # Update accessed_at to 30 days ago directly in DB
    old_time = datetime.now(timezone.utc) - timedelta(days=30)
    await pool.execute(
        "UPDATE memories SET accessed_at = $1 WHERE id = $2",
        old_time,
        mem.id,
    )

    result = await build_primer(pool, budget_tokens=4000, recent_days=7)

    # Should NOT appear in recent_work since it was accessed 30 days ago
    assert len(result["recent_work"]) == 0
    # It's a fact, not a preference, so it shouldn't be in preferences either
    assert len(result["preferences"]) == 0


async def test_primer_project_scoping(pool):
    """With project_id, primer should include both project-scoped and global memories."""
    # Global preference (no project_id)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers verbose logging",
        topic=["config"],
        source=MemorySource.conversation,
        confidence=1.0,
        project_id=None,
    ))

    # Project-scoped fact
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Project Alpha uses microservices",
        topic=["architecture"],
        source=MemorySource.conversation,
        confidence=0.9,
        project_id="proj-alpha",
    ))

    # Different project fact (should not appear)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Project Beta uses monolith",
        topic=["architecture"],
        source=MemorySource.conversation,
        confidence=0.9,
        project_id="proj-beta",
    ))

    result = await build_primer(pool, project_id="proj-alpha", budget_tokens=4000)

    # Global preference should be included
    pref_contents = [p["content"] for p in result["preferences"]]
    assert any("verbose logging" in c for c in pref_contents)

    # Project-scoped fact should appear in recent_work, ideas, or relevant
    all_contents = (
        [m["content"] for m in result["recent_work"]]
        + [m["content"] for m in result["ideas"]]
        + [m["content"] for m in result["relevant"]]
    )
    assert any("microservices" in c for c in all_contents)

    # Other project's fact should NOT appear
    all_all = (
        [m["content"] for m in result["preferences"]]
        + [m["content"] for m in result["recent_work"]]
        + [m["content"] for m in result["ideas"]]
        + [m["content"] for m in result["relevant"]]
    )
    assert not any("monolith" in c for c in all_all)


async def test_primer_no_duplicates_across_sections(pool):
    """A preference memory should NOT appear again in recent_work or relevant."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers tabs over spaces",
        topic=["code_style"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Preference should only be in preferences section
    assert len(result["preferences"]) == 1

    # Should not also appear in recent_work
    recent_ids = [m["id"] for m in result["recent_work"]]
    pref_id = result["preferences"][0]["id"]
    assert pref_id not in recent_ids

    # Should not appear in relevant either
    relevant_ids = [m["id"] for m in result["relevant"]]
    assert pref_id not in relevant_ids


async def test_primer_return_structure(pool):
    """Verify all expected keys are present in the return dict."""
    result = await build_primer(pool, budget_tokens=4000)

    expected_keys = {
        "pinned",
        "handoff",
        "preferences",
        "recent_work",
        "ideas",
        "relevant",
        "total_tokens",
        "budget_tokens",
        "budget_remaining",
    }
    assert set(result.keys()) == expected_keys

    # Type checks
    assert isinstance(result["pinned"], list)
    assert isinstance(result["handoff"], list)
    assert isinstance(result["preferences"], list)
    assert isinstance(result["recent_work"], list)
    assert isinstance(result["ideas"], list)
    assert isinstance(result["relevant"], list)
    assert isinstance(result["total_tokens"], int)
    assert isinstance(result["budget_tokens"], int)
    assert isinstance(result["budget_remaining"], int)

    # Budget invariant
    assert result["total_tokens"] + result["budget_remaining"] == result["budget_tokens"]


async def test_primer_splits_ideas_from_recent_work(pool):
    """Memories with idea/improvement topics go to ideas section, not recent_work."""
    # Concrete work
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Fixed stale connection pool bug",
        topic=["postgres", "bug", "fixed"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    # Improvement idea
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Weft should inject memories at loom_claim time",
        topic=["weft", "improvement", "agent-context"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))
    # Another idea with "idea" topic
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Cross-project issue log for field observations",
        topic=["loom", "idea", "cross-project"],
        source=MemorySource.conversation,
        confidence=0.7,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Concrete work in recent_work
    work_contents = [m["content"] for m in result["recent_work"]]
    assert any("connection pool" in c for c in work_contents)

    # Ideas in ideas section
    idea_contents = [m["content"] for m in result["ideas"]]
    assert any("loom_claim" in c for c in idea_contents)
    assert any("issue log" in c for c in idea_contents)

    # Ideas should NOT be in recent_work
    assert not any("loom_claim" in c for c in work_contents)
    assert not any("issue log" in c for c in work_contents)


async def test_primer_surfaces_most_recent_handoff(pool):
    """Most recent handoff appears in the handoff section; older ones do not."""
    # Older handoff
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Fixed caching bugs",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))
    # Newer handoff
    import asyncio
    await asyncio.sleep(0.01)  # ensure different created_at
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Shipped project detection\n\n**Next Steps:** Build cross-project pattern transfer",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Only the most recent handoff should appear
    assert len(result["handoff"]) == 1
    assert "project detection" in result["handoff"][0]["content"]
    assert "caching bugs" not in result["handoff"][0]["content"]

    # Handoff should NOT also appear in recent_work
    work_contents = [m["content"] for m in result["recent_work"]]
    assert not any("Session Handoff" in c for c in work_contents)


async def test_primer_project_scoped_recent_work_first(pool):
    """When project_id is set, project-scoped memories appear before globals in recent_work."""
    # Global memory (created first, so it's older)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Global fact about Redis caching patterns",
        topic=["redis"],
        source=MemorySource.conversation,
        confidence=0.8,
        project_id=None,
    ))
    # Project-scoped memory (created second but should appear first)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft uses fastembed for local embeddings",
        topic=["embeddings"],
        source=MemorySource.conversation,
        confidence=0.8,
        project_id="weft",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=4000)

    work = result["recent_work"]
    assert len(work) == 2
    # Project-scoped should come first
    assert work[0]["project_id"] == "weft"
    assert work[1]["project_id"] is None
