"""Tests for weft.primer — session primer context assembly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import Memory, MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import (
    _MAX_GLOBAL_PREFS,
    _MAX_GLOBAL_RECENT,
    _MAX_RECENT_ITEMS,
    _REFERENCE_THRESHOLD,
    _is_completed,
    _novelty_score,
    build_primer,
)
from weft.store import store_memory
from weft.tokens import estimate_tokens


def _make_mem(content: str, topic: list[str] | None = None) -> Memory:
    """Build a minimal Memory for unit-testing helper functions."""
    now = datetime.now(timezone.utc)
    return Memory(
        id="weft-test", type=MemoryType.fact, topic=topic or [],
        content=content, source=MemorySource.conversation, confidence=0.9,
        token_count=10, created_at=now, updated_at=now, accessed_at=now,
        access_count=1, project_id=None, agent_id=None, status=MemoryStatus.active,
        pinned=False, usefulness_score=1, usefulness_count=0,
    )


class TestIsCompleted:
    def test_done_marker(self):
        assert _is_completed(_make_mem("Weft Improvement (DONE): Pinned memories")) is True

    def test_fixed_marker(self):
        assert _is_completed(_make_mem("Weft Bug (FIXED): stale pool")) is True

    def test_fixed_topic(self):
        assert _is_completed(_make_mem("Some bug was fixed", topic=["bug", "fixed"])) is True

    def test_done_improvement_topic(self):
        assert _is_completed(_make_mem("Auto-extract shipped", topic=["improvement", "done"])) is True

    def test_live_item_not_completed(self):
        assert _is_completed(_make_mem("Redis cache has 1 hour TTL")) is False

    def test_open_idea_not_completed(self):
        assert _is_completed(_make_mem("Inject memories at claim time", topic=["improvement"])) is False


async def test_primer_empty_db(pool):
    """No memories -> all sections empty, budget_remaining = budget_tokens."""
    result = await build_primer(pool, budget_tokens=4000)

    assert result["preferences"] == []
    assert result["recent_work"] == []
    assert result["active_issues"] == {"count": 0, "items": []}
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
        + result["active_issues"]["count"]
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

    # Project-scoped fact should appear in recent_work
    work_contents = [m["content"] for m in result["recent_work"]]
    assert any("microservices" in c for c in work_contents)

    # Other project's fact should NOT appear
    all_all = (
        [m["content"] for m in result["preferences"]]
        + [m["content"] for m in result["recent_work"]]
        + [m["content"] for m in result["active_issues"]["items"]]
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

    # Should not appear in active_issues either
    issue_ids = [m["id"] for m in result["active_issues"]["items"]]
    assert pref_id not in issue_ids


async def test_primer_return_structure(pool):
    """Verify all expected keys are present in the return dict."""
    result = await build_primer(pool, budget_tokens=4000)

    expected_keys = {
        "pinned",
        "handoff",
        "preferences",
        "recent_work",
        "active_issues",
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
    assert isinstance(result["active_issues"], dict)
    assert isinstance(result["active_issues"]["count"], int)
    assert isinstance(result["active_issues"]["items"], list)
    assert isinstance(result["total_tokens"], int)
    assert isinstance(result["budget_tokens"], int)
    assert isinstance(result["budget_remaining"], int)

    # Budget invariant
    assert result["total_tokens"] + result["budget_remaining"] == result["budget_tokens"]


async def test_primer_excludes_ideas_from_recent_work(pool):
    """Memories with idea/improvement topics are excluded from recent_work entirely."""
    # Concrete work (live, not completed)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Connection pool uses keepalive pings every 5 minutes",
        topic=["postgres", "infrastructure"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    # Improvement idea — should be excluded
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Weft should inject memories at loom_claim time",
        topic=["weft", "improvement", "agent-context"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))
    # Another idea with "idea" topic — should be excluded
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
    assert any("keepalive" in c for c in work_contents)

    # Ideas should NOT be in recent_work
    assert not any("loom_claim" in c for c in work_contents)
    assert not any("issue log" in c for c in work_contents)

    # Only the concrete fact should be present
    assert len(result["recent_work"]) == 1


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


async def test_primer_handoff_fallback_by_topic(pool):
    """Handoff stored as wrong type but with topic 'session-handoff' still surfaces."""
    # Simulate pre-handoff-type memory: stored as fact with session-handoff topic
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="## Session Handoff\n\n**Summary:** Mistyped handoff from old server",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Should appear in handoff section via fallback
    assert len(result["handoff"]) == 1
    assert "Mistyped handoff" in result["handoff"][0]["content"]

    # Should NOT also appear in recent_work
    work_contents = [m["content"] for m in result["recent_work"]]
    assert not any("Session Handoff" in c for c in work_contents)


async def test_primer_handoff_typed_takes_priority_over_fallback(pool):
    """When both typed handoff and topic-based exist, typed one wins."""
    import asyncio

    # Mistyped handoff (older)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="## Session Handoff\n\n**Summary:** Old mistyped one",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))
    await asyncio.sleep(0.01)
    # Properly typed handoff (newer)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Properly typed one",
        topic=["session-handoff"],
        source=MemorySource.conversation,
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    # Typed handoff should win — fallback not triggered
    assert len(result["handoff"]) == 1
    assert "Properly typed one" in result["handoff"][0]["content"]


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


async def test_primer_completed_items_excluded(pool):
    """Memories with DONE/FIXED markers are excluded from recent_work entirely."""
    # Live item (no completion marker)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Redis cache has 1 hour TTL for memories",
        topic=["weft", "caching"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    # Completed item — should not appear
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft Bug (FIXED): Stale asyncpg connection pool with no auto-reconnect",
        topic=["weft-feedback", "bug", "fixed"],
        source=MemorySource.conversation,
        confidence=0.95,
    ))
    # Another live item
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Weft uses pgvector for semantic search",
        topic=["weft", "architecture"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    work = result["recent_work"]
    # Only the 2 live items — completed item excluded entirely
    assert len(work) == 2
    for item in work:
        assert "(FIXED)" not in item["content"]


async def test_primer_active_issues_section(pool):
    """Active issues appear in the active_issues section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="Connection pool timeout after 47 hours of uptime",
        topic=["postgres", "bug"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="MCP topic coercion misses nested arrays",
        topic=["mcp", "bug"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    issues = result["active_issues"]
    assert issues["count"] == 2
    assert len(issues["items"]) == 2
    contents = [i["content"] for i in issues["items"]]
    assert any("Connection pool" in c for c in contents)
    assert any("topic coercion" in c for c in contents)

    # Issues should NOT appear in recent_work
    work_contents = [m["content"] for m in result["recent_work"]]
    assert not any("Connection pool" in c for c in work_contents)


async def test_primer_completed_items_excluded_regardless_of_budget(pool):
    """Completed items are excluded even when budget is generous."""
    # Live item (small)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Redis cache has 1 hour TTL",
        topic=["caching"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    # Completed item — excluded regardless of budget
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft Bug (FIXED): Stale asyncpg connection pool. After containers run for 47 hours the connection dies. Fix: added pool keepalive background task in server.py that pings every 5 minutes.",
        topic=["bug", "fixed"],
        source=MemorySource.conversation,
        confidence=0.95,
    ))

    # Generous budget — completed items still excluded
    result = await build_primer(pool, budget_tokens=4000)

    work = result["recent_work"]
    assert len(work) == 1
    assert "(FIXED)" not in work[0]["content"]


# --- Cross-project cap tests ---


async def test_primer_caps_global_recent_work(pool):
    """When project_id is set, non-project items in recent_work are capped."""
    import asyncio

    # Create more global items than the cap
    for i in range(_MAX_GLOBAL_RECENT + 3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Global fact number {i} about general patterns",
            topic=["patterns"],
            source=MemorySource.conversation,
            confidence=0.8,
            project_id=None,
        ))
        await asyncio.sleep(0.005)

    # Create project-scoped items (should all appear)
    for i in range(3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Weft-specific fact {i} about architecture",
            topic=["architecture"],
            source=MemorySource.conversation,
            confidence=0.8,
            project_id="weft",
        ))

    result = await build_primer(pool, project_id="weft", budget_tokens=4000)

    work = result["recent_work"]
    project_items = [m for m in work if m["project_id"] == "weft"]
    global_items = [m for m in work if m["project_id"] is None]

    # All project items should appear
    assert len(project_items) == 3
    # Global items capped at _MAX_GLOBAL_RECENT
    assert len(global_items) <= _MAX_GLOBAL_RECENT


async def test_primer_active_issues_empty(pool):
    """When no issues exist, active_issues has count=0 and empty items."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Some regular fact, not an issue",
        topic=["general"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    assert result["active_issues"] == {"count": 0, "items": []}


async def test_primer_active_issues_budget_aware(pool):
    """Active issues respect the token budget like all other sections."""
    # Fill up most of the budget with preferences
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Preference {i}: " + "x" * 200,
            topic=["prefs"],
            source=MemorySource.conversation,
            confidence=0.9,
        ))

    # Add an issue
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="A bug that might not fit: " + "y" * 200,
        topic=["bug"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))

    # Tiny budget — issue likely won't fit after preferences
    result = await build_primer(pool, budget_tokens=100)

    assert result["total_tokens"] <= 100


async def test_primer_caps_global_preferences(pool):
    """When project_id is set, non-project preferences are capped."""
    import asyncio

    # Create more global preferences than the cap
    for i in range(_MAX_GLOBAL_PREFS + 3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Global preference {i} about coding style",
            topic=["coding"],
            source=MemorySource.conversation,
            confidence=0.9,
            project_id=None,
        ))
        await asyncio.sleep(0.005)

    # Create project-scoped preferences (should all appear)
    for i in range(2):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Weft preference {i} about embeddings",
            topic=["embeddings"],
            source=MemorySource.conversation,
            confidence=0.9,
            project_id="weft",
        ))

    result = await build_primer(pool, project_id="weft", budget_tokens=4000)

    prefs = result["preferences"]
    project_prefs = [m for m in prefs if m["project_id"] == "weft"]
    global_prefs = [m for m in prefs if m["project_id"] is None]

    assert len(project_prefs) == 2
    assert len(global_prefs) <= _MAX_GLOBAL_PREFS


async def test_primer_no_caps_without_project_id(pool):
    """Without project_id, no global caps are applied."""
    import asyncio

    # Create many global items
    for i in range(_MAX_GLOBAL_RECENT + 5):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Global fact {i} about various topics",
            topic=["various"],
            source=MemorySource.conversation,
            confidence=0.8,
            project_id=None,
        ))
        await asyncio.sleep(0.005)

    # No project_id — all items should appear (budget permitting)
    result = await build_primer(pool, budget_tokens=4000)

    work = result["recent_work"]
    # More than the cap should be present since no project filter
    assert len(work) > _MAX_GLOBAL_RECENT


async def test_primer_preferences_project_scoped_first(pool):
    """Project-scoped preferences sort before globals in preferences section."""
    import asyncio

    # Global preference (high confidence)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Global: always use pytest for testing",
        topic=["testing"],
        source=MemorySource.conversation,
        confidence=1.0,
        project_id=None,
    ))
    await asyncio.sleep(0.005)
    # Project preference (lower confidence but should appear first)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Weft: use fastembed as default embedding provider",
        topic=["embeddings"],
        source=MemorySource.conversation,
        confidence=0.9,
        project_id="weft",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=4000)

    prefs = result["preferences"]
    assert len(prefs) == 2
    # Project preference first despite lower confidence
    assert prefs[0]["project_id"] == "weft"
    assert prefs[1]["project_id"] is None


# --- Novelty scoring tests ---


class TestNoveltyScore:
    """Unit tests for the _novelty_score ranking function."""

    def test_low_access_count_no_penalty(self):
        """Memories below the reference threshold get no penalty."""
        mem = _make_mem("some fact")
        mem.access_count = 1
        score_low = _novelty_score(mem)

        mem.access_count = _REFERENCE_THRESHOLD
        score_at_threshold = _novelty_score(mem)

        # At or below threshold, score equals raw accessed_at timestamp
        assert score_low == score_at_threshold == mem.accessed_at.timestamp()

    def test_high_access_count_penalized(self):
        """Memories above the reference threshold get a lower novelty score."""
        mem = _make_mem("reference material")
        mem.access_count = 1
        score_low = _novelty_score(mem)

        mem.access_count = _REFERENCE_THRESHOLD + 10
        score_high = _novelty_score(mem)

        # Same accessed_at but high access_count => lower score
        assert score_high < score_low

    def test_penalty_grows_with_access_count(self):
        """Higher access counts get progressively larger penalties."""
        mem = _make_mem("frequently touched")

        mem.access_count = _REFERENCE_THRESHOLD + 2
        score_a = _novelty_score(mem)

        mem.access_count = _REFERENCE_THRESHOLD + 20
        score_b = _novelty_score(mem)

        assert score_b < score_a

    def test_very_recent_high_access_can_still_rank(self):
        """A high-access memory touched just now ranks above a low-access
        memory touched several days ago — the penalty is bounded."""
        now = datetime.now(timezone.utc)
        five_days_ago = now - timedelta(days=5)

        recent_ref = _make_mem("reference, just touched")
        recent_ref.accessed_at = now
        recent_ref.access_count = 15

        old_novel = _make_mem("novel but stale")
        old_novel.accessed_at = five_days_ago
        old_novel.access_count = 1

        # The high-access memory touched NOW should still beat
        # a low-access memory touched 5 days ago
        assert _novelty_score(recent_ref) > _novelty_score(old_novel)


async def test_primer_novelty_ranking_in_recent_work(pool):
    """High access_count memories rank below low access_count memories
    with similar accessed_at in the recent_work section."""
    import asyncio

    # Memory A: low access count, accessed recently
    mem_a = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="New discovery: connection pool needs keepalive",
        topic=["postgres"],
        source=MemorySource.conversation,
        confidence=0.8,
    ))
    await asyncio.sleep(0.01)

    # Memory B: high access count (reference material), accessed at similar time
    mem_b = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Redis cache has 1 hour TTL for memories",
        topic=["caching"],
        source=MemorySource.conversation,
        confidence=0.9,
    ))
    # Simulate many accesses (bumps accessed_at to now each time)
    for _ in range(12):
        await pool.execute(
            "UPDATE memories SET accessed_at = now(), access_count = access_count + 1 WHERE id = $1",
            mem_b.id,
        )

    result = await build_primer(pool, budget_tokens=4000)

    work = result["recent_work"]
    assert len(work) == 2
    work_ids = [m["id"] for m in work]
    # Low-access memory should rank first (more novel)
    assert work_ids[0] == mem_a.id
    assert work_ids[1] == mem_b.id


async def test_primer_recent_work_hard_cap(pool):
    """Recent work section respects _MAX_RECENT_ITEMS even with generous budget."""
    import asyncio

    # Create more items than the cap
    for i in range(_MAX_RECENT_ITEMS + 4):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Recent work item {i} with enough content to be meaningful",
            topic=["work"],
            source=MemorySource.conversation,
            confidence=0.8,
        ))
        await asyncio.sleep(0.005)

    result = await build_primer(pool, budget_tokens=10000)  # generous budget

    # Hard cap should limit recent_work
    assert len(result["recent_work"]) <= _MAX_RECENT_ITEMS


async def test_primer_hard_cap_still_respects_budget(pool):
    """Budget enforcement still works even below the hard cap."""
    import asyncio

    # Create a few items with large content
    for i in range(3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Item {i}: " + "x" * 500,  # ~130 tokens each
            topic=["work"],
            source=MemorySource.conversation,
            confidence=0.8,
        ))
        await asyncio.sleep(0.005)

    # Tight budget — fewer than cap should appear
    result = await build_primer(pool, budget_tokens=150)

    assert len(result["recent_work"]) < 3
    assert result["total_tokens"] <= 150
