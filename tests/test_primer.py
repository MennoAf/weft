"""Tests for weft.primer — session primer context assembly."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from weft.behaviors import store_behavior
from weft.models import BehaviorCreate, MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import (
    _CAP_DECISIONS,
    _CAP_GROUNDING,
    _CAP_HANDOFF,
    _CAP_ISSUES,
    _CAP_RECENT_WORK,
    _CAP_RULES,
    _COLD_START_THRESHOLD,
    _GROUNDING_TOPIC,
    _MAX_DECISIONS,
    _MAX_RECENT_WORK,
    _ONBOARDING_TEXT,
    _SECTION_HINTS,
    _newest_created_at,
    build_primer,
)
from weft.store import list_memories, record_feedback, store_memory
from weft.tokens import estimate_tokens


# --- Structure & basics ---


async def test_primer_empty_db(pool):
    """No memories -> all sections empty, budget_remaining = budget_tokens,
    all hints present, onboarding shown (cold start)."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["grounding"] is None
    assert result["rules"] == []
    assert result["handoff"] == []
    assert result["recent_work"] == []
    assert result["issues"] == {"count": 0, "items": []}
    assert result["anti_patterns"] == []
    assert result["decisions"] == []
    assert result["total_tokens"] == 0
    assert result["budget_tokens"] == 1800
    assert result["budget_remaining"] == 1800
    assert result["excluded"] == 0
    assert result["freshness_hours"] is None
    assert result["section_tokens"] == {
        "grounding": 0, "rules": 0, "behaviors": 0, "handoff": 0,
        "recent_work": 0, "issues": 0, "anti_patterns": 0, "decisions": 0, "entities": 0,
    }
    assert result["behaviors"] == []
    assert result["entities"] == []
    # Empty DB = all hints + onboarding
    assert set(result["hints"].keys()) == {"rules", "behaviors", "handoff", "recent_work", "issues", "decisions", "loom"}
    assert result["onboarding"] is not None


async def test_primer_return_structure(pool):
    """Verify all expected keys are present in the return dict."""
    result = await build_primer(pool, budget_tokens=1800)

    expected_keys = {
        "grounding", "rules", "behaviors", "handoff", "recent_work", "issues", "decisions",
        "entities", "changes_since", "total_tokens", "budget_tokens", "budget_remaining",
        "excluded", "freshness_hours", "section_tokens", "hints", "onboarding",
    }
    assert expected_keys.issubset(set(result.keys()))

    assert result["grounding"] is None or isinstance(result["grounding"], str)
    assert isinstance(result["rules"], list)
    assert isinstance(result["handoff"], list)
    assert isinstance(result["recent_work"], list)
    assert isinstance(result["issues"], dict)
    assert isinstance(result["issues"]["count"], int)
    assert isinstance(result["issues"]["items"], list)
    assert isinstance(result["decisions"], list)
    assert isinstance(result["total_tokens"], int)
    assert isinstance(result["budget_tokens"], int)
    assert isinstance(result["budget_remaining"], int)
    assert isinstance(result["excluded"], int)
    assert result["freshness_hours"] is None or isinstance(result["freshness_hours"], float)
    assert isinstance(result["section_tokens"], dict)
    assert isinstance(result["hints"], dict)
    assert result["onboarding"] is None or isinstance(result["onboarding"], str)

    # Budget invariant
    assert result["total_tokens"] + result["budget_remaining"] == result["budget_tokens"]


async def test_primer_budget_enforcement(pool):
    """Total tokens should not exceed budget."""
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: " + "x" * 200,
            source=MemorySource.conversation,
            confidence=0.7,
        ))

    small_budget = 100
    result = await build_primer(pool, budget_tokens=small_budget)

    assert result["total_tokens"] <= small_budget
    assert result["budget_remaining"] >= 0
    assert result["budget_remaining"] == small_budget - result["total_tokens"]


# --- Rules section (pinned only) ---


async def test_primer_rules_are_pinned_only(pool):
    """Only pinned memories appear in the rules section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Always use weft_remember, never flat-file memory",
        confidence=1.0,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers dark mode",
        confidence=1.0,
        pinned=False,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["rules"]) == 1
    assert "weft_remember" in result["rules"][0]["content"]


async def test_primer_unpinned_preferences_not_in_primer(pool):
    """Non-pinned preferences do NOT appear anywhere in the primer."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers verbose logging",
        confidence=1.0,
        pinned=False,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    all_contents = (
        [m["content"] for m in result["rules"]]
        + [m["content"] for m in result["handoff"]]
        + [m["summary"] for m in result["recent_work"]]
        + [m["content"] for m in result["issues"]["items"]]
        + [m["content"] for m in result["decisions"]]
    )
    assert not any("verbose logging" in c for c in all_contents)


async def test_primer_unpinned_facts_not_in_primer(pool):
    """Architecture facts, patterns, and other reference material
    do NOT appear in the primer — they belong in recall."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.architecture,
        content="Weft uses pgvector for semantic search",
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Redis cache has 1 hour TTL",
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.pattern,
        content="testcontainers pattern works well for integration tests",
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    all_contents = (
        [m["content"] for m in result["rules"]]
        + [m["content"] for m in result["handoff"]]
        + [m["summary"] for m in result["recent_work"]]
        + [m["content"] for m in result["issues"]["items"]]
        + [m["content"] for m in result["decisions"]]
    )
    assert not any("pgvector" in c for c in all_contents)
    assert not any("Redis cache" in c for c in all_contents)
    assert not any("testcontainers" in c for c in all_contents)


async def test_primer_rules_sorted_by_confidence(pool):
    """Rules (pinned memories) are sorted by confidence, highest first."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Low confidence rule",
        confidence=0.7,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="High confidence rule",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["rules"]) == 2
    assert result["rules"][0]["confidence"] >= result["rules"][1]["confidence"]


# --- Handoff section ---


async def test_primer_surfaces_most_recent_handoff(pool):
    """Most recent handoff appears in the handoff section; older ones do not."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Fixed caching bugs",
        topic=["session-handoff"],
        confidence=1.0,
    ))
    await asyncio.sleep(0.01)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Shipped project detection",
        topic=["session-handoff"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["handoff"]) == 1
    assert "project detection" in result["handoff"][0]["content"]
    assert "caching bugs" not in result["handoff"][0]["content"]


async def test_primer_handoff_fallback_by_topic(pool):
    """Handoff stored as wrong type but with topic 'session-handoff' still surfaces."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="## Session Handoff\n\n**Summary:** Mistyped handoff from old server",
        topic=["session-handoff"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["handoff"]) == 1
    assert "Mistyped handoff" in result["handoff"][0]["content"]


async def test_primer_handoff_typed_takes_priority_over_fallback(pool):
    """When both typed handoff and topic-based exist, typed one wins."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="## Session Handoff\n\n**Summary:** Old mistyped one",
        topic=["session-handoff"],
        confidence=1.0,
    ))
    await asyncio.sleep(0.01)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Properly typed one",
        topic=["session-handoff"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["handoff"]) == 1
    assert "Properly typed one" in result["handoff"][0]["content"]


# --- Issues section ---


async def test_primer_issues_section(pool):
    """Active issues appear in the issues section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="Connection pool timeout after 47 hours",
        topic=["postgres", "bug"],
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="MCP topic coercion misses nested arrays",
        topic=["mcp", "bug"],
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    issues = result["issues"]
    assert issues["count"] == 2
    assert len(issues["items"]) == 2
    contents = [i["content"] for i in issues["items"]]
    assert any("Connection pool" in c for c in contents)
    assert any("topic coercion" in c for c in contents)


async def test_primer_issues_empty(pool):
    """When no issues exist, issues has count=0 and empty items."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Some regular fact, not an issue",
        confidence=0.8,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert result["issues"] == {"count": 0, "items": []}


async def test_primer_issues_budget_aware(pool):
    """Issues respect the token budget."""
    # Fill budget with pinned rules
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Rule {i}: " + "x" * 200,
            confidence=0.9,
            pinned=True,
        ))

    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="A bug that might not fit: " + "y" * 200,
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=100)

    assert result["total_tokens"] <= 100


# --- Decisions section ---


async def test_primer_decisions_section(pool):
    """Closed decisions appear in the decisions section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Don't suggest mocks — use real integration tests with testcontainers",
        topic=["testing"],
        confidence=0.9,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Don't split store.py — monolithic store is deliberate",
        topic=["architecture"],
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["decisions"]) == 2
    contents = [d["content"] for d in result["decisions"]]
    assert any("mocks" in c for c in contents)
    assert any("store.py" in c for c in contents)


async def test_primer_decisions_capped(pool):
    """Decisions section respects _MAX_DECISIONS hard cap."""
    for i in range(_MAX_DECISIONS + 3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: don't do thing {i}",
            confidence=0.9,
        ))

    result = await build_primer(pool, budget_tokens=4000)

    assert len(result["decisions"]) <= _MAX_DECISIONS


async def test_primer_decisions_project_scoped_first(pool):
    """Project-scoped decisions appear before global ones."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Global: don't use print() for debugging",
        confidence=0.9,
        project_id=None,
    ))
    await asyncio.sleep(0.01)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Weft: don't split store.py",
        confidence=0.9,
        project_id="weft",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=1500)

    decisions = result["decisions"]
    assert len(decisions) == 2
    assert decisions[0]["project_id"] == "weft"
    assert decisions[1]["project_id"] is None


# --- No duplicates ---


async def test_primer_no_duplicates_across_sections(pool):
    """A pinned issue shouldn't appear in both rules and issues."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="Critical bug: connection leak",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    all_ids = (
        [m["id"] for m in result["rules"]]
        + [m["id"] for m in result["handoff"]]
        + [m["id"] for m in result["recent_work"]]
        + [m["id"] for m in result["issues"]["items"]]
        + [m["id"] for m in result["decisions"]]
    )
    assert len(all_ids) == len(set(all_ids)), "Memory duplicated across sections"


# --- Project scoping ---


async def test_primer_project_scoping(pool):
    """With project_id, primer includes both project-scoped and global pinned memories."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Global rule: always use weft_remember",
        confidence=1.0,
        pinned=True,
        project_id=None,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft rule: store.py is the only Postgres writer",
        confidence=0.9,
        pinned=True,
        project_id="weft",
    ))
    # Different project (should not appear)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Loom rule: orchestrator.py owns task state",
        confidence=0.9,
        pinned=True,
        project_id="loom",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=1500)

    rule_contents = [r["content"] for r in result["rules"]]
    assert any("weft_remember" in c for c in rule_contents)
    assert any("store.py" in c for c in rule_contents)
    assert not any("orchestrator" in c for c in rule_contents)


async def test_primer_default_budget_is_2400(pool):
    """Default budget is 2400 tokens."""
    result = await build_primer(pool)
    assert result["budget_tokens"] == 2400


# --- Grounding section ---


async def test_primer_grounding_with_project(pool):
    """Project grounding shows up when project_id is set and memory exists."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft: Python MCP server for persistent agent memory, using asyncpg + pgvector",
        topic=[_GROUNDING_TOPIC],
        confidence=1.0,
        project_id="weft",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=1500)

    assert result["grounding"] is not None
    assert "Python MCP server" in result["grounding"]


async def test_primer_grounding_none_without_project(pool):
    """Without project_id, grounding is always None."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Some grounding text",
        topic=[_GROUNDING_TOPIC],
        confidence=1.0,
        project_id="weft",
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert result["grounding"] is None


async def test_primer_grounding_none_when_no_memory(pool):
    """With project_id but no grounding memory, grounding is None."""
    result = await build_primer(pool, project_id="weft", budget_tokens=1500)

    assert result["grounding"] is None


async def test_primer_grounding_not_cross_project(pool):
    """Grounding from a different project doesn't leak in."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Loom: task orchestration system",
        topic=[_GROUNDING_TOPIC],
        confidence=1.0,
        project_id="loom",
    ))

    result = await build_primer(pool, project_id="weft", budget_tokens=1500)

    assert result["grounding"] is None


# --- Excluded count ---


async def test_primer_excluded_count_when_budget_full(pool):
    """Excluded count reflects memories that didn't fit in budget."""
    # Create a pinned rule that fills most of the budget
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="A rule: " + "x" * 300,
        confidence=1.0,
        pinned=True,
    ))
    # Create a decision that won't fit
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="A decision: " + "y" * 300,
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=100)

    assert result["excluded"] >= 1


async def test_primer_excluded_zero_when_all_fit(pool):
    """Excluded is 0 when everything fits in budget."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Short rule",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert result["excluded"] == 0


# --- Handoff age ---


async def test_primer_handoff_has_age_hours(pool):
    """Handoff entries include an age_hours field."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Just finished some work",
        topic=["session-handoff"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["handoff"]) == 1
    assert "age_hours" in result["handoff"][0]
    # Just created, should be very recent
    assert result["handoff"][0]["age_hours"] < 1.0


# --- Handoff auto-prune (store-level pattern) ---


async def test_handoff_prune_archives_old_handoffs(pool):
    """Simulates weft_handoff auto-prune: archiving previous handoffs leaves
    only the newest one active, and the primer surfaces it correctly."""
    from weft.store import update_memory

    ids = []
    for i in range(5):
        m = await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content=f"## Session Handoff\n\n**Summary:** Session {i}",
            topic=["session-handoff"],
            confidence=1.0,
            project_id="test-proj",
        ))
        ids.append(m.id)
        await asyncio.sleep(0.01)

    newest_id = ids[-1]

    # Archive all except the newest (mimics auto-prune in weft_handoff)
    prev = await list_memories(
        pool, memory_type=MemoryType.handoff,
        status=MemoryStatus.active, project_id="test-proj", limit=100,
    )
    archived = 0
    for old in prev:
        if old.id != newest_id:
            await update_memory(pool, old.id, status=MemoryStatus.archived)
            archived += 1

    assert archived == 4

    # Only the newest handoff should appear in the primer
    result = await build_primer(pool, project_id="test-proj", budget_tokens=1500)
    assert len(result["handoff"]) == 1
    assert "Session 4" in result["handoff"][0]["content"]


async def test_handoff_prune_scoped_to_project(pool):
    """Auto-prune only archives handoffs for the same project."""
    from weft.store import update_memory

    # Store handoffs for two different projects
    proj_a = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Project A old",
        topic=["session-handoff"], confidence=1.0, project_id="proj-a",
    ))
    await asyncio.sleep(0.01)
    proj_a_new = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Project A new",
        topic=["session-handoff"], confidence=1.0, project_id="proj-a",
    ))
    proj_b = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Project B",
        topic=["session-handoff"], confidence=1.0, project_id="proj-b",
    ))

    # Prune only proj-a handoffs (mimics scoped auto-prune)
    prev = await list_memories(
        pool, memory_type=MemoryType.handoff,
        status=MemoryStatus.active, project_id="proj-a", limit=100,
    )
    for old in prev:
        if old.id != proj_a_new.id:
            await update_memory(pool, old.id, status=MemoryStatus.archived)

    # proj-b handoff should be untouched
    result_b = await build_primer(pool, project_id="proj-b", budget_tokens=1500)
    assert len(result_b["handoff"]) == 1
    assert "Project B" in result_b["handoff"][0]["content"]


# --- Freshness indicator ---


async def test_primer_freshness_hours_with_memories(pool):
    """freshness_hours reflects the age of the newest included memory."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="A pinned rule for freshness test",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=1500)

    assert result["freshness_hours"] is not None
    # Just created, should be very recent
    assert result["freshness_hours"] < 1.0


async def test_primer_freshness_hours_none_when_empty(pool):
    """freshness_hours is None when no memories are included."""
    result = await build_primer(pool, budget_tokens=1500)

    assert result["freshness_hours"] is None


async def test_primer_freshness_hours_reflects_newest(pool):
    """freshness_hours should reflect the most recently created memory, not the oldest."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Old decision",
        confidence=0.9,
    ))
    await asyncio.sleep(0.01)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="New decision",
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["freshness_hours"] is not None
    assert result["freshness_hours"] < 1.0


# --- Recent work (milestones) ---


async def test_primer_recent_work_empty_when_no_milestones(pool):
    """recent_work is empty when no milestone memories exist."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["recent_work"] == []
    assert result["section_tokens"]["recent_work"] == 0


async def test_primer_recent_work_surfaces_milestones(pool):
    """Milestone memories appear in the recent_work section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="Shipped primer optimization (319 tests)",
        topic=["loom-abc123"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["recent_work"]) == 1
    entry = result["recent_work"][0]
    assert entry["summary"] == "Shipped primer optimization (319 tests)"
    assert entry["age_hours"] < 1.0
    assert "loom-abc123" in entry["refs"]
    assert "id" in entry


async def test_primer_recent_work_max_items(pool):
    """recent_work is capped at _MAX_RECENT_WORK items."""
    for i in range(_MAX_RECENT_WORK + 2):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.milestone,
            content=f"Milestone {i}",
            topic=[f"task-{i}"],
            confidence=1.0,
        ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["recent_work"]) <= _MAX_RECENT_WORK


async def test_primer_recent_work_most_recent_first(pool):
    """recent_work entries are ordered by most recent first."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="Older milestone",
        confidence=1.0,
    ))
    await asyncio.sleep(0.01)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="Newer milestone",
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["recent_work"]) == 2
    assert result["recent_work"][0]["summary"] == "Newer milestone"
    assert result["recent_work"][1]["summary"] == "Older milestone"


async def test_primer_recent_work_respects_section_cap(pool):
    """recent_work section respects its token cap."""
    # Create a milestone that's very long — should hit the cap
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="Big milestone: " + "x" * 1000,
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["section_tokens"]["recent_work"] <= _CAP_RECENT_WORK


async def test_primer_recent_work_not_in_other_sections(pool):
    """Milestones only appear in recent_work, not in other sections."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="A completed milestone",
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    # Should be in recent_work
    assert len(result["recent_work"]) == 1
    # Should NOT be in rules, handoff, issues, or decisions
    other_contents = (
        [m["content"] for m in result["rules"]]
        + [m["content"] for m in result["handoff"]]
        + [m["content"] for m in result["issues"]["items"]]
        + [m["content"] for m in result["decisions"]]
    )
    assert not any("completed milestone" in c for c in other_contents)


# --- Per-section token caps ---


async def test_primer_section_tokens_in_response(pool):
    """section_tokens dict is always present with all section keys."""
    result = await build_primer(pool, budget_tokens=1800)

    assert "section_tokens" in result
    expected_sections = {"grounding", "rules", "behaviors", "handoff", "recent_work", "issues", "anti_patterns", "decisions", "entities"}
    assert set(result["section_tokens"].keys()) == expected_sections


async def test_primer_rules_section_cap(pool):
    """Rules section respects its per-section cap even with budget remaining."""
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Rule {i}: " + "x" * 50,
            confidence=0.9,
            pinned=True,
        ))

    result = await build_primer(pool, budget_tokens=4000)

    assert result["section_tokens"]["rules"] <= _CAP_RULES


async def test_primer_issues_section_cap(pool):
    """Issues section respects its per-section cap."""
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.issue,
            content=f"Issue {i}: " + "y" * 50,
            confidence=0.9,
        ))

    result = await build_primer(pool, budget_tokens=4000)

    assert result["section_tokens"]["issues"] <= _CAP_ISSUES


async def test_primer_decisions_section_cap(pool):
    """Decisions section respects its per-section cap."""
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: " + "z" * 50,
            confidence=0.9,
        ))

    result = await build_primer(pool, budget_tokens=4000)

    assert result["section_tokens"]["decisions"] <= _CAP_DECISIONS


async def test_primer_section_caps_dont_block_other_sections(pool):
    """Hitting one section's cap doesn't prevent other sections from filling."""
    # Fill rules to its cap
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=f"Rule {i}: " + "x" * 50,
            confidence=0.9,
            pinned=True,
        ))
    # Add a decision — should still fit
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="A decision that should still appear",
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=4000)

    assert result["section_tokens"]["rules"] <= _CAP_RULES
    assert len(result["decisions"]) == 1


# --- review_after lifecycle flagging ---


async def test_primer_rule_with_overdue_review_after(pool):
    """A pinned rule past its review_after date is flagged with review_due=True."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Old rule that needs review",
        confidence=1.0,
        pinned=True,
        review_after=past,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["rules"]) == 1
    rule = result["rules"][0]
    assert rule["review_due"] is True
    assert "review_after" in rule


async def test_primer_rule_with_future_review_after(pool):
    """A pinned rule with future review_after shows the date but review_due=False."""
    future = datetime.now(timezone.utc) + timedelta(days=30)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Fresh rule with future review",
        confidence=1.0,
        pinned=True,
        review_after=future,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["rules"]) == 1
    rule = result["rules"][0]
    assert rule["review_due"] is False
    assert "review_after" in rule


async def test_primer_rule_without_review_after(pool):
    """A pinned rule with no review_after has no review_due field."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Rule with no lifecycle date",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["rules"]) == 1
    rule = result["rules"][0]
    assert "review_due" not in rule


async def test_primer_decision_with_overdue_review_after(pool):
    """A decision past its review_after date is flagged with review_due=True."""
    past = datetime.now(timezone.utc) - timedelta(days=7)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Old decision that should be reviewed",
        confidence=0.9,
        review_after=past,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert decision["review_due"] is True


async def test_primer_decision_without_review_after(pool):
    """A decision with no review_after has no review_due field."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Decision with no lifecycle date",
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert len(result["decisions"]) == 1
    assert "review_due" not in result["decisions"][0]


# --- Usefulness-score ranking ---


async def test_primer_rules_usefulness_tiebreaker(pool):
    """When two pinned rules have equal confidence, higher usefulness ranks first."""
    m1 = await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Less useful rule",
        confidence=0.9,
        pinned=True,
    ))
    await asyncio.sleep(0.01)
    m2 = await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="More useful rule",
        confidence=0.9,
        pinned=True,
    ))

    # Downgrade m1's usefulness
    await record_feedback(pool, m1.id, helpful=False)
    # Upgrade m2's usefulness
    await record_feedback(pool, m2.id, helpful=True)

    result = await build_primer(pool, budget_tokens=1500)

    assert len(result["rules"]) == 2
    assert result["rules"][0]["content"] == "More useful rule"
    assert result["rules"][1]["content"] == "Less useful rule"


async def test_primer_issues_ranked_by_usefulness(pool):
    """Issues with higher usefulness_score rank before less useful ones."""
    m1 = await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="Rarely confirmed issue",
        confidence=0.8,
    ))
    await asyncio.sleep(0.01)
    m2 = await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="Frequently confirmed issue",
        confidence=0.8,
    ))

    # m1 gets negative feedback, m2 gets positive
    await record_feedback(pool, m1.id, helpful=False)
    await record_feedback(pool, m2.id, helpful=True)

    result = await build_primer(pool, budget_tokens=1500)

    items = result["issues"]["items"]
    assert len(items) == 2
    assert items[0]["content"] == "Frequently confirmed issue"
    assert items[1]["content"] == "Rarely confirmed issue"


async def test_primer_decisions_ranked_by_usefulness(pool):
    """Decisions with higher usefulness_score rank before less useful ones (same scope)."""
    m1 = await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Rarely helpful decision",
        confidence=0.9,
    ))
    await asyncio.sleep(0.01)
    m2 = await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Very helpful decision",
        confidence=0.9,
    ))

    await record_feedback(pool, m1.id, helpful=False)
    await record_feedback(pool, m2.id, helpful=True)

    result = await build_primer(pool, budget_tokens=1500)

    decisions = result["decisions"]
    assert len(decisions) == 2
    assert decisions[0]["content"] == "Very helpful decision"
    assert decisions[1]["content"] == "Rarely helpful decision"


# --- Empty-section hints ---


async def test_primer_hints_all_present_when_empty(pool):
    """All five section hints appear when database is empty."""
    result = await build_primer(pool, budget_tokens=1800)

    hints = result["hints"]
    assert "rules" in hints
    assert "handoff" in hints
    assert "recent_work" in hints
    assert "issues" in hints
    assert "decisions" in hints
    # Each hint should mention the relevant tool
    assert "weft_remember" in hints["rules"]
    assert "weft_handoff" in hints["handoff"]
    assert "weft_learn" in hints["recent_work"]
    assert "weft_remember" in hints["issues"]
    assert "weft_remember" in hints["decisions"]


async def test_primer_hints_disappear_when_populated(pool):
    """Hints for populated sections are absent; hints for empty sections remain."""
    # Populate rules and decisions
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Always use weft_remember",
        confidence=1.0,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Don't split store.py",
        confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    hints = result["hints"]
    # Populated sections have no hint
    assert "rules" not in hints
    assert "decisions" not in hints
    # Empty sections still have hints
    assert "handoff" in hints
    assert "recent_work" in hints
    assert "issues" in hints


async def test_primer_no_hints_when_all_populated(pool):
    """Hints dict is empty when all sections have content."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference, content="A rule",
        confidence=1.0, pinned=True,
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when testing", action="use pytest",
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Some work",
        topic=["session-handoff"], confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone, content="Did a thing",
        confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue, content="A bug", confidence=0.8,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision, content="A choice", confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["hints"] == {}


async def test_primer_hint_text_matches_constants(pool):
    """Empty-section hint text matches the _SECTION_HINTS constants exactly."""
    result = await build_primer(pool, budget_tokens=1800)

    for section_name, expected_hint in _SECTION_HINTS.items():
        assert result["hints"][section_name] == expected_hint


# --- Cold-start onboarding ---


async def test_primer_onboarding_on_cold_start(pool):
    """Onboarding text appears when DB is empty (classic cold start)."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is not None
    assert result["onboarding"] == _ONBOARDING_TEXT
    # Check key tool names are mentioned
    assert "weft_remember" in result["onboarding"]
    assert "weft_recall" in result["onboarding"]
    assert "weft_learn" in result["onboarding"]
    assert "weft_handoff" in result["onboarding"]
    assert "weft_feedback" in result["onboarding"]


async def test_primer_onboarding_absent_with_handoff(pool):
    """Onboarding disappears as soon as a handoff exists (agent has prior session)."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** First session done",
        topic=["session-handoff"],
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is None


async def test_primer_onboarding_absent_with_many_memories(pool):
    """Onboarding disappears when enough memories exist even without a handoff."""
    # Create enough items to exceed _COLD_START_THRESHOLD
    for i in range(_COLD_START_THRESHOLD + 1):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: an established choice",
            confidence=0.9,
        ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is None


async def test_primer_onboarding_with_few_memories_no_handoff(pool):
    """Onboarding still shows with <= threshold items and no handoff."""
    # Add exactly threshold items (should still be cold start)
    for i in range(_COLD_START_THRESHOLD):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: early choice",
            confidence=0.9,
        ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is not None


async def test_primer_onboarding_token_budget(pool):
    """Onboarding text stays within ~200 token budget (proxy: < 250 words)."""
    word_count = len(_ONBOARDING_TEXT.split())
    assert word_count < 250, f"Onboarding text is {word_count} words, should be < 250"
    assert word_count > 20, f"Onboarding text is only {word_count} words, seems too short"


async def test_primer_cold_start_has_both_hints_and_onboarding(pool):
    """On cold start, both hints and onboarding are present simultaneously."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is not None
    assert len(result["hints"]) == 7  # all six sections empty + loom cold-start hint


async def test_primer_established_agent_has_neither(pool):
    """An established agent (handoff + populated sections) sees neither hints nor onboarding."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference, content="A rule",
        confidence=1.0, pinned=True,
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when testing", action="use pytest",
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Returning agent",
        topic=["session-handoff"], confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone, content="Did work",
        confidence=1.0,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue, content="A bug", confidence=0.8,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision, content="A choice", confidence=0.9,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is None
    assert result["hints"] == {}


# --- Loom cold-start hint ---


async def test_primer_loom_hint_on_cold_start(pool):
    """Loom project hint appears on cold start to prevent task scoping mistakes."""
    result = await build_primer(pool, budget_tokens=1800)

    assert "loom" in result["hints"]
    assert "loom_create_project" in result["hints"]["loom"]
    assert "loom_decompose" in result["hints"]["loom"]


async def test_primer_loom_hint_absent_for_established_agent(pool):
    """Loom hint does not appear once a handoff exists (not a cold start)."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Session Handoff\n\n**Summary:** Returning",
        topic=["session-handoff"], confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=1800)

    assert "loom" not in result["hints"]


async def test_primer_onboarding_mentions_loom(pool):
    """Onboarding text includes Loom integration guidance."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["onboarding"] is not None
    assert "loom_create_project" in result["onboarding"]
