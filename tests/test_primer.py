"""Tests for weft.primer — session primer context assembly."""

from __future__ import annotations

import asyncio

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import (
    _CAP_DECISIONS,
    _CAP_GROUNDING,
    _CAP_HANDOFF,
    _CAP_ISSUES,
    _CAP_RECENT_WORK,
    _CAP_RULES,
    _GROUNDING_TOPIC,
    _MAX_DECISIONS,
    _MAX_RECENT_WORK,
    _newest_created_at,
    build_primer,
)
from weft.store import store_memory
from weft.tokens import estimate_tokens


# --- Structure & basics ---


async def test_primer_empty_db(pool):
    """No memories -> all sections empty, budget_remaining = budget_tokens."""
    result = await build_primer(pool, budget_tokens=1800)

    assert result["grounding"] is None
    assert result["rules"] == []
    assert result["handoff"] == []
    assert result["recent_work"] == []
    assert result["issues"] == {"count": 0, "items": []}
    assert result["decisions"] == []
    assert result["total_tokens"] == 0
    assert result["budget_tokens"] == 1800
    assert result["budget_remaining"] == 1800
    assert result["excluded"] == 0
    assert result["freshness_hours"] is None
    assert result["section_tokens"] == {
        "grounding": 0, "rules": 0, "handoff": 0,
        "recent_work": 0, "issues": 0, "decisions": 0,
    }


async def test_primer_return_structure(pool):
    """Verify all expected keys are present in the return dict."""
    result = await build_primer(pool, budget_tokens=1800)

    expected_keys = {
        "grounding", "rules", "handoff", "recent_work", "issues", "decisions",
        "total_tokens", "budget_tokens", "budget_remaining", "excluded",
        "freshness_hours", "section_tokens",
    }
    assert set(result.keys()) == expected_keys

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


async def test_primer_default_budget_is_1800(pool):
    """Default budget is 1800 tokens."""
    result = await build_primer(pool)
    assert result["budget_tokens"] == 1800


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
    expected_sections = {"grounding", "rules", "handoff", "recent_work", "issues", "decisions"}
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
