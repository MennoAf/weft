"""Equivalence tests: section builders vs monolithic build_primer.

These tests store test data, run both the monolithic build_primer and
the individual section builders, and assert identical output.  This is
the safety gate before we can swap the orchestrator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.behaviors import store_behavior
from weft.entities import store_entity
from weft.models import (
    BehaviorCreate,
    EntityCreate,
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from weft.primer import build_primer
from weft.primer_sections.anti_patterns import build_anti_patterns_section
from weft.primer_sections.behaviors import build_behaviors_section
from weft.primer_sections.context import PrimerContext
from weft.primer_sections.decisions import build_decisions_section
from weft.primer_sections.entities import build_entities_section
from weft.primer_sections.grounding import build_grounding_section
from weft.primer_sections.handoff import build_handoff_section
from weft.primer_sections.issues import build_issues_section
from weft.primer_sections.recent_work import build_recent_work_section
from weft.primer_sections.rules import build_rules_section
from weft.store import store_memory


def _make_ctx(pool, *, project_id=None, budget_tokens=4000):
    """Create a PrimerContext with generous budget for equivalence testing."""
    return PrimerContext(
        user_id="test-user", project_id=project_id, agent_id=None,
        pool=pool, budget_tokens=budget_tokens, query=None, query_vec=None,
        disclosure="full", mode=None,
    )


# ---------------------------------------------------------------------------
# Fixture: populate a realistic set of test data
# ---------------------------------------------------------------------------

@pytest.fixture
async def populated_pool(pool):
    """Pool with a representative set of data across all section types."""
    # Grounding
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Test project: an equivalence testing harness.",
        topic=["project-grounding"],
        confidence=1.0,
        project_id="equiv-proj",
    ))

    # Pinned rules
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Always run tests before committing.",
        confidence=1.0,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="Use uv for Python package management.",
        confidence=0.9,
        pinned=True,
    ))

    # Handoff
    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="## Handoff\nCompleted grounding section. Next: rules.",
        confidence=1.0,
    ))

    # Issues
    await store_memory(pool, MemoryCreate(
        type=MemoryType.issue,
        content="weekly_recap test fails with project filter.",
        confidence=0.8,
    ))

    # Anti-patterns
    await store_memory(pool, MemoryCreate(
        type=MemoryType.anti_pattern,
        content="Do not enable FASTMCP_STATELESS_HTTP for production.",
        confidence=0.95,
    ))

    # Decisions
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Use pure Python Pearson correlation, not scipy.",
        confidence=0.85,
        project_id="equiv-proj",
    ))

    # Milestones (recent work)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone,
        content="Shipped primer section stubs and design doc.",
        confidence=0.9,
    ))

    # Behaviors
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when writing tests",
        action="use pytest with testcontainers",
        confidence=0.9,
        priority=10,
    ))

    # Entities
    await store_entity(pool, EntityCreate(
        name="Jason", entity_type="person", description="Project owner",
    ))
    await store_entity(pool, EntityCreate(
        name="Weft", entity_type="project", description="Memory system",
    ))

    return pool


# ---------------------------------------------------------------------------
# Section-by-section equivalence
# ---------------------------------------------------------------------------


async def test_grounding_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, project_id="equiv-proj",
                              budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool, project_id="equiv-proj")
    result = await build_grounding_section(ctx)

    if result.items:
        assert mono["grounding"] == result.items[0]["grounding_line"]
    else:
        assert mono["grounding"] is None
    assert result.tokens_used == mono["section_tokens"]["grounding"]


async def test_rules_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    result = await build_rules_section(ctx)

    assert len(result.items) == len(mono["rules"])
    for mono_item, sect_item in zip(mono["rules"], result.items):
        assert mono_item["id"] == sect_item["id"]
        assert mono_item["content"] == sect_item["content"]
    assert result.tokens_used == mono["section_tokens"]["rules"]


async def test_behaviors_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    # Behaviors come after rules in budget order — simulate preceding sections
    # by running rules first to match token usage
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    result = await build_behaviors_section(ctx)

    assert len(result.items) == len(mono["behaviors"])
    for mono_item, sect_item in zip(mono["behaviors"], result.items):
        assert mono_item["id"] == sect_item["id"]
        assert mono_item["trigger"] == sect_item["trigger"]
    assert result.tokens_used == mono["section_tokens"]["behaviors"]


async def test_handoff_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    # Run preceding sections to match budget state
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    result = await build_handoff_section(ctx)

    assert len(result.items) == len(mono["handoff"])
    if result.items:
        assert result.items[0]["id"] == mono["handoff"][0]["id"]
        assert result.items[0]["content"] == mono["handoff"][0]["content"]
    assert result.tokens_used == mono["section_tokens"]["handoff"]


async def test_issues_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    # Run preceding sections
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    result = await build_issues_section(ctx)

    mono_items = mono["issues"]["items"]
    assert len(result.items) == len(mono_items)
    for mono_item, sect_item in zip(mono_items, result.items):
        assert mono_item["id"] == sect_item["id"]
    assert result.tokens_used == mono["section_tokens"]["issues"]


async def test_anti_patterns_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    await build_issues_section(ctx)
    result = await build_anti_patterns_section(ctx)

    assert len(result.items) == len(mono["anti_patterns"])
    for mono_item, sect_item in zip(mono["anti_patterns"], result.items):
        assert mono_item["id"] == sect_item["id"]
    assert result.tokens_used == mono["section_tokens"]["anti_patterns"]


async def test_decisions_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, project_id="equiv-proj",
                              budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool, project_id="equiv-proj")
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    await build_issues_section(ctx)
    await build_anti_patterns_section(ctx)
    result = await build_decisions_section(ctx)

    assert len(result.items) == len(mono["decisions"])
    for mono_item, sect_item in zip(mono["decisions"], result.items):
        assert mono_item["id"] == sect_item["id"]
    assert result.tokens_used == mono["section_tokens"]["decisions"]


async def test_entities_equivalence(populated_pool):
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")
    ctx = _make_ctx(pool)
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    await build_issues_section(ctx)
    await build_anti_patterns_section(ctx)
    await build_decisions_section(ctx)
    result = await build_entities_section(ctx)

    mono_names = {e["name"] for e in mono["entities"]}
    sect_names = {e["name"] for e in result.items}
    assert mono_names == sect_names
    assert result.tokens_used == mono["section_tokens"]["entities"]


# ---------------------------------------------------------------------------
# Full pipeline equivalence: all sections in order, same total tokens
# ---------------------------------------------------------------------------


async def test_full_pipeline_token_equivalence(populated_pool):
    """Total tokens from running all sections sequentially matches monolithic."""
    pool = populated_pool
    mono = await build_primer(pool, budget_tokens=4000, disclosure="full")

    ctx = _make_ctx(pool)
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    await build_issues_section(ctx)
    await build_anti_patterns_section(ctx)
    await build_decisions_section(ctx)
    await build_entities_section(ctx)

    assert ctx.used_tokens == mono["total_tokens"]
    assert ctx.section_tokens == mono["section_tokens"]


async def test_empty_db_equivalence(pool):
    """Empty DB: section builders produce same state as monolithic."""
    mono = await build_primer(pool, budget_tokens=2400, disclosure="full")

    ctx = _make_ctx(pool, budget_tokens=2400)
    await build_grounding_section(ctx)
    await build_rules_section(ctx)
    await build_behaviors_section(ctx)
    await build_handoff_section(ctx)
    await build_recent_work_section(ctx)
    await build_issues_section(ctx)
    await build_anti_patterns_section(ctx)
    await build_decisions_section(ctx)
    await build_entities_section(ctx)

    assert ctx.used_tokens == 0
    assert ctx.used_tokens == mono["total_tokens"]
    assert ctx.section_tokens == mono["section_tokens"]
