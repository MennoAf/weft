"""Tests for weft.primer_sections.grounding — grounding section builder.

These tests verify that build_grounding_section produces identical behavior
to the grounding block in the monolithic build_primer (primer.py L332-345).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.primer import build_primer
from weft.primer_sections.context import GROUNDING_TOPIC, PrimerContext, SectionResult
from weft.primer_sections.grounding import build_grounding_section
from weft.store import store_memory
from weft.tokens import estimate_tokens


def _make_ctx(pool, *, project_id=None, budget_tokens=2400):
    """Create a minimal PrimerContext for testing."""
    return PrimerContext(
        user_id="test-user",
        project_id=project_id,
        agent_id=None,
        pool=pool,
        budget_tokens=budget_tokens,
        query=None,
        query_vec=None,
        disclosure="full",
        mode=None,
    )


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


async def test_skips_when_no_project_id(pool):
    """No project_id → skipped=True, zero tokens."""
    ctx = _make_ctx(pool, project_id=None)
    result = await build_grounding_section(ctx)

    assert result.skipped is True
    assert result.skip_reason == "no project_id"
    assert result.items == []
    assert result.tokens_used == 0
    assert ctx.used_tokens == 0


async def test_empty_when_no_grounding_memory(pool):
    """project_id set but no grounding memory → empty, not skipped."""
    ctx = _make_ctx(pool, project_id="some-project")
    result = await build_grounding_section(ctx)

    assert result.skipped is False
    assert result.items == []
    assert result.tokens_used == 0


async def test_returns_grounding_line(pool):
    """Stores a grounding memory and verifies the section returns it."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft is a persistent memory system for AI agents.",
        source=MemorySource.conversation,
        confidence=1.0,
        topic=[GROUNDING_TOPIC],
        project_id="test-proj",
    ))

    ctx = _make_ctx(pool, project_id="test-proj")
    result = await build_grounding_section(ctx)

    assert result.skipped is False
    assert len(result.items) == 1
    assert result.items[0]["grounding_line"] == "Weft is a persistent memory system for AI agents."
    assert result.tokens_used > 0


async def test_updates_context_state(pool):
    """Verify section updates used_tokens, seen_ids, section_tokens."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Test project description.",
        source=MemorySource.conversation,
        confidence=1.0,
        topic=[GROUNDING_TOPIC],
        project_id="proj-1",
    ))

    ctx = _make_ctx(pool, project_id="proj-1")
    result = await build_grounding_section(ctx)

    assert ctx.used_tokens == result.tokens_used
    assert ctx.used_tokens > 0
    assert len(ctx.seen_ids) == 1
    assert ctx.section_tokens["grounding"] == result.tokens_used


async def test_excluded_when_over_budget(pool):
    """Grounding memory too large for budget → excluded."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="A very long grounding that exceeds the tiny budget " + "x" * 500,
        source=MemorySource.conversation,
        confidence=1.0,
        topic=[GROUNDING_TOPIC],
        project_id="proj-2",
    ))

    ctx = _make_ctx(pool, project_id="proj-2", budget_tokens=10)
    result = await build_grounding_section(ctx)

    assert result.items == []
    assert result.tokens_used == 0
    assert ctx.excluded == 1


# ---------------------------------------------------------------------------
# Equivalence with monolithic build_primer
# ---------------------------------------------------------------------------


async def test_matches_monolithic_primer(pool):
    """Grounding from section builder matches the monolithic build_primer."""
    content = "Weft — persistent agent memory."
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content=content,
        source=MemorySource.conversation,
        confidence=1.0,
        topic=[GROUNDING_TOPIC],
        project_id="equiv-proj",
    ))

    # Monolithic
    mono = await build_primer(pool, project_id="equiv-proj",
                              budget_tokens=2400, disclosure="full")

    # Section builder
    ctx = _make_ctx(pool, project_id="equiv-proj", budget_tokens=2400)
    result = await build_grounding_section(ctx)

    # Both should produce the same grounding line
    assert mono["grounding"] == content
    if result.items:
        assert result.items[0]["grounding_line"] == content
    assert result.tokens_used == mono["section_tokens"]["grounding"]
