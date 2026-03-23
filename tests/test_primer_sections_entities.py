"""Tests for weft.primer_sections.entities — entities section builder."""

from __future__ import annotations

import pytest

from weft.entities import store_entity
from weft.models import EntityCreate
from weft.primer import build_primer
from weft.primer_sections.context import PrimerContext
from weft.primer_sections.entities import build_entities_section


def _make_ctx(pool, *, project_id=None, budget_tokens=2400):
    return PrimerContext(
        user_id="test-user", project_id=project_id, agent_id=None,
        pool=pool, budget_tokens=budget_tokens, query=None, query_vec=None,
        disclosure="full", mode=None,
    )


async def test_empty_when_no_entities(pool):
    ctx = _make_ctx(pool)
    result = await build_entities_section(ctx)

    assert result.items == []
    assert result.tokens_used == 0
    assert result.skipped is False


async def test_returns_entities(pool):
    await store_entity(pool, EntityCreate(
        name="Jason", entity_type="person", description="Project owner",
    ))
    await store_entity(pool, EntityCreate(
        name="Weft", entity_type="project", description="Memory system",
    ))

    ctx = _make_ctx(pool)
    result = await build_entities_section(ctx)

    assert len(result.items) == 2
    names = {e["name"] for e in result.items}
    assert "Jason" in names
    assert "Weft" in names
    assert result.tokens_used > 0


async def test_respects_max_items(pool):
    """Entity count is capped at 10."""
    for i in range(15):
        await store_entity(pool, EntityCreate(
            name=f"Entity-{i}", entity_type="tool",
        ))

    ctx = _make_ctx(pool)
    result = await build_entities_section(ctx)

    assert len(result.items) <= 10


async def test_updates_context_state(pool):
    await store_entity(pool, EntityCreate(
        name="Loom", entity_type="project", description="Task orchestrator",
    ))

    ctx = _make_ctx(pool)
    result = await build_entities_section(ctx)

    assert ctx.used_tokens == result.tokens_used
    assert ctx.section_tokens["entities"] == result.tokens_used


async def test_entity_boost_zero_excludes_all(pool):
    """entity_boost=0 means cap=0, no entities should be included."""
    await store_entity(pool, EntityCreate(
        name="Test", entity_type="tool", description="Something",
    ))

    ctx = _make_ctx(pool)
    ctx.entity_boost = 0.0
    result = await build_entities_section(ctx)

    assert result.items == []


async def test_matches_monolithic_primer(pool):
    """Entities from section builder match the monolithic build_primer."""
    await store_entity(pool, EntityCreate(
        name="Jason", entity_type="person", description="Owner",
    ))
    await store_entity(pool, EntityCreate(
        name="Weft", entity_type="project", description="Memory",
    ))

    mono = await build_primer(pool, budget_tokens=2400, disclosure="full")
    ctx = _make_ctx(pool, budget_tokens=2400)
    result = await build_entities_section(ctx)

    # Same items (order may vary)
    mono_names = {e["name"] for e in mono["entities"]}
    sect_names = {e["name"] for e in result.items}
    assert mono_names == sect_names
    assert result.tokens_used == mono["section_tokens"]["entities"]
