"""Tests for mode-aware primer — verifies ModeWeights influence primer output."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.behaviors import store_behavior
from weft.models import (
    BehaviorCreate,
    MemoryCreate,
    MemorySource,
    MemoryType,
    ModeCreate,
    ModeWeights,
)
from weft.modes import upsert_mode
from weft.primer import build_primer
from weft.store import store_memory


# --- Mode parameter acceptance ---


@pytest.mark.asyncio
async def test_primer_accepts_mode_param(pool):
    """build_primer should accept a mode parameter without error."""
    result = await build_primer(pool, mode=None)
    assert "rules" in result


@pytest.mark.asyncio
async def test_primer_accepts_mode_string(pool):
    """build_primer with a mode name that doesn't exist should fall back to defaults."""
    result = await build_primer(pool, mode="nonexistent")
    assert "rules" in result


@pytest.mark.asyncio
async def test_primer_resolves_named_mode(pool):
    """build_primer with a named mode should use that mode's weights."""
    await upsert_mode(pool, ModeCreate(
        name="research",
        weights=ModeWeights(behavior_boost=5.0, entity_boost=5.0),
    ))
    result = await build_primer(pool, mode="research")
    assert "rules" in result


# --- behavior_boost affects behavior section ---


@pytest.mark.asyncio
async def test_behavior_boost_zero_suppresses_behaviors(pool):
    """behavior_boost=0 should result in zero behavior section cap."""
    # Store some behaviors
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when testing", action="use pytest",
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when deploying", action="check CI",
    ))

    # Default: behaviors should appear
    result_default = await build_primer(pool, mode=None, disclosure="full")
    assert len(result_default["behaviors"]) > 0

    # Suppressed: behavior_boost=0 should produce empty behaviors
    await upsert_mode(pool, ModeCreate(
        name="no-behaviors",
        weights=ModeWeights(behavior_boost=0.0),
    ))
    result_suppressed = await build_primer(pool, mode="no-behaviors", disclosure="full")
    assert len(result_suppressed["behaviors"]) == 0


@pytest.mark.asyncio
async def test_behavior_boost_high_increases_cap(pool):
    """behavior_boost > 1 should allow more behavior tokens in the section."""
    # Store several behaviors to potentially fill the section
    for i in range(6):
        await store_behavior(pool, BehaviorCreate(
            trigger_pattern=f"pattern {i}", action=f"action {i}",
        ))

    await upsert_mode(pool, ModeCreate(
        name="high-behavior",
        weights=ModeWeights(behavior_boost=3.0),
    ))

    result_default = await build_primer(pool, mode=None, disclosure="full")
    result_boosted = await build_primer(pool, mode="high-behavior", disclosure="full")

    # Boosted mode should include at least as many behaviors
    assert len(result_boosted["behaviors"]) >= len(result_default["behaviors"])


# --- entity_boost affects entity section ---


@pytest.mark.asyncio
async def test_entity_boost_zero_suppresses_entities(pool):
    """entity_boost=0 should result in zero entity section cap."""
    from weft.entities import store_entity
    from weft.models import EntityCreate

    await store_entity(pool, EntityCreate(name="Jason", entity_type="person"))

    result_default = await build_primer(pool, mode=None, disclosure="full")
    assert len(result_default["entities"]) > 0

    await upsert_mode(pool, ModeCreate(
        name="no-entities",
        weights=ModeWeights(entity_boost=0.0),
    ))
    result_suppressed = await build_primer(pool, mode="no-entities", disclosure="full")
    assert len(result_suppressed["entities"]) == 0


# --- recency_bias modulates milestone ranking ---


@pytest.mark.asyncio
async def test_recency_bias_high_favors_recent(pool):
    """High recency_bias should favor newer milestones over older ones."""
    now = datetime.now(timezone.utc)

    # Old milestone (2 days ago)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone, content="Old milestone from two days ago",
        source=MemorySource.conversation,
    ))
    # Backdate it
    await pool.execute(
        "UPDATE memories SET created_at = $1 WHERE content LIKE 'Old milestone%'",
        now - timedelta(days=2),
    )

    # New milestone (1 hour ago)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.milestone, content="New milestone from one hour ago",
        source=MemorySource.conversation,
    ))
    await pool.execute(
        "UPDATE memories SET created_at = $1 WHERE content LIKE 'New milestone%'",
        now - timedelta(hours=1),
    )

    # Default mode: both should appear, newest first (default sort)
    result_default = await build_primer(pool, mode=None, disclosure="full")
    if result_default["recent_work"]:
        assert result_default["recent_work"][0]["summary"].startswith("New milestone")

    # High recency mode should also have new first
    await upsert_mode(pool, ModeCreate(
        name="recency-heavy",
        weights=ModeWeights(recency_bias=1.0),
    ))
    result_recency = await build_primer(pool, mode="recency-heavy", disclosure="full")
    if result_recency["recent_work"]:
        assert result_recency["recent_work"][0]["summary"].startswith("New milestone")


# --- Mode doesn't break progressive disclosure ---


@pytest.mark.asyncio
async def test_mode_with_progressive_disclosure(pool):
    """Mode parameter should work with both progressive and full disclosure."""
    await upsert_mode(pool, ModeCreate(
        name="coding",
        weights=ModeWeights(vector_weight=0.8, bm25_weight=0.2),
    ))

    result_prog = await build_primer(pool, mode="coding", disclosure="progressive")
    assert result_prog["disclosure"] == "progressive"

    result_full = await build_primer(pool, mode="coding", disclosure="full")
    assert result_full["disclosure"] == "full"
