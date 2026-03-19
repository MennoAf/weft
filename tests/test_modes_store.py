"""Tests for mode store layer — CRUD and active weight resolution."""

from __future__ import annotations

import json

import pytest

from weft.models import Mode, ModeCreate, ModeWeights


# ── CRUD tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_mode_creates_new(pool):
    from weft.modes import upsert_mode

    create = ModeCreate(name="research", description="Deep dive mode")
    mode = await upsert_mode(pool, create)
    assert isinstance(mode, Mode)
    assert mode.name == "research"
    assert mode.description == "Deep dive mode"
    assert mode.weights.vector_weight == 0.5  # default


@pytest.mark.asyncio
async def test_upsert_mode_updates_existing(pool):
    from weft.modes import upsert_mode

    await upsert_mode(pool, ModeCreate(name="research"))
    updated = await upsert_mode(
        pool,
        ModeCreate(name="research", weights=ModeWeights(vector_weight=0.9)),
    )
    assert updated.weights.vector_weight == 0.9

    # Only one row should exist
    count = await pool.fetchval("SELECT count(*) FROM modes WHERE name = 'research'")
    assert count == 1


@pytest.mark.asyncio
async def test_get_mode(pool):
    from weft.modes import get_mode, upsert_mode

    await upsert_mode(pool, ModeCreate(name="coding"))
    mode = await get_mode(pool, "coding")
    assert mode is not None
    assert mode.name == "coding"


@pytest.mark.asyncio
async def test_get_mode_not_found(pool):
    from weft.modes import get_mode

    mode = await get_mode(pool, "nonexistent")
    assert mode is None


@pytest.mark.asyncio
async def test_list_modes_empty(pool):
    from weft.modes import list_modes

    modes = await list_modes(pool)
    assert modes == []


@pytest.mark.asyncio
async def test_list_modes_returns_all_ordered(pool):
    from weft.modes import list_modes, upsert_mode

    await upsert_mode(pool, ModeCreate(name="cooking"))
    await upsert_mode(pool, ModeCreate(name="admin"))
    await upsert_mode(pool, ModeCreate(name="research"))

    modes = await list_modes(pool)
    assert len(modes) == 3
    assert [m.name for m in modes] == ["admin", "cooking", "research"]


@pytest.mark.asyncio
async def test_delete_mode_returns_true(pool):
    from weft.modes import delete_mode, get_mode, upsert_mode

    await upsert_mode(pool, ModeCreate(name="temp"))
    deleted = await delete_mode(pool, "temp")
    assert deleted is True

    mode = await get_mode(pool, "temp")
    assert mode is None


@pytest.mark.asyncio
async def test_delete_mode_nonexistent_returns_false(pool):
    from weft.modes import delete_mode

    deleted = await delete_mode(pool, "nonexistent")
    assert deleted is False


# ── get_active_weights tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_active_weights_none_mode(pool):
    from weft.modes import get_active_weights

    weights = await get_active_weights(pool, None)
    assert isinstance(weights, ModeWeights)
    assert weights.vector_weight == 0.5


@pytest.mark.asyncio
async def test_get_active_weights_empty_string(pool):
    from weft.modes import get_active_weights

    weights = await get_active_weights(pool, "")
    assert isinstance(weights, ModeWeights)


@pytest.mark.asyncio
async def test_get_active_weights_not_found(pool):
    from weft.modes import get_active_weights

    weights = await get_active_weights(pool, "unknown")
    assert isinstance(weights, ModeWeights)
    assert weights.vector_weight == 0.5


@pytest.mark.asyncio
async def test_get_active_weights_found(pool):
    from weft.modes import get_active_weights, upsert_mode

    await upsert_mode(
        pool,
        ModeCreate(name="coding", weights=ModeWeights(vector_weight=0.8, bm25_weight=0.2)),
    )
    weights = await get_active_weights(pool, "coding")
    assert weights.vector_weight == 0.8
    assert weights.bm25_weight == 0.2


@pytest.mark.asyncio
async def test_get_active_weights_malformed_json(pool):
    """Malformed weights in DB should fall back to defaults, not raise."""
    from weft.modes import get_active_weights

    # Insert directly with invalid weights shape
    await pool.execute(
        """
        INSERT INTO modes (id, user_id, name, weights)
        VALUES ('m-bad', NULL, 'broken', $1::jsonb)
        """,
        json.dumps({"vector_weight": "not-a-number"}),
    )
    weights = await get_active_weights(pool, "broken")
    assert isinstance(weights, ModeWeights)
    assert weights.vector_weight == 0.5  # fell back to default
