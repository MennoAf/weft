"""Tests for Mode/Persona system — DB schema, Pydantic models, and RLS."""

from __future__ import annotations

import pytest

from weft.models import Mode, ModeCreate, ModeWeights


# ── Pydantic model tests (no DB needed) ──────────────────────────────


class TestModeWeights:
    def test_defaults(self):
        w = ModeWeights()
        assert w.vector_weight == 0.5
        assert w.bm25_weight == 0.5
        assert w.recency_bias == 0.0
        assert w.entity_boost == 1.0
        assert w.behavior_boost == 1.0

    def test_partial_override(self):
        w = ModeWeights(vector_weight=0.8)
        assert w.vector_weight == 0.8
        assert w.bm25_weight == 0.5  # default preserved

    def test_from_empty_dict(self):
        """Empty JSONB '{}' from DB should produce all defaults."""
        w = ModeWeights(**{})
        assert w.vector_weight == 0.5
        assert w.entity_boost == 1.0

    def test_rejects_negative_weight(self):
        with pytest.raises(Exception):
            ModeWeights(vector_weight=-0.1)

    def test_rejects_over_max(self):
        with pytest.raises(Exception):
            ModeWeights(entity_boost=11.0)


class TestModeCreate:
    def test_minimal(self):
        mc = ModeCreate(name="research")
        assert mc.name == "research"
        assert mc.description is None
        assert isinstance(mc.weights, ModeWeights)

    def test_with_weights(self):
        mc = ModeCreate(
            name="cooking",
            weights=ModeWeights(recency_bias=0.8, entity_boost=2.0),
        )
        assert mc.weights.recency_bias == 0.8


class TestModeModel:
    def test_round_trip(self):
        m = Mode(
            name="research",
            user_id="user-1",
            weights=ModeWeights(vector_weight=0.9),
        )
        d = m.model_dump(mode="json")
        m2 = Mode.model_validate(d)
        assert m2.name == m.name
        assert m2.weights.vector_weight == 0.9

    def test_empty_weights_hydrates_defaults(self):
        """Mode with weights={} should still have ModeWeights defaults."""
        m = Mode.model_validate({
            "name": "test",
            "user_id": "user-1",
            "weights": {},
        })
        assert m.weights.vector_weight == 0.5
        assert m.weights.entity_boost == 1.0

    def test_to_dict(self):
        m = Mode(name="coding", user_id="user-1")
        d = m.to_dict()
        assert d["name"] == "coding"
        assert "weights" in d
        assert isinstance(d["weights"], dict)


# ── DB migration tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_modes_table_exists(pool):
    """Migration 20 should create the modes table."""
    exists = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'modes')"
    )
    assert exists


@pytest.mark.asyncio
async def test_modes_columns(pool):
    """Modes table should have the expected columns."""
    rows = await pool.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'modes'
        ORDER BY ordinal_position
        """
    )
    cols = {r["column_name"]: r["data_type"] for r in rows}
    assert "id" in cols
    assert "user_id" in cols
    assert "name" in cols
    assert "description" in cols
    assert "weights" in cols  # JSONB
    assert "created_at" in cols
    assert "updated_at" in cols


@pytest.mark.asyncio
async def test_modes_unique_constraint(pool):
    """Each user can have only one mode with a given name."""
    await pool.execute(
        """
        INSERT INTO modes (id, user_id, name, weights)
        VALUES ('m-1', 'user-a', 'research', '{}')
        """
    )
    with pytest.raises(Exception):  # UniqueViolationError
        await pool.execute(
            """
            INSERT INTO modes (id, user_id, name, weights)
            VALUES ('m-2', 'user-a', 'research', '{}')
            """
        )


@pytest.mark.asyncio
async def test_modes_different_users_same_name(pool):
    """Different users can have modes with the same name."""
    await pool.execute(
        "INSERT INTO modes (id, user_id, name, weights) VALUES ('m-1', 'user-a', 'research', '{}')"
    )
    await pool.execute(
        "INSERT INTO modes (id, user_id, name, weights) VALUES ('m-2', 'user-b', 'research', '{}')"
    )
    count = await pool.fetchval("SELECT count(*) FROM modes")
    assert count == 2


@pytest.mark.asyncio
async def test_modes_rls_enabled(pool):
    """RLS should be enabled on the modes table."""
    rls = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'modes'"
    )
    assert rls is True


@pytest.mark.asyncio
async def test_modes_rls_policy_exists(pool):
    """RLS policies should exist for the modes table."""
    policies = await pool.fetch(
        "SELECT policyname FROM pg_policies WHERE tablename = 'modes'"
    )
    names = {r["policyname"] for r in policies}
    assert "modes_select" in names
    assert "modes_insert" in names
    assert "modes_update" in names
    assert "modes_delete" in names
