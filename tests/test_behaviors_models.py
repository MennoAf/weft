"""Tests for behavior models and migration."""

from __future__ import annotations

import pytest

from weft.models import (
    Behavior,
    BehaviorCreate,
    BehaviorMatch,
    BehaviorScope,
)


# --- BehaviorScope enum ---


def test_behavior_scope_values():
    assert BehaviorScope.global_.value == "global"
    assert BehaviorScope.project.value == "project"
    assert BehaviorScope.agent.value == "agent"


# --- BehaviorCreate validation ---


def test_behavior_create_minimal():
    bc = BehaviorCreate(
        trigger_pattern="when user asks about testing",
        action="recommend pytest with testcontainers",
    )
    assert bc.trigger_pattern == "when user asks about testing"
    assert bc.action == "recommend pytest with testcontainers"
    assert bc.confidence == 0.7
    assert bc.scope == BehaviorScope.global_
    assert bc.project_id is None
    assert bc.agent_id is None
    assert bc.user_id is None
    assert bc.priority == 0
    assert bc.enabled is True


def test_behavior_create_full():
    bc = BehaviorCreate(
        trigger_pattern="when writing Python code",
        action="use type hints and docstrings",
        confidence=0.9,
        scope=BehaviorScope.project,
        project_id="proj-1",
        agent_id="warp",
        user_id="user-1",
        priority=5,
        enabled=True,
    )
    assert bc.confidence == 0.9
    assert bc.scope == BehaviorScope.project
    assert bc.project_id == "proj-1"
    assert bc.priority == 5


def test_behavior_create_confidence_bounds():
    with pytest.raises(Exception):
        BehaviorCreate(
            trigger_pattern="t", action="a", confidence=1.5,
        )
    with pytest.raises(Exception):
        BehaviorCreate(
            trigger_pattern="t", action="a", confidence=-0.1,
        )


def test_behavior_create_confidence_edge_values():
    bc_zero = BehaviorCreate(trigger_pattern="t", action="a", confidence=0.0)
    assert bc_zero.confidence == 0.0
    bc_one = BehaviorCreate(trigger_pattern="t", action="a", confidence=1.0)
    assert bc_one.confidence == 1.0


# --- Behavior model ---


def test_behavior_defaults():
    b = Behavior(trigger_pattern="when X", action="do Y")
    assert b.id.startswith("weft-")
    assert b.confidence == 0.7
    assert b.scope == BehaviorScope.global_
    assert b.enabled is True
    assert b.access_count == 0
    assert b.token_count == 0
    assert b.status == "active"
    assert b.priority == 0


def test_behavior_to_dict():
    b = Behavior(
        trigger_pattern="when deploying",
        action="run tests first",
        scope=BehaviorScope.project,
        project_id="proj-1",
        priority=3,
    )
    d = b.to_dict()
    assert d["trigger_pattern"] == "when deploying"
    assert d["action"] == "run tests first"
    assert d["scope"] == "project"
    assert d["project_id"] == "proj-1"
    assert d["priority"] == 3
    assert "id" in d
    assert "created_at" in d


def test_behavior_to_dict_global_scope():
    b = Behavior(trigger_pattern="t", action="a")
    d = b.to_dict()
    assert d["scope"] == "global"


# --- BehaviorMatch ---


def test_behavior_match_to_dict():
    b = Behavior(trigger_pattern="when testing", action="use pytest")
    bm = BehaviorMatch(behavior=b, similarity=0.8523)
    d = bm.to_dict()
    assert d["similarity"] == 0.8523
    assert d["trigger_pattern"] == "when testing"
    assert d["action"] == "use pytest"


def test_behavior_match_default_similarity():
    b = Behavior(trigger_pattern="t", action="a")
    bm = BehaviorMatch(behavior=b)
    assert bm.similarity == 0.0


# --- Migration (behaviors table creation) ---


async def test_behaviors_table_exists(pool):
    """Migration 10 creates the behaviors table."""
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'behaviors'
        )
        """
    )
    assert exists is True


async def test_behaviors_table_columns(pool):
    """Behaviors table has all expected columns."""
    rows = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'behaviors'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "trigger_pattern" in columns
    assert "action" in columns
    assert "confidence" in columns
    assert "scope" in columns
    assert "project_id" in columns
    assert "agent_id" in columns
    assert "user_id" in columns
    assert "priority" in columns
    assert "enabled" in columns
    assert "access_count" in columns
    assert "token_count" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "embedding" in columns
    assert "status" in columns

    # NOT NULL constraints
    assert columns["trigger_pattern"]["is_nullable"] == "NO"
    assert columns["action"]["is_nullable"] == "NO"
    assert columns["confidence"]["is_nullable"] == "NO"
    assert columns["scope"]["is_nullable"] == "NO"
    assert columns["enabled"]["is_nullable"] == "NO"
    assert columns["status"]["is_nullable"] == "NO"


async def test_behaviors_table_indexes(pool):
    """Behaviors table has expected indexes."""
    rows = await pool.fetch(
        """
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'behaviors'
        """
    )
    index_names = {r["indexname"] for r in rows}

    assert "behaviors_pkey" in index_names
    assert "idx_behaviors_scope" in index_names
    assert "idx_behaviors_project" in index_names
    assert "idx_behaviors_agent" in index_names
    assert "idx_behaviors_enabled" in index_names
    assert "idx_behaviors_status" in index_names
    assert "idx_behaviors_embedding_hnsw" in index_names


async def test_behaviors_table_insert_and_read(pool):
    """Can insert and read a behavior row."""
    await pool.execute(
        """
        INSERT INTO behaviors (id, trigger_pattern, action, confidence, scope, priority)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        "test-b-1",
        "when writing tests",
        "use pytest with fixtures",
        0.9,
        "global",
        5,
    )

    row = await pool.fetchrow("SELECT * FROM behaviors WHERE id = $1", "test-b-1")
    assert row is not None
    assert row["trigger_pattern"] == "when writing tests"
    assert row["action"] == "use pytest with fixtures"
    assert float(row["confidence"]) == pytest.approx(0.9)
    assert row["scope"] == "global"
    assert row["priority"] == 5
    assert row["enabled"] is True
    assert row["access_count"] == 0
    assert row["status"] == "active"


async def test_behaviors_table_defaults(pool):
    """Default values are applied correctly."""
    await pool.execute(
        """
        INSERT INTO behaviors (id, trigger_pattern, action)
        VALUES ($1, $2, $3)
        """,
        "test-b-2",
        "trigger",
        "action",
    )

    row = await pool.fetchrow("SELECT * FROM behaviors WHERE id = $1", "test-b-2")
    assert float(row["confidence"]) == pytest.approx(0.7)
    assert row["scope"] == "global"
    assert row["priority"] == 0
    assert row["enabled"] is True
    assert row["access_count"] == 0
    assert row["token_count"] == 0
    assert row["status"] == "active"
    assert row["project_id"] is None
    assert row["agent_id"] is None
    # Migration 34: user_id default fills from session app.user_id.
    assert row["user_id"] == "test-user-default"


async def test_behaviors_soft_delete(pool):
    """Soft delete by setting status to archived."""
    await pool.execute(
        """
        INSERT INTO behaviors (id, trigger_pattern, action)
        VALUES ($1, $2, $3)
        """,
        "test-b-3",
        "trigger",
        "action",
    )

    await pool.execute(
        "UPDATE behaviors SET status = 'archived' WHERE id = $1",
        "test-b-3",
    )

    row = await pool.fetchrow("SELECT * FROM behaviors WHERE id = $1", "test-b-3")
    assert row["status"] == "archived"

    # Active-only query excludes it
    active = await pool.fetch(
        "SELECT * FROM behaviors WHERE status = 'active'"
    )
    assert len(active) == 0
