"""Tests for autonomy policy models and migration."""

from __future__ import annotations

import pytest

from weft.autonomy import (
    ActionPolicy,
    ActionPolicyCreate,
    AutonomyTier,
    PolicyCalibrationEvent,
    PolicyCalibrationEventCreate,
)


# --- AutonomyTier enum ---


def test_autonomy_tier_values():
    assert AutonomyTier.never.value == "never"
    assert AutonomyTier.earned.value == "earned"
    assert AutonomyTier.always.value == "always"


def test_autonomy_tier_count():
    assert len(AutonomyTier) == 3


# --- ActionPolicyCreate validation ---


def test_action_policy_create_minimal():
    apc = ActionPolicyCreate(action="send_slack_message")
    assert apc.action == "send_slack_message"
    assert apc.tier == AutonomyTier.never
    assert apc.description is None
    assert apc.conditions == {}
    assert apc.project_id is None
    assert apc.agent_id is None
    assert apc.user_id is None
    assert apc.enabled is True


def test_action_policy_create_full():
    apc = ActionPolicyCreate(
        action="deploy",
        tier=AutonomyTier.earned,
        description="Deploy to staging",
        conditions={"environment": "staging"},
        project_id="weft",
        agent_id="warp",
        user_id="user-1",
        enabled=False,
    )
    assert apc.action == "deploy"
    assert apc.tier == AutonomyTier.earned
    assert apc.description == "Deploy to staging"
    assert apc.conditions == {"environment": "staging"}
    assert apc.enabled is False


# --- ActionPolicy model ---


def test_action_policy_defaults():
    ap = ActionPolicy(action="recall_memory")
    assert ap.id.startswith("weft-")
    assert ap.action == "recall_memory"
    assert ap.tier == AutonomyTier.never
    assert ap.description is None
    assert ap.conditions == {}
    assert ap.enabled is True
    assert ap.project_id is None
    assert ap.agent_id is None
    assert ap.user_id is None


def test_action_policy_to_dict():
    ap = ActionPolicy(
        action="create_pr",
        tier=AutonomyTier.earned,
        description="Create pull requests",
        project_id="weft",
    )
    d = ap.to_dict()
    assert d["action"] == "create_pr"
    assert d["tier"] == "earned"
    assert d["description"] == "Create pull requests"
    assert d["project_id"] == "weft"
    assert "id" in d
    assert "created_at" in d
    assert "updated_at" in d


def test_action_policy_to_dict_tier_serialized_as_string():
    ap = ActionPolicy(action="status_check", tier=AutonomyTier.always)
    d = ap.to_dict()
    assert d["tier"] == "always"
    assert isinstance(d["tier"], str)


# --- PolicyCalibrationEventCreate validation ---


def test_calibration_event_create_minimal():
    ce = PolicyCalibrationEventCreate(
        policy_id="weft-abc12345",
        previous_tier=AutonomyTier.never,
        new_tier=AutonomyTier.earned,
    )
    assert ce.policy_id == "weft-abc12345"
    assert ce.previous_tier == AutonomyTier.never
    assert ce.new_tier == AutonomyTier.earned
    assert ce.reason is None
    assert ce.agent_id is None
    assert ce.user_id is None


def test_calibration_event_create_full():
    ce = PolicyCalibrationEventCreate(
        policy_id="weft-abc12345",
        previous_tier=AutonomyTier.earned,
        new_tier=AutonomyTier.always,
        reason="Approved after 10 successful runs",
        agent_id="warp",
        user_id="user-1",
    )
    assert ce.reason == "Approved after 10 successful runs"
    assert ce.agent_id == "warp"


# --- PolicyCalibrationEvent model ---


def test_calibration_event_defaults():
    ce = PolicyCalibrationEvent(
        policy_id="weft-abc12345",
        previous_tier=AutonomyTier.never,
        new_tier=AutonomyTier.earned,
    )
    assert ce.id.startswith("weft-")
    assert ce.policy_id == "weft-abc12345"
    assert ce.reason is None
    assert ce.created_at is not None


def test_calibration_event_to_dict():
    ce = PolicyCalibrationEvent(
        policy_id="weft-abc12345",
        previous_tier=AutonomyTier.never,
        new_tier=AutonomyTier.always,
        reason="Fast-tracked",
    )
    d = ce.to_dict()
    assert d["previous_tier"] == "never"
    assert d["new_tier"] == "always"
    assert d["reason"] == "Fast-tracked"
    assert isinstance(d["previous_tier"], str)
    assert isinstance(d["new_tier"], str)


# --- Migration (autonomy_policies + policy_calibration_events tables) ---


async def test_autonomy_policies_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'autonomy_policies'
        )
        """
    )
    assert exists is True


async def test_policy_calibration_events_table_exists(pool):
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'policy_calibration_events'
        )
        """
    )
    assert exists is True


async def test_autonomy_policies_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'autonomy_policies'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "action" in columns
    assert "tier" in columns
    assert "description" in columns
    assert "conditions" in columns
    assert "project_id" in columns
    assert "agent_id" in columns
    assert "user_id" in columns
    assert "enabled" in columns
    assert "created_at" in columns
    assert "updated_at" in columns

    assert columns["action"]["is_nullable"] == "NO"
    assert columns["tier"]["is_nullable"] == "NO"
    assert columns["enabled"]["is_nullable"] == "NO"
    assert columns["description"]["is_nullable"] == "YES"
    assert columns["project_id"]["is_nullable"] == "YES"
    # Migration 34 made user_id NOT NULL (kills implicit-global path).
    assert columns["user_id"]["is_nullable"] == "NO"


async def test_policy_calibration_events_table_columns(pool):
    rows = await pool.fetch(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'policy_calibration_events'
        ORDER BY ordinal_position
        """
    )
    columns = {r["column_name"]: r for r in rows}

    assert "id" in columns
    assert "policy_id" in columns
    assert "previous_tier" in columns
    assert "new_tier" in columns
    assert "reason" in columns
    assert "agent_id" in columns
    assert "user_id" in columns
    assert "created_at" in columns

    assert columns["policy_id"]["is_nullable"] == "NO"
    assert columns["previous_tier"]["is_nullable"] == "NO"
    assert columns["new_tier"]["is_nullable"] == "NO"
    assert columns["reason"]["is_nullable"] == "YES"


async def test_autonomy_policies_table_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'autonomy_policies'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "autonomy_policies_pkey" in index_names
    assert "idx_autonomy_policies_action" in index_names
    assert "idx_autonomy_policies_tier" in index_names
    assert "idx_autonomy_policies_user" in index_names
    assert "idx_autonomy_policies_project" in index_names
    assert "idx_autonomy_policies_agent" in index_names


async def test_calibration_events_indexes(pool):
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'policy_calibration_events'"
    )
    index_names = {r["indexname"] for r in rows}

    assert "policy_calibration_events_pkey" in index_names
    assert "idx_calibration_events_policy" in index_names
    assert "idx_calibration_events_user" in index_names
    assert "idx_calibration_events_created" in index_names


async def test_autonomy_policies_insert_and_read(pool):
    await pool.execute(
        """
        INSERT INTO autonomy_policies (id, action, tier, description, project_id)
        VALUES ($1, $2, $3, $4, $5)
        """,
        "test-pol-1", "send_slack_message", "never", "Send messages in Slack", "weft",
    )

    row = await pool.fetchrow(
        "SELECT * FROM autonomy_policies WHERE id = $1", "test-pol-1",
    )
    assert row is not None
    assert row["action"] == "send_slack_message"
    assert row["tier"] == "never"
    assert row["enabled"] is True


async def test_autonomy_policies_defaults(pool):
    await pool.execute(
        "INSERT INTO autonomy_policies (id, action) VALUES ($1, $2)",
        "test-pol-2", "recall_memory",
    )

    row = await pool.fetchrow(
        "SELECT * FROM autonomy_policies WHERE id = $1", "test-pol-2",
    )
    assert row["tier"] == "never"
    assert row["enabled"] is True
    # asyncpg returns JSONB as string without codec setup
    import json
    assert json.loads(row["conditions"]) == {}
    assert row["description"] is None
    assert row["project_id"] is None
    # Migration 34: user_id column DEFAULT picks up the session's
    # app.user_id; in tests the pool's setup callback puts the
    # DEFAULT_TEST_USER_ID there.
    assert row["user_id"] == "test-user-default"


async def test_autonomy_policies_conditions_jsonb(pool):
    import json

    conditions = {"environment": "staging", "max_cost": 10.0}
    await pool.execute(
        """
        INSERT INTO autonomy_policies (id, action, conditions)
        VALUES ($1, $2, $3::jsonb)
        """,
        "test-pol-3", "deploy", json.dumps(conditions),
    )

    row = await pool.fetchrow(
        "SELECT * FROM autonomy_policies WHERE id = $1", "test-pol-3",
    )
    result = row["conditions"]
    if isinstance(result, str):
        result = json.loads(result)
    assert result == conditions


async def test_calibration_event_insert_and_read(pool):
    # Create parent policy first
    await pool.execute(
        "INSERT INTO autonomy_policies (id, action) VALUES ($1, $2)",
        "test-pol-cal-1", "create_pr",
    )

    await pool.execute(
        """
        INSERT INTO policy_calibration_events
            (id, policy_id, previous_tier, new_tier, reason)
        VALUES ($1, $2, $3, $4, $5)
        """,
        "test-cal-1", "test-pol-cal-1", "never", "earned",
        "Approved after successful trial",
    )

    row = await pool.fetchrow(
        "SELECT * FROM policy_calibration_events WHERE id = $1", "test-cal-1",
    )
    assert row is not None
    assert row["policy_id"] == "test-pol-cal-1"
    assert row["previous_tier"] == "never"
    assert row["new_tier"] == "earned"
    assert row["reason"] == "Approved after successful trial"


async def test_calibration_event_cascade_on_policy_delete(pool):
    """Deleting a policy cascades to its calibration events."""
    await pool.execute(
        "INSERT INTO autonomy_policies (id, action) VALUES ($1, $2)",
        "test-pol-cas-1", "deploy",
    )
    await pool.execute(
        """
        INSERT INTO policy_calibration_events
            (id, policy_id, previous_tier, new_tier)
        VALUES ($1, $2, $3, $4)
        """,
        "test-cal-cas-1", "test-pol-cas-1", "never", "earned",
    )

    await pool.execute(
        "DELETE FROM autonomy_policies WHERE id = $1", "test-pol-cas-1",
    )

    events = await pool.fetch(
        "SELECT * FROM policy_calibration_events WHERE policy_id = $1",
        "test-pol-cas-1",
    )
    assert len(events) == 0


async def test_calibration_event_fk_constraint(pool):
    """Cannot insert a calibration event for a non-existent policy."""
    with pytest.raises(Exception):
        await pool.execute(
            """
            INSERT INTO policy_calibration_events
                (id, policy_id, previous_tier, new_tier)
            VALUES ($1, $2, $3, $4)
            """,
            "test-cal-fk-1", "nonexistent-policy", "never", "earned",
        )
