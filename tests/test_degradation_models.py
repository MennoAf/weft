"""Tests for degradation policy schema, models, and store layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import (
    DegradationAction,
    DegradationPolicy,
    DegradationPolicyCreate,
    DegradationPolicyStatus,
    DegradationTriggerType,
)
from weft.degradation import (
    create_policy,
    delete_policy,
    get_active_policies,
    get_policy,
    is_cooldown_elapsed,
    list_policies,
    record_fire,
    update_policy,
)


# --- Helpers ---


async def _make_policy(pool, name="test policy", **kwargs):
    defaults = {
        "trigger_type": DegradationTriggerType.low_confidence,
        "condition": {"threshold": 0.3},
        "action": DegradationAction.escalate,
    }
    defaults.update(kwargs)
    return await create_policy(pool, DegradationPolicyCreate(name=name, **defaults))


# --- Model validation ---


class TestModelValidation:
    def test_low_confidence_requires_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.low_confidence,
                condition={},
                action=DegradationAction.pause,
            )

    def test_low_confidence_threshold_must_be_number(self):
        with pytest.raises(ValueError, match="threshold must be a number"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.low_confidence,
                condition={"threshold": "high"},
                action=DegradationAction.pause,
            )

    def test_low_confidence_threshold_range(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.low_confidence,
                condition={"threshold": 1.5},
                action=DegradationAction.pause,
            )

    def test_api_error_requires_max_errors(self):
        with pytest.raises(ValueError, match="max_errors"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.api_error,
                condition={"window_minutes": 10},
                action=DegradationAction.restart,
            )

    def test_api_error_requires_window_minutes(self):
        with pytest.raises(ValueError, match="window_minutes"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.api_error,
                condition={"max_errors": 5},
                action=DegradationAction.restart,
            )

    def test_api_error_max_errors_positive(self):
        with pytest.raises(ValueError, match="positive integer"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.api_error,
                condition={"max_errors": 0, "window_minutes": 10},
                action=DegradationAction.restart,
            )

    def test_context_decay_requires_max_age_hours(self):
        with pytest.raises(ValueError, match="max_age_hours"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.context_decay,
                condition={},
                action=DegradationAction.restrict,
            )

    def test_context_decay_max_age_positive(self):
        with pytest.raises(ValueError, match="positive number"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.context_decay,
                condition={"max_age_hours": -1},
                action=DegradationAction.restrict,
            )

    def test_budget_breach_requires_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.budget_breach,
                condition={},
                action=DegradationAction.pause,
            )

    def test_budget_breach_max_tokens_positive(self):
        with pytest.raises(ValueError, match="positive integer"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.budget_breach,
                condition={"max_tokens": -100},
                action=DegradationAction.pause,
            )

    def test_valid_low_confidence(self):
        p = DegradationPolicyCreate(
            name="ok",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
        )
        assert p.trigger_type == DegradationTriggerType.low_confidence
        assert p.condition["threshold"] == 0.3

    def test_valid_api_error(self):
        p = DegradationPolicyCreate(
            name="ok",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 5, "window_minutes": 10},
            action=DegradationAction.restart,
        )
        assert p.action == DegradationAction.restart

    def test_valid_context_decay(self):
        p = DegradationPolicyCreate(
            name="ok",
            trigger_type=DegradationTriggerType.context_decay,
            condition={"max_age_hours": 24},
            action=DegradationAction.restrict,
        )
        assert p.trigger_type == DegradationTriggerType.context_decay

    def test_valid_budget_breach(self):
        p = DegradationPolicyCreate(
            name="ok",
            trigger_type=DegradationTriggerType.budget_breach,
            condition={"max_tokens": 100000},
            action=DegradationAction.pause,
        )
        assert p.condition["max_tokens"] == 100000

    def test_to_dict_serialization(self):
        p = DegradationPolicy(
            name="test",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 3, "window_minutes": 5},
            action=DegradationAction.escalate,
        )
        d = p.to_dict()
        assert d["trigger_type"] == "api_error"
        assert d["action"] == "escalate"
        assert d["status"] == "active"


# --- CRUD store tests ---


class TestCreateAndGet:
    async def test_create_minimal(self, pool):
        p = await _make_policy(pool)
        assert p.id.startswith("weft-")
        assert p.name == "test policy"
        assert p.trigger_type == DegradationTriggerType.low_confidence
        assert p.action == DegradationAction.escalate
        assert p.status == DegradationPolicyStatus.active
        assert p.fire_count == 0
        assert p.last_fired_at is None

    async def test_create_with_all_fields(self, pool):
        p = await _make_policy(
            pool,
            name="full policy",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 5, "window_minutes": 10},
            action=DegradationAction.restart,
            description="Restart on repeated API failures",
            cooldown_minutes=30.0,
            max_fires=3,
            project_id="proj-1",
            agent_id="agent-1",
        )
        assert p.condition == {"max_errors": 5, "window_minutes": 10}
        assert p.description == "Restart on repeated API failures"
        assert p.cooldown_minutes == 30.0
        assert p.max_fires == 3
        assert p.project_id == "proj-1"

    async def test_get_by_id(self, pool):
        p = await _make_policy(pool)
        fetched = await get_policy(pool, p.id)
        assert fetched is not None
        assert fetched.id == p.id
        assert fetched.name == p.name

    async def test_get_not_found(self, pool):
        result = await get_policy(pool, "weft-nonexistent")
        assert result is None


class TestList:
    async def test_list_all(self, pool):
        await _make_policy(pool, name="p1")
        await _make_policy(pool, name="p2")
        result = await list_policies(pool)
        assert len(result) == 2

    async def test_list_by_trigger_type(self, pool):
        await _make_policy(pool, name="low-conf")
        await _make_policy(
            pool,
            name="api-err",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 3, "window_minutes": 5},
        )
        result = await list_policies(
            pool, trigger_type=DegradationTriggerType.low_confidence
        )
        assert len(result) == 1
        assert result[0].name == "low-conf"

    async def test_list_by_action(self, pool):
        await _make_policy(pool, name="esc", action=DegradationAction.escalate)
        await _make_policy(pool, name="pause", action=DegradationAction.pause)
        result = await list_policies(pool, action=DegradationAction.pause)
        assert len(result) == 1
        assert result[0].name == "pause"

    async def test_list_by_status(self, pool):
        p = await _make_policy(pool, name="active")
        await _make_policy(pool, name="will-disable")
        await update_policy(
            pool, (await list_policies(pool))[0].id,
            status=DegradationPolicyStatus.disabled,
        )
        result = await list_policies(
            pool, status=DegradationPolicyStatus.active
        )
        assert len(result) == 1

    async def test_list_with_limit(self, pool):
        for i in range(5):
            await _make_policy(pool, name=f"p{i}")
        result = await list_policies(pool, limit=2)
        assert len(result) == 2


class TestUpdate:
    async def test_update_name(self, pool):
        p = await _make_policy(pool, name="old")
        updated = await update_policy(pool, p.id, name="new")
        assert updated is not None
        assert updated.name == "new"

    async def test_update_status(self, pool):
        p = await _make_policy(pool)
        updated = await update_policy(
            pool, p.id, status=DegradationPolicyStatus.disabled
        )
        assert updated is not None
        assert updated.status == DegradationPolicyStatus.disabled

    async def test_update_cooldown(self, pool):
        p = await _make_policy(pool)
        updated = await update_policy(pool, p.id, cooldown_minutes=15.0)
        assert updated is not None
        assert updated.cooldown_minutes == 15.0

    async def test_update_not_found(self, pool):
        result = await update_policy(pool, "weft-nonexistent", name="x")
        assert result is None

    async def test_update_no_changes(self, pool):
        p = await _make_policy(pool)
        result = await update_policy(pool, p.id)
        assert result is not None
        assert result.id == p.id


class TestDelete:
    async def test_delete_existing(self, pool):
        p = await _make_policy(pool)
        assert await delete_policy(pool, p.id) is True
        assert await get_policy(pool, p.id) is None

    async def test_delete_nonexistent(self, pool):
        assert await delete_policy(pool, "weft-nonexistent") is False


# --- Firing ---


class TestFiring:
    async def test_record_fire(self, pool):
        p = await _make_policy(pool)
        fired = await record_fire(pool, p.id)
        assert fired is not None
        assert fired.fire_count == 1
        assert fired.last_fired_at is not None

    async def test_record_fire_increments(self, pool):
        p = await _make_policy(pool)
        await record_fire(pool, p.id)
        fired = await record_fire(pool, p.id)
        assert fired is not None
        assert fired.fire_count == 2

    async def test_record_fire_max_fires_transitions_status(self, pool):
        p = await _make_policy(pool, max_fires=2)
        await record_fire(pool, p.id)
        fired = await record_fire(pool, p.id)
        assert fired is not None
        assert fired.status == DegradationPolicyStatus.fired

    async def test_record_fire_not_found(self, pool):
        result = await record_fire(pool, "weft-nonexistent")
        assert result is None

    async def test_cooldown_no_limit(self, pool):
        p = await _make_policy(pool)
        assert is_cooldown_elapsed(p) is True

    async def test_cooldown_never_fired(self, pool):
        p = await _make_policy(pool, cooldown_minutes=60.0)
        assert is_cooldown_elapsed(p) is True

    async def test_cooldown_not_elapsed(self, pool):
        p = await _make_policy(pool, cooldown_minutes=60.0)
        fired = await record_fire(pool, p.id)
        assert fired is not None
        assert is_cooldown_elapsed(fired) is False

    async def test_cooldown_elapsed(self, pool):
        p = DegradationPolicy(
            name="test",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
            cooldown_minutes=60.0,
            last_fired_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        assert is_cooldown_elapsed(p) is True


# --- Active policy queries ---


class TestActivePolicies:
    async def test_get_active_basic(self, pool):
        await _make_policy(pool, name="active")
        result = await get_active_policies(pool)
        assert len(result) == 1
        assert result[0].name == "active"

    async def test_excludes_disabled(self, pool):
        p = await _make_policy(pool)
        await update_policy(pool, p.id, status=DegradationPolicyStatus.disabled)
        result = await get_active_policies(pool)
        assert len(result) == 0

    async def test_excludes_max_fires_reached(self, pool):
        p = await _make_policy(pool, max_fires=1)
        await record_fire(pool, p.id)
        result = await get_active_policies(pool)
        assert len(result) == 0

    async def test_filter_by_trigger_type(self, pool):
        await _make_policy(pool, name="low-conf")
        await _make_policy(
            pool,
            name="api-err",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 3, "window_minutes": 5},
        )
        result = await get_active_policies(
            pool, trigger_type=DegradationTriggerType.low_confidence
        )
        assert len(result) == 1
        assert result[0].name == "low-conf"

    async def test_filter_by_project(self, pool):
        await _make_policy(pool, name="global")
        await _make_policy(pool, name="proj", project_id="proj-1")
        result = await get_active_policies(pool, project_id="proj-1")
        assert len(result) == 2  # global (NULL) + project-specific
