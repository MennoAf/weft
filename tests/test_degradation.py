"""Tests for degradation MCP tools, state evaluation, and primer section."""

from __future__ import annotations

import pytest

from weft.degradation import (
    _evaluate_condition,
    create_policy,
    get_active_policies,
    record_fire,
    update_degradation_state,
    update_policy,
)
from weft.models import (
    DegradationAction,
    DegradationPolicy,
    DegradationPolicyCreate,
    DegradationPolicyStatus,
    DegradationTriggerType,
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


# --- Condition evaluation (unit) ---


class TestEvaluateCondition:
    def test_low_confidence_triggers(self):
        policy = DegradationPolicy(
            name="lc",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
        )
        result = _evaluate_condition(policy, {"confidence": 0.2})
        assert result is not None
        assert "0.20" in result

    def test_low_confidence_no_trigger(self):
        policy = DegradationPolicy(
            name="lc",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
        )
        assert _evaluate_condition(policy, {"confidence": 0.5}) is None

    def test_low_confidence_missing_metric(self):
        policy = DegradationPolicy(
            name="lc",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
        )
        assert _evaluate_condition(policy, {"error_count": 10}) is None

    def test_api_error_triggers(self):
        policy = DegradationPolicy(
            name="ae",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 5, "window_minutes": 10},
            action=DegradationAction.restart,
        )
        result = _evaluate_condition(policy, {"error_count": 7})
        assert result is not None
        assert "7" in result

    def test_api_error_no_trigger(self):
        policy = DegradationPolicy(
            name="ae",
            trigger_type=DegradationTriggerType.api_error,
            condition={"max_errors": 5, "window_minutes": 10},
            action=DegradationAction.restart,
        )
        assert _evaluate_condition(policy, {"error_count": 3}) is None

    def test_context_decay_triggers(self):
        policy = DegradationPolicy(
            name="cd",
            trigger_type=DegradationTriggerType.context_decay,
            condition={"max_age_hours": 24},
            action=DegradationAction.restrict,
        )
        result = _evaluate_condition(policy, {"context_age_hours": 30.0})
        assert result is not None

    def test_budget_breach_triggers(self):
        policy = DegradationPolicy(
            name="bb",
            trigger_type=DegradationTriggerType.budget_breach,
            condition={"max_tokens": 100000},
            action=DegradationAction.pause,
        )
        result = _evaluate_condition(policy, {"tokens_used": 150000})
        assert result is not None

    def test_budget_breach_no_trigger(self):
        policy = DegradationPolicy(
            name="bb",
            trigger_type=DegradationTriggerType.budget_breach,
            condition={"max_tokens": 100000},
            action=DegradationAction.pause,
        )
        assert _evaluate_condition(policy, {"tokens_used": 50000}) is None


# --- update_degradation_state (integration) ---


class TestUpdateDegradationState:
    async def test_no_policies_returns_empty(self, pool):
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert result == []

    async def test_triggers_matching_policy(self, pool):
        await _make_policy(
            pool,
            name="low-conf-escalate",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.3},
            action=DegradationAction.escalate,
        )
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert len(result) == 1
        assert result[0]["action"] == "escalate"
        assert result[0]["name"] == "low-conf-escalate"
        assert result[0]["fire_count"] == 1

    async def test_does_not_trigger_unmatched(self, pool):
        await _make_policy(
            pool,
            name="high-threshold",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.1},
            action=DegradationAction.pause,
        )
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.5}
        )
        assert result == []

    async def test_fires_multiple_policies(self, pool):
        await _make_policy(
            pool,
            name="conf-policy",
            trigger_type=DegradationTriggerType.low_confidence,
            condition={"threshold": 0.5},
            action=DegradationAction.escalate,
        )
        await _make_policy(
            pool,
            name="budget-policy",
            trigger_type=DegradationTriggerType.budget_breach,
            condition={"max_tokens": 1000},
            action=DegradationAction.pause,
        )
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.2, "tokens_used": 5000}
        )
        assert len(result) == 2
        actions = {r["action"] for r in result}
        assert actions == {"escalate", "pause"}

    async def test_respects_max_fires(self, pool):
        p = await _make_policy(
            pool,
            name="one-shot",
            max_fires=1,
        )
        # First check triggers
        r1 = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert len(r1) == 1
        assert r1[0]["status"] == "fired"

        # Second check should not trigger (max_fires reached)
        r2 = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert len(r2) == 0

    async def test_skips_disabled_policies(self, pool):
        p = await _make_policy(pool, name="disabled-policy")
        await update_policy(pool, p.id, status=DegradationPolicyStatus.disabled)
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert result == []

    async def test_triggered_result_shape(self, pool):
        await _make_policy(pool, name="shape-test")
        result = await update_degradation_state(
            pool, metrics={"confidence": 0.1}
        )
        assert len(result) == 1
        entry = result[0]
        assert "policy_id" in entry
        assert "name" in entry
        assert "trigger_type" in entry
        assert "action" in entry
        assert "reason" in entry
        assert "fire_count" in entry
        assert "status" in entry


# --- Primer section ---


class TestDegradationPrimerSection:
    async def test_skipped_when_no_policies(self, pool):
        from weft.primer_sections.degradation import build_degradation_section
        from weft.primer_sections.context import PrimerContext

        ctx = PrimerContext(
            user_id="test-user",
            project_id=None,
            agent_id=None,
            pool=pool,
            budget_tokens=2400,
            query=None,
            query_vec=None,
            disclosure="progressive",
            mode=None,
        )
        result = await build_degradation_section(ctx)
        assert result.skipped is True

    async def test_shows_active_policies(self, pool):
        from weft.primer_sections.degradation import build_degradation_section
        from weft.primer_sections.context import PrimerContext

        await _make_policy(pool, name="active-guardrail")

        ctx = PrimerContext(
            user_id="test-user",
            project_id=None,
            agent_id=None,
            pool=pool,
            budget_tokens=2400,
            query=None,
            query_vec=None,
            disclosure="full",
            mode=None,
        )
        result = await build_degradation_section(ctx)
        assert result.skipped is False
        assert len(result.items) >= 1
        assert result.items[0]["name"] == "active-guardrail"
        assert result.items[0]["status"] == "active"

    async def test_includes_fired_policies(self, pool):
        from weft.primer_sections.degradation import build_degradation_section
        from weft.primer_sections.context import PrimerContext

        p = await _make_policy(pool, name="fired-policy", max_fires=1)
        await record_fire(pool, p.id)

        ctx = PrimerContext(
            user_id="test-user",
            project_id=None,
            agent_id=None,
            pool=pool,
            budget_tokens=2400,
            query=None,
            query_vec=None,
            disclosure="full",
            mode=None,
        )
        result = await build_degradation_section(ctx)
        assert result.skipped is False
        fired_items = [i for i in result.items if i["status"] == "fired"]
        assert len(fired_items) == 1
        assert fired_items[0]["fire_count"] == 1
