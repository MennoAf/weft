"""Integration tests for primer sections: triggers, calibration, cost, degradation.

Tests that each section renders correct content from its respective store,
sections are independently togglable (skipped when no data), and the primer
integrates them properly under both progressive and full disclosure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.primer_sections.context import PrimerContext, SectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(**overrides) -> PrimerContext:
    """Build a PrimerContext with sensible defaults for testing."""
    defaults = dict(
        user_id="test-user",
        project_id="proj-1",
        agent_id=None,
        pool=MagicMock(),
        budget_tokens=2400,
        query=None,
        query_vec=None,
        disclosure="full",
        mode=None,
        now=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return PrimerContext(**defaults)


# ---------------------------------------------------------------------------
# Triggers section
# ---------------------------------------------------------------------------


class TestTriggersSection:
    """Tests for build_triggers_section."""

    @pytest.mark.asyncio
    async def test_skipped_when_no_triggers(self):
        from weft.primer_sections.triggers import build_triggers_section

        ctx = _make_ctx()
        with patch("weft.primer_sections.triggers.list_triggers", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            result = await build_triggers_section(ctx)

        assert result.skipped is True
        assert result.items == []
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_renders_enabled_triggers(self):
        from weft.models import TriggerConditionType, TriggerStatus
        from weft.primer_sections.triggers import build_triggers_section

        trigger = MagicMock()
        trigger.id = "trg-1"
        trigger.name = "Daily check"
        trigger.condition_type = TriggerConditionType.time
        trigger.action = "Send summary"
        trigger.fire_count = 3
        trigger.cooldown_hours = 24.0
        trigger.max_fires = 10
        trigger.status = TriggerStatus.enabled

        ctx = _make_ctx()
        with patch("weft.primer_sections.triggers.list_triggers", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [trigger]
            result = await build_triggers_section(ctx)

        assert not result.skipped
        assert len(result.items) == 1
        item = result.items[0]
        assert item["id"] == "trg-1"
        assert item["name"] == "Daily check"
        assert item["condition_type"] == "time"
        assert item["action"] == "Send summary"
        assert item["fire_count"] == 3
        assert item["cooldown_hours"] == 24.0
        assert item["max_fires"] == 10

    @pytest.mark.asyncio
    async def test_omits_optional_fields_when_none(self):
        from weft.models import TriggerConditionType, TriggerStatus
        from weft.primer_sections.triggers import build_triggers_section

        trigger = MagicMock()
        trigger.id = "trg-2"
        trigger.name = "Event trigger"
        trigger.condition_type = TriggerConditionType.event
        trigger.action = "Notify"
        trigger.fire_count = 0
        trigger.cooldown_hours = None
        trigger.max_fires = None
        trigger.status = TriggerStatus.enabled

        ctx = _make_ctx()
        with patch("weft.primer_sections.triggers.list_triggers", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [trigger]
            result = await build_triggers_section(ctx)

        item = result.items[0]
        assert "cooldown_hours" not in item
        assert "max_fires" not in item

    @pytest.mark.asyncio
    async def test_respects_budget_cap(self):
        from weft.models import TriggerConditionType, TriggerStatus
        from weft.primer_sections.triggers import build_triggers_section

        def _trigger(i):
            t = MagicMock()
            t.id = f"trg-{i}"
            t.name = f"Trigger {i} with a fairly long name to consume tokens"
            t.condition_type = TriggerConditionType.threshold
            t.action = f"Action {i} with enough text to use budget"
            t.fire_count = 0
            t.cooldown_hours = None
            t.max_fires = None
            t.status = TriggerStatus.enabled
            return t

        ctx = _make_ctx(budget_tokens=50)  # very tight budget
        with patch("weft.primer_sections.triggers.list_triggers", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [_trigger(i) for i in range(20)]
            result = await build_triggers_section(ctx)

        # Guarantee at least 1 item, but budget should constrain the rest
        assert len(result.items) >= 1
        assert len(result.items) < 20

    @pytest.mark.asyncio
    async def test_updates_ctx_state(self):
        from weft.models import TriggerConditionType, TriggerStatus
        from weft.primer_sections.triggers import build_triggers_section

        trigger = MagicMock()
        trigger.id = "trg-1"
        trigger.name = "Test"
        trigger.condition_type = TriggerConditionType.time
        trigger.action = "Act"
        trigger.fire_count = 0
        trigger.cooldown_hours = None
        trigger.max_fires = None
        trigger.status = TriggerStatus.enabled

        ctx = _make_ctx()
        initial_tokens = ctx.used_tokens
        with patch("weft.primer_sections.triggers.list_triggers", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [trigger]
            result = await build_triggers_section(ctx)

        assert ctx.used_tokens > initial_tokens
        assert "triggers" in ctx.section_tokens
        assert ctx.section_tokens["triggers"] == result.tokens_used


# ---------------------------------------------------------------------------
# Cost section
# ---------------------------------------------------------------------------


class TestCostSection:
    """Tests for build_cost_section."""

    @pytest.mark.asyncio
    async def test_skipped_when_no_entries(self):
        from weft.cost_tracking import CostSummary
        from weft.primer_sections.cost import build_cost_section

        empty_summary = CostSummary(
            total_entries=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
        )

        ctx = _make_ctx()
        with patch("weft.primer_sections.cost.get_cost_summary", new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = empty_summary
            result = await build_cost_section(ctx)

        assert result.skipped is True
        assert result.items == []

    @pytest.mark.asyncio
    async def test_renders_cost_summary(self):
        from weft.cost_tracking import CostSummary
        from weft.primer_sections.cost import build_cost_section

        summary = CostSummary(
            total_entries=5,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_tokens=15000,
            total_cost_usd=0.1234,
        )

        ctx = _make_ctx()
        with patch("weft.primer_sections.cost.get_cost_summary", new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = summary
            result = await build_cost_section(ctx)

        assert not result.skipped
        assert len(result.items) == 1
        item = result.items[0]
        assert item["window_hours"] == 24
        assert item["total_entries"] == 5
        assert item["total_tokens"] == 15000
        assert item["total_cost_usd"] == 0.1234

    @pytest.mark.asyncio
    async def test_updates_ctx_state(self):
        from weft.cost_tracking import CostSummary
        from weft.primer_sections.cost import build_cost_section

        summary = CostSummary(
            total_entries=1,
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            total_cost_usd=0.01,
        )

        ctx = _make_ctx()
        initial_tokens = ctx.used_tokens
        with patch("weft.primer_sections.cost.get_cost_summary", new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = summary
            result = await build_cost_section(ctx)

        assert ctx.used_tokens > initial_tokens
        assert "cost" in ctx.section_tokens
        assert ctx.section_tokens["cost"] == result.tokens_used


# ---------------------------------------------------------------------------
# Calibration section
# ---------------------------------------------------------------------------


class TestCalibrationSection:
    """Tests for build_calibration_section."""

    @pytest.mark.asyncio
    async def test_skipped_when_no_records(self):
        from weft.primer_sections.calibration import build_calibration_section

        summary = {
            "total": 0,
            "approved": 0,
            "rejected": 0,
            "modified": 0,
            "approval_rate": 0.0,
            "by_category": {},
        }

        ctx = _make_ctx()
        with patch("weft.primer_sections.calibration.get_calibration_summary", new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = summary
            result = await build_calibration_section(ctx)

        assert result.skipped is True
        assert result.items == []

    @pytest.mark.asyncio
    async def test_renders_calibration_overview(self):
        from weft.primer_sections.calibration import build_calibration_section

        summary = {
            "total": 10,
            "approved": 8,
            "rejected": 1,
            "modified": 1,
            "approval_rate": 0.8,
            "by_category": {},
        }

        ctx = _make_ctx()
        with patch("weft.primer_sections.calibration.get_calibration_summary", new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = summary
            result = await build_calibration_section(ctx)

        assert not result.skipped
        assert len(result.items) >= 1
        item = result.items[0]
        assert item["total"] == 10
        assert item["approved"] == 8
        assert item["approval_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_includes_tier_change_recommendations(self):
        from weft.primer_sections.calibration import build_calibration_section

        summary = {
            "total": 10,
            "approved": 8,
            "rejected": 1,
            "modified": 1,
            "approval_rate": 0.8,
            "by_category": {
                "send_slack": {"total": 6, "approved": 5, "rejected": 1, "modified": 0},
            },
        }

        evaluation = {
            "recommendation": "promote",
            "reason": "High approval rate",
        }

        ctx = _make_ctx()
        with (
            patch("weft.primer_sections.calibration.get_calibration_summary", new_callable=AsyncMock) as mock_summary,
            patch("weft.primer_sections.calibration.evaluate_tier_change", new_callable=AsyncMock) as mock_eval,
        ):
            mock_summary.return_value = summary
            mock_eval.return_value = evaluation
            result = await build_calibration_section(ctx)

        assert len(result.items) >= 2
        rec_item = result.items[1]
        assert rec_item["action_category"] == "send_slack"
        assert rec_item["recommendation"] == "promote"


# ---------------------------------------------------------------------------
# Degradation section
# ---------------------------------------------------------------------------


class TestDegradationSection:
    """Tests for build_degradation_section."""

    @pytest.mark.asyncio
    async def test_skipped_when_no_policies(self):
        from weft.primer_sections.degradation import build_degradation_section

        ctx = _make_ctx()
        with patch("weft.primer_sections.degradation.list_policies", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            result = await build_degradation_section(ctx)

        assert result.skipped is True
        assert result.items == []

    @pytest.mark.asyncio
    async def test_renders_active_policies(self):
        from weft.models import (
            DegradationAction,
            DegradationPolicyStatus,
            DegradationTriggerType,
        )
        from weft.primer_sections.degradation import build_degradation_section

        policy = MagicMock()
        policy.id = "deg-1"
        policy.name = "Budget guard"
        policy.trigger_type = DegradationTriggerType.budget_breach
        policy.action = DegradationAction.pause
        policy.status = DegradationPolicyStatus.active
        policy.fire_count = 2
        policy.description = "Pause when over budget"

        ctx = _make_ctx()
        with patch("weft.primer_sections.degradation.list_policies", new_callable=AsyncMock) as mock_list:
            # First call: active policies, second call: fired policies
            mock_list.side_effect = [[policy], []]
            result = await build_degradation_section(ctx)

        assert not result.skipped
        assert len(result.items) == 1
        item = result.items[0]
        assert item["id"] == "deg-1"
        assert item["name"] == "Budget guard"
        assert item["trigger_type"] == "budget_breach"
        assert item["action"] == "pause"
        assert item["fire_count"] == 2
        assert item["description"] == "Pause when over budget"

    @pytest.mark.asyncio
    async def test_includes_fired_policies(self):
        from weft.models import (
            DegradationAction,
            DegradationPolicyStatus,
            DegradationTriggerType,
        )
        from weft.primer_sections.degradation import build_degradation_section

        active = MagicMock()
        active.id = "deg-a"
        active.name = "Active policy"
        active.trigger_type = DegradationTriggerType.low_confidence
        active.action = DegradationAction.escalate
        active.status = DegradationPolicyStatus.active
        active.fire_count = 0
        active.description = None

        fired = MagicMock()
        fired.id = "deg-f"
        fired.name = "Fired policy"
        fired.trigger_type = DegradationTriggerType.api_error
        fired.action = DegradationAction.restart
        fired.status = DegradationPolicyStatus.fired
        fired.fire_count = 1
        fired.description = None

        ctx = _make_ctx()
        with patch("weft.primer_sections.degradation.list_policies", new_callable=AsyncMock) as mock_list:
            # First call returns active, second returns fired
            mock_list.side_effect = [[active], [fired]]
            result = await build_degradation_section(ctx)

        assert len(result.items) == 2
        ids = {item["id"] for item in result.items}
        assert "deg-a" in ids
        assert "deg-f" in ids


# ---------------------------------------------------------------------------
# Section contract tests (all four sections)
# ---------------------------------------------------------------------------


class TestSectionContracts:
    """Verify all four sections return SectionResult and are async."""

    SECTIONS = [
        "weft.primer_sections.triggers:build_triggers_section",
        "weft.primer_sections.cost:build_cost_section",
        "weft.primer_sections.calibration:build_calibration_section",
        "weft.primer_sections.degradation:build_degradation_section",
    ]

    @pytest.mark.parametrize("path", SECTIONS)
    def test_is_async(self, path):
        import importlib
        import inspect

        module_path, func_name = path.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        assert inspect.iscoroutinefunction(func)

    @pytest.mark.parametrize("path", SECTIONS)
    def test_first_param_is_ctx(self, path):
        import importlib
        import inspect

        module_path, func_name = path.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        assert params[0].name == "ctx"


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


class TestNewSectionReexports:
    """Verify the new sections are exported from the package."""

    def test_triggers_importable(self):
        from weft.primer_sections import build_triggers_section  # noqa: F401

    def test_cost_importable(self):
        from weft.primer_sections import build_cost_section  # noqa: F401

    def test_calibration_importable(self):
        from weft.primer_sections import build_calibration_section  # noqa: F401

    def test_degradation_importable(self):
        from weft.primer_sections import build_degradation_section  # noqa: F401

    def test_autonomy_importable(self):
        from weft.primer_sections import build_autonomy_section  # noqa: F401
