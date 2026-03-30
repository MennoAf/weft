"""Tests for calibration MCP tools, tier evaluation, and primer section."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.calibration import (
    evaluate_tier_change,
    get_calibration_summary,
    list_calibrations,
    record_calibration,
)
from weft.models import (
    CalibrationCreate,
    CalibrationOutcome,
    CalibrationRecord,
)


# --- Helpers ---


async def _make_calibration(pool, category="test_action", outcome=CalibrationOutcome.approved, **kwargs):
    defaults = {
        "action_category": category,
        "action_description": f"{category} action",
        "outcome": outcome,
    }
    defaults.update(kwargs)
    return await record_calibration(pool, CalibrationCreate(**defaults))


# --- Record creation and listing ---


class TestRecordAndList:
    async def test_record_calibration(self, pool):
        r = await _make_calibration(pool)
        assert r.id.startswith("weft-")
        assert r.action_category == "test_action"
        assert r.outcome == CalibrationOutcome.approved

    async def test_record_with_context(self, pool):
        r = await _make_calibration(
            pool,
            context={"reason": "good action"},
        )
        assert r.context == {"reason": "good action"}

    async def test_list_by_category(self, pool):
        await _make_calibration(pool, category="cat_a")
        await _make_calibration(pool, category="cat_b")
        result = await list_calibrations(pool, action_category="cat_a")
        assert len(result) == 1
        assert result[0].action_category == "cat_a"

    async def test_list_by_outcome(self, pool):
        await _make_calibration(pool, outcome=CalibrationOutcome.approved)
        await _make_calibration(pool, outcome=CalibrationOutcome.rejected)
        result = await list_calibrations(pool, outcome=CalibrationOutcome.rejected)
        assert len(result) == 1
        assert result[0].outcome == CalibrationOutcome.rejected


# --- Summary ---


class TestSummary:
    async def test_empty_summary(self, pool):
        summary = await get_calibration_summary(pool)
        assert summary["total"] == 0
        assert summary["approval_rate"] == 0.0

    async def test_summary_counts(self, pool):
        await _make_calibration(pool, outcome=CalibrationOutcome.approved)
        await _make_calibration(pool, outcome=CalibrationOutcome.approved)
        await _make_calibration(pool, outcome=CalibrationOutcome.rejected)
        summary = await get_calibration_summary(pool)
        assert summary["total"] == 3
        assert summary["approved"] == 2
        assert summary["rejected"] == 1
        assert summary["approval_rate"] == pytest.approx(2 / 3)

    async def test_summary_by_category(self, pool):
        await _make_calibration(pool, category="deploy")
        await _make_calibration(pool, category="deploy", outcome=CalibrationOutcome.rejected)
        await _make_calibration(pool, category="send_message")
        summary = await get_calibration_summary(pool)
        assert "deploy" in summary["by_category"]
        assert summary["by_category"]["deploy"]["total"] == 2
        assert "send_message" in summary["by_category"]


# --- Tier evaluation ---


class TestTierEvaluation:
    async def test_no_records(self, pool):
        result = await evaluate_tier_change(pool, "nonexistent")
        assert result["recommendation"] == "no_change"
        assert "No calibration records" in result["reason"]

    async def test_promotion_threshold(self, pool):
        """5+ records with 80%+ approval should recommend promotion."""
        for _ in range(5):
            await _make_calibration(pool, category="safe_action")
        result = await evaluate_tier_change(pool, "safe_action")
        assert result["recommendation"] == "promote"
        assert result["stats"]["total"] == 5

    async def test_demotion_threshold(self, pool):
        """3+ records with 50%+ rejection should recommend demotion."""
        await _make_calibration(pool, category="risky", outcome=CalibrationOutcome.rejected)
        await _make_calibration(pool, category="risky", outcome=CalibrationOutcome.rejected)
        await _make_calibration(pool, category="risky", outcome=CalibrationOutcome.approved)
        result = await evaluate_tier_change(pool, "risky")
        assert result["recommendation"] == "demote"

    async def test_insufficient_evidence(self, pool):
        """Below thresholds should return no_change."""
        await _make_calibration(pool, category="new_action")
        await _make_calibration(pool, category="new_action")
        result = await evaluate_tier_change(pool, "new_action")
        assert result["recommendation"] == "no_change"
        assert "Insufficient evidence" in result["reason"]

    async def test_demotion_takes_priority(self, pool):
        """Demotion check happens before promotion even with many records."""
        # 3 rejected, 3 approved = 50% rejection rate = demote
        for _ in range(3):
            await _make_calibration(pool, category="mixed", outcome=CalibrationOutcome.rejected)
        for _ in range(3):
            await _make_calibration(pool, category="mixed", outcome=CalibrationOutcome.approved)
        result = await evaluate_tier_change(pool, "mixed")
        assert result["recommendation"] == "demote"

    async def test_mixed_below_both_thresholds(self, pool):
        """4 approved, 1 rejected = 80% approval but only 5 records;
        rejection rate 20% below 50% threshold. Should promote."""
        for _ in range(4):
            await _make_calibration(pool, category="mostly_good", outcome=CalibrationOutcome.approved)
        await _make_calibration(pool, category="mostly_good", outcome=CalibrationOutcome.rejected)
        result = await evaluate_tier_change(pool, "mostly_good")
        assert result["recommendation"] == "promote"

    async def test_evaluation_stats_included(self, pool):
        for _ in range(5):
            await _make_calibration(pool, category="action_x")
        result = await evaluate_tier_change(pool, "action_x")
        assert "stats" in result
        assert result["stats"]["total"] == 5
        assert result["stats"]["window_days"] == 30


# --- Primer section ---


class TestCalibrationPrimerSection:
    async def test_skipped_when_no_records(self, pool):
        from weft.primer_sections.calibration import build_calibration_section
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
        result = await build_calibration_section(ctx)
        assert result.skipped is True

    async def test_includes_overview_when_records_exist(self, pool):
        from weft.primer_sections.calibration import build_calibration_section
        from weft.primer_sections.context import PrimerContext

        for _ in range(3):
            await _make_calibration(pool, category="deploy")

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
        result = await build_calibration_section(ctx)
        assert result.skipped is False
        assert len(result.items) >= 1
        assert result.items[0]["total"] == 3

    async def test_includes_recommendations(self, pool):
        from weft.primer_sections.calibration import build_calibration_section
        from weft.primer_sections.context import PrimerContext

        # Create enough records to trigger promotion
        for _ in range(6):
            await _make_calibration(pool, category="safe_deploy")

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
        result = await build_calibration_section(ctx)
        assert result.skipped is False
        # Should have overview + at least one recommendation
        recs = [i for i in result.items if "recommendation" in i]
        assert len(recs) >= 1
        assert recs[0]["recommendation"] == "promote"
