"""Tests for weft/health_check.py — health evaluator aggregator.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Unit: run_all_evaluators with all evaluators mocked — happy path
# ---------------------------------------------------------------------------


class TestRunAllEvaluatorsHappyPath:
    @pytest.mark.asyncio
    async def test_returns_health_summary_with_findings(self):
        from weft.health_check import HealthFinding, HealthSummary, run_all_evaluators

        checkin_findings = [
            {"source": "check_in_alerts", "severity": "warning",
             "message": "Low mood streak: 3 days", "metadata": {}},
        ]
        loom_findings = [
            {"source": "loom_alerts", "severity": "info",
             "message": "1 epic ready to close", "metadata": {}},
        ]
        hygiene_findings = [
            {"source": "memory_hygiene", "severity": "warning",
             "message": "Consolidation overdue", "metadata": {}},
        ]

        pool = AsyncMock()

        with patch("weft.health_check._evaluate_checkin", new_callable=AsyncMock, return_value=checkin_findings), \
             patch("weft.health_check._evaluate_loom", new_callable=AsyncMock, return_value=loom_findings), \
             patch("weft.health_check._evaluate_hygiene", new_callable=AsyncMock, return_value=hygiene_findings):
            result = await run_all_evaluators(pool)

        assert isinstance(result, HealthSummary)
        assert result.total_findings == 3
        assert len(result.findings) == 3
        assert len(result.errors) == 0
        assert result.evaluated_at.tzinfo is not None  # timezone-aware

    @pytest.mark.asyncio
    async def test_empty_evaluators_return_zero_findings(self):
        from weft.health_check import run_all_evaluators

        pool = AsyncMock()

        with patch("weft.health_check._evaluate_checkin", new_callable=AsyncMock, return_value=[]), \
             patch("weft.health_check._evaluate_loom", new_callable=AsyncMock, return_value=[]), \
             patch("weft.health_check._evaluate_hygiene", new_callable=AsyncMock, return_value=[]):
            result = await run_all_evaluators(pool)

        assert result.total_findings == 0
        assert result.findings == []
        assert result.errors == []


# ---------------------------------------------------------------------------
# 2. Unit: one evaluator raises, others still run
# ---------------------------------------------------------------------------


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_one_evaluator_error_others_still_run(self):
        from weft.health_check import run_all_evaluators

        loom_findings = [
            {"source": "loom_alerts", "severity": "info",
             "message": "1 epic ready to close", "metadata": {}},
        ]

        pool = AsyncMock()

        with patch("weft.health_check._evaluate_checkin", new_callable=AsyncMock,
                    side_effect=RuntimeError("DB connection lost")), \
             patch("weft.health_check._evaluate_loom", new_callable=AsyncMock, return_value=loom_findings), \
             patch("weft.health_check._evaluate_hygiene", new_callable=AsyncMock, return_value=[]):
            result = await run_all_evaluators(pool)

        assert result.total_findings == 1
        assert len(result.errors) == 1
        assert result.errors[0]["source"] == "check_in_alerts"
        assert result.errors[0]["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_all_evaluators_raise(self):
        from weft.health_check import run_all_evaluators

        pool = AsyncMock()

        with patch("weft.health_check._evaluate_checkin", new_callable=AsyncMock,
                    side_effect=RuntimeError("fail 1")), \
             patch("weft.health_check._evaluate_loom", new_callable=AsyncMock,
                    side_effect=ConnectionError("fail 2")), \
             patch("weft.health_check._evaluate_hygiene", new_callable=AsyncMock,
                    side_effect=ValueError("fail 3")):
            result = await run_all_evaluators(pool)

        assert result.total_findings == 0
        assert len(result.errors) == 3
        error_sources = {e["source"] for e in result.errors}
        assert error_sources == {"check_in_alerts", "loom_alerts", "memory_hygiene"}


# ---------------------------------------------------------------------------
# 3. Unit: evaluator returns None instead of list
# ---------------------------------------------------------------------------


class TestNoneNormalization:
    @pytest.mark.asyncio
    async def test_none_return_normalized_to_empty(self):
        from weft.health_check import run_all_evaluators

        pool = AsyncMock()

        with patch("weft.health_check._evaluate_checkin", new_callable=AsyncMock, return_value=None), \
             patch("weft.health_check._evaluate_loom", new_callable=AsyncMock, return_value=None), \
             patch("weft.health_check._evaluate_hygiene", new_callable=AsyncMock, return_value=None):
            result = await run_all_evaluators(pool)

        assert result.total_findings == 0
        assert result.findings == []


# ---------------------------------------------------------------------------
# 4. Unit: HealthSummary.total_findings auto-computed
# ---------------------------------------------------------------------------


class TestHealthSummaryPostInit:
    def test_total_findings_auto_computed(self):
        from weft.health_check import HealthFinding, HealthSummary

        findings = [
            HealthFinding(source="a", severity="info", message="x"),
            HealthFinding(source="b", severity="warning", message="y"),
        ]
        summary = HealthSummary(
            findings=findings,
            errors=[],
            evaluated_at=datetime.now(timezone.utc),
        )
        assert summary.total_findings == 2

    def test_total_findings_zero_when_empty(self):
        from weft.health_check import HealthSummary

        summary = HealthSummary(
            findings=[],
            errors=[],
            evaluated_at=datetime.now(timezone.utc),
        )
        assert summary.total_findings == 0


# ---------------------------------------------------------------------------
# 5. Unit: severity normalization
# ---------------------------------------------------------------------------


class TestSeverityNormalization:
    def test_known_severities_pass_through(self):
        from weft.health_check import normalize_severity

        assert normalize_severity("info") == "info"
        assert normalize_severity("warning") == "warning"
        assert normalize_severity("critical") == "critical"

    def test_alternate_severities_mapped(self):
        from weft.health_check import normalize_severity

        assert normalize_severity("high") == "critical"
        assert normalize_severity("medium") == "warning"
        assert normalize_severity("low") == "info"

    def test_unknown_defaults_to_info(self):
        from weft.health_check import normalize_severity

        assert normalize_severity("banana") == "info"


# ---------------------------------------------------------------------------
# 6. Unit: to_serializable produces JSON-safe dict
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_summary_to_dict_is_json_serializable(self):
        from weft.health_check import HealthFinding, HealthSummary, summary_to_dict

        summary = HealthSummary(
            findings=[
                HealthFinding(source="test", severity="info", message="ok"),
            ],
            errors=[],
            evaluated_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        )
        d = summary_to_dict(summary)

        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # evaluated_at must be ISO string
        assert d["evaluated_at"] == "2026-03-25T12:00:00+00:00"

        # summary field for quick scanning
        assert "summary" in d
        assert "1 finding" in d["summary"]

    def test_summary_with_errors_includes_error_count_in_summary(self):
        from weft.health_check import HealthSummary, summary_to_dict

        summary = HealthSummary(
            findings=[],
            errors=[{"source": "test", "error": "boom", "error_type": "RuntimeError"}],
            evaluated_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        )
        d = summary_to_dict(summary)
        assert "1 error" in d["summary"]


# ---------------------------------------------------------------------------
# 7. Unit: MCP tool handler returns JSON-serializable response
# ---------------------------------------------------------------------------


class TestMCPToolHandler:
    @pytest.mark.asyncio
    async def test_weft_check_health_returns_serializable(self):
        from weft.health_check import HealthSummary

        mock_summary = HealthSummary(
            findings=[],
            errors=[],
            evaluated_at=datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
        )

        pool = AsyncMock()

        with patch("weft.health_check.run_all_evaluators", new_callable=AsyncMock, return_value=mock_summary):
            from weft.health_check import run_all_evaluators, summary_to_dict
            result = await run_all_evaluators(pool)
            d = summary_to_dict(result)

        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["evaluated_at"] == "2026-03-25T12:00:00+00:00"
        assert d["total_findings"] == 0
