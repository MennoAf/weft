"""Tests for calibration records store — CRUD, filtering, and summary aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.calibration import (
    delete_calibration,
    get_calibration,
    get_calibration_summary,
    list_calibrations,
    record_calibration,
)
from weft.models import CalibrationCreate, CalibrationOutcome


# --- Helpers ---


async def _make_calibration(pool, **kwargs):
    defaults = {
        "action_category": "send_message",
        "action_description": "Send Slack message to #general",
        "outcome": CalibrationOutcome.approved,
    }
    defaults.update(kwargs)
    return await record_calibration(pool, CalibrationCreate(**defaults))


# --- create / get ---


async def test_create_and_get(pool):
    rec = await _make_calibration(pool)
    assert rec.id.startswith("weft-")
    assert rec.action_category == "send_message"
    assert rec.action_description == "Send Slack message to #general"
    assert rec.outcome == CalibrationOutcome.approved
    assert rec.context == {}

    fetched = await get_calibration(pool, rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.action_category == rec.action_category


async def test_create_with_context(pool):
    rec = await _make_calibration(
        pool,
        context={"channel": "#ops", "urgency": "high"},
        agent_id="agent-1",
        project_id="proj-a",
    )
    assert rec.context == {"channel": "#ops", "urgency": "high"}
    assert rec.agent_id == "agent-1"
    assert rec.project_id == "proj-a"


async def test_get_not_found(pool):
    result = await get_calibration(pool, "weft-nonexistent")
    assert result is None


# --- list with filters ---


async def test_list_all(pool):
    await _make_calibration(pool, action_category="a")
    await _make_calibration(pool, action_category="b")
    await _make_calibration(pool, action_category="c")
    result = await list_calibrations(pool)
    assert len(result) == 3


async def test_list_by_category(pool):
    await _make_calibration(pool, action_category="deploy")
    await _make_calibration(pool, action_category="deploy")
    await _make_calibration(pool, action_category="send_message")

    result = await list_calibrations(pool, action_category="deploy")
    assert len(result) == 2
    assert all(r.action_category == "deploy" for r in result)


async def test_list_by_outcome(pool):
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, outcome=CalibrationOutcome.rejected)
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)

    result = await list_calibrations(pool, outcome=CalibrationOutcome.rejected)
    assert len(result) == 1
    assert result[0].outcome == CalibrationOutcome.rejected


async def test_list_by_project(pool):
    await _make_calibration(pool, project_id="alpha")
    await _make_calibration(pool, project_id="beta")

    result = await list_calibrations(pool, project_id="alpha")
    categories = {r.project_id for r in result}
    assert "alpha" in categories


# --- summary aggregation ---


async def test_summary_basic(pool):
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, outcome=CalibrationOutcome.rejected)
    await _make_calibration(pool, outcome=CalibrationOutcome.modified)

    summary = await get_calibration_summary(pool)
    assert summary["total"] == 4
    assert summary["approved"] == 2
    assert summary["rejected"] == 1
    assert summary["modified"] == 1
    assert summary["approval_rate"] == pytest.approx(0.5)


async def test_summary_approval_rate_all_approved(pool):
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)

    summary = await get_calibration_summary(pool)
    assert summary["approval_rate"] == pytest.approx(1.0)


async def test_summary_empty(pool):
    summary = await get_calibration_summary(pool)
    assert summary["total"] == 0
    assert summary["approval_rate"] == pytest.approx(0.0)
    assert summary["by_category"] == {}


async def test_summary_by_category_breakdown(pool):
    await _make_calibration(pool, action_category="deploy", outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, action_category="deploy", outcome=CalibrationOutcome.rejected)
    await _make_calibration(pool, action_category="send_message", outcome=CalibrationOutcome.approved)

    summary = await get_calibration_summary(pool)
    assert "deploy" in summary["by_category"]
    assert "send_message" in summary["by_category"]

    deploy = summary["by_category"]["deploy"]
    assert deploy["total"] == 2
    assert deploy["approved"] == 1
    assert deploy["rejected"] == 1
    assert deploy["approval_rate"] == pytest.approx(0.5)

    msg = summary["by_category"]["send_message"]
    assert msg["total"] == 1
    assert msg["approval_rate"] == pytest.approx(1.0)


async def test_summary_with_since_filter(pool):
    await _make_calibration(pool, outcome=CalibrationOutcome.rejected)
    await _make_calibration(pool, outcome=CalibrationOutcome.approved)

    since = datetime.now(timezone.utc) - timedelta(seconds=1)
    summary = await get_calibration_summary(pool, since=since)
    assert summary["total"] >= 1


async def test_summary_filtered_by_category(pool):
    await _make_calibration(pool, action_category="deploy", outcome=CalibrationOutcome.approved)
    await _make_calibration(pool, action_category="send_message", outcome=CalibrationOutcome.rejected)

    summary = await get_calibration_summary(pool, action_category="deploy")
    assert summary["total"] == 1
    assert summary["approved"] == 1
    assert summary["rejected"] == 0


# --- delete ---


async def test_delete(pool):
    rec = await _make_calibration(pool)
    assert await delete_calibration(pool, rec.id) is True
    assert await get_calibration(pool, rec.id) is None


async def test_delete_not_found(pool):
    assert await delete_calibration(pool, "weft-nonexistent") is False


# --- to_dict ---


async def test_to_dict(pool):
    rec = await _make_calibration(
        pool,
        context={"key": "value"},
        outcome=CalibrationOutcome.modified,
    )
    d = rec.to_dict()
    assert d["outcome"] == "modified"
    assert d["context"] == {"key": "value"}
    assert d["action_category"] == "send_message"
