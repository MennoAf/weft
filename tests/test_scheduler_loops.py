"""Tests for scheduler loops — loom_awareness_loop and memory_hygiene_loop.

Tests verify that each loop:
1. Calls its evaluator and handles findings
2. Handles the evaluator returning empty results
3. Handles evaluator exceptions without crashing
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_one_iteration(coro_fn, pool, **kwargs):
    """Run a scheduler loop for one iteration then cancel it."""
    task = asyncio.create_task(coro_fn(pool, interval=0, **kwargs))
    # Give the loop one iteration
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Loom awareness loop
# ---------------------------------------------------------------------------


class TestLoomAwarenessLoop:
    @pytest.mark.asyncio
    async def test_calls_evaluator_with_findings(self):
        from weft.scheduler import loom_awareness_loop

        mock_alerts = [{"alert_type": "loom_stale_claim", "title": "test"}]
        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ) as mock_eval:
            await _run_one_iteration(loom_awareness_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_handles_empty_findings(self):
        from weft.scheduler import loom_awareness_loop

        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_eval:
            await _run_one_iteration(loom_awareness_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_survives_evaluator_exception(self):
        from weft.scheduler import loom_awareness_loop

        with patch(
            "weft.loom_alerts.evaluate_loom_alerts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            # Should not raise — the loop catches exceptions
            await _run_one_iteration(loom_awareness_loop, AsyncMock())


# ---------------------------------------------------------------------------
# Memory hygiene loop
# ---------------------------------------------------------------------------


class TestMemoryHygieneLoop:
    @pytest.mark.asyncio
    async def test_calls_evaluator_with_findings(self):
        from weft.scheduler import memory_hygiene_loop

        mock_alerts = [{"alert_type": "memory_consolidation_overdue", "title": "test"}]
        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ) as mock_eval:
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_handles_empty_findings(self):
        from weft.scheduler import memory_hygiene_loop

        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_eval:
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())
            assert mock_eval.call_count >= 1

    @pytest.mark.asyncio
    async def test_survives_evaluator_exception(self):
        from weft.scheduler import memory_hygiene_loop

        with patch(
            "weft.memory_hygiene_alerts.evaluate_memory_hygiene_alerts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            await _run_one_iteration(memory_hygiene_loop, AsyncMock())
