"""Tests for scheduler loops — loom_awareness_loop, memory_hygiene_loop,
and the outbound event dispatch layer.

Tests verify that each loop:
1. Calls its evaluator and handles findings
2. Handles the evaluator returning empty results
3. Handles evaluator exceptions without crashing
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# Outbound event dispatch layer
# ---------------------------------------------------------------------------


class TestOutboundEventDispatch:
    """Tests for the event registry / emit_outbound_event layer."""

    @pytest.mark.asyncio
    async def test_emit_calls_active_connector(self, monkeypatch):
        """emit_outbound_event routes the event to the registered connector."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("test_event", "fake_connector", handler)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "fake_connector")

        await emit_outbound_event("test_event", channel="#test", brief_result=None)

        handler.assert_awaited_once_with(channel="#test", brief_result=None)

    @pytest.mark.asyncio
    async def test_slack_connector_registered(self):
        """The slack connector for daily_brief is registered at module load."""
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        assert "daily_brief" in _OUTBOUND_EVENT_REGISTRY
        assert "slack" in _OUTBOUND_EVENT_REGISTRY["daily_brief"]

    @pytest.mark.asyncio
    async def test_only_active_connector_receives_event(self, monkeypatch):
        """Only the env-selected connector receives the event; others are skipped."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler_a = AsyncMock()
        handler_b = AsyncMock()
        register_outbound_handler("singleton_event", "connector_a", handler_a)
        register_outbound_handler("singleton_event", "connector_b", handler_b)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "connector_a")

        await emit_outbound_event("singleton_event", x=1)

        handler_a.assert_awaited_once_with(x=1)
        handler_b.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_connector_unset(self, monkeypatch):
        """emit_outbound_event is a no-op when WEFT_OUTBOUND_CONNECTOR is unset."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("noop_event", "slack", handler)
        monkeypatch.delenv("WEFT_OUTBOUND_CONNECTOR", raising=False)

        # Must not raise
        await emit_outbound_event("noop_event", channel="#foo", brief_result=None)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_connector_is_none_string(self, monkeypatch):
        """emit_outbound_event is a no-op when WEFT_OUTBOUND_CONNECTOR='none'."""
        from weft.scheduler import emit_outbound_event, register_outbound_handler

        handler = AsyncMock()
        register_outbound_handler("none_event", "slack", handler)
        monkeypatch.setenv("WEFT_OUTBOUND_CONNECTOR", "none")

        await emit_outbound_event("none_event", channel="#foo", brief_result=None)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_daily_brief_loop_emits_event(self, monkeypatch):
        """daily_brief_loop emits a daily_brief event via emit_outbound_event."""
        from weft.scheduler import daily_brief_loop

        # Fake brief result with the fields _post_brief_to_slack expects
        fake_result = SimpleNamespace(
            markdown="Morning brief",
            slack_blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}],
        )

        # daily_brief_loop uses asyncio.sleep(_BRIEF_POLL_INTERVAL) at the end of
        # each cycle. Patch it to raise CancelledError so the loop exits cleanly
        # after exactly one iteration (the CancelledError propagates out of the
        # while-loop's outer try/except to the asyncio.CancelledError handler).
        async def sleep_then_cancel(_):
            raise asyncio.CancelledError("test cancel")

        mock_emit = AsyncMock()

        with (
            # is_daily_brief_due is imported into weft.scheduler at module load
            patch("weft.scheduler.is_daily_brief_due", return_value=True),
            # get/set_last_brief_date are imported inside the function body
            patch("weft.brief_state.get_last_brief_date", return_value=None),
            patch("weft.brief_state.set_last_brief_date"),
            # assemble_daily_brief is imported inside the function body
            patch(
                "weft.daily_brief.assemble_daily_brief",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
            patch("weft.scheduler.emit_outbound_event", mock_emit),
            patch("weft.scheduler.asyncio.sleep", side_effect=sleep_then_cancel),
        ):
            with pytest.raises(asyncio.CancelledError):
                await daily_brief_loop(
                    AsyncMock(),
                    brief_time="08:00",
                    brief_tz="UTC",
                    brief_channel="#test",
                )

        # The loop must have called emit_outbound_event with the daily_brief event
        assert mock_emit.call_count >= 1
        call_kwargs = mock_emit.call_args
        assert call_kwargs[0][0] == "daily_brief"
        assert call_kwargs[1]["channel"] == "#test"
        assert call_kwargs[1]["brief_result"] is fake_result

    @pytest.mark.asyncio
    async def test_slack_handler_calls_post_brief(self, monkeypatch):
        """The registered slack handler calls _post_brief_to_slack with correct args."""
        from weft.scheduler import _OUTBOUND_EVENT_REGISTRY

        fake_result = SimpleNamespace(
            markdown="x" * 400,
            slack_blocks=[],
        )

        slack_handler = _OUTBOUND_EVENT_REGISTRY["daily_brief"]["slack"]

        with patch(
            "weft.scheduler._post_brief_to_slack",
            new_callable=AsyncMock,
        ) as mock_post:
            await slack_handler(channel="#mychannel", brief_result=fake_result)

        mock_post.assert_awaited_once_with("#mychannel", fake_result)
