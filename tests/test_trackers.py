"""Tracker store + MCP tools — Wick Phase 3 (open-loop primitive).

Covers:
- create / get / update / close / dismiss / snooze / list / due
- state machine transitions (open ↔ open, open → terminal, terminal locked)
- nudge mode semantics (none / once / recur)
- snooze suppression of due()
- dismiss bumps last_touch and rolls recurring nudge_after forward
- list sugar (append / check / remove) operates on context.items
- kind=trace context shape for orchestrator promotion
- CHECK constraints (invalid kind/state at DB level)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.models import (
    NudgeMode,
    TrackerCreate,
    TrackerKind,
    TrackerState,
)
from weft.trackers import (
    close_tracker,
    create_tracker,
    dismiss_tracker,
    due_trackers,
    get_tracker,
    list_append,
    list_check,
    list_remove,
    list_trackers,
    snooze_tracker,
    update_tracker,
)


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool, cache=NullCache(),
        embedding=MagicMock(),  # trackers don't use embeddings
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# ---------------------------------------------------------------------------
# Store layer — direct
# ---------------------------------------------------------------------------


class TestTrackerCreate:
    async def test_create_minimal(self, pool):
        tr = await create_tracker(
            pool,
            TrackerCreate(kind=TrackerKind.outreach, title="Ping Brandon"),
        )
        assert tr.id.startswith("tr-")
        assert tr.kind == TrackerKind.outreach
        assert tr.state == TrackerState.in_progress
        assert tr.nudge_mode == NudgeMode.none
        # state_history records the initial state
        assert len(tr.state_history) == 1
        assert tr.state_history[0]["from"] is None
        assert tr.state_history[0]["to"] == "in_progress"
        assert tr.state_history[0]["note"] == "created"

    async def test_create_with_nudge_recur(self, pool):
        in_5min = datetime.now(timezone.utc) + timedelta(minutes=5)
        tr = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.outreach,
                title="Follow up Tory Burch",
                state=TrackerState.awaiting_reply,
                nudge_mode=NudgeMode.recur,
                nudge_after=in_5min,
                nudge_interval=timedelta(days=7),
            ),
        )
        assert tr.nudge_mode == NudgeMode.recur
        assert tr.nudge_interval == timedelta(days=7)
        assert tr.nudge_after.replace(microsecond=0) == in_5min.replace(microsecond=0)

    async def test_get_returns_full_record(self, pool):
        tr = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.shopping_list, title="Groceries",
                context={"items": [{"text": "milk", "checked": False}]},
            ),
        )
        fetched = await get_tracker(pool, tr.id)
        assert fetched is not None
        assert fetched.title == "Groceries"
        assert fetched.context["items"] == [{"text": "milk", "checked": False}]

    async def test_invalid_kind_rejected(self, pool):
        # CHECK constraint at the DB level
        import asyncpg
        with pytest.raises((asyncpg.CheckViolationError, ValueError)):
            await pool.execute(
                "INSERT INTO trackers (id, kind, title) VALUES ($1, $2, $3)",
                "tr-bad", "not_a_kind", "x",
            )


class TestStateMachine:
    async def test_open_to_open(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="x"),
        )
        updated = await update_tracker(
            pool, tr.id,
            state=TrackerState.awaiting_reply,
            state_note="sent the email",
        )
        assert updated.state == TrackerState.awaiting_reply
        assert len(updated.state_history) == 2
        assert updated.state_history[-1]["from"] == "in_progress"
        assert updated.state_history[-1]["to"] == "awaiting_reply"
        assert updated.state_history[-1]["note"] == "sent the email"

    async def test_open_to_terminal(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="x"),
        )
        closed = await close_tracker(pool, tr.id, final_state=TrackerState.done)
        assert closed.state == TrackerState.done
        assert closed.is_open() is False

    async def test_terminal_to_open_rejected(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="x"),
        )
        await close_tracker(pool, tr.id)
        with pytest.raises(ValueError, match="terminal state"):
            await update_tracker(pool, tr.id, state=TrackerState.in_progress)

    async def test_close_requires_terminal_state(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="x"),
        )
        with pytest.raises(ValueError, match="terminal state"):
            await close_tracker(
                pool, tr.id, final_state=TrackerState.in_progress,
            )


class TestNudgeAndDismiss:
    async def test_dismiss_bumps_last_touch_no_nudge(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="x"),
        )
        before = tr.last_touch
        # asyncpg timestamps don't have sub-microsecond resolution, so sleep
        import asyncio
        await asyncio.sleep(0.01)
        dismissed = await dismiss_tracker(pool, tr.id)
        assert dismissed.last_touch > before
        assert dismissed.nudge_mode == NudgeMode.none

    async def test_dismiss_recur_rolls_nudge_forward(self, pool):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        tr = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.outreach, title="check-in",
                nudge_mode=NudgeMode.recur,
                nudge_after=past,
                nudge_interval=timedelta(days=7),
            ),
        )
        dismissed = await dismiss_tracker(pool, tr.id)
        assert dismissed.nudge_mode == NudgeMode.recur
        # nudge_after is now ~7 days in the future
        delta = dismissed.nudge_after - datetime.now(timezone.utc)
        assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)

    async def test_dismiss_once_silences(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        tr = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task, title="x",
                nudge_mode=NudgeMode.once,
                nudge_after=past,
            ),
        )
        dismissed = await dismiss_tracker(pool, tr.id)
        assert dismissed.nudge_mode == NudgeMode.none


class TestSnoozeAndDue:
    async def test_due_returns_open_with_past_nudge(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.task, title="due-now",
                nudge_mode=NudgeMode.once, nudge_after=past,
            ),
        )
        await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.task, title="due-later",
                nudge_mode=NudgeMode.once, nudge_after=future,
            ),
        )
        await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.task, title="no-nudge", nudge_mode=NudgeMode.none,
            ),
        )

        due = await due_trackers(pool)
        titles = {t.title for t in due}
        assert titles == {"due-now"}

    async def test_terminal_trackers_excluded_from_due(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        tr = await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.task, title="closed",
                nudge_mode=NudgeMode.once, nudge_after=past,
            ),
        )
        await close_tracker(pool, tr.id)
        due = await due_trackers(pool)
        assert all(t.id != tr.id for t in due)

    async def test_snooze_suppresses_due(self, pool):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        tr = await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.task, title="snoozable",
                nudge_mode=NudgeMode.once, nudge_after=past,
            ),
        )
        # Confirms it was due before snoozing
        assert any(t.id == tr.id for t in await due_trackers(pool))
        await snooze_tracker(pool, tr.id, future)
        # No longer in due()
        assert all(t.id != tr.id for t in await due_trackers(pool))


class TestListSugar:
    async def test_append_check_remove(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.shopping_list, title="Groceries"),
        )
        tr = await list_append(pool, tr.id, {"text": "milk", "checked": False})
        tr = await list_append(pool, tr.id, {"text": "eggs", "checked": False})
        tr = await list_append(pool, tr.id, {"text": "bread", "checked": False})
        assert len(tr.context["items"]) == 3
        assert tr.context["items"][0]["text"] == "milk"

        tr = await list_check(pool, tr.id, 1, checked=True)
        assert tr.context["items"][1]["checked"] is True
        assert tr.context["items"][0]["checked"] is False

        tr = await list_remove(pool, tr.id, 0)
        assert len(tr.context["items"]) == 2
        assert [i["text"] for i in tr.context["items"]] == ["eggs", "bread"]

    async def test_check_index_out_of_range(self, pool):
        tr = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.list, title="empty"),
        )
        with pytest.raises(ValueError, match="out of range"):
            await list_check(pool, tr.id, 5)


class TestKindTrace:
    async def test_trace_context_shape(self, pool):
        """Orchestrator stores trace payload in context — full round-trip preserves shape."""
        trace_payload = {
            "chain": ["agent-a", "agent-b"],
            "dispatches": [{"to": "agent-a", "at": "2026-04-27T12:00:00Z"}],
            "cost_accumulated_usd": 0.42,
            "status": "awaiting_external",
        }
        tr = await create_tracker(
            pool, TrackerCreate(
                kind=TrackerKind.trace,
                title="long-running orchestrator chain abc",
                context=trace_payload,
            ),
        )
        fetched = await get_tracker(pool, tr.id)
        assert fetched.kind == TrackerKind.trace
        assert fetched.context == trace_payload


class TestListQueries:
    async def test_filter_by_kind(self, pool):
        await create_tracker(pool, TrackerCreate(kind=TrackerKind.outreach, title="a"))
        await create_tracker(pool, TrackerCreate(kind=TrackerKind.task, title="b"))
        await create_tracker(pool, TrackerCreate(kind=TrackerKind.task, title="c"))
        result = await list_trackers(pool, kind=TrackerKind.task)
        assert len(result) == 2
        assert {t.title for t in result} == {"b", "c"}

    async def test_open_only(self, pool):
        tr_open = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="open"),
        )
        tr_closed = await create_tracker(
            pool, TrackerCreate(kind=TrackerKind.task, title="closed"),
        )
        await close_tracker(pool, tr_closed.id)
        result = await list_trackers(pool, open_only=True)
        ids = {t.id for t in result}
        assert tr_open.id in ids
        assert tr_closed.id not in ids

    async def test_filter_by_context_single_key(self, pool):
        # All three are kind=trace (the Wick catchall); only context.wick_kind
        # distinguishes them — exactly the partition Wick's read path needs.
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="skip-a",
            context={"wick_kind": "authority_skip", "action": "dispatch"},
        ))
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="skip-b",
            context={"wick_kind": "authority_skip", "action": "merge"},
        ))
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="approval",
            context={"wick_kind": "approval_outcome", "action": "dispatch"},
        ))
        result = await list_trackers(
            pool, kind=TrackerKind.trace,
            context_filter={"wick_kind": "authority_skip"},
        )
        assert {t.title for t in result} == {"skip-a", "skip-b"}

    async def test_filter_by_context_multi_key_ands(self, pool):
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="approve-dispatch",
            context={"wick_kind": "approval_outcome", "action": "dispatch"},
        ))
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="approve-merge",
            context={"wick_kind": "approval_outcome", "action": "merge"},
        ))
        result = await list_trackers(
            pool,
            context_filter={"wick_kind": "approval_outcome", "action": "merge"},
        )
        assert {t.title for t in result} == {"approve-merge"}

    async def test_filter_by_context_no_match(self, pool):
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="skip",
            context={"wick_kind": "authority_skip"},
        ))
        result = await list_trackers(
            pool, context_filter={"wick_kind": "does_not_exist"},
        )
        assert result == []

    async def test_filter_since_window(self, pool):
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.task, title="before",
        ))
        cutoff = datetime.now(timezone.utc)
        after = await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.task, title="after",
        ))
        result = await list_trackers(pool, since=cutoff)
        ids = {t.id for t in result}
        assert after.id in ids
        assert {t.title for t in result} == {"after"}

    async def test_context_filter_combines_with_project(self, pool):
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="warp-skip", project_id="warp",
            context={"wick_kind": "authority_skip"},
        ))
        await create_tracker(pool, TrackerCreate(
            kind=TrackerKind.trace, title="other-skip", project_id="other",
            context={"wick_kind": "authority_skip"},
        ))
        result = await list_trackers(
            pool, project_id="warp",
            context_filter={"wick_kind": "authority_skip"},
        )
        assert {t.title for t in result} == {"warp-skip"}


# ---------------------------------------------------------------------------
# MCP tool layer — verifies the wrappers parse args and route correctly.
# ---------------------------------------------------------------------------


class TestMCPLayer:
    async def test_create_via_tool(self, ctx):
        from weft.mcp.tools import weft_tracker_create

        result = await weft_tracker_create(
            ctx, kind="outreach", title="Reach out to Brandon",
        )
        assert "error" not in result
        assert result["kind"] == "outreach"
        assert result["state"] == "in_progress"

    async def test_full_lifecycle_via_tools(self, ctx):
        from weft.mcp.tools import (
            weft_tracker_close,
            weft_tracker_create,
            weft_tracker_dismiss,
            weft_tracker_get,
            weft_tracker_update,
        )

        created = await weft_tracker_create(
            ctx, kind="task", title="Ship trackers",
            nudge_mode="recur", nudge_interval="3d",
        )
        tid = created["id"]

        updated = await weft_tracker_update(
            ctx, tracker_id=tid,
            state="awaiting_reply", state_note="reviewer pinged",
        )
        assert updated["state"] == "awaiting_reply"

        dismissed = await weft_tracker_dismiss(ctx, tracker_id=tid)
        assert "error" not in dismissed

        fetched = await weft_tracker_get(ctx, tracker_id=tid)
        assert fetched["id"] == tid

        closed = await weft_tracker_close(
            ctx, tracker_id=tid, final_state="done", note="merged",
        )
        assert closed["state"] == "done"

    async def test_list_via_tool_context_filter_and_since(self, ctx):
        from weft.mcp.tools import weft_tracker_create, weft_tracker_list

        await weft_tracker_create(
            ctx, kind="trace", title="skip",
            context={"wick_kind": "authority_skip"},
        )
        cutoff = datetime.now(timezone.utc).isoformat()
        await weft_tracker_create(
            ctx, kind="trace", title="approval",
            context={"wick_kind": "approval_outcome"},
        )

        # context_filter partitions the kind=trace catchall.
        skips = await weft_tracker_list(
            ctx, kind="trace", context_filter={"wick_kind": "authority_skip"},
        )
        assert {t["title"] for t in skips["trackers"]} == {"skip"}

        # since (ISO string) bounds the window; only the post-cutoff row.
        recent = await weft_tracker_list(ctx, kind="trace", since=cutoff)
        assert {t["title"] for t in recent["trackers"]} == {"approval"}

    async def test_due_via_tool(self, ctx):
        from weft.mcp.tools import weft_tracker_create, weft_tracker_due

        past_iso = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        await weft_tracker_create(
            ctx, kind="task", title="overdue",
            nudge_mode="once", nudge_after=past_iso,
        )
        await weft_tracker_create(ctx, kind="task", title="not-due")

        result = await weft_tracker_due(ctx)
        titles = {t["title"] for t in result["trackers"]}
        assert titles == {"overdue"}

    async def test_list_sugar_via_tool(self, ctx):
        from weft.mcp.tools import (
            weft_list_append,
            weft_list_check,
            weft_list_remove,
            weft_tracker_create,
        )

        ws = await weft_tracker_create(ctx, kind="shopping_list", title="Groceries")
        tid = ws["id"]

        await weft_list_append(ctx, tracker_id=tid, text="milk")
        await weft_list_append(ctx, tracker_id=tid, text="eggs")
        result = await weft_list_check(ctx, tracker_id=tid, index=0, checked=True)
        assert result["context"]["items"][0]["checked"] is True
        result = await weft_list_remove(ctx, tracker_id=tid, index=1)
        assert len(result["context"]["items"]) == 1

    async def test_terminal_state_locks_tool_path(self, ctx):
        from weft.mcp.tools import weft_tracker_close, weft_tracker_create, weft_tracker_update

        created = await weft_tracker_create(ctx, kind="task", title="x")
        await weft_tracker_close(ctx, tracker_id=created["id"])
        result = await weft_tracker_update(
            ctx, tracker_id=created["id"], state="in_progress",
        )
        assert result.get("error") == "Invalid input"
