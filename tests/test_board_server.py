"""Tests for weft.board_server — the localhost overlay (weft-board-epic Task 9).

Drives the ASGI app directly over `httpx.ASGITransport` (no real socket)
against the real testcontainers Postgres `pool` fixture. Covers:
- GET /board returns JSON with the four bucket keys under `buckets`.
- POST /act with a close descriptor returns 200, and the closed item is
  gone from a subsequent GET /board.
- POST /act with a tool NOT on the write-tool allowlist returns 4xx and
  performs no write (no underlying row change, no board_triage_events row).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from weft.board_server import create_app
from weft.models import NudgeMode, TrackerCreate, TrackerKind


@pytest.fixture
def now():
    return datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
async def client(pool):
    app = create_app(pool)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver",
    ) as c:
        yield c


class TestGetBoard:
    async def test_returns_four_bucket_keys(self, client):
        response = await client.get("/board")
        assert response.status_code == 200

        body = response.json()
        assert set(body["buckets"].keys()) == {
            "overdue", "due_soon", "pending", "no_date",
        }


class TestPostActClose:
    async def test_close_returns_200_and_item_omitted_from_subsequent_board(
        self, client, pool, now,
    ):
        from weft.trackers import create_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Overlay close target",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )

        # Confirm it's actually on the board before closing it.
        before = await client.get("/board")
        before_ids = {item["id"] for item in before.json()["items"]}
        assert tracker.id in before_ids

        response = await client.post(
            "/act",
            json={
                "tool": "weft_tracker_close",
                "args": {"tracker_id": tracker.id},
                "item_id": tracker.id,
                "source": "tracker",
                "kind": "task",
                "urgency_at_surface": "overdue",
                "age_days_at_surface": 0.5,
                "verb": "close",
            },
        )
        assert response.status_code == 200

        after = await client.get("/board")
        after_ids = {item["id"] for item in after.json()["items"]}
        assert tracker.id not in after_ids


class TestGetPage:
    async def test_root_serves_the_dashboard_html(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        body = response.text
        assert "<title>Weft Board</title>" in body
        # The page must speak the board contract it renders from.
        assert "/board" in body and "/act" in body

    async def test_days_param_forwarded_to_assemble(self, client):
        response = await client.get("/board?days=30")
        assert response.status_code == 200
        assert response.json()["horizon_days"] == 30


class TestActionDescriptorRoundTrip:
    """The whole point of `actions[]`: a descriptor the board emits must be
    directly dispatchable through /act. Fetch the board, take an item's own
    action, POST exactly that, and verify the item is gone — no hand-built
    payload."""

    async def test_board_supplied_close_descriptor_dispatches(
        self, client, pool, now,
    ):
        from weft.trackers import create_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Round-trip close target",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )

        board = (await client.get("/board")).json()
        item = next(i for i in board["items"] if i["id"] == tracker.id)
        close = next(a for a in item["actions"] if a["verb"] == "close")
        assert close["tool"] == "weft_tracker_close"  # board told us the tool

        # Build the /act body straight from the descriptor + item — exactly
        # what the dashboard's fireAct() does.
        resp = await client.post("/act", json={
            "tool": close["tool"], "args": close["args"],
            "item_id": item["id"], "source": item["source"], "kind": item["kind"],
            "urgency_at_surface": item["urgency"],
            "age_days_at_surface": item["age_days"], "verb": close["verb"],
        })
        assert resp.status_code == 200

        after = (await client.get("/board")).json()
        assert tracker.id not in {i["id"] for i in after["items"]}


class TestPostActRejectsDisallowedTool:
    async def test_disallowed_tool_returns_4xx_and_performs_no_write(
        self, client, pool, now,
    ):
        from weft.trackers import create_tracker, get_tracker

        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.task,
                title="Untouched by disallowed tool",
                nudge_mode=NudgeMode.once,
                nudge_after=now - timedelta(hours=1),
            ),
        )
        tracker_before = await get_tracker(pool, tracker.id)
        events_before = await pool.fetchval(
            "SELECT count(*) FROM board_triage_events",
        )

        response = await client.post(
            "/act",
            json={
                "tool": "weft_forget",  # not a triage write tool
                "args": {"tracker_id": tracker.id},
                "item_id": tracker.id,
                "source": "tracker",
                "kind": "task",
                "urgency_at_surface": "overdue",
                "age_days_at_surface": 0.5,
                "verb": "close",
            },
        )

        assert 400 <= response.status_code < 500

        tracker_after = await get_tracker(pool, tracker.id)
        assert tracker_after.state == tracker_before.state
        assert tracker_after.nudge_mode == tracker_before.nudge_mode

        events_after = await pool.fetchval(
            "SELECT count(*) FROM board_triage_events",
        )
        assert events_after == events_before
