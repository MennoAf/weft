"""Tests for Weft skills — query functions for structured insights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.board import BOARD_TIMEZONE
from weft.skills import meal_plan, project_status, search_all, up_next, weekly_recap


def _make_row(
    id="weft-test",
    type="fact",
    content="Test content",
    topic=None,
    source="conversation",
    confidence=0.7,
    created_at=None,
    project_id=None,
    status="active",
    usefulness_score=0.7,
    embedding=None,
):
    """Create a dict mimicking an asyncpg Row."""
    if topic is None:
        topic = ["test"]
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return {
        "id": id,
        "type": type,
        "content": content,
        "topic": topic,
        "source": source,
        "confidence": confidence,
        "created_at": created_at,
        "updated_at": created_at,
        "accessed_at": created_at,
        "access_count": 0,
        "project_id": project_id,
        "agent_id": None,
        "status": status,
        "pinned": False,
        "usefulness_score": usefulness_score,
        "usefulness_count": 0,
        "review_after": None,
        "token_count": 10,
        "embedding": embedding,
    }


def _mock_pool(rows=None):
    """Create a mock asyncpg pool that returns given rows from fetch."""
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows or [])
    return pool


class TestWeeklyRecap:
    @pytest.mark.asyncio
    async def test_empty_results(self):
        pool = _mock_pool([])
        result = await weekly_recap(pool, days=7)
        assert result["total_memories"] == 0
        assert result["by_type"] == {}
        assert result["recent"] == []

    @pytest.mark.asyncio
    async def test_groups_by_type(self):
        rows = [
            _make_row(id="1", type="decision", content="Decided to use Postgres"),
            _make_row(id="2", type="fact", content="Redis is fast"),
            _make_row(id="3", type="decision", content="Use fastembed"),
        ]
        pool = _mock_pool(rows)
        result = await weekly_recap(pool, days=7)
        assert result["total_memories"] == 3
        assert result["by_type"]["decision"] == 2
        assert result["by_type"]["fact"] == 1
        assert len(result["decisions"]) == 2

    @pytest.mark.asyncio
    async def test_counts_sources(self):
        rows = [
            _make_row(id="1", source="conversation"),
            _make_row(id="2", source="ingest"),
            _make_row(id="3", source="conversation"),
        ]
        pool = _mock_pool(rows)
        result = await weekly_recap(pool, days=7)
        assert result["by_source"]["conversation"] == 2
        assert result["by_source"]["ingest"] == 1

    @pytest.mark.asyncio
    async def test_top_topics_excludes_file_paths(self):
        rows = [
            _make_row(id="1", topic=["obsidian", "file:notes/test.md", "recipes"]),
            _make_row(id="2", topic=["obsidian", "recipes"]),
        ]
        pool = _mock_pool(rows)
        result = await weekly_recap(pool, days=7)
        topics_dict = dict(result["top_topics"])
        assert "file:notes/test.md" not in topics_dict
        assert topics_dict["recipes"] == 2

    @pytest.mark.asyncio
    async def test_with_project_filter(self):
        pool = _mock_pool([])
        await weekly_recap(pool, days=7, project_id="weft")
        # Verify the first fetch (memories query) included project_id param
        first_call = pool.fetch.call_args_list[0]
        assert "weft" in first_call[0]


class TestSearchAll:
    @pytest.mark.asyncio
    async def test_requires_at_least_one_filter(self):
        pool = _mock_pool()
        result = await search_all(pool, None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_topic_filter(self):
        rows = [
            _make_row(id="1", topic=["recipes"], content="Pasta recipe"),
        ]
        pool = _mock_pool(rows)
        result = await search_all(pool, None, topic="recipes")
        assert result["count"] == 1
        assert result["filters"]["topic"] == "recipes"

    @pytest.mark.asyncio
    async def test_days_filter(self):
        rows = [
            _make_row(id="1", content="Recent note"),
        ]
        pool = _mock_pool(rows)
        result = await search_all(pool, None, days=7)
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_type_filter(self):
        rows = [
            _make_row(id="1", type="decision", content="Important decision"),
        ]
        pool = _mock_pool(rows)
        result = await search_all(pool, None, memory_type="decision")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_semantic_search(self):
        pool = _mock_pool()
        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[0.1] * 768)

        # Mock search_by_vector via the pool — but search_all calls search_by_vector directly
        # We need to patch it
        from unittest.mock import patch

        mock_recall = MagicMock()
        mock_recall.memory = MagicMock()
        mock_recall.memory.created_at = datetime.now(timezone.utc)
        mock_recall.to_dict = MagicMock(return_value={"id": "1", "content": "test"})
        mock_recall.similarity = 0.9

        with patch("weft.skills.search_by_vector", return_value=[mock_recall]):
            result = await search_all(pool, embedding_provider, query="test query")

        assert result["count"] == 1
        embedding_provider.embed.assert_called_once_with("test query")


class TestProjectStatus:
    @pytest.mark.asyncio
    async def test_empty_project(self):
        pool = _mock_pool([])
        result = await project_status(pool, project_id="nonexistent")
        assert result["total_memories"] == 0
        assert result["decisions"] == []

    @pytest.mark.asyncio
    async def test_weights_decisions_first(self):
        rows = [
            _make_row(id="1", type="fact", content="A fact"),
            _make_row(id="2", type="decision", content="A decision"),
            _make_row(id="3", type="issue", content="An issue"),
        ]
        pool = _mock_pool(rows)
        result = await project_status(pool, project_id="weft")
        assert len(result["decisions"]) == 1
        assert len(result["issues"]) == 1

    @pytest.mark.asyncio
    async def test_recent_activity(self):
        rows = [_make_row(id=str(i)) for i in range(15)]
        pool = _mock_pool(rows)
        result = await project_status(pool, project_id="weft")
        assert len(result["recent_activity"]) == 10


class TestMealPlan:
    @pytest.mark.asyncio
    async def test_returns_recipes(self):
        rows = [
            _make_row(
                id="1",
                topic=["obsidian", "recipes"],
                content="# Pasta Carbonara\n\nDelicious pasta\n\nCuisine: Italian\nLissy approved: yes",
            ),
        ]
        pool = _mock_pool(rows)
        result = await meal_plan(pool)
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_lissy_approved_filter(self):
        rows = [
            _make_row(
                id="1",
                topic=["recipes"],
                content="# Pasta\n\nLissy approved: yes",
            ),
            _make_row(
                id="2",
                topic=["recipes"],
                content="# Spicy Ramen\n\nLissy approved: no",
            ),
        ]
        pool = _mock_pool(rows)
        result = await meal_plan(pool, lissy_approved=True)
        assert result["count"] == 1
        assert "Pasta" in result["recipes"][0]["content"]

    @pytest.mark.asyncio
    async def test_cuisine_filter(self):
        rows = [
            _make_row(
                id="1",
                topic=["recipes"],
                content="# Pasta\n\nCuisine: Italian",
            ),
            _make_row(
                id="2",
                topic=["recipes"],
                content="# Tacos\n\nCuisine: Mexican",
            ),
        ]
        pool = _mock_pool(rows)
        result = await meal_plan(pool, cuisine="Italian")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_empty_recipes(self):
        pool = _mock_pool([])
        result = await meal_plan(pool)
        assert result["count"] == 0
        assert result["recipes"] == []


class TestUpNext:
    @pytest.mark.asyncio
    async def test_finds_due_tasks(self):
        # due:-tags are interpreted as BOARD_TIMEZONE calendar dates
        # (Ghost-F3 fix, weft.board.task_due_at) — anchor "tomorrow" to that
        # zone's current date so this test can't straddle the UTC/ET day
        # boundary and misclassify near midnight.
        tomorrow = (
            datetime.now(BOARD_TIMEZONE) + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        rows = [
            _make_row(
                id="1",
                topic=["tasks", f"due:{tomorrow}", "priority:high"],
                content=f"Task: Buy groceries\nDue: {tomorrow}\nPriority: high",
            ),
        ]
        pool = _mock_pool(rows)
        result = await up_next(pool, days=7)
        assert result["due_soon_count"] == 1
        assert result["due_soon"][0]["due"] == tomorrow
        assert result["due_soon"][0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_finds_overdue_tasks(self):
        # Anchor to BOARD_TIMEZONE's current date (see test_finds_due_tasks)
        # so "yesterday" is unambiguously a full elapsed day regardless of
        # the UTC/ET offset at the moment this test happens to run.
        yesterday = (
            datetime.now(BOARD_TIMEZONE) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        rows = [
            _make_row(
                id="1",
                topic=["tasks", f"due:{yesterday}"],
                content=f"Task: Overdue thing\nDue: {yesterday}",
            ),
        ]
        pool = _mock_pool(rows)
        result = await up_next(pool, days=7)
        assert result["overdue_count"] == 1

    @pytest.mark.asyncio
    async def test_excludes_tasks_beyond_window(self):
        far_future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        rows = [
            _make_row(
                id="1",
                topic=["tasks", f"due:{far_future}"],
                content=f"Task: Later\nDue: {far_future}",
            ),
        ]
        pool = _mock_pool(rows)
        result = await up_next(pool, days=7)
        assert result["due_soon_count"] == 0
        assert result["overdue_count"] == 0

    @pytest.mark.asyncio
    async def test_no_date_tasks(self):
        rows = [
            _make_row(
                id="1",
                topic=["tasks"],
                content="Task: No deadline",
            ),
        ]
        pool = _mock_pool(rows)

        result_excluded = await up_next(pool, days=7, include_no_date=False)
        assert "no_date" not in result_excluded

        result_included = await up_next(pool, days=7, include_no_date=True)
        assert result_included["no_date_count"] == 1

    @pytest.mark.asyncio
    async def test_sorts_by_date_then_priority(self):
        # Anchor to BOARD_TIMEZONE's current date (see test_finds_due_tasks)
        # so "today"/"tomorrow" can't straddle the UTC/ET day boundary.
        today = datetime.now(BOARD_TIMEZONE).strftime("%Y-%m-%d")
        tomorrow = (datetime.now(BOARD_TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")
        rows = [
            _make_row(
                id="1",
                topic=["tasks", f"due:{tomorrow}", "priority:low"],
                content="Task: Tomorrow low",
            ),
            _make_row(
                id="2",
                topic=["tasks", f"due:{today}", "priority:high"],
                content="Task: Today high",
            ),
            _make_row(
                id="3",
                topic=["tasks", f"due:{today}", "priority:low"],
                content="Task: Today low",
            ),
        ]
        pool = _mock_pool(rows)
        result = await up_next(pool, days=7)
        dues = result["due_soon"]
        assert len(dues) == 3
        # Today high should come before today low
        assert dues[0]["due"] == today
        assert dues[0]["priority"] == "high"
        assert dues[1]["due"] == today
        assert dues[1]["priority"] == "low"
        # Tomorrow comes last
        assert dues[2]["due"] == tomorrow

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        pool = _mock_pool([])
        result = await up_next(pool, days=7)
        assert result["overdue_count"] == 0
        assert result["due_soon_count"] == 0
