"""Tests for the recent_memories tier-1 primer section.

The section answers a different question from handoff and recent_work:
*what other memories are alive in this user's brain right now*. It
ranks by recency + project-match + pinned, excludes the types other
sections own (handoff, milestone) and ingest noise, and dedups
against any tier-1 memory already packed via ``ctx.seen_ids``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.models import MemorySource, MemoryStatus, MemoryType
from weft.primer_sections.context import PrimerContext, SectionFetch


def _ctx(**overrides) -> PrimerContext:
    defaults = dict(
        user_id="u",
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


def _mem(
    *,
    mid: str,
    mtype: MemoryType = MemoryType.fact,
    project_id: str | None = "proj-1",
    pinned: bool = False,
    age_hours: float = 1.0,
    content: str = "x" * 100,
    source: MemorySource = MemorySource.conversation,
):
    """Build a Memory-like MagicMock the section can rank."""
    now = datetime.now(timezone.utc)
    m = MagicMock()
    m.id = mid
    m.type = mtype
    m.project_id = project_id
    m.pinned = pinned
    m.created_at = now - timedelta(hours=age_hours)
    m.updated_at = now - timedelta(hours=age_hours)
    m.content = content
    m.topic = ["t"]
    m.confidence = 0.8
    m.token_count = 0
    m.review_after = None
    m.source = source
    return m


class TestExcludedTypes:
    @pytest.mark.asyncio
    async def test_section_owned_types_filtered_out(self):
        # Other sections (handoff, recent_work, decisions, issues,
        # anti_patterns) own their canonical types. recent_memories
        # must not claim those — only the leaf types nobody else owns.
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        candidates = [
            _mem(mid="h", mtype=MemoryType.handoff),
            _mem(mid="m", mtype=MemoryType.milestone),
            _mem(mid="d", mtype=MemoryType.decision),
            _mem(mid="i", mtype=MemoryType.issue),
            _mem(mid="a", mtype=MemoryType.anti_pattern),
            _mem(mid="f", mtype=MemoryType.fact),
            _mem(mid="p", mtype=MemoryType.preference),
            _mem(mid="s", mtype=MemoryType.solution),
        ]
        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        ids = {item["id"] for item in result.items}
        assert ids == {"f", "p", "s"}


class TestRanking:
    @pytest.mark.asyncio
    async def test_pinned_outranks_more_recent_unpinned(self):
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        # Both inside the recency window. Pinned older one should win.
        candidates = [
            _mem(mid="fresh", pinned=False, age_hours=1.0),
            _mem(mid="pinned-older", pinned=True, age_hours=24.0),
        ]
        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        assert result.items[0]["id"] == "pinned-older"

    @pytest.mark.asyncio
    async def test_project_match_outranks_global_at_equal_recency(self):
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        candidates = [
            _mem(mid="global", project_id=None, age_hours=2.0),
            _mem(mid="local", project_id="proj-1", age_hours=2.0),
        ]
        ctx = _ctx(project_id="proj-1")
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        assert result.items[0]["id"] == "local"

    @pytest.mark.asyncio
    async def test_pinned_bypasses_recency_window(self):
        # 30 days old, pinned — should still appear.
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        ancient_pinned = _mem(mid="ancient", pinned=True, age_hours=30 * 24)
        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = [ancient_pinned]
            result = await build_recent_memories_section(ctx)

        assert [item["id"] for item in result.items] == ["ancient"]


class TestDedup:
    @pytest.mark.asyncio
    async def test_seen_ids_filtered(self):
        # Sections that ran first (handoff, rules, etc.) put their ids
        # in ctx.seen_ids — recent_memories must respect that to avoid
        # duplicating handoff content.
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        candidates = [
            _mem(mid="already-shown"),
            _mem(mid="new"),
        ]
        ctx = _ctx()
        ctx.seen_ids.add("already-shown")
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        ids = {item["id"] for item in result.items}
        assert ids == {"new"}

    @pytest.mark.asyncio
    async def test_unscoped_ingest_filtered_when_project_scoped(self):
        # Cross-project ingest noise (project_id=None, source=ingest)
        # should not surface in a project-scoped primer.
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        candidates = [
            _mem(
                mid="ingest-noise",
                project_id=None,
                source=MemorySource.ingest,
            ),
            _mem(mid="legit"),
        ]
        ctx = _ctx(project_id="proj-1")
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        ids = {item["id"] for item in result.items}
        assert ids == {"legit"}


class TestBudget:
    @pytest.mark.asyncio
    async def test_max_items_cap_honored(self):
        # Pool of 12, _MAX is 5 — must drop the rest into excluded.
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        candidates = [_mem(mid=f"m{i}") for i in range(12)]
        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = candidates
            result = await build_recent_memories_section(ctx)

        assert len(result.items) == 5
        # Excluded count should account for the drops.
        assert ctx.excluded >= 7

    @pytest.mark.asyncio
    async def test_section_tokens_recorded(self):
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = [_mem(mid="a"), _mem(mid="b")]
            result = await build_recent_memories_section(ctx)

        assert "recent_memories" in ctx.section_tokens
        assert ctx.section_tokens["recent_memories"] > 0
        assert ctx.section_tokens["recent_memories"] == result.tokens_used

    @pytest.mark.asyncio
    async def test_empty_when_no_candidates(self):
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = []
            result = await build_recent_memories_section(ctx)

        assert result.items == []
        assert result.tokens_used == 0
        assert result.skipped is False
        assert ctx.section_tokens["recent_memories"] == 0


class TestEntryShape:
    @pytest.mark.asyncio
    async def test_entry_has_expected_fields(self):
        from weft.primer_sections.recent_memories import (
            build_recent_memories_section,
        )

        m = _mem(mid="x", mtype=MemoryType.preference, pinned=True)
        ctx = _ctx()
        with patch(
            "weft.primer_sections.recent_memories.list_memories",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = [m]
            result = await build_recent_memories_section(ctx)

        entry = result.items[0]
        assert entry["id"] == "x"
        assert entry["type"] == "preference"
        assert entry["pinned"] is True
        assert entry["content"] == m.content
        assert entry["topic"] == ["t"]
        assert "age_hours" in entry
        assert "confidence" in entry
