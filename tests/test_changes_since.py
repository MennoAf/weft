"""Tests for changes_since — memory diff and 'what changed' section.

Covers store helpers (_get_last_handoff_timestamp, get_memory_changes_since),
primer integration (What Changed section), and focus integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.store import store_memory, update_memory


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


async def test_get_last_handoff_timestamp_no_handoff(pool):
    """No handoff memory -> returns None."""
    from weft.store import get_last_handoff_timestamp

    result = await get_last_handoff_timestamp(pool, project_id="test-proj")
    assert result is None


async def test_get_last_handoff_timestamp_returns_most_recent(pool):
    """Multiple handoffs -> returns the most recent created_at."""
    from weft.store import get_last_handoff_timestamp

    # Create two handoffs at different times
    old = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Old handoff",
        topic=["session-handoff"],
        project_id="test-proj",
    ))
    # Force a later timestamp
    await pool.execute(
        "UPDATE memories SET created_at = created_at + interval '1 hour' WHERE id = $1",
        old.id,
    )
    new = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="New handoff",
        topic=["session-handoff"],
        project_id="test-proj",
    ))

    result = await get_last_handoff_timestamp(pool, project_id="test-proj")
    assert result is not None
    assert result >= new.created_at


async def test_get_last_handoff_timestamp_project_scoped(pool):
    """Handoff in another project is not returned."""
    from weft.store import get_last_handoff_timestamp

    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Other project handoff",
        topic=["session-handoff"],
        project_id="other-proj",
    ))

    result = await get_last_handoff_timestamp(pool, project_id="test-proj")
    assert result is None


async def test_get_last_handoff_timestamp_global(pool):
    """project_id=None -> queries global handoffs (project_id IS NULL)."""
    from weft.store import get_last_handoff_timestamp

    await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Global handoff",
        topic=["session-handoff"],
        project_id=None,
    ))

    result = await get_last_handoff_timestamp(pool, project_id=None)
    assert result is not None


async def test_get_memory_changes_since_counts(pool):
    """Correct counts for created, archived, revised memories."""
    from weft.store import get_memory_changes_since

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=2)

    # Pre-existing memory (created before since, will be revised)
    pre = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Pre-existing fact",
        topic=["test"],
        project_id="test-proj",
    ))
    # Backdate its created_at
    await pool.execute(
        "UPDATE memories SET created_at = $1, updated_at = $1 WHERE id = $2",
        since - timedelta(hours=1), pre.id,
    )

    # New memory (created after since)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="New fact",
        topic=["test"],
        project_id="test-proj",
    ))

    # Archive a memory (status='archived', updated_at > since)
    archived = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Archived fact",
        topic=["test"],
        project_id="test-proj",
    ))
    await update_memory(pool, archived.id, status=MemoryStatus.archived)

    # Revise the pre-existing memory (update content, created_at <= since, updated_at > since)
    await update_memory(pool, pre.id, content="Pre-existing fact (revised)")

    result = await get_memory_changes_since(pool, since=since, project_id="test-proj")
    assert result["memories_created"] == 1  # the new fact (not the archived one, it was also new but archived)
    assert result["memories_archived"] == 1
    assert result["memories_revised"] == 1
    assert result["since"] is not None


async def test_get_memory_changes_since_boundary_exclusion(pool):
    """Memory created exactly AT the since timestamp is NOT counted (strict >)."""
    from weft.store import get_memory_changes_since

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=1)

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Boundary memory",
        topic=["test"],
        project_id="test-proj",
    ))
    # Set created_at to exactly `since`
    await pool.execute(
        "UPDATE memories SET created_at = $1, updated_at = $1 WHERE id = $2",
        since, mem.id,
    )

    result = await get_memory_changes_since(pool, since=since, project_id="test-proj")
    assert result["memories_created"] == 0


async def test_get_memory_changes_since_global_scope(pool):
    """project_id=None -> counts only global memories (project_id IS NULL)."""
    from weft.store import get_memory_changes_since

    since = datetime.now(timezone.utc) - timedelta(hours=1)

    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Global memory",
        topic=["test"],
        project_id=None,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Scoped memory",
        topic=["test"],
        project_id="some-proj",
    ))

    result = await get_memory_changes_since(pool, since=since, project_id=None)
    assert result["memories_created"] == 1


async def test_get_memory_changes_since_no_overlap(pool):
    """A memory created AND updated since the timestamp appears only in created, not revised."""
    from weft.store import get_memory_changes_since

    since = datetime.now(timezone.utc) - timedelta(hours=1)

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="New fact",
        topic=["test"],
        project_id="test-proj",
    ))
    # Update it (created_at > since AND updated_at > since)
    await update_memory(pool, mem.id, content="New fact updated")

    result = await get_memory_changes_since(pool, since=since, project_id="test-proj")
    assert result["memories_created"] == 1
    assert result["memories_revised"] == 0  # Not in revised — it's new


# ---------------------------------------------------------------------------
# Primer integration
# ---------------------------------------------------------------------------


async def test_primer_includes_changes_since_section(pool):
    """Primer output includes changes_since when a handoff exists."""
    from weft.primer import build_primer

    # Create a handoff and a new memory after it
    handoff = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Session handoff summary",
        topic=["session-handoff"],
        project_id="test-proj",
    ))
    # Backdate handoff
    await pool.execute(
        "UPDATE memories SET created_at = created_at - interval '2 hours' WHERE id = $1",
        handoff.id,
    )
    # New memory after handoff
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="A new fact since last session",
        topic=["test"],
        project_id="test-proj",
    ))

    result = await build_primer(pool, project_id="test-proj", budget_tokens=3000)
    assert "changes_since" in result
    assert result["changes_since"] is not None
    assert result["changes_since"]["memories_created"] >= 1


async def test_primer_omits_changes_since_on_first_session(pool):
    """No handoff -> changes_since is None in primer output."""
    from weft.primer import build_primer

    result = await build_primer(pool, project_id="test-proj", budget_tokens=3000)
    assert "changes_since" in result
    assert result["changes_since"] is None


async def test_primer_changes_since_with_git(pool):
    """When git commits exist, they appear in changes_since."""
    from weft.primer import build_primer

    # Create a handoff
    handoff = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Session handoff",
        topic=["session-handoff"],
        project_id="test-proj",
    ))
    await pool.execute(
        "UPDATE memories SET created_at = created_at - interval '1 hour' WHERE id = $1",
        handoff.id,
    )

    mock_commits = ["abc1234 Add feature X", "def5678 Fix bug Y"]
    with patch("weft.primer_sections.changes_since.get_recent_commits", new_callable=AsyncMock, return_value=mock_commits):
        result = await build_primer(pool, project_id="test-proj", budget_tokens=3000)

    assert result["changes_since"] is not None
    assert result["changes_since"]["recent_commits"] == mock_commits


async def test_primer_changes_since_git_unavailable(pool):
    """Git unavailable -> changes_since still shows memory counts."""
    from weft.primer import build_primer

    handoff = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Session handoff",
        topic=["session-handoff"],
        project_id="test-proj",
    ))
    await pool.execute(
        "UPDATE memories SET created_at = created_at - interval '1 hour' WHERE id = $1",
        handoff.id,
    )

    with patch("weft.primer_sections.changes_since.get_recent_commits", new_callable=AsyncMock, return_value=[]):
        result = await build_primer(pool, project_id="test-proj", budget_tokens=3000)

    assert result["changes_since"] is not None
    assert result["changes_since"]["recent_commits"] == []


# ---------------------------------------------------------------------------
# Focus integration
# ---------------------------------------------------------------------------


async def test_focus_includes_changes_since(pool):
    """Focus output includes changes_since when called standalone (no pre-computed)."""
    from weft.focus import build_focus

    # Create a handoff
    handoff = await store_memory(pool, MemoryCreate(
        type=MemoryType.handoff,
        content="Session handoff",
        topic=["session-handoff"],
        project_id="test-proj",
    ))
    await pool.execute(
        "UPDATE memories SET created_at = created_at - interval '1 hour' WHERE id = $1",
        handoff.id,
    )
    # New memory
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="New fact for focus",
        topic=["test"],
        project_id="test-proj",
    ))

    async def mock_embed(text):
        return [0.0] * 1536

    with patch("weft.focus.get_recent_commits", new_callable=AsyncMock, return_value=[]):
        result = await build_focus(
            pool,
            intent="test focus",
            embedding_fn=mock_embed,
            project_id="test-proj",
            exclude_memory_ids=[],
        )

    d = result.to_dict()
    assert "changes_since" in d
    assert d["changes_since"] is not None
    assert d["changes_since"]["memories_created"] >= 1


async def test_focus_accepts_precomputed_changes(pool):
    """Focus uses pre-computed changes_since when provided."""
    from weft.focus import build_focus

    precomputed = {
        "memories_created": 5,
        "memories_archived": 2,
        "memories_revised": 1,
        "since": datetime.now(timezone.utc).isoformat(),
        "recent_commits": [],
    }

    async def mock_embed(text):
        return [0.0] * 1536

    with patch("weft.focus.get_recent_commits", new_callable=AsyncMock, return_value=[]):
        result = await build_focus(
            pool,
            intent="test focus",
            embedding_fn=mock_embed,
            project_id="test-proj",
            exclude_memory_ids=[],
            changes_since=precomputed,
        )

    d = result.to_dict()
    assert d["changes_since"] == precomputed
