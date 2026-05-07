"""Tests for list_projects_with_handoffs — cross-project handoff index."""

from __future__ import annotations

from weft.models import MemoryCreate, MemoryType
from weft.skills import _extract_handoff_summary, list_projects_with_handoffs
from weft.store import delete_memory, store_memory


async def _make_handoff(pool, project_id: str, summary: str):
    content = (
        "## Session Handoff\n\n"
        f"**Summary:** {summary}\n\n"
        "**Next Steps:** keep going."
    )
    return await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.handoff,
            content=content,
            topic=["session-handoff"],
            project_id=project_id,
        ),
    )


async def _make_fact(pool, project_id: str, content: str = "fact"):
    return await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact, content=content, project_id=project_id,
        ),
    )


async def test_happy_path_three_projects_two_with_handoffs(pool):
    """Three projects, two have handoffs, one is handoff-less. All show up."""
    await _make_handoff(pool, "proj-alpha", "Alpha summary line.")
    await _make_fact(pool, "proj-alpha", "alpha fact")
    await _make_handoff(pool, "proj-beta", "Beta summary line.")
    await _make_fact(pool, "proj-gamma", "gamma fact")  # no handoff

    result = await list_projects_with_handoffs(pool)

    assert len(result) == 3
    by_project = {r["project_id"]: r for r in result}
    assert by_project["proj-alpha"]["last_handoff_at"] is not None
    assert by_project["proj-alpha"]["last_handoff_summary"] == "Alpha summary line."
    assert by_project["proj-alpha"]["memory_count"] == 2
    assert by_project["proj-beta"]["last_handoff_at"] is not None
    assert by_project["proj-beta"]["last_handoff_summary"] == "Beta summary line."
    assert by_project["proj-beta"]["memory_count"] == 1
    assert by_project["proj-gamma"]["last_handoff_at"] is None
    assert by_project["proj-gamma"]["last_handoff_summary"] is None
    assert by_project["proj-gamma"]["memory_count"] == 1


async def test_project_with_no_handoff_returns_null_handoff(pool):
    """A project that exists but has no handoff returns last_handoff_at=None."""
    await _make_fact(pool, "proj-no-handoff")

    result = await list_projects_with_handoffs(pool)

    assert len(result) == 1
    assert result[0]["project_id"] == "proj-no-handoff"
    assert result[0]["last_handoff_at"] is None
    assert result[0]["last_handoff_summary"] is None
    assert result[0]["last_activity_at"] is not None
    assert result[0]["memory_count"] == 1


async def test_archived_handoff_does_not_count(pool):
    """A project whose only handoff has status='archived' returns last_handoff_at=None."""
    handoff = await _make_handoff(pool, "proj-archived", "Old summary.")
    await _make_fact(pool, "proj-archived")
    archived = await delete_memory(pool, handoff.id)  # soft-delete = archive
    assert archived is True

    result = await list_projects_with_handoffs(pool)

    assert len(result) == 1
    assert result[0]["project_id"] == "proj-archived"
    assert result[0]["last_handoff_at"] is None
    assert result[0]["last_handoff_summary"] is None
    assert result[0]["memory_count"] == 1  # only the active fact counts


async def test_ordering_recent_handoff_first_null_last(pool):
    """Recent-handoff projects sort first; no-handoff projects sort last."""
    older = await _make_handoff(pool, "proj-older", "Older.")
    # Backdate the older handoff so newer wins deterministically.
    await pool.execute(
        "UPDATE memories SET created_at = created_at - interval '1 day' WHERE id = $1",
        older.id,
    )
    await _make_handoff(pool, "proj-newer", "Newer.")
    await _make_fact(pool, "proj-no-handoff")

    result = await list_projects_with_handoffs(pool)

    project_order = [r["project_id"] for r in result]
    assert project_order == ["proj-newer", "proj-older", "proj-no-handoff"]
    # And the no-handoff project really has null
    assert result[2]["last_handoff_at"] is None


async def test_extract_handoff_summary_truncates_long():
    """Summaries longer than max_chars are truncated with an ellipsis."""
    long_summary = "x" * 500
    body = f"## Session Handoff\n\n**Summary:** {long_summary}\n\n**Next:** y"
    out = _extract_handoff_summary(body, max_chars=50)
    assert out is not None
    assert len(out) == 50
    assert out.endswith("…")


async def test_extract_handoff_summary_returns_none_when_pattern_missing():
    """Legacy or malformed handoff content returns None rather than crashing."""
    assert _extract_handoff_summary("") is None
    assert _extract_handoff_summary(None) is None
    assert _extract_handoff_summary("just some text, no marker") is None
    # Empty Summary block returns None too
    assert _extract_handoff_summary("**Summary:**   \n\n**Next:** x") is None


async def test_global_memories_excluded(pool):
    """Memories with project_id=None don't surface as their own project entry."""
    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="global", project_id=None),
    )
    await _make_handoff(pool, "proj-real", "Real project summary.")

    result = await list_projects_with_handoffs(pool)

    project_ids = [r["project_id"] for r in result]
    assert project_ids == ["proj-real"]
    assert None not in project_ids
