"""Tests for the working memory primer section (open episodes)."""

from __future__ import annotations

from datetime import datetime, timezone

from weft.episodes import add_memory_to_episode, close_episode, create_episode
from weft.models import EpisodeCreate, MemoryCreate, MemoryType
from weft.primer import build_primer
from weft.primer_sections.context import PrimerContext, SectionResult
from weft.primer_sections.working_memory import build_working_memory_section
from weft.store import store_memory


# --- Helpers ---

async def _make_episode(pool, title="test episode", **kwargs):
    return await create_episode(pool, EpisodeCreate(title=title, **kwargs))


async def _make_memory(pool, content="test memory"):
    return await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content=content,
    ))


def _ctx(pool, **overrides):
    """Build a minimal PrimerContext for section-level tests."""
    defaults = dict(
        user_id="",
        project_id=None,
        agent_id=None,
        pool=pool,
        budget_tokens=2400,
        query=None,
        query_vec=None,
        disclosure="full",
        mode=None,
        now=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return PrimerContext(**defaults)


# --- Section builder tests ---


async def test_working_memory_empty_when_no_episodes(pool):
    """Section is skipped when there are no open episodes."""
    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert result.skipped is True
    assert result.items == []
    assert result.tokens_used == 0


async def test_working_memory_shows_open_episodes(pool):
    """Open episodes appear in the working memory section."""
    ep = await _make_episode(pool, title="Debug auth flow")
    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert result.skipped is False
    assert len(result.items) == 1
    assert result.items[0]["id"] == ep.id
    assert result.items[0]["title"] == "Debug auth flow"
    assert result.items[0]["memory_count"] == 0


async def test_working_memory_includes_linked_memories(pool):
    """Episodes show their linked memories."""
    ep = await _make_episode(pool, title="Investigate crash")
    mem = await _make_memory(pool, content="Stack trace shows null pointer in auth module")
    await add_memory_to_episode(pool, ep.id, mem.id)

    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert result.items[0]["memory_count"] == 1
    assert len(result.items[0]["memories"]) == 1
    assert result.items[0]["memories"][0]["id"] == mem.id


async def test_working_memory_excludes_closed_episodes(pool):
    """Closed episodes do not appear in working memory."""
    ep = await _make_episode(pool, title="Old episode")
    await close_episode(pool, ep.id, summary="Done")

    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert result.skipped is True
    assert result.items == []


async def test_working_memory_includes_summary(pool):
    """Episode summary is included when present."""
    ep = await _make_episode(pool, title="Plan sprint", summary="Sprint 5 planning")
    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert result.items[0]["summary"] == "Sprint 5 planning"


async def test_working_memory_no_summary_key_when_absent(pool):
    """Summary key is omitted when episode has no summary."""
    await _make_episode(pool, title="Quick fix")
    ctx = _ctx(pool)
    result = await build_working_memory_section(ctx)
    assert "summary" not in result.items[0]


async def test_working_memory_project_scoped(pool):
    """Only episodes matching the project scope appear."""
    await _make_episode(pool, title="Project A work", project_id="proj-a")
    await _make_episode(pool, title="Project B work", project_id="proj-b")

    ctx = _ctx(pool, project_id="proj-a")
    result = await build_working_memory_section(ctx)
    titles = [item["title"] for item in result.items]
    assert "Project A work" in titles
    # proj-b should not appear (list_episodes uses OR-NULL scoping,
    # so only proj-a and NULL project_id episodes show up)
    assert "Project B work" not in titles


# --- Full primer integration tests ---


async def test_primer_includes_working_memory_when_episodes_exist(pool):
    """build_primer includes working_memory section when open episodes exist."""
    await _make_episode(pool, title="Active investigation")
    result = await build_primer(pool, disclosure="full")
    assert "working_memory" in result
    assert len(result["working_memory"]) == 1
    assert result["working_memory"][0]["title"] == "Active investigation"


async def test_primer_omits_working_memory_when_no_episodes(pool):
    """build_primer omits working_memory key when no open episodes exist."""
    result = await build_primer(pool, disclosure="full")
    assert "working_memory" not in result


async def test_primer_progressive_defers_working_memory(pool):
    """In progressive mode, working_memory is deferred with a count."""
    await _make_episode(pool, title="Active work")
    result = await build_primer(pool, disclosure="progressive")
    assert "working_memory" in result
    wm = result["working_memory"]
    assert wm["deferred"] is True
    assert wm["count"] == 1


async def test_primer_progressive_omits_working_memory_when_empty(pool):
    """In progressive mode, working_memory is omitted when no open episodes."""
    result = await build_primer(pool, disclosure="progressive")
    assert "working_memory" not in result
