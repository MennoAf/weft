"""Tests for session-scoped memory access tracking and implicit usefulness signals."""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.session_tracking import (
    IMPLICIT_ACCESS_BOOST,
    USEFULNESS_CAP,
    _session_id,
    boost_session_memories,
    get_session_id,
    get_session_memory_ids,
    log_memory_access,
    set_session_id,
)
from weft.store import get_memory, store_memory


# --- Session ID management ---


def test_get_session_id_creates_on_first_call():
    """First call creates a session ID with ses- prefix."""
    token = _session_id.set("will-be-reset")
    _session_id.reset(token)
    # After reset, LookupError → generates new ID
    sid = get_session_id()
    assert sid.startswith("ses-")
    assert len(sid) == 16  # "ses-" + 12 hex chars


def test_get_session_id_stable_within_context():
    """Same context returns the same session ID."""
    token = _session_id.set("test-stable")
    try:
        assert get_session_id() == "test-stable"
        assert get_session_id() == "test-stable"
    finally:
        _session_id.reset(token)


def test_set_session_id():
    """Explicitly set session ID is returned by get."""
    set_session_id("custom-session-42")
    assert get_session_id() == "custom-session-42"


# --- log_memory_access ---


@pytest.mark.asyncio
async def test_log_access_empty_list_is_noop(pool):
    """Empty memory_ids list should not write any rows."""
    await log_memory_access(pool, [], "recall", session_id="ses-empty")
    count = await pool.fetchval(
        "SELECT count(*) FROM memory_access_log WHERE session_id = $1",
        "ses-empty",
    )
    assert count == 0


@pytest.mark.asyncio
async def test_log_access_inserts_rows(pool):
    """Log access inserts one row per memory ID."""
    mem1 = await _create_memory(pool, "First memory")
    mem2 = await _create_memory(pool, "Second memory")

    await log_memory_access(
        pool, [mem1.id, mem2.id], "recall", session_id="ses-insert",
    )

    rows = await pool.fetch(
        "SELECT memory_id, tool_name FROM memory_access_log WHERE session_id = $1",
        "ses-insert",
    )
    assert len(rows) == 2
    ids = {r["memory_id"] for r in rows}
    assert ids == {mem1.id, mem2.id}
    assert all(r["tool_name"] == "recall" for r in rows)


@pytest.mark.asyncio
async def test_log_access_deduplicates_within_session(pool):
    """Same memory accessed twice in one session → one row."""
    mem = await _create_memory(pool, "Dedup test")

    await log_memory_access(pool, [mem.id], "recall", session_id="ses-dedup")
    await log_memory_access(pool, [mem.id], "context", session_id="ses-dedup")

    count = await pool.fetchval(
        "SELECT count(*) FROM memory_access_log WHERE session_id = $1",
        "ses-dedup",
    )
    assert count == 1  # ON CONFLICT DO NOTHING


@pytest.mark.asyncio
async def test_log_access_different_sessions_both_recorded(pool):
    """Same memory in different sessions → separate rows."""
    mem = await _create_memory(pool, "Multi-session test")

    await log_memory_access(pool, [mem.id], "recall", session_id="ses-a")
    await log_memory_access(pool, [mem.id], "recall", session_id="ses-b")

    count = await pool.fetchval(
        "SELECT count(*) FROM memory_access_log WHERE memory_id = $1",
        mem.id,
    )
    assert count == 2


@pytest.mark.asyncio
async def test_log_access_db_failure_does_not_raise(pool):
    """DB errors are caught and logged, not propagated."""
    # Close the pool to simulate DB failure
    await pool.close()
    # Should not raise
    await log_memory_access(pool, ["fake-id"], "recall", session_id="ses-fail")


# --- boost_session_memories ---


@pytest.mark.asyncio
async def test_boost_updates_usefulness_score(pool):
    """Boost increases usefulness_score by the configured amount."""
    mem = await _create_memory(pool, "Boost me")
    original_score = mem.usefulness_score

    await log_memory_access(pool, [mem.id], "recall", session_id="ses-boost")
    result = await boost_session_memories(pool, session_id="ses-boost")

    assert result["boosted"] == 1
    assert result["boost"] == IMPLICIT_ACCESS_BOOST

    updated = await get_memory(pool, mem.id)
    assert updated.usefulness_score == pytest.approx(
        original_score + IMPLICIT_ACCESS_BOOST
    )


@pytest.mark.asyncio
async def test_boost_deduplicates_across_multiple_accesses(pool):
    """Memory accessed 3 times in session gets ONE boost, not three."""
    mem = await _create_memory(pool, "Dedup boost")
    original_score = mem.usefulness_score

    await log_memory_access(pool, [mem.id], "recall", session_id="ses-3x")
    await log_memory_access(pool, [mem.id], "context", session_id="ses-3x")
    await log_memory_access(pool, [mem.id], "prime", session_id="ses-3x")

    await boost_session_memories(pool, session_id="ses-3x")

    updated = await get_memory(pool, mem.id)
    # Only one boost applied, not three
    assert updated.usefulness_score == pytest.approx(
        original_score + IMPLICIT_ACCESS_BOOST
    )


@pytest.mark.asyncio
async def test_boost_caps_at_maximum(pool):
    """Usefulness score never exceeds USEFULNESS_CAP."""
    mem = await _create_memory(pool, "Cap test")

    # Set score just below cap
    await pool.execute(
        "UPDATE memories SET usefulness_score = $1 WHERE id = $2",
        USEFULNESS_CAP - 0.005,
        mem.id,
    )

    await log_memory_access(pool, [mem.id], "recall", session_id="ses-cap")
    await boost_session_memories(pool, session_id="ses-cap")

    updated = await get_memory(pool, mem.id)
    assert updated.usefulness_score == pytest.approx(USEFULNESS_CAP)


@pytest.mark.asyncio
async def test_boost_empty_session_is_noop(pool):
    """Session with no access log entries → boost 0, no error."""
    result = await boost_session_memories(pool, session_id="ses-ghost")
    assert result["boosted"] == 0
    assert result["session_id"] == "ses-ghost"


@pytest.mark.asyncio
async def test_boost_skips_deleted_memories(pool):
    """If a memory was deleted between access and boost, skip it silently."""
    mem = await _create_memory(pool, "Will be deleted")
    await log_memory_access(pool, [mem.id], "recall", session_id="ses-del")

    # Hard-delete the memory (CASCADE deletes access log row too)
    await pool.execute("DELETE FROM memories WHERE id = $1", mem.id)

    result = await boost_session_memories(pool, session_id="ses-del")
    assert result["boosted"] == 0  # Cascaded delete removed access log row


@pytest.mark.asyncio
async def test_boost_multiple_memories(pool):
    """Multiple memories in session all get boosted."""
    mems = [await _create_memory(pool, f"Multi {i}") for i in range(3)]
    original_scores = {m.id: m.usefulness_score for m in mems}

    await log_memory_access(
        pool, [m.id for m in mems], "prime", session_id="ses-multi",
    )
    result = await boost_session_memories(pool, session_id="ses-multi")

    assert result["boosted"] == 3
    for m in mems:
        updated = await get_memory(pool, m.id)
        assert updated.usefulness_score == pytest.approx(
            original_scores[m.id] + IMPLICIT_ACCESS_BOOST
        )


@pytest.mark.asyncio
async def test_boost_second_session_stacks(pool):
    """Two sessions boosting the same memory → score increases twice."""
    mem = await _create_memory(pool, "Double session")
    original_score = mem.usefulness_score

    # Session 1
    await log_memory_access(pool, [mem.id], "recall", session_id="ses-s1")
    await boost_session_memories(pool, session_id="ses-s1")

    # Session 2
    await log_memory_access(pool, [mem.id], "recall", session_id="ses-s2")
    await boost_session_memories(pool, session_id="ses-s2")

    updated = await get_memory(pool, mem.id)
    assert updated.usefulness_score == pytest.approx(
        original_score + 2 * IMPLICIT_ACCESS_BOOST
    )


# --- get_session_memory_ids ---


@pytest.mark.asyncio
async def test_get_session_memory_ids(pool):
    """Returns distinct memory IDs for a session."""
    mem1 = await _create_memory(pool, "Session mem 1")
    mem2 = await _create_memory(pool, "Session mem 2")

    await log_memory_access(
        pool, [mem1.id, mem2.id], "recall", session_id="ses-ids",
    )
    ids = await get_session_memory_ids(pool, session_id="ses-ids")
    assert set(ids) == {mem1.id, mem2.id}


@pytest.mark.asyncio
async def test_get_session_memory_ids_empty_session(pool):
    """Empty session returns empty list."""
    ids = await get_session_memory_ids(pool, session_id="ses-none")
    assert ids == []


# --- Helper ---


async def _create_memory(pool, content: str):
    """Create a test memory and return it."""
    create = MemoryCreate(
        type=MemoryType.fact,
        content=content,
        topic=["test"],
        source=MemorySource.conversation,
    )
    return await store_memory(pool, create)
