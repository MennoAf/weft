"""Tests for session-scoped memory access tracking and implicit usefulness signals."""

from __future__ import annotations

import pytest

from weft.auth import current_caller_mode, current_user_id
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.session_tracking import (
    IMPLICIT_ACCESS_BOOST,
    USEFULNESS_CAP,
    _session_id,
    boost_session_memories,
    boost_session_turns,
    get_session_id,
    get_session_memory_ids,
    log_memory_access,
    log_turn_access,
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


# --- Read-side audit log (mig 41) ---


@pytest.mark.asyncio
async def test_log_access_stamps_reader_user_id_from_contextvar(pool):
    """The audit row pulls reader_user_id from current_user_id contextvar."""
    mem = await _create_memory(pool, "Audit reader stamp")
    uid_token = current_user_id.set("user-incident-A")
    try:
        await log_memory_access(
            pool, [mem.id], "recall", session_id="ses-audit-A",
        )
    finally:
        current_user_id.reset(uid_token)

    row = await pool.fetchrow(
        """
        SELECT reader_user_id, reader_caller_mode
        FROM memory_access_log
        WHERE session_id = $1 AND memory_id = $2
        """,
        "ses-audit-A", mem.id,
    )
    assert row["reader_user_id"] == "user-incident-A"
    # No caller_mode set → defaults to supervisor (per get_caller_mode())
    assert row["reader_caller_mode"] == "supervisor"


@pytest.mark.asyncio
async def test_log_access_stamps_caller_mode_agent(pool):
    """An agent-mode caller's reads are tagged so a supervisor can filter for
    agent-mode reads of a confirmed-poisoned memory during incident response."""
    mem = await _create_memory(pool, "Audit agent-mode read")
    uid_token = current_user_id.set("user-incident-B")
    mode_token = current_caller_mode.set("agent")
    try:
        await log_memory_access(
            pool, [mem.id], "recall", session_id="ses-audit-agent",
            retrieval_mode="code",
        )
    finally:
        current_caller_mode.reset(mode_token)
        current_user_id.reset(uid_token)

    row = await pool.fetchrow(
        """
        SELECT reader_user_id, reader_caller_mode, retrieval_mode
        FROM memory_access_log
        WHERE session_id = $1 AND memory_id = $2
        """,
        "ses-audit-agent", mem.id,
    )
    assert row["reader_user_id"] == "user-incident-B"
    assert row["reader_caller_mode"] == "agent"
    assert row["retrieval_mode"] == "code"


@pytest.mark.asyncio
async def test_log_access_records_retrieval_mode(pool):
    """retrieval_mode flows through to the row so an audit query can
    distinguish 'face' reads (Jason's queries) from 'code' reads (agent
    in repo context)."""
    mem = await _create_memory(pool, "retrieval_mode trace")
    await log_memory_access(
        pool, [mem.id], "recall", session_id="ses-mode-face",
        retrieval_mode="face",
    )
    mode = await pool.fetchval(
        """
        SELECT retrieval_mode FROM memory_access_log
        WHERE session_id = $1 AND memory_id = $2
        """,
        "ses-mode-face", mem.id,
    )
    assert mode == "face"


@pytest.mark.asyncio
async def test_log_access_unauthenticated_records_null_user(pool):
    """If no auth context is set, reader_user_id is NULL — the row still
    records the access for anomaly detection (a read with no caller_mode
    binding is itself a signal)."""
    mem = await _create_memory(pool, "No auth read")
    # Explicitly clear auth context for this test
    uid_token = current_user_id.set(None)
    try:
        await log_memory_access(
            pool, [mem.id], "recall", session_id="ses-no-auth",
        )
    finally:
        current_user_id.reset(uid_token)

    row = await pool.fetchrow(
        """
        SELECT reader_user_id, reader_caller_mode
        FROM memory_access_log
        WHERE session_id = $1 AND memory_id = $2
        """,
        "ses-no-auth", mem.id,
    )
    assert row["reader_user_id"] is None
    # caller_mode still gets a value (defaults to supervisor) because the
    # contextvar always has a default — there's no "unset" state.
    assert row["reader_caller_mode"] == "supervisor"


@pytest.mark.asyncio
async def test_log_access_dedup_keeps_first_caller_metadata(pool):
    """Session-level dedup (existing PK semantics) means the FIRST caller's
    metadata wins. This is intentional — for a poisoned memory, the first
    reader is the one who pulled it into context. Subsequent reads in the
    same session add no forensic value."""
    mem = await _create_memory(pool, "Dedup audit metadata")
    # First read: supervisor mode
    uid_token = current_user_id.set("user-first")
    try:
        await log_memory_access(
            pool, [mem.id], "recall", session_id="ses-dedup-meta",
        )
    finally:
        current_user_id.reset(uid_token)

    # Second read from same session, different metadata — should NOT overwrite
    uid_token = current_user_id.set("user-second")
    mode_token = current_caller_mode.set("agent")
    try:
        await log_memory_access(
            pool, [mem.id], "context", session_id="ses-dedup-meta",
        )
    finally:
        current_caller_mode.reset(mode_token)
        current_user_id.reset(uid_token)

    row = await pool.fetchrow(
        """
        SELECT reader_user_id, reader_caller_mode, tool_name
        FROM memory_access_log
        WHERE session_id = $1 AND memory_id = $2
        """,
        "ses-dedup-meta", mem.id,
    )
    # First write wins via ON CONFLICT DO NOTHING
    assert row["reader_user_id"] == "user-first"
    assert row["reader_caller_mode"] == "supervisor"
    assert row["tool_name"] == "recall"


@pytest.mark.asyncio
async def test_audit_query_who_read_memory(pool):
    """Incident-response query: given a poisoned memory, find every
    (user_id, session_id, accessed_at) that read it. Uses the
    idx_access_log_memory_recent index added in mig 41."""
    mem = await _create_memory(pool, "Hot poisoned memory")

    # Multiple users, multiple sessions read this memory
    for uid, sid in [
        ("user-X", "ses-X1"),
        ("user-Y", "ses-Y1"),
        ("user-X", "ses-X2"),
    ]:
        token = current_user_id.set(uid)
        try:
            await log_memory_access(
                pool, [mem.id], "recall", session_id=sid,
            )
        finally:
            current_user_id.reset(token)

    rows = await pool.fetch(
        """
        SELECT reader_user_id, session_id, accessed_at
        FROM memory_access_log
        WHERE memory_id = $1
        ORDER BY accessed_at DESC
        """,
        mem.id,
    )
    assert len(rows) == 3
    readers = {(r["reader_user_id"], r["session_id"]) for r in rows}
    assert readers == {
        ("user-X", "ses-X1"),
        ("user-Y", "ses-Y1"),
        ("user-X", "ses-X2"),
    }


@pytest.mark.asyncio
async def test_audit_query_what_user_read(pool):
    """Incident-response query: given a user_id, find everything they read
    (across sessions) ordered by recency. Uses the
    idx_access_log_user_recent partial index."""
    mems = [await _create_memory(pool, f"User-X read {i}") for i in range(3)]

    uid_token = current_user_id.set("user-trace-X")
    try:
        await log_memory_access(
            pool, [m.id for m in mems], "recall",
            session_id="ses-trace-X",
        )
    finally:
        current_user_id.reset(uid_token)

    rows = await pool.fetch(
        """
        SELECT memory_id FROM memory_access_log
        WHERE reader_user_id = $1
        ORDER BY accessed_at DESC
        """,
        "user-trace-X",
    )
    assert len(rows) == 3
    assert {r["memory_id"] for r in rows} == {m.id for m in mems}


@pytest.mark.asyncio
async def test_caller_mode_check_constraint_rejects_invalid(pool):
    """The CHECK constraint on reader_caller_mode rejects anything that
    isn't supervisor / agent / NULL — defends against future mig drift."""
    mem = await _create_memory(pool, "Constraint test")
    import asyncpg
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await pool.execute(
            """
            INSERT INTO memory_access_log
                (session_id, memory_id, tool_name, reader_caller_mode)
            VALUES ($1, $2, $3, $4)
            """,
            "ses-bad-mode", mem.id, "recall", "root",
        )


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


# --- Turn-tier session tracking (P1.A2) ---
#
# Mirrors the belief-tier suite above one-for-one against the turn-tier
# substrate landed in P1.A1 (mig 46): turn_access_log table +
# episode_turns.{usefulness_score, usefulness_count, last_boosted_at}.
# Boost loop is intentionally symmetric with boost_session_memories — the
# only deliberate asymmetry is that we leave usefulness_count alone here
# too, because the belief tier reserves count bumps for explicit feedback.


async def _create_turn(pool, content: str):
    """Create a test episode + turn and return the EpisodeTurn.

    Each turn lives in its own episode so cross-test cleanup is trivial
    and the turn-level tests don't accidentally couple through shared
    episode state.
    """
    from weft.episode_turns import append_turn
    from weft.episodes import create_episode
    from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole

    ep = await create_episode(pool, EpisodeCreate(title=f"turn-test: {content[:40]}"))
    return await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content=content,
        ),
    )


# --- log_turn_access ---


@pytest.mark.asyncio
async def test_log_turn_access_empty_list_is_noop(pool):
    """Empty turn_ids list should not write any rows."""
    await log_turn_access(
        pool, [], tool_name="recall", session_id="ses-turn-empty",
    )
    count = await pool.fetchval(
        "SELECT count(*) FROM turn_access_log WHERE session_id = $1",
        "ses-turn-empty",
    )
    assert count == 0


@pytest.mark.asyncio
async def test_log_turn_access_inserts_rows(pool):
    """Log access inserts one row per turn ID."""
    t1 = await _create_turn(pool, "First turn")
    t2 = await _create_turn(pool, "Second turn")

    await log_turn_access(
        pool, [t1.id, t2.id],
        tool_name="recall", session_id="ses-turn-insert",
    )

    rows = await pool.fetch(
        "SELECT turn_id, tool_name FROM turn_access_log WHERE session_id = $1",
        "ses-turn-insert",
    )
    assert len(rows) == 2
    ids = {r["turn_id"] for r in rows}
    assert ids == {t1.id, t2.id}
    assert all(r["tool_name"] == "recall" for r in rows)


@pytest.mark.asyncio
async def test_log_turn_access_deduplicates_within_session(pool):
    """Same turn accessed twice in one session → one row."""
    t = await _create_turn(pool, "Dedup turn test")

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-dedup",
    )
    await log_turn_access(
        pool, [t.id], tool_name="context", session_id="ses-turn-dedup",
    )

    count = await pool.fetchval(
        "SELECT count(*) FROM turn_access_log WHERE session_id = $1",
        "ses-turn-dedup",
    )
    assert count == 1  # ON CONFLICT DO NOTHING


@pytest.mark.asyncio
async def test_log_turn_access_different_sessions_both_recorded(pool):
    """Same turn in different sessions → separate rows."""
    t = await _create_turn(pool, "Multi-session turn")

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-a",
    )
    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-b",
    )

    count = await pool.fetchval(
        "SELECT count(*) FROM turn_access_log WHERE turn_id = $1",
        t.id,
    )
    assert count == 2


@pytest.mark.asyncio
async def test_log_turn_access_db_failure_does_not_raise(pool):
    """DB errors are caught and logged, not propagated."""
    await pool.close()
    # Should not raise even with a closed pool.
    await log_turn_access(
        pool, ["fake-turn"],
        tool_name="recall", session_id="ses-turn-fail",
    )


@pytest.mark.asyncio
async def test_log_turn_access_tool_name_threaded_through(pool):
    """The tool_name kwarg lands verbatim on the row — supervisor needs to
    distinguish recall reads from temporal_anchor reads from prime reads."""
    t = await _create_turn(pool, "Tool-name passthrough")
    await log_turn_access(
        pool, [t.id], tool_name="temporal_anchor",
        session_id="ses-turn-tool",
    )
    tool_name = await pool.fetchval(
        """
        SELECT tool_name FROM turn_access_log
        WHERE session_id = $1 AND turn_id = $2
        """,
        "ses-turn-tool", t.id,
    )
    assert tool_name == "temporal_anchor"


# --- boost_session_turns ---


@pytest.mark.asyncio
async def test_boost_turns_updates_usefulness_score(pool):
    """Boost increases usefulness_score by the configured amount."""
    t = await _create_turn(pool, "Boost me")
    original_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-boost",
    )
    result = await boost_session_turns(pool, session_id="ses-turn-boost")

    assert result["boosted"] == 1
    assert result["boost"] == IMPLICIT_ACCESS_BOOST

    updated_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )
    assert updated_score == pytest.approx(
        original_score + IMPLICIT_ACCESS_BOOST
    )


@pytest.mark.asyncio
async def test_boost_turns_sets_last_boosted_at(pool):
    """Boost stamps last_boosted_at — feeds the time-decay term."""
    t = await _create_turn(pool, "last_boosted_at trace")
    pre = await pool.fetchval(
        "SELECT last_boosted_at FROM episode_turns WHERE id = $1", t.id,
    )
    assert pre is None  # default per migration

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-stamp",
    )
    await boost_session_turns(pool, session_id="ses-turn-stamp")

    post = await pool.fetchval(
        "SELECT last_boosted_at FROM episode_turns WHERE id = $1", t.id,
    )
    assert post is not None


@pytest.mark.asyncio
async def test_boost_turns_does_not_increment_usefulness_count(pool):
    """Boost loop leaves usefulness_count alone — count is reserved for
    explicit-feedback events, mirroring boost_session_memories which also
    leaves the count untouched in the implicit-access path."""
    t = await _create_turn(pool, "Count untouched")
    pre_count = await pool.fetchval(
        "SELECT usefulness_count FROM episode_turns WHERE id = $1", t.id,
    )

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-count",
    )
    await boost_session_turns(pool, session_id="ses-turn-count")

    post_count = await pool.fetchval(
        "SELECT usefulness_count FROM episode_turns WHERE id = $1", t.id,
    )
    assert post_count == pre_count


@pytest.mark.asyncio
async def test_boost_turns_deduplicates_across_multiple_accesses(pool):
    """Turn accessed 3 times in session gets ONE boost, not three."""
    t = await _create_turn(pool, "Dedup boost turn")
    original_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-3x",
    )
    await log_turn_access(
        pool, [t.id], tool_name="context", session_id="ses-turn-3x",
    )
    await log_turn_access(
        pool, [t.id], tool_name="prime", session_id="ses-turn-3x",
    )

    await boost_session_turns(pool, session_id="ses-turn-3x")

    updated_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )
    assert updated_score == pytest.approx(
        original_score + IMPLICIT_ACCESS_BOOST
    )


@pytest.mark.asyncio
async def test_boost_turns_caps_at_maximum(pool):
    """Usefulness score never exceeds USEFULNESS_CAP."""
    t = await _create_turn(pool, "Turn cap test")

    await pool.execute(
        "UPDATE episode_turns SET usefulness_score = $1 WHERE id = $2",
        USEFULNESS_CAP - 0.005,
        t.id,
    )

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-cap",
    )
    await boost_session_turns(pool, session_id="ses-turn-cap")

    updated_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )
    assert updated_score == pytest.approx(USEFULNESS_CAP)


@pytest.mark.asyncio
async def test_boost_turns_empty_session_is_noop(pool):
    """Session with no access log entries → boost 0, no error."""
    result = await boost_session_turns(pool, session_id="ses-turn-ghost")
    assert result["boosted"] == 0
    assert result["session_id"] == "ses-turn-ghost"


@pytest.mark.asyncio
async def test_boost_turns_skips_deleted_turns(pool):
    """If a turn was deleted between access and boost, FK cascade removes
    the access log row — boost reports zero, no error."""
    t = await _create_turn(pool, "Will be deleted turn")
    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-del",
    )

    # Hard-delete the turn (CASCADE deletes access log row too)
    await pool.execute("DELETE FROM episode_turns WHERE id = $1", t.id)

    result = await boost_session_turns(pool, session_id="ses-turn-del")
    assert result["boosted"] == 0


@pytest.mark.asyncio
async def test_boost_turns_multiple_turns(pool):
    """Multiple turns in session all get boosted."""
    turns = [await _create_turn(pool, f"Turn multi {i}") for i in range(3)]
    original = {
        t.id: await pool.fetchval(
            "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
        )
        for t in turns
    }

    await log_turn_access(
        pool, [t.id for t in turns],
        tool_name="prime", session_id="ses-turn-multi",
    )
    result = await boost_session_turns(pool, session_id="ses-turn-multi")

    assert result["boosted"] == 3
    for t in turns:
        updated = await pool.fetchval(
            "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
        )
        assert updated == pytest.approx(
            original[t.id] + IMPLICIT_ACCESS_BOOST
        )


@pytest.mark.asyncio
async def test_boost_turns_second_session_stacks(pool):
    """Two sessions boosting the same turn → score increases twice."""
    t = await _create_turn(pool, "Double session turn")
    original_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-s1",
    )
    await boost_session_turns(pool, session_id="ses-turn-s1")

    await log_turn_access(
        pool, [t.id], tool_name="recall", session_id="ses-turn-s2",
    )
    await boost_session_turns(pool, session_id="ses-turn-s2")

    updated_score = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t.id,
    )
    assert updated_score == pytest.approx(
        original_score + 2 * IMPLICIT_ACCESS_BOOST
    )


@pytest.mark.asyncio
async def test_boost_turns_does_not_cross_contaminate_sessions(pool):
    """A turn accessed in session A does NOT get boosted by session B's
    end-of-session call. Mirrors the per-session dedup contract."""
    t_a = await _create_turn(pool, "Session A turn")
    t_b = await _create_turn(pool, "Session B turn")

    await log_turn_access(
        pool, [t_a.id], tool_name="recall", session_id="ses-turn-A",
    )
    await log_turn_access(
        pool, [t_b.id], tool_name="recall", session_id="ses-turn-B",
    )

    # End session A only.
    result = await boost_session_turns(pool, session_id="ses-turn-A")
    assert result["boosted"] == 1

    score_a = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t_a.id,
    )
    score_b = await pool.fetchval(
        "SELECT usefulness_score FROM episode_turns WHERE id = $1", t_b.id,
    )
    # A boosted, B untouched.
    assert score_a == pytest.approx(0.7 + IMPLICIT_ACCESS_BOOST)
    assert score_b == pytest.approx(0.7)
    last_boosted_b = await pool.fetchval(
        "SELECT last_boosted_at FROM episode_turns WHERE id = $1", t_b.id,
    )
    assert last_boosted_b is None
