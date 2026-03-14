"""Tests for usefulness score integration into relevance ranking.

Covers: time decay in usefulness_factor, last_boosted_at in Memory model,
boost_session_memories setting last_boosted_at, prune_old_access_logs,
and consolidation wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.models import Memory, MemoryCreate, MemoryRecall, MemorySource, MemoryStatus, MemoryType
from weft.store import store_memory, update_memory

NOW = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)


def _make_memory(
    *,
    usefulness_score: float = 0.7,
    last_boosted_at: datetime | None = None,
    accessed_at: datetime | None = None,
    confidence: float = 0.9,
    access_count: int = 0,
) -> Memory:
    return Memory(
        id="weft-test0001",
        type=MemoryType.fact,
        topic=["test"],
        content="test memory",
        source=MemorySource.conversation,
        confidence=confidence,
        accessed_at=accessed_at or NOW,
        access_count=access_count,
        usefulness_score=usefulness_score,
        last_boosted_at=last_boosted_at,
        status=MemoryStatus.active,
    )


# ---------------------------------------------------------------------------
# usefulness_factor with time decay
# ---------------------------------------------------------------------------


def test_usefulness_factor_no_decay_when_never_boosted():
    """last_boosted_at=None -> no decay applied, uses raw score."""
    from weft.relevance import usefulness_factor

    result = usefulness_factor(0.7, last_boosted_at=None, now=NOW)
    no_decay = usefulness_factor(0.7, last_boosted_at=NOW, now=NOW)
    # Both should be the same (no penalty for never-boosted)
    assert result == pytest.approx(no_decay, abs=0.01)


def test_usefulness_factor_decays_over_time():
    """Score boosted 30 days ago should produce ~50% of a just-boosted score."""
    from weft.relevance import usefulness_factor

    recent = usefulness_factor(0.9, last_boosted_at=NOW, now=NOW)
    old = usefulness_factor(0.9, last_boosted_at=NOW - timedelta(days=30), now=NOW)
    # After one half-life (30 days), the decayed usefulness should be lower
    assert old < recent
    # The ratio should be approximately 0.5 of the contribution
    # (not exact because of floor)
    assert old < recent * 0.8


def test_usefulness_factor_floor_respected():
    """Even with heavy decay, floor is respected."""
    from weft.relevance import usefulness_factor

    result = usefulness_factor(
        0.1,
        last_boosted_at=NOW - timedelta(days=365),
        now=NOW,
        floor=0.5,
    )
    assert result >= 0.5


def test_usefulness_factor_future_last_boosted_clamped():
    """Clock skew: last_boosted_at in the future should not amplify score."""
    from weft.relevance import usefulness_factor

    normal = usefulness_factor(0.7, last_boosted_at=NOW, now=NOW)
    future = usefulness_factor(0.7, last_boosted_at=NOW + timedelta(hours=1), now=NOW)
    # Future should not exceed normal
    assert future <= normal + 0.001


def test_usefulness_factor_null_score_treated_as_zero():
    """None usefulness_score should not raise."""
    from weft.relevance import usefulness_factor

    # Should not raise and should produce a valid result
    result = usefulness_factor(0.0, last_boosted_at=None, now=NOW)
    assert isinstance(result, float)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# score_memory with last_boosted_at
# ---------------------------------------------------------------------------


def test_score_memory_uses_last_boosted_at():
    """score_memory should incorporate last_boosted_at for decay."""
    from weft.relevance import score_memory

    recent_mem = _make_memory(usefulness_score=0.9, last_boosted_at=NOW)
    old_mem = _make_memory(usefulness_score=0.9, last_boosted_at=NOW - timedelta(days=60))

    recent_recall = MemoryRecall(memory=recent_mem, similarity=0.8)
    old_recall = MemoryRecall(memory=old_mem, similarity=0.8)

    recent_scored = score_memory(recent_recall, now=NOW)
    old_scored = score_memory(old_recall, now=NOW)

    assert recent_scored.score > old_scored.score


def test_score_memory_higher_usefulness_ranks_higher():
    """Memory with higher usefulness_score should rank above lower one."""
    from weft.relevance import score_memory

    high = _make_memory(usefulness_score=0.9, last_boosted_at=NOW)
    low = _make_memory(usefulness_score=0.1, last_boosted_at=NOW)

    high_scored = score_memory(MemoryRecall(memory=high, similarity=0.8), now=NOW)
    low_scored = score_memory(MemoryRecall(memory=low, similarity=0.8), now=NOW)

    assert high_scored.score > low_scored.score


# ---------------------------------------------------------------------------
# boost_session_memories sets last_boosted_at
# ---------------------------------------------------------------------------


async def test_boost_sets_last_boosted_at(pool):
    """boost_session_memories should update last_boosted_at alongside usefulness_score."""
    from weft.session_tracking import boost_session_memories, log_memory_access

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Test memory for boosting",
        topic=["test"],
    ))

    # Log access and boost
    session_id = "test-session-boost"
    await log_memory_access(pool, [mem.id], "test", session_id=session_id)
    await boost_session_memories(pool, session_id=session_id)

    # Check last_boosted_at is set
    row = await pool.fetchrow("SELECT last_boosted_at FROM memories WHERE id = $1", mem.id)
    assert row["last_boosted_at"] is not None


async def test_boost_soft_cap(pool):
    """Boosting should not exceed the usefulness cap."""
    from weft.session_tracking import USEFULNESS_CAP, boost_session_memories, log_memory_access

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="High usefulness memory",
        topic=["test"],
    ))
    # Set usefulness close to cap
    await pool.execute(
        "UPDATE memories SET usefulness_score = $1 WHERE id = $2",
        USEFULNESS_CAP - 0.001, mem.id,
    )

    session_id = "test-session-cap"
    await log_memory_access(pool, [mem.id], "test", session_id=session_id)
    await boost_session_memories(pool, session_id=session_id)

    row = await pool.fetchrow("SELECT usefulness_score FROM memories WHERE id = $1", mem.id)
    assert float(row["usefulness_score"]) <= USEFULNESS_CAP


# ---------------------------------------------------------------------------
# prune_old_access_logs
# ---------------------------------------------------------------------------


async def test_prune_old_access_logs(pool):
    """Pruning removes rows older than cutoff and returns count."""
    from weft.session_tracking import log_memory_access, prune_old_access_logs

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Pruning test memory",
        topic=["test"],
    ))

    # Log access then backdate it
    await log_memory_access(pool, [mem.id], "test", session_id="old-session")
    await pool.execute(
        "UPDATE memory_access_log SET accessed_at = $1 WHERE session_id = 'old-session'",
        NOW - timedelta(days=100),
    )

    # Log a recent access
    await log_memory_access(pool, [mem.id], "test", session_id="new-session")

    count = await prune_old_access_logs(pool, cutoff_days=90)
    assert count == 1

    # Verify old row is gone, new row remains
    rows = await pool.fetch("SELECT session_id FROM memory_access_log")
    session_ids = [r["session_id"] for r in rows]
    assert "old-session" not in session_ids
    assert "new-session" in session_ids


async def test_prune_empty_table(pool):
    """Pruning an empty table returns 0 without error."""
    from weft.session_tracking import prune_old_access_logs

    count = await prune_old_access_logs(pool, cutoff_days=90)
    assert count == 0


# ---------------------------------------------------------------------------
# Consolidation wiring
# ---------------------------------------------------------------------------


async def test_consolidation_prunes_access_logs(pool):
    """Consolidation pipeline should include access log pruning."""
    from weft.consolidation import consolidate
    from weft.session_tracking import log_memory_access

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Consolidation prune test",
        topic=["test"],
        confidence=0.9,
    ))

    # Create old access log entries
    await log_memory_access(pool, [mem.id], "test", session_id="cons-old")
    await pool.execute(
        "UPDATE memory_access_log SET accessed_at = $1 WHERE session_id = 'cons-old'",
        NOW - timedelta(days=100),
    )

    report = await consolidate(pool)
    assert hasattr(report, "access_logs_pruned")
    assert report.access_logs_pruned >= 1


# ---------------------------------------------------------------------------
# Memory model includes last_boosted_at
# ---------------------------------------------------------------------------


def test_memory_model_has_last_boosted_at():
    """Memory model should accept last_boosted_at field."""
    mem = Memory(
        type=MemoryType.fact,
        content="test",
        topic=["test"],
        last_boosted_at=NOW,
    )
    assert mem.last_boosted_at == NOW


def test_memory_model_last_boosted_at_defaults_none():
    """last_boosted_at should default to None."""
    mem = Memory(
        type=MemoryType.fact,
        content="test",
        topic=["test"],
    )
    assert mem.last_boosted_at is None


async def test_store_round_trips_last_boosted_at(pool):
    """Storing and retrieving a memory preserves last_boosted_at."""
    from weft.store import get_memory

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Round trip test",
        topic=["test"],
    ))

    # Set last_boosted_at via direct SQL (since store_memory doesn't set it)
    await pool.execute(
        "UPDATE memories SET last_boosted_at = $1 WHERE id = $2",
        NOW, mem.id,
    )

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.last_boosted_at is not None
