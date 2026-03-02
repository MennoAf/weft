"""Tests for the feedback MCP tool and usefulness integration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weft.models import Memory, MemoryCreate, MemoryRecall, MemorySource, MemoryStatus, MemoryType
from weft.relevance import score_memory, usefulness_factor
from weft.store import get_memory, record_feedback, store_memory, touch_memory

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)


# --- DB tests using the shared pool fixture from conftest ---


async def test_feedback_helpful_increases_score(pool):
    """Record helpful feedback; verify usefulness_score increases from default."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="helpful memory",
        topic=["test"],
    ))
    # Default usefulness_score is 0.7; helpful feedback should increase it
    # EMA: 0.3 * 1.0 + 0.7 * 0.7 = 0.79
    result = await record_feedback(pool, mem.id, helpful=True)
    assert result["usefulness_score"] == pytest.approx(0.79, abs=0.01)
    assert result["usefulness_count"] == 1

    # Additional helpful feedback keeps pushing toward 1.0
    r2 = await record_feedback(pool, mem.id, helpful=True)
    assert r2["usefulness_score"] > result["usefulness_score"]

    # Unhelpful feedback lowers the score
    r3 = await record_feedback(pool, mem.id, helpful=False)
    assert r3["usefulness_score"] < r2["usefulness_score"]

    # But helpful feedback recovers it
    r4 = await record_feedback(pool, mem.id, helpful=True)
    assert r4["usefulness_score"] > r3["usefulness_score"]


async def test_feedback_unhelpful_decreases_score(pool):
    """Record unhelpful feedback; verify usefulness_score decreases."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="unhelpful memory",
        topic=["test"],
    ))
    result = await record_feedback(pool, mem.id, helpful=False)
    # EMA: (1 - 0.3) * 0.7 + 0.3 * 0.0 = 0.49
    assert result["usefulness_score"] == pytest.approx(0.49, abs=0.01)
    assert result["usefulness_count"] == 1


async def test_feedback_multiple_rounds(pool):
    """Multiple feedback rounds converge correctly."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="multi feedback memory",
        topic=["test"],
    ))
    # 5 rounds of unhelpful feedback (starting from 0.7)
    for _ in range(5):
        result = await record_feedback(pool, mem.id, helpful=False)

    # Score should be very low after repeated unhelpful from 0.7 start
    assert result["usefulness_score"] < 0.15
    assert result["usefulness_count"] == 5

    # Now give helpful feedback to push it back up
    for _ in range(10):
        result = await record_feedback(pool, mem.id, helpful=True)

    # Should have recovered significantly toward 1.0
    assert result["usefulness_score"] > 0.8
    assert result["usefulness_count"] == 15


async def test_feedback_nonexistent_memory_raises(pool):
    """ValueError for missing memory_id."""
    with pytest.raises(ValueError, match="not found"):
        await record_feedback(pool, "weft-nonexistent", helpful=True)


# --- Pure function tests (no DB) ---


def test_usefulness_factor_range():
    """Test usefulness_factor() at boundary values with default floor=0.5."""
    # 0.0 -> floor = 0.5
    assert usefulness_factor(0.0) == pytest.approx(0.5)
    # 0.5 -> 0.5 + 0.5 * 0.5 = 0.75
    assert usefulness_factor(0.5) == pytest.approx(0.75)
    # 1.0 -> 1.0
    assert usefulness_factor(1.0) == pytest.approx(1.0)


def test_usefulness_factor_clamps():
    """Test with out-of-range values — should clamp."""
    # Negative -> clamp to 0.0 -> factor = floor
    assert usefulness_factor(-0.5) == pytest.approx(0.5)
    # > 1.0 -> clamp to 1.0 -> factor 1.0
    assert usefulness_factor(1.5) == pytest.approx(1.0)


def test_usefulness_factor_custom_floor():
    """Test usefulness_factor() with custom floor values."""
    # floor=0.0 -> factor equals raw score
    assert usefulness_factor(0.0, floor=0.0) == pytest.approx(0.0)
    assert usefulness_factor(0.5, floor=0.0) == pytest.approx(0.5)
    assert usefulness_factor(1.0, floor=0.0) == pytest.approx(1.0)

    # floor=1.0 -> factor always 1.0
    assert usefulness_factor(0.0, floor=1.0) == pytest.approx(1.0)
    assert usefulness_factor(0.5, floor=1.0) == pytest.approx(1.0)

    # floor=0.3 -> 0.3 + 0.7 * score
    assert usefulness_factor(0.0, floor=0.3) == pytest.approx(0.3)
    assert usefulness_factor(1.0, floor=0.3) == pytest.approx(1.0)
    assert usefulness_factor(0.5, floor=0.3) == pytest.approx(0.65)


def _make_memory(
    *,
    confidence: float = 0.9,
    accessed_at: datetime | None = None,
    access_count: int = 0,
    content: str = "test memory",
    usefulness_score: float = 1.0,
    topic: list[str] | None = None,
) -> Memory:
    return Memory(
        id="weft-test0001",
        type=MemoryType.fact,
        topic=topic or ["test"],
        content=content,
        source=MemorySource.conversation,
        confidence=confidence,
        accessed_at=accessed_at or NOW,
        access_count=access_count,
        usefulness_score=usefulness_score,
        status=MemoryStatus.active,
    )


def _make_recall(similarity: float = 0.8, **kwargs) -> MemoryRecall:
    return MemoryRecall(memory=_make_memory(**kwargs), similarity=similarity)


def test_scored_memory_includes_usefulness():
    """score_memory returns ScoredMemory with usefulness_factor field."""
    recall = _make_recall(similarity=0.8, confidence=0.9, usefulness_score=0.6)
    scored = score_memory(recall, now=NOW)

    assert hasattr(scored, "usefulness_factor")
    # usefulness_factor(0.6) = 0.5 + 0.5 * 0.6 = 0.8
    assert scored.usefulness_factor == pytest.approx(0.8)


def test_low_usefulness_reduces_relevance():
    """Two identical memories, one with low usefulness_score, verify lower relevance."""
    high = _make_recall(similarity=0.8, confidence=0.9, usefulness_score=1.0)
    low = _make_recall(similarity=0.8, confidence=0.9, usefulness_score=0.2)

    high_scored = score_memory(high, now=NOW)
    low_scored = score_memory(low, now=NOW)

    assert high_scored.score > low_scored.score
    # usefulness_factor(1.0)=1.0, usefulness_factor(0.2)=0.6
    # So ratio should be ~0.6
    assert low_scored.score / high_scored.score == pytest.approx(0.6, abs=0.01)


def test_feedback_mcp_tool_exists():
    """Import and verify weft_feedback exists in tools module."""
    from weft.mcp import tools
    assert hasattr(tools, "weft_feedback")
    assert callable(tools.weft_feedback)


# --- Implicit usefulness bump via touch_memory ---


async def test_touch_memory_implicit_bump(pool):
    """touch_memory should apply a mild positive usefulness bump."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="implicitly useful memory",
        topic=["test"],
    ))
    # Default score is 0.7
    assert mem.usefulness_score == pytest.approx(0.7)

    await touch_memory(pool, mem.id)
    updated = await get_memory(pool, mem.id)

    # EMA: (1 - 0.05) * 0.7 + 0.05 * 1.0 = 0.665 + 0.05 = 0.715
    assert updated.usefulness_score == pytest.approx(0.715, abs=0.001)
    assert updated.access_count == 1


async def test_touch_memory_repeated_bumps_converge(pool):
    """Repeated touches should converge toward 1.0 monotonically."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="frequently accessed memory",
        topic=["test"],
    ))

    prev_score = 0.7
    for i in range(20):
        await touch_memory(pool, mem.id)
        updated = await get_memory(pool, mem.id)
        assert updated.usefulness_score >= prev_score  # monotonically increasing
        prev_score = updated.usefulness_score

    # After 20 touches from 0.7 with alpha=0.05: converges to ~0.89
    assert updated.usefulness_score > 0.85
    assert updated.usefulness_score <= 1.0


async def test_touch_memory_at_max_stays_capped(pool):
    """A memory already at 1.0 should not exceed 1.0 after touch."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="maxed out memory",
        topic=["test"],
    ))
    # Push to 1.0 via explicit feedback
    for _ in range(20):
        await record_feedback(pool, mem.id, helpful=True)

    pre = await get_memory(pool, mem.id)
    assert pre.usefulness_score == pytest.approx(1.0, abs=0.01)

    await touch_memory(pool, mem.id)
    post = await get_memory(pool, mem.id)
    assert post.usefulness_score <= 1.0


def test_default_usefulness_score_is_0_7():
    """New Memory instances should default to 0.7, not 1.0."""
    mem = Memory(
        type=MemoryType.fact,
        content="test default",
    )
    assert mem.usefulness_score == pytest.approx(0.7)
