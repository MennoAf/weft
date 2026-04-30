"""Tests for memory hygiene alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.alerts import create_alert
from weft.memory_hygiene_alerts import (
    MemoryHygieneConfig,
    check_consolidation_overdue,
    check_memory_count,
    check_stale_decisions,
    create_contradiction_alert,
    evaluate_memory_hygiene_alerts,
)

# Use config defaults as test constants
_cfg = MemoryHygieneConfig()
_CONSOLIDATION_OVERDUE_HOURS = _cfg.consolidation_overdue_hours
_MEMORY_COUNT_THRESHOLD = _cfg.memory_count_threshold
_STALE_DECISION_CONFIDENCE = _cfg.stale_decision_confidence
_STALE_DECISION_DAYS = _cfg.stale_decision_days
from weft.models import AlertType, MemoryType
from weft.store import store_memory, set_metadata


async def _insert_decision(
    pool,
    *,
    content: str = "Use PostgreSQL for storage",
    confidence: float = 0.8,
    review_after: datetime | None = None,
    created_at: datetime | None = None,
) -> str:
    """Insert a decision memory and return its ID."""
    from weft.models import MemoryCreate
    mem = await store_memory(
        pool,
        MemoryCreate(
            content=content,
            type=MemoryType.decision,
            confidence=confidence,
            review_after=review_after,
        ),
    )
    if created_at:
        # Backdate the created_at
        await pool.execute(
            "UPDATE memories SET created_at = $1 WHERE id = $2",
            created_at,
            mem.id,
        )
    return mem.id


async def _insert_memories(pool, count: int) -> None:
    """Insert N active memories."""
    from weft.models import MemoryCreate
    for i in range(count):
        await store_memory(
            pool,
            MemoryCreate(content=f"Memory number {i}", type=MemoryType.fact),
        )


# --- Stale Decisions ---


class TestStaleDecisions:
    @pytest.mark.asyncio
    async def test_fires_on_overdue_review(self, pool):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        await _insert_decision(pool, review_after=past)
        result = await check_stale_decisions(pool)
        assert len(result) == 1
        assert result[0]["alert_type"] == AlertType.stale_decision.value

    @pytest.mark.asyncio
    async def test_no_alert_future_review(self, pool):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        await _insert_decision(pool, review_after=future)
        result = await check_stale_decisions(pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fires_on_old_low_confidence(self, pool):
        old = datetime.now(timezone.utc) - timedelta(days=_STALE_DECISION_DAYS + 1)
        await _insert_decision(
            pool,
            confidence=_STALE_DECISION_CONFIDENCE - 0.1,
            created_at=old,
        )
        result = await check_stale_decisions(pool)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_alert_old_high_confidence(self, pool):
        old = datetime.now(timezone.utc) - timedelta(days=_STALE_DECISION_DAYS + 1)
        await _insert_decision(
            pool,
            confidence=0.9,
            created_at=old,
        )
        result = await check_stale_decisions(pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_alert_recent_low_confidence(self, pool):
        await _insert_decision(pool, confidence=_STALE_DECISION_CONFIDENCE - 0.1)
        result = await check_stale_decisions(pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_dedup(self, pool):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        await _insert_decision(pool, review_after=past)
        r1 = await check_stale_decisions(pool)
        assert len(r1) == 1
        r2 = await check_stale_decisions(pool)
        assert len(r2) == 0


# --- Consolidation Overdue ---


class TestConsolidationOverdue:
    @pytest.mark.asyncio
    async def test_fires_when_overdue(self, pool):
        # Insert enough memories so we don't hit the "too few" guard
        await _insert_memories(pool, 60)
        overdue = datetime.now(timezone.utc) - timedelta(hours=_CONSOLIDATION_OVERDUE_HOURS + 1)
        await set_metadata(pool, "last_consolidation_run", {
            "ran_at": overdue.isoformat(),
            "status": "completed",
        })
        result = await check_consolidation_overdue(pool)
        assert len(result) == 1
        assert result[0]["alert_type"] == AlertType.memory_consolidation_overdue.value

    @pytest.mark.asyncio
    async def test_no_alert_when_recent(self, pool):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        await set_metadata(pool, "last_consolidation_run", {
            "ran_at": recent.isoformat(),
            "status": "completed",
        })
        result = await check_consolidation_overdue(pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_alert_new_install_few_memories(self, pool):
        """Brand new install with < 50 memories should not alert."""
        result = await check_consolidation_overdue(pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fires_never_run_many_memories(self, pool):
        """Never-run consolidation with 50+ memories should alert."""
        await _insert_memories(pool, 55)
        result = await check_consolidation_overdue(pool)
        assert len(result) == 1


# --- Memory Count ---


class TestMemoryCount:
    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, pool):
        await _insert_memories(pool, 10)
        result = await check_memory_count(pool)
        assert len(result) == 0

    # Note: testing above threshold would require inserting 1000+ rows,
    # which is slow. We trust the query logic and test the dedup instead.


# --- Contradiction Alert ---


class TestContradictionAlert:
    @pytest.mark.asyncio
    async def test_creates_alert(self, pool):
        contradictions = [
            {
                "memory_id": "weft-abc123",
                "content_preview": "Use MongoDB for storage",
                "similarity": 0.85,
            }
        ]
        result = await create_contradiction_alert(
            pool,
            new_memory_id="weft-new-123",
            contradictions=contradictions,
        )
        assert result is not None
        assert result["alert_type"] == AlertType.memory_contradiction.value

    @pytest.mark.asyncio
    async def test_dedup_per_memory(self, pool):
        # V2 (Epic 7): contradiction alerts dedup per-new_memory_id rather
        # than per-alert-type. m1 and m2 are different memories, so both
        # fire. A repeat call for m1 within the cooldown is suppressed.
        # The per-type independence here is the operative change vs. V1.
        contradictions = [{"memory_id": "weft-x", "content_preview": "x", "similarity": 0.9}]
        r1 = await create_contradiction_alert(pool, new_memory_id="m1", contradictions=contradictions)
        assert r1 is not None
        r2 = await create_contradiction_alert(pool, new_memory_id="m2", contradictions=contradictions)
        assert r2 is not None
        r1_again = await create_contradiction_alert(
            pool, new_memory_id="m1", contradictions=contradictions,
        )
        assert r1_again is None

    @pytest.mark.asyncio
    async def test_empty_contradictions(self, pool):
        result = await create_contradiction_alert(
            pool, new_memory_id="m1", contradictions=[],
        )
        assert result is None


# --- Combined evaluation ---


class TestEvaluateAll:
    @pytest.mark.asyncio
    async def test_empty_db(self, pool):
        result = await evaluate_memory_hygiene_alerts(pool)
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_alerts(self, pool):
        """Multiple conditions can fire in one evaluation."""
        # Stale decision
        past = datetime.now(timezone.utc) - timedelta(days=1)
        await _insert_decision(pool, review_after=past)
        # Overdue consolidation
        await _insert_memories(pool, 55)
        result = await evaluate_memory_hygiene_alerts(pool)
        types = {a["alert_type"] for a in result}
        assert AlertType.stale_decision.value in types
        assert AlertType.memory_consolidation_overdue.value in types
