"""Tests for alert deduplication + suppression and the refactored producers.

Covers:
  - should_fire / record_fire cooldown semantics
  - per-key independence (firing on key A doesn't affect key B)
  - suppress / clear_suppression and their interaction with cooldowns
  - fire_count increments across multiple fires
  - producers (loom_alerts, memory_hygiene_alerts) use per-key dedup
  - per-user RLS isolation of state rows
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from weft.alert_dedup import (
    AlertState,
    clear_suppression,
    list_state,
    record_fire,
    should_fire,
    suppress,
)
from weft.alerts import create_alert, list_alerts
from weft.auth import current_user_id
from weft.config import AlertCooldownConfig
from weft.db.connection import acquire
from weft.memory_hygiene_alerts import (
    MemoryHygieneConfig,
    check_consolidation_overdue,
    check_memory_count,
    create_contradiction_alert,
)
from weft.models import (
    AlertCreate,
    AlertStatus,
    AlertType,
    Memory,
    MemorySource,
    MemoryType,
)


# =========================================================================
# should_fire / record_fire core
# =========================================================================


class TestShouldFire:
    @pytest.mark.asyncio
    async def test_first_call_fires(self, pool):
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:abc", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_within_cooldown_blocks(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "task:abc")
        assert not await should_fire(
            pool, AlertType.loom_stale_claim, "task:abc", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_after_cooldown_fires(self, pool):
        # Insert an old fire by manipulating the row directly
        await record_fire(pool, AlertType.loom_stale_claim, "task:abc")
        await pool.execute(
            "UPDATE alert_state SET last_fired_at = now() - interval '2 hours' "
            "WHERE alert_type = $1 AND dedup_key = $2",
            AlertType.loom_stale_claim.value, "task:abc",
        )
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:abc", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_per_key_independence(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "task:a")
        # Key A is in cooldown, but key B is fresh.
        assert not await should_fire(
            pool, AlertType.loom_stale_claim, "task:a", cooldown_minutes=60,
        )
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:b", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_per_type_independence(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "global")
        # Same key but different alert_type — independent state row.
        assert await should_fire(
            pool, AlertType.stale_decision, "global", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_explicit_now_param(self, pool):
        # Use the now= injection to test cooldown elapse without sleeping
        await record_fire(pool, AlertType.loom_stale_claim, "task:x")
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:x",
            cooldown_minutes=60, now=future,
        )


# =========================================================================
# record_fire — fire_count increments + idempotent upsert
# =========================================================================


class TestRecordFire:
    @pytest.mark.asyncio
    async def test_first_fire_creates_row_with_count_1(self, pool):
        state = await record_fire(pool, AlertType.loom_stale_claim, "task:x")
        assert state.fire_count == 1
        assert state.last_fired_at is not None

    @pytest.mark.asyncio
    async def test_repeat_fires_increment_count(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "task:x")
        await record_fire(pool, AlertType.loom_stale_claim, "task:x")
        s = await record_fire(pool, AlertType.loom_stale_claim, "task:x")
        assert s.fire_count == 3

    @pytest.mark.asyncio
    async def test_with_alert_id(self, pool):
        s = await record_fire(
            pool, AlertType.loom_stale_claim, "task:y", alert_id="weft-abc123",
        )
        assert s.last_alert_id == "weft-abc123"

    @pytest.mark.asyncio
    async def test_does_not_clear_suppression(self, pool):
        # If a manual mute is active, a stray record_fire should NOT lift it.
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await suppress(pool, AlertType.loom_stale_claim, "task:z", until=until)
        await record_fire(pool, AlertType.loom_stale_claim, "task:z")

        states = await list_state(pool, alert_type=AlertType.loom_stale_claim)
        z = next(s for s in states if s.dedup_key == "task:z")
        assert z.suppressed_until is not None
        assert z.fire_count == 1


# =========================================================================
# suppress / clear_suppression
# =========================================================================


class TestSuppression:
    @pytest.mark.asyncio
    async def test_suppress_blocks_fire(self, pool):
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await suppress(
            pool, AlertType.loom_stale_claim, "task:m",
            until=until, reason="muted",
        )
        assert not await should_fire(
            pool, AlertType.loom_stale_claim, "task:m", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_suppress_outranks_cooldown_in_both_directions(self, pool):
        # Even past cooldown, suppression keeps it muted
        await record_fire(pool, AlertType.loom_stale_claim, "task:n")
        await pool.execute(
            "UPDATE alert_state SET last_fired_at = now() - interval '5 hours' "
            "WHERE alert_type = $1 AND dedup_key = $2",
            AlertType.loom_stale_claim.value, "task:n",
        )
        # Without suppression: would fire (past cooldown)
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:n", cooldown_minutes=60,
        )
        # Add suppression
        await suppress(
            pool, AlertType.loom_stale_claim, "task:n",
            until=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        # Now should not fire
        assert not await should_fire(
            pool, AlertType.loom_stale_claim, "task:n", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_clear_suppression_unblocks_immediately(self, pool):
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await suppress(pool, AlertType.loom_stale_claim, "task:p", until=until)
        cleared = await clear_suppression(pool, AlertType.loom_stale_claim, "task:p")
        assert cleared is True
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:p", cooldown_minutes=60,
        )

    @pytest.mark.asyncio
    async def test_clear_suppression_returns_false_when_no_row(self, pool):
        cleared = await clear_suppression(
            pool, AlertType.loom_stale_claim, "task:never_existed",
        )
        assert cleared is False

    @pytest.mark.asyncio
    async def test_suppress_until_must_be_future(self, pool):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        with pytest.raises(ValueError, match="future"):
            await suppress(
                pool, AlertType.loom_stale_claim, "task:q", until=past,
            )

    @pytest.mark.asyncio
    async def test_expired_suppression_no_longer_blocks(self, pool):
        # Seed an already-expired suppression directly; should_fire ignores it.
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await suppress(pool, AlertType.loom_stale_claim, "task:e", until=until)
        await pool.execute(
            "UPDATE alert_state SET suppressed_until = now() - interval '10 minutes' "
            "WHERE alert_type = $1 AND dedup_key = $2",
            AlertType.loom_stale_claim.value, "task:e",
        )
        assert await should_fire(
            pool, AlertType.loom_stale_claim, "task:e", cooldown_minutes=60,
        )


# =========================================================================
# list_state
# =========================================================================


class TestListState:
    @pytest.mark.asyncio
    async def test_list_filters_by_alert_type(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "k1")
        await record_fire(pool, AlertType.stale_decision, "k2")
        loom = await list_state(pool, alert_type=AlertType.loom_stale_claim)
        assert all(s.alert_type == AlertType.loom_stale_claim for s in loom)
        assert any(s.dedup_key == "k1" for s in loom)
        assert not any(s.dedup_key == "k2" for s in loom)

    @pytest.mark.asyncio
    async def test_suppressed_only(self, pool):
        await record_fire(pool, AlertType.loom_stale_claim, "k1")
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await suppress(pool, AlertType.loom_stale_claim, "k2", until=until)
        sup = await list_state(pool, suppressed_only=True)
        keys = {s.dedup_key for s in sup}
        assert "k2" in keys
        assert "k1" not in keys


# =========================================================================
# Producer integration: per-key dedup actually used
# =========================================================================


class TestProducerIntegration:
    @pytest.mark.asyncio
    async def test_consolidation_overdue_singleton_dedup(self, pool):
        # Seed enough memories to trigger the count-based fallback path.
        from weft.store import store_memory

        for i in range(60):
            mem = Memory(
                type=MemoryType.fact, content=f"fact {i}",
                source=MemorySource.conversation,
            )
            await store_memory(pool, mem, embedding=None)

        cd = AlertCooldownConfig()
        # First call fires
        first = await check_consolidation_overdue(
            pool, config=MemoryHygieneConfig(), cooldowns=cd,
        )
        assert len(first) == 1
        # Second call within cooldown — same singleton key — does not fire
        second = await check_consolidation_overdue(
            pool, config=MemoryHygieneConfig(), cooldowns=cd,
        )
        assert second == []

    @pytest.mark.asyncio
    async def test_memory_count_threshold_singleton_dedup(self, pool):
        from weft.store import store_memory

        for i in range(15):
            mem = Memory(
                type=MemoryType.fact, content=f"f{i}",
                source=MemorySource.conversation,
            )
            await store_memory(pool, mem, embedding=None)

        cfg = MemoryHygieneConfig(memory_count_threshold=10)
        cd = AlertCooldownConfig()
        first = await check_memory_count(pool, config=cfg, cooldowns=cd)
        assert len(first) == 1
        second = await check_memory_count(pool, config=cfg, cooldowns=cd)
        assert second == []

    @pytest.mark.asyncio
    async def test_contradiction_alert_per_memory_independence(self, pool):
        # First memory contradiction fires
        r1 = await create_contradiction_alert(
            pool,
            new_memory_id="weft-mem001",
            contradictions=[{"memory_id": "x", "content_preview": "x", "similarity": 0.9}],
        )
        assert r1 is not None
        # Second memory contradiction *also* fires — different dedup_key.
        # V1 would have suppressed this for 24h.
        r2 = await create_contradiction_alert(
            pool,
            new_memory_id="weft-mem002",
            contradictions=[{"memory_id": "y", "content_preview": "y", "similarity": 0.9}],
        )
        assert r2 is not None
        # Same memory again — within 60min cooldown — suppressed
        r3 = await create_contradiction_alert(
            pool,
            new_memory_id="weft-mem001",
            contradictions=[{"memory_id": "z", "content_preview": "z", "similarity": 0.9}],
        )
        assert r3 is None


# =========================================================================
# Per-user RLS isolation
# =========================================================================


class TestUserIsolation:
    """Per-user isolation. The test container runs Postgres as a superuser,
    which bypasses RLS read filtering, so we verify isolation the same way
    test_auth_integration.py does: by checking that the user_id column was
    populated correctly via the GUC default. RLS is enforced in production
    where Weft runs as a non-superuser; the read-side contract is pinned
    by tests/test_rls_invariants.py at the policy/schema level.
    """

    @pytest.mark.asyncio
    async def test_two_users_get_correct_user_id_on_record_fire(self, pool):
        tok_a = current_user_id.set("user-aaa")
        try:
            async with acquire(pool):
                await record_fire(pool, AlertType.loom_stale_claim, "task:shared")
        finally:
            current_user_id.reset(tok_a)

        tok_b = current_user_id.set("user-bbb")
        try:
            async with acquire(pool):
                await record_fire(pool, AlertType.loom_stale_claim, "task:shared")
        finally:
            current_user_id.reset(tok_b)

        # The unique constraint is (alert_type, dedup_key, user_id), so
        # both users own a distinct row for the same dedup_key. If user_id
        # weren't being scoped per request, the second insert would have
        # hit ON CONFLICT and incremented fire_count instead of creating
        # a separate row.
        rows = await pool.fetch(
            "SELECT user_id, fire_count FROM alert_state "
            "WHERE alert_type = $1 AND dedup_key = $2 "
            "ORDER BY user_id",
            AlertType.loom_stale_claim.value, "task:shared",
        )
        assert len(rows) == 2
        users = sorted(r["user_id"] for r in rows)
        assert users == ["user-aaa", "user-bbb"]
        assert all(r["fire_count"] == 1 for r in rows)

    @pytest.mark.asyncio
    async def test_user_b_suppression_writes_under_b(self, pool):
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        tok = current_user_id.set("user-bbb")
        try:
            async with acquire(pool):
                await suppress(
                    pool, AlertType.loom_stale_claim, "task:s",
                    until=until, reason="vacation",
                )
        finally:
            current_user_id.reset(tok)

        row = await pool.fetchrow(
            "SELECT user_id, suppressed_until, suppression_reason "
            "FROM alert_state "
            "WHERE alert_type = $1 AND dedup_key = $2",
            AlertType.loom_stale_claim.value, "task:s",
        )
        assert row["user_id"] == "user-bbb"
        assert row["suppression_reason"] == "vacation"
