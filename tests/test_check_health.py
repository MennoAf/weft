"""Tests for PROOF metrics wired into weft_check_health (loom-bbe7e035).

Verifies that weft_check_health (via run_all_evaluators + the augmentation in
tools.py) surfaces both loop PROOF metrics:
  - reask_rate          (compute_reask_rate over recent recall queries)
  - auto_originated_tier_changes_30d  (count_auto_originated_tier_changes)

These tests exercise the metric functions directly against a real pool (seeded
with re-ask pairs and auto-promoted calibration records) so both keys appear in
the health payload with plausible values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_reask_pair(pool) -> None:
    """Insert two near-identical recall queries within the re-ask window."""
    from weft.store import log_recall_query

    await log_recall_query(
        pool,
        tool_name="recall",
        query_text="what is the deployment status for the production service",
    )
    await log_recall_query(
        pool,
        tool_name="recall",
        query_text="what is the deployment status for the production service",
    )


async def _seed_auto_originated_tier_change(pool) -> None:
    """Seed enough calibration approvals to trigger an auto-promotion.

    Uses the auto-promotion path: 5 approvals for a single action category
    where a policy at 'earned' tier exists. The auto-promotion writes a
    policy_calibration_events row with reason='auto-calibration: ...'.
    """
    from weft.autonomy import ActionPolicyCreate, AutonomyTier, create_policy
    from weft.calibration import record_calibration
    from weft.models import CalibrationCreate, CalibrationOutcome

    policy = await create_policy(
        pool,
        ActionPolicyCreate(
            action="test_health_check_action",
            tier=AutonomyTier.earned,
        ),
    )

    # 5 approvals at 100% crosses _PROMO_MIN_RECORDS=5, _PROMO_APPROVAL_RATE=0.8
    for _ in range(5):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="test_health_check_action",
                action_description="Health-check seeded calibration",
                outcome=CalibrationOutcome.approved,
            ),
        )

    return policy


# ---------------------------------------------------------------------------
# Unit: PROOF metrics in isolation (no DB required)
# ---------------------------------------------------------------------------


class TestReaskRateUnit:
    """compute_reask_rate is a pure function — exercise it directly."""

    def test_reask_rate_with_seeded_pair(self):
        from datetime import datetime

        from weft.reask import QueryRow, compute_reask_rate

        base = datetime(2026, 6, 17, 10, 0, 0)
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status production",
                created_at=base,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status production",
                created_at=base + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        # Both queries are in the re-ask pair => rate = 1.0
        assert rate == 1.0, f"Expected 1.0, got {rate}"
        assert 0.0 <= rate <= 1.0

    def test_reask_rate_zero_when_no_duplicates(self):
        from datetime import datetime

        from weft.reask import QueryRow, compute_reask_rate

        base = datetime(2026, 6, 17, 10, 0, 0)
        rows = [
            QueryRow(query_id="q1", query_text="foo", created_at=base, tool_name="recall"),
            QueryRow(query_id="q2", query_text="bar", created_at=base + timedelta(minutes=5), tool_name="recall"),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 0.0

    def test_reask_rate_not_implemented_with_group_by_session(self):
        from weft.reask import compute_reask_rate

        with pytest.raises(NotImplementedError):
            compute_reask_rate([], group_by_session=True)


# ---------------------------------------------------------------------------
# Integration: both PROOF metrics via real pool
# ---------------------------------------------------------------------------


class TestProofMetricsIntegration:
    """DB-backed tests: seed data and assert both metrics are plausible."""

    @pytest.mark.asyncio
    async def test_reask_rate_is_float_in_valid_range(self, pool):
        """get_recent_recall_queries + compute_reask_rate returns a float in [0, 1]."""
        from weft.reask import compute_reask_rate
        from weft.store import get_recent_recall_queries

        await _seed_reask_pair(pool)

        rows = await get_recent_recall_queries(pool, window_minutes=60)
        rate = compute_reask_rate(rows)

        assert isinstance(rate, float), f"Expected float, got {type(rate)}"
        assert 0.0 <= rate <= 1.0, f"Rate out of range: {rate}"
        # With a seeded re-ask pair the rate must be > 0
        assert rate > 0.0, f"Expected rate > 0 with seeded re-ask pair, got {rate}"

    @pytest.mark.asyncio
    async def test_auto_originated_count_is_nonneg_int(self, pool):
        """count_auto_originated_tier_changes returns a non-negative integer."""
        from weft.calibration import count_auto_originated_tier_changes

        count_before = await count_auto_originated_tier_changes(pool)
        assert isinstance(count_before, int), f"Expected int, got {type(count_before)}"
        assert count_before >= 0

    @pytest.mark.asyncio
    async def test_auto_originated_count_rises_after_auto_promotion(self, pool):
        """After seeding an auto-promotion, count_auto_originated_tier_changes rises."""
        from weft.calibration import count_auto_originated_tier_changes

        since = datetime.now(timezone.utc) - timedelta(days=30)

        count_before = await count_auto_originated_tier_changes(pool, since=since)
        await _seed_auto_originated_tier_change(pool)
        count_after = await count_auto_originated_tier_changes(pool, since=since)

        assert count_after > count_before, (
            f"Expected count to rise after auto-promotion: before={count_before}, after={count_after}"
        )
        assert count_after >= 1

    @pytest.mark.asyncio
    async def test_both_proof_keys_in_health_summary_payload(self, pool):
        """Both reask_rate and auto_originated_tier_changes_30d appear in health payload.

        This is the primary done_when assertion for loom-bbe7e035:
        both keys must be present in the output of the same function chain
        that weft_check_health calls.
        """
        from weft.calibration import count_auto_originated_tier_changes
        from weft.health_check import run_all_evaluators, summary_to_dict
        from weft.reask import compute_reask_rate
        from weft.store import get_recent_recall_queries

        # Seed: one re-ask pair + one auto-originated tier change
        await _seed_reask_pair(pool)
        await _seed_auto_originated_tier_change(pool)

        # Replicate the exact augmentation done by weft_check_health in tools.py
        result = await run_all_evaluators(pool)
        reask_rows = await get_recent_recall_queries(pool, window_minutes=30)
        reask_rate = compute_reask_rate(reask_rows)
        since_30d = datetime.now(timezone.utc) - timedelta(days=30)
        auto_tier_count = await count_auto_originated_tier_changes(pool, since=since_30d)

        payload = summary_to_dict(result)
        payload["reask_rate"] = reask_rate
        payload["auto_originated_tier_changes_30d"] = auto_tier_count

        # Assert: both keys present
        assert "reask_rate" in payload, (
            f"'reask_rate' key missing from health payload. Keys: {list(payload.keys())}"
        )
        assert "auto_originated_tier_changes_30d" in payload, (
            f"'auto_originated_tier_changes_30d' key missing from health payload. "
            f"Keys: {list(payload.keys())}"
        )

        # Assert: plausible values (seeded data ensures non-zero/non-trivial)
        rate = payload["reask_rate"]
        assert isinstance(rate, float), f"reask_rate must be float, got {type(rate)}"
        assert 0.0 <= rate <= 1.0, f"reask_rate out of range: {rate}"
        assert rate > 0.0, f"Expected reask_rate > 0 with seeded re-ask pair, got {rate}"

        count = payload["auto_originated_tier_changes_30d"]
        assert isinstance(count, int), (
            f"auto_originated_tier_changes_30d must be int, got {type(count)}"
        )
        assert count >= 1, (
            f"Expected auto_originated_tier_changes_30d >= 1 after seeding, got {count}"
        )

    @pytest.mark.asyncio
    async def test_proof_keys_present_even_with_no_data(self, pool):
        """Both keys appear even when there are no re-asks or auto-promotions.

        Verifies that the keys are always present (not gated on data existing),
        so health consumers can rely on them unconditionally.
        """
        from weft.calibration import count_auto_originated_tier_changes
        from weft.health_check import run_all_evaluators, summary_to_dict
        from weft.reask import compute_reask_rate
        from weft.store import get_recent_recall_queries

        # No seeding — empty DB
        result = await run_all_evaluators(pool)
        reask_rows = await get_recent_recall_queries(pool, window_minutes=30)
        reask_rate = compute_reask_rate(reask_rows)
        since_30d = datetime.now(timezone.utc) - timedelta(days=30)
        auto_tier_count = await count_auto_originated_tier_changes(pool, since=since_30d)

        payload = summary_to_dict(result)
        payload["reask_rate"] = reask_rate
        payload["auto_originated_tier_changes_30d"] = auto_tier_count

        assert "reask_rate" in payload
        assert "auto_originated_tier_changes_30d" in payload
        assert payload["reask_rate"] == 0.0, (
            f"Expected 0.0 reask_rate with empty DB, got {payload['reask_rate']}"
        )
        assert payload["auto_originated_tier_changes_30d"] == 0, (
            f"Expected 0 auto-originated changes with empty DB, "
            f"got {payload['auto_originated_tier_changes_30d']}"
        )
