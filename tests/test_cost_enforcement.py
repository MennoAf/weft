"""Tests for the cost-to-autonomy/degradation enforcement feedback loop.

Covers:
  - autonomy override store + strictest-wins resolver
  - daily idempotency via cost_enforcement_state row
  - threshold ladder firing across multiple bands in one tick
  - cost_breach DegradationTriggerType evaluation
  - daily reset semantics
  - graceful disable when config off / no limit / no thresholds
  - federation invariant: NEVER baseline policy is never *escalated* by an override
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.autonomy import (
    ActionPolicyCreate,
    AutonomyOverrideCreate,
    AutonomyTier,
    OverrideSource,
    create_override,
    create_policy,
    expire_overrides,
    get_active_overrides,
    get_effective_tier,
    get_tier_for_action,
)
from weft.config import CostEnforcementConfig, CostThreshold, CostThresholdLiteral
from weft.cost_enforcement import (
    EnforcementReport,
    _next_daily_reset,
    enforce_cost_thresholds,
)
from weft.cost_tracking import CostEntryCreate, record_cost
from weft.degradation import _evaluate_condition, create_policy as create_deg_policy
from weft.models import (
    DegradationAction,
    DegradationPolicy,
    DegradationPolicyCreate,
    DegradationTriggerType,
)


# =========================================================================
# Helpers
# =========================================================================


def _config(
    *,
    enabled: bool = True,
    daily_limit_usd: float = 10.0,
    thresholds: list[CostThreshold] | None = None,
) -> CostEnforcementConfig:
    """Default config for enforcement tests — enabled, $10 limit, 3-band ladder."""
    if thresholds is None:
        thresholds = [
            CostThreshold(pct_used=50.0, notify=True),
            CostThreshold(
                pct_used=80.0,
                demote_actions=["send_message"],
                demote_to=CostThresholdLiteral.earned,
            ),
            CostThreshold(
                pct_used=100.0,
                demote_actions=["send_message"],
                demote_to=CostThresholdLiteral.never,
            ),
        ]
    return CostEnforcementConfig(
        enabled=enabled,
        daily_limit_usd=daily_limit_usd,
        thresholds=thresholds,
    )


async def _spend(pool, dollars: float) -> None:
    """Record a cost entry contributing to today's spend."""
    await record_cost(
        pool,
        CostEntryCreate(estimated_cost_usd=dollars),
    )


# =========================================================================
# Autonomy override model + resolver
# =========================================================================


class TestAutonomyOverrides:
    @pytest.mark.asyncio
    async def test_create_override_persists(self, pool):
        ov = await create_override(
            pool,
            AutonomyOverrideCreate(
                action="send_message",
                effective_tier=AutonomyTier.earned,
                source=OverrideSource.manual,
                reason="manual lock for review",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
        assert ov.id.startswith("weft-")
        assert ov.action == "send_message"
        assert ov.effective_tier == AutonomyTier.earned
        assert ov.source == OverrideSource.manual

    @pytest.mark.asyncio
    async def test_expires_at_must_be_future(self, pool):
        with pytest.raises(ValueError, match="expires_at"):
            await create_override(
                pool,
                AutonomyOverrideCreate(
                    action="send_message",
                    effective_tier=AutonomyTier.earned,
                    source=OverrideSource.manual,
                    expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                ),
            )

    @pytest.mark.asyncio
    async def test_get_active_overrides_excludes_expired(self, pool):
        # Create one already-expired by manipulating directly; create_override
        # refuses past expires_at, so insert via SQL to seed an expired row.
        await pool.execute(
            """
            INSERT INTO autonomy_overrides
              (id, action, effective_tier, source, reason, expires_at, user_id)
            VALUES ('weft-expired', 'send_message', 'earned', 'manual', 'old',
                    now() - interval '1 hour',
                    nullif(current_setting('app.user_id', true), ''))
            """
        )
        await create_override(
            pool,
            AutonomyOverrideCreate(
                action="send_message",
                effective_tier=AutonomyTier.earned,
                source=OverrideSource.manual,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
        active = await get_active_overrides(pool, action="send_message")
        ids = [o.id for o in active]
        assert "weft-expired" not in ids
        assert len(active) == 1

    @pytest.mark.asyncio
    async def test_resolver_uses_baseline_when_no_overrides(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="ship_code", tier=AutonomyTier.always),
        )
        tier = await get_effective_tier(pool, "ship_code")
        assert tier == AutonomyTier.always

    @pytest.mark.asyncio
    async def test_resolver_strictest_override_wins(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        # Two simultaneous overrides — earned and never. Never must win.
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        await create_override(
            pool,
            AutonomyOverrideCreate(
                action="send_message",
                effective_tier=AutonomyTier.earned,
                source=OverrideSource.cost_enforcement,
                expires_at=expires,
            ),
        )
        await create_override(
            pool,
            AutonomyOverrideCreate(
                action="send_message",
                effective_tier=AutonomyTier.never,
                source=OverrideSource.degradation_policy,
                expires_at=expires,
            ),
        )
        assert await get_effective_tier(pool, "send_message") == AutonomyTier.never

    @pytest.mark.asyncio
    async def test_override_cannot_escalate_baseline(self, pool):
        # Federation-safe invariant: if baseline says EARNED, an override
        # claiming ALWAYS cannot promote — circuit breakers may only restrict.
        await create_policy(
            pool,
            ActionPolicyCreate(action="risky_op", tier=AutonomyTier.earned),
        )
        await create_override(
            pool,
            AutonomyOverrideCreate(
                action="risky_op",
                effective_tier=AutonomyTier.always,
                source=OverrideSource.manual,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
        assert await get_effective_tier(pool, "risky_op") == AutonomyTier.earned

    @pytest.mark.asyncio
    async def test_resolver_falls_back_to_earned_when_no_policy_or_override(
        self, pool,
    ):
        # No autonomy_policies row, no overrides — default is EARNED (asks).
        assert await get_effective_tier(pool, "unknown_action") == AutonomyTier.earned

    @pytest.mark.asyncio
    async def test_expire_overrides_deletes_only_past(self, pool):
        # One expired (seeded), one live.
        await pool.execute(
            """
            INSERT INTO autonomy_overrides
              (id, action, effective_tier, source, expires_at, user_id)
            VALUES ('weft-old', 'a', 'earned', 'manual',
                    now() - interval '2 days',
                    nullif(current_setting('app.user_id', true), ''))
            """
        )
        await create_override(
            pool,
            AutonomyOverrideCreate(
                action="a",
                effective_tier=AutonomyTier.earned,
                source=OverrideSource.manual,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
        deleted = await expire_overrides(pool)
        assert deleted == 1
        # Live one survives
        rows = await pool.fetch("SELECT id FROM autonomy_overrides")
        assert len(rows) == 1


# =========================================================================
# DegradationTriggerType.cost_breach
# =========================================================================


class TestCostBreachTrigger:
    def test_cost_breach_fires_when_pct_meets_threshold(self):
        policy = DegradationPolicy(
            name="cb",
            trigger_type=DegradationTriggerType.cost_breach,
            condition={"pct_used": 90.0},
            action=DegradationAction.escalate,
        )
        result = _evaluate_condition(policy, {"cost_pct_used": 95.0})
        assert result is not None
        assert "95" in result

    def test_cost_breach_no_trigger_below_threshold(self):
        policy = DegradationPolicy(
            name="cb",
            trigger_type=DegradationTriggerType.cost_breach,
            condition={"pct_used": 90.0},
            action=DegradationAction.escalate,
        )
        assert _evaluate_condition(policy, {"cost_pct_used": 50.0}) is None

    def test_cost_breach_validation_requires_pct_used(self):
        with pytest.raises(ValueError, match="pct_used"):
            DegradationPolicyCreate(
                name="bad",
                trigger_type=DegradationTriggerType.cost_breach,
                condition={},
                action=DegradationAction.escalate,
            )


# =========================================================================
# enforce_cost_thresholds — main enforcement
# =========================================================================


class TestEnforceCostThresholds:
    @pytest.mark.asyncio
    async def test_disabled_returns_skipped(self, pool):
        report = await enforce_cost_thresholds(
            pool, config=CostEnforcementConfig(enabled=False),
        )
        assert report.skipped_reason == "disabled"
        assert report.actions == []

    @pytest.mark.asyncio
    async def test_no_limit_returns_skipped(self, pool):
        report = await enforce_cost_thresholds(
            pool, config=CostEnforcementConfig(enabled=True, daily_limit_usd=0),
        )
        assert report.skipped_reason == "no_limit"

    @pytest.mark.asyncio
    async def test_no_thresholds_returns_skipped(self, pool):
        report = await enforce_cost_thresholds(
            pool,
            config=CostEnforcementConfig(
                enabled=True, daily_limit_usd=10, thresholds=[],
            ),
        )
        assert report.skipped_reason == "no_thresholds"

    @pytest.mark.asyncio
    async def test_below_lowest_band_no_actions(self, pool):
        await _spend(pool, 1.0)  # 10% of $10 limit
        report = await enforce_cost_thresholds(pool, config=_config())
        assert report.actions == []
        assert report.threshold_band_active is None
        assert report.state_row_id is not None

    @pytest.mark.asyncio
    async def test_crossing_80_band_creates_override_for_send_message(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 8.5)  # 85% — crosses 50% and 80%
        report = await enforce_cost_thresholds(pool, config=_config())

        # Both 50% (no demote_actions) and 80% (demote_actions=[send_message])
        # bands fire. Only 80% creates an override.
        override_actions = [a for a in report.actions if a.action_type == "override_created"]
        assert len(override_actions) == 1
        assert override_actions[0].target == "send_message"
        assert override_actions[0].threshold_pct == 80.0

        tier = await get_effective_tier(pool, "send_message")
        assert tier == AutonomyTier.earned  # demoted from always

    @pytest.mark.asyncio
    async def test_idempotent_within_band(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 8.5)
        first = await enforce_cost_thresholds(pool, config=_config())
        assert len(first.actions) >= 1

        # Second tick at same spend: no new actions, but state row updates.
        second = await enforce_cost_thresholds(pool, config=_config())
        assert second.actions == []
        assert second.state_row_id == first.state_row_id

    @pytest.mark.asyncio
    async def test_crossing_100_after_already_at_80_fires_only_new_band(
        self, pool,
    ):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 8.5)
        first = await enforce_cost_thresholds(pool, config=_config())
        first_override_count = len(
            [a for a in first.actions if a.action_type == "override_created"]
        )
        assert first_override_count == 1

        # Push past 100%
        await _spend(pool, 2.0)  # now 105%
        second = await enforce_cost_thresholds(pool, config=_config())
        new_overrides = [a for a in second.actions if a.action_type == "override_created"]
        assert len(new_overrides) == 1
        assert new_overrides[0].threshold_pct == 100.0

        # Strictest of the two stacked overrides wins.
        tier = await get_effective_tier(pool, "send_message")
        assert tier == AutonomyTier.never

    @pytest.mark.asyncio
    async def test_first_tick_above_top_band_fires_all_intermediate_bands(
        self, pool,
    ):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 11.0)  # 110% — above all bands
        report = await enforce_cost_thresholds(pool, config=_config())

        override_pcts = sorted(
            a.threshold_pct
            for a in report.actions
            if a.action_type == "override_created"
        )
        # 50% has no demote_actions; 80% and 100% each create one override.
        assert override_pcts == [80.0, 100.0]

    @pytest.mark.asyncio
    async def test_wildcard_demote_targets_all_enabled_policies(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="action_a", tier=AutonomyTier.always),
        )
        await create_policy(
            pool,
            ActionPolicyCreate(action="action_b", tier=AutonomyTier.always),
        )
        # Disabled policy must not be demoted.
        await create_policy(
            pool,
            ActionPolicyCreate(
                action="action_c", tier=AutonomyTier.always, enabled=False,
            ),
        )

        await _spend(pool, 8.5)
        cfg = _config(thresholds=[
            CostThreshold(
                pct_used=80.0,
                demote_actions=["*"],
                demote_to=CostThresholdLiteral.earned,
            ),
        ])
        report = await enforce_cost_thresholds(pool, config=cfg)
        targets = sorted(
            a.target for a in report.actions if a.action_type == "override_created"
        )
        assert targets == ["action_a", "action_b"]

    @pytest.mark.asyncio
    async def test_baseline_policies_never_demoted_by_enforcement(self, pool):
        # Federation invariant: enforcement never mutates the policy table.
        # Only autonomy_overrides are written. Calibration history stays clean.
        policy = await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 11.0)
        await enforce_cost_thresholds(pool, config=_config())

        baseline = await get_tier_for_action(pool, "send_message")
        assert baseline == AutonomyTier.always  # untouched

        # No calibration events recorded by enforcement
        events = await pool.fetch(
            "SELECT * FROM policy_calibration_events WHERE policy_id = $1",
            policy.id,
        )
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_cost_breach_degradation_policy_fires_when_band_crossed(
        self, pool,
    ):
        await create_deg_policy(
            pool,
            DegradationPolicyCreate(
                name="kill_switch",
                trigger_type=DegradationTriggerType.cost_breach,
                condition={"pct_used": 80.0},
                action=DegradationAction.escalate,
            ),
        )
        await _spend(pool, 8.5)
        report = await enforce_cost_thresholds(pool, config=_config())

        deg_fires = [a for a in report.actions if a.action_type == "degradation_fired"]
        assert len(deg_fires) >= 1
        assert any(f.metadata.get("policy_name") == "kill_switch" for f in deg_fires)

    @pytest.mark.asyncio
    async def test_overrides_expire_at_next_daily_reset(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 8.5)
        await enforce_cost_thresholds(pool, config=_config())

        rows = await pool.fetch(
            "SELECT expires_at FROM autonomy_overrides WHERE source = 'cost_enforcement'"
        )
        assert len(rows) >= 1
        # Expiration must be after now and at most 24h+ away
        expected = _next_daily_reset()
        for r in rows:
            assert r["expires_at"] == expected

    @pytest.mark.asyncio
    async def test_state_row_records_actions_log(self, pool):
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        await _spend(pool, 8.5)
        report = await enforce_cost_thresholds(pool, config=_config())

        row = await pool.fetchrow(
            "SELECT actions_taken, max_threshold_fired_pct, last_pct_used "
            "FROM cost_enforcement_state WHERE id = $1",
            report.state_row_id,
        )
        import json
        actions = row["actions_taken"]
        if isinstance(actions, str):
            actions = json.loads(actions)
        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert row["max_threshold_fired_pct"] == 80.0
        assert 84.0 <= float(row["last_pct_used"]) <= 86.0

    @pytest.mark.asyncio
    async def test_existing_override_from_other_source_not_disturbed(self, pool):
        # A manual lock pre-exists. Cost enforcement adds its own override
        # for the same action. Both should coexist; resolver picks strictest.
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        manual = await create_override(
            pool,
            AutonomyOverrideCreate(
                action="send_message",
                effective_tier=AutonomyTier.never,
                source=OverrideSource.manual,
                reason="manual review hold",
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            ),
        )
        await _spend(pool, 8.5)
        await enforce_cost_thresholds(pool, config=_config())

        # Both overrides survive
        active = await get_active_overrides(pool, action="send_message")
        sources = sorted(o.source.value for o in active)
        assert sources == ["cost_enforcement", "manual"]
        # Strictest still wins
        assert await get_effective_tier(pool, "send_message") == AutonomyTier.never

    @pytest.mark.asyncio
    async def test_skipped_band_when_higher_already_max(self, pool):
        # Backfill scenario: state row says we already fired at 100% earlier
        # today (say from a manual catch-up). A subsequent tick at 85% must
        # not "downgrade" the max — and must not re-fire 80%.
        await create_policy(
            pool,
            ActionPolicyCreate(action="send_message", tier=AutonomyTier.always),
        )
        # Seed state at max_threshold=100
        await pool.execute(
            """
            INSERT INTO cost_enforcement_state
              (id, state_date, user_id, max_threshold_fired_pct,
               daily_limit_usd, last_pct_used, actions_taken)
            VALUES ('weft-seed', CURRENT_DATE,
                    nullif(current_setting('app.user_id', true), ''),
                    100.0, 10.0, 105.0, '[]'::jsonb)
            """
        )
        await _spend(pool, 8.5)  # current spend lands at 80% band
        report = await enforce_cost_thresholds(pool, config=_config())
        assert report.actions == []  # already past this band today
