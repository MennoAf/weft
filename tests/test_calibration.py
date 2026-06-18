"""Tests for auto-calibration promotion wired into record_calibration.

Verifies that when enough approving calibration records are recorded for an
action category that has a matching autonomy policy, the policy tier is
automatically promoted and a policy_calibration_events row with reason
prefixed 'auto-calibration' is written — with NO manual call to
weft_autonomy_calibrate (update_policy_tier).
"""

from __future__ import annotations

import pytest

from weft.alerts import list_alerts
from weft.autonomy import (
    ActionPolicyCreate,
    AutonomyTier,
    create_policy,
    get_policy,
    list_calibration_events,
)
from weft.calibration import count_auto_originated_tier_changes, record_calibration
from weft.models import CalibrationCreate, CalibrationOutcome

# Must cross _PROMO_MIN_RECORDS=5 with _PROMO_APPROVAL_RATE=0.8 (80%)
_PROMO_COUNT = 5


@pytest.mark.asyncio
async def test_auto_promotion_on_sufficient_approvals(pool):
    """After enough approvals, record_calibration auto-promotes the policy tier."""
    # Arrange: create a policy at 'earned' tier for the action category
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="send_slack_message", tier=AutonomyTier.earned),
    )
    assert policy.tier == AutonomyTier.earned

    # Act: record enough approved calibrations to cross the promotion threshold
    # _PROMO_MIN_RECORDS=5, _PROMO_APPROVAL_RATE=0.8 → all 5 approved (100% ≥ 80%)
    for _ in range(_PROMO_COUNT):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="send_slack_message",
                action_description="Send a message to #ops",
                outcome=CalibrationOutcome.approved,
            ),
        )

    # Assert: policy tier was auto-promoted to 'always'
    updated_policy = await get_policy(pool, policy.id)
    assert updated_policy is not None
    assert updated_policy.tier == AutonomyTier.always, (
        f"Expected tier 'always' after auto-calibration, got '{updated_policy.tier}'"
    )

    # Assert: policy_calibration_events row exists with reason prefixed 'auto-calibration'
    events = await list_calibration_events(pool, policy.id)
    auto_events = [e for e in events if e.reason and e.reason.startswith("auto-calibration")]
    assert len(auto_events) >= 1, (
        f"Expected at least one policy_calibration_events row with reason "
        f"starting 'auto-calibration', found: {[e.reason for e in events]}"
    )
    assert auto_events[0].previous_tier == AutonomyTier.earned
    assert auto_events[0].new_tier == AutonomyTier.always


@pytest.mark.asyncio
async def test_no_auto_promotion_below_threshold(pool):
    """Fewer than _PROMO_MIN_RECORDS approvals should NOT trigger promotion."""
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="deploy_staging", tier=AutonomyTier.earned),
    )

    # Record only 4 approvals — one below the threshold of 5
    for _ in range(_PROMO_COUNT - 1):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="deploy_staging",
                action_description="Deploy to staging",
                outcome=CalibrationOutcome.approved,
            ),
        )

    # Tier should remain 'earned'
    updated = await get_policy(pool, policy.id)
    assert updated is not None
    assert updated.tier == AutonomyTier.earned

    events = await list_calibration_events(pool, policy.id)
    assert len(events) == 0


@pytest.mark.asyncio
async def test_no_auto_promotion_on_demotion_recommendation(pool):
    """High rejection rate triggers 'demote' recommendation — must NOT be auto-applied."""
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="delete_file", tier=AutonomyTier.earned),
    )

    # Record 3 rejections — crosses _DEMO_MIN_RECORDS=3 at _DEMO_REJECTION_RATE=0.5
    for _ in range(3):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="delete_file",
                action_description="Delete a file",
                outcome=CalibrationOutcome.rejected,
            ),
        )

    # Tier should remain 'earned' — demotions are NOT auto-applied here
    updated = await get_policy(pool, policy.id)
    assert updated is not None
    assert updated.tier == AutonomyTier.earned

    events = await list_calibration_events(pool, policy.id)
    assert len(events) == 0


@pytest.mark.asyncio
async def test_no_auto_promotion_when_no_policy(pool):
    """If no policy exists for the action category, record_calibration succeeds silently."""
    # No policy created for 'mystery_action'
    record = await record_calibration(
        pool,
        CalibrationCreate(
            action_category="mystery_action",
            action_description="An action with no policy",
            outcome=CalibrationOutcome.approved,
        ),
    )
    assert record is not None
    assert record.action_category == "mystery_action"


@pytest.mark.asyncio
async def test_already_at_always_tier_no_duplicate_event(pool):
    """If policy is already at 'always', no additional calibration event is written."""
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="read_file", tier=AutonomyTier.always),
    )

    for _ in range(_PROMO_COUNT):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="read_file",
                action_description="Read a file",
                outcome=CalibrationOutcome.approved,
            ),
        )

    updated = await get_policy(pool, policy.id)
    assert updated is not None
    assert updated.tier == AutonomyTier.always

    # No calibration events (policy was already at always, no tier change needed)
    events = await list_calibration_events(pool, policy.id)
    assert len(events) == 0


# Must cross _DEMO_MIN_RECORDS=3 with _DEMO_REJECTION_RATE=0.5 (50%)
_DEMO_COUNT = 3


@pytest.mark.asyncio
async def test_demotion_alert_created_tier_unchanged(pool):
    """When rejection rate crosses the demote threshold, an alert is created
    proposing the demotion — but the autonomy tier must remain UNCHANGED.
    """
    # Arrange: create a policy at 'earned' tier
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="archive_project", tier=AutonomyTier.earned),
    )
    assert policy.tier == AutonomyTier.earned

    # Act: record enough rejecting calibrations to cross the demotion threshold
    # _DEMO_MIN_RECORDS=3, _DEMO_REJECTION_RATE=0.5 → all 3 rejected (100% ≥ 50%)
    for _ in range(_DEMO_COUNT):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="archive_project",
                action_description="Archive a project",
                outcome=CalibrationOutcome.rejected,
            ),
        )

    # Assert: autonomy tier is UNCHANGED — demotions must never be auto-applied
    updated_policy = await get_policy(pool, policy.id)
    assert updated_policy is not None
    assert updated_policy.tier == AutonomyTier.earned, (
        f"Expected tier to remain 'earned' after demotion recommendation, "
        f"got '{updated_policy.tier}'"
    )

    # Assert: no policy_calibration_events rows (no tier change was applied)
    events = await list_calibration_events(pool, policy.id)
    assert len(events) == 0, (
        f"Expected no calibration events (no tier change), got {len(events)}"
    )

    # Assert: a demotion-proposal alert was created
    alerts = await list_alerts(pool)
    demotion_alerts = [
        a for a in alerts
        if a.payload.get("action_category") == "archive_project"
    ]
    assert len(demotion_alerts) >= 1, (
        f"Expected at least one demotion alert for 'archive_project', "
        f"found {len(alerts)} total alerts"
    )
    alert = demotion_alerts[0]
    assert "archive_project" in alert.body, (
        f"Alert body should mention the action category, got: {alert.body!r}"
    )
    assert "rejection" in alert.body.lower(), (
        f"Alert body should mention rejection rate, got: {alert.body!r}"
    )


@pytest.mark.asyncio
async def test_count_auto_originated_tier_changes(pool):
    """Verify count_auto_originated_tier_changes counts only auto-calibration events.

    Seeds a policy with calibration records to trigger auto-promotion, then
    verifies that count_auto_originated_tier_changes returns the correct count
    (only events with reason LIKE 'auto-calibration%') and excludes manual
    tier changes.
    """
    # Arrange: create a policy at 'earned' tier
    policy = await create_policy(
        pool,
        ActionPolicyCreate(action="test_auto_origination", tier=AutonomyTier.earned),
    )
    assert policy.tier == AutonomyTier.earned

    # Act: record enough approved calibrations to trigger auto-promotion
    # _PROMO_MIN_RECORDS=5, _PROMO_APPROVAL_RATE=0.8 → all 5 approved (100% ≥ 80%)
    for _ in range(_PROMO_COUNT):
        await record_calibration(
            pool,
            CalibrationCreate(
                action_category="test_auto_origination",
                action_description="Test action",
                outcome=CalibrationOutcome.approved,
            ),
        )

    # Assert: policy was auto-promoted and a calibration event with auto-calibration reason exists
    updated_policy = await get_policy(pool, policy.id)
    assert updated_policy is not None
    assert updated_policy.tier == AutonomyTier.always

    events = await list_calibration_events(pool, policy.id)
    auto_events = [e for e in events if e.reason and e.reason.startswith("auto-calibration")]
    assert len(auto_events) >= 1

    # Act: count auto-originated tier changes using the new function (no time window)
    count = await count_auto_originated_tier_changes(pool)

    # Assert: count should be at least 1 (the auto-promotion we just triggered)
    assert count >= 1, (
        f"Expected at least 1 auto-originated tier change, got {count}"
    )

    # Act: manually create a second policy and manually change its tier to simulate
    # a human-initiated change (non auto-calibration reason)
    from weft.autonomy import update_policy_tier
    policy2 = await create_policy(
        pool,
        ActionPolicyCreate(action="test_manual_change", tier=AutonomyTier.earned),
    )
    await update_policy_tier(
        pool,
        policy2.id,
        AutonomyTier.always,
        reason="manual: human review required",
        agent_id="test-agent",
    )

    # Act: count again
    count_after_manual = await count_auto_originated_tier_changes(pool)

    # Assert: count should be unchanged (manual event should not be counted)
    assert count_after_manual == count, (
        f"Expected count to remain {count} after manual tier change, "
        f"got {count_after_manual}"
    )
