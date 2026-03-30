"""Tests for autonomy policy store layer (CRUD + tier query)."""

from __future__ import annotations

import pytest

from weft.autonomy import (
    ActionPolicyCreate,
    AutonomyTier,
    create_policy,
    delete_policy,
    get_policy,
    get_policy_by_action,
    get_tier_for_action,
    list_calibration_events,
    list_policies,
    update_policy_tier,
)


class TestCreateAndGetPolicy:
    @pytest.mark.asyncio
    async def test_create_and_get(self, pool):
        apc = ActionPolicyCreate(
            action="send_slack_message",
            tier=AutonomyTier.earned,
            description="Send messages in Slack",
        )
        policy = await create_policy(pool, apc)
        assert policy.action == "send_slack_message"
        assert policy.tier == AutonomyTier.earned
        assert policy.id.startswith("weft-")
        assert policy.enabled is True

        fetched = await get_policy(pool, policy.id)
        assert fetched is not None
        assert fetched.action == "send_slack_message"
        assert fetched.tier == AutonomyTier.earned

    @pytest.mark.asyncio
    async def test_create_with_conditions(self, pool):
        apc = ActionPolicyCreate(
            action="deploy",
            tier=AutonomyTier.earned,
            conditions={"environment": "staging", "max_cost": 10.0},
        )
        policy = await create_policy(pool, apc)
        assert policy.conditions == {"environment": "staging", "max_cost": 10.0}

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, pool):
        result = await get_policy(pool, "weft-nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_defaults_to_never(self, pool):
        apc = ActionPolicyCreate(action="recall_memory")
        policy = await create_policy(pool, apc)
        assert policy.tier == AutonomyTier.never


class TestGetPolicyByAction:
    @pytest.mark.asyncio
    async def test_finds_enabled_policy(self, pool):
        apc = ActionPolicyCreate(
            action="unique_action_find",
            tier=AutonomyTier.always,
        )
        created = await create_policy(pool, apc)
        found = await get_policy_by_action(pool, "unique_action_find")
        assert found is not None
        assert found.id == created.id
        assert found.tier == AutonomyTier.always

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_action(self, pool):
        result = await get_policy_by_action(pool, "totally_unknown_action")
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_disabled_policy(self, pool):
        apc = ActionPolicyCreate(
            action="disabled_action_test",
            tier=AutonomyTier.always,
            enabled=False,
        )
        await create_policy(pool, apc)
        result = await get_policy_by_action(pool, "disabled_action_test")
        assert result is None


class TestListPolicies:
    @pytest.mark.asyncio
    async def test_list_all(self, pool):
        for action in ["list_a", "list_b", "list_c"]:
            await create_policy(pool, ActionPolicyCreate(action=action))
        policies = await list_policies(pool)
        actions = {p.action for p in policies}
        assert "list_a" in actions
        assert "list_b" in actions
        assert "list_c" in actions

    @pytest.mark.asyncio
    async def test_list_by_tier(self, pool):
        await create_policy(pool, ActionPolicyCreate(
            action="tier_filter_always", tier=AutonomyTier.always,
        ))
        await create_policy(pool, ActionPolicyCreate(
            action="tier_filter_earned", tier=AutonomyTier.earned,
        ))
        always_policies = await list_policies(pool, tier=AutonomyTier.always)
        actions = {p.action for p in always_policies}
        assert "tier_filter_always" in actions
        assert "tier_filter_earned" not in actions

    @pytest.mark.asyncio
    async def test_list_with_limit(self, pool):
        for i in range(5):
            await create_policy(pool, ActionPolicyCreate(action=f"limit_{i}"))
        policies = await list_policies(pool, limit=2)
        assert len(policies) == 2


class TestGetTierForAction:
    @pytest.mark.asyncio
    async def test_returns_tier_for_known_action(self, pool):
        await create_policy(pool, ActionPolicyCreate(
            action="known_tier_action", tier=AutonomyTier.always,
        ))
        tier = await get_tier_for_action(pool, "known_tier_action")
        assert tier == AutonomyTier.always

    @pytest.mark.asyncio
    async def test_defaults_to_earned_for_unknown(self, pool):
        tier = await get_tier_for_action(pool, "completely_unknown_action")
        assert tier == AutonomyTier.earned


class TestUpdatePolicyTier:
    @pytest.mark.asyncio
    async def test_update_earned_to_always(self, pool):
        policy = await create_policy(pool, ActionPolicyCreate(
            action="promote_test", tier=AutonomyTier.earned,
        ))
        updated = await update_policy_tier(
            pool, policy.id, AutonomyTier.always,
            reason="Approved after 10 successful runs",
        )
        assert updated.tier == AutonomyTier.always

        # Verify calibration event was recorded
        events = await list_calibration_events(pool, policy.id)
        assert len(events) == 1
        assert events[0].previous_tier == AutonomyTier.earned
        assert events[0].new_tier == AutonomyTier.always
        assert events[0].reason == "Approved after 10 successful runs"

    @pytest.mark.asyncio
    async def test_never_tier_is_immutable(self, pool):
        policy = await create_policy(pool, ActionPolicyCreate(
            action="hard_stop_test", tier=AutonomyTier.never,
        ))
        with pytest.raises(ValueError, match="NEVER hard-stop"):
            await update_policy_tier(pool, policy.id, AutonomyTier.earned)

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises_lookup_error(self, pool):
        with pytest.raises(LookupError, match="not found"):
            await update_policy_tier(
                pool, "weft-nonexistent", AutonomyTier.always,
            )

    @pytest.mark.asyncio
    async def test_multiple_calibration_events(self, pool):
        policy = await create_policy(pool, ActionPolicyCreate(
            action="multi_cal_test", tier=AutonomyTier.earned,
        ))
        await update_policy_tier(
            pool, policy.id, AutonomyTier.always, reason="first promotion",
        )
        # Demote back — now tier is ALWAYS, which is mutable
        await update_policy_tier(
            pool, policy.id, AutonomyTier.earned, reason="demotion",
        )

        events = await list_calibration_events(pool, policy.id)
        assert len(events) == 2
        # Newest first
        assert events[0].reason == "demotion"
        assert events[1].reason == "first promotion"


class TestDeletePolicy:
    @pytest.mark.asyncio
    async def test_delete_existing(self, pool):
        policy = await create_policy(pool, ActionPolicyCreate(
            action="delete_me",
        ))
        deleted = await delete_policy(pool, policy.id)
        assert deleted is True
        assert await get_policy(pool, policy.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, pool):
        deleted = await delete_policy(pool, "weft-nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_cascades_calibration_events(self, pool):
        policy = await create_policy(pool, ActionPolicyCreate(
            action="cascade_delete_test", tier=AutonomyTier.earned,
        ))
        await update_policy_tier(
            pool, policy.id, AutonomyTier.always, reason="test",
        )
        events_before = await list_calibration_events(pool, policy.id)
        assert len(events_before) == 1

        await delete_policy(pool, policy.id)
        events_after = await list_calibration_events(pool, policy.id)
        assert len(events_after) == 0
