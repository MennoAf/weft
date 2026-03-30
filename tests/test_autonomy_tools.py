"""Tests for autonomy MCP tools — weft_autonomy_check, weft_autonomy_set,
weft_autonomy_list, weft_autonomy_calibrate."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


class FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# ── Registration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_autonomy_tools_registered():
    from weft.mcp.tools import mcp

    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "weft_autonomy_check" in tool_names
    assert "weft_autonomy_set" in tool_names
    assert "weft_autonomy_list" in tool_names
    assert "weft_autonomy_calibrate" in tool_names


# ── weft_autonomy_check ────────────────────────────────────────────


class TestAutonomyCheck:
    @pytest.mark.asyncio
    async def test_check_unknown_defaults_to_earned(self, ctx):
        from weft.mcp.tools import weft_autonomy_check

        result = await weft_autonomy_check(ctx, action="unknown_action")
        assert result["tier"] == "earned"
        assert result["requires_approval"] is True
        assert result["permitted"] is False
        assert result["blocked"] is False
        assert result["policy_id"] is None
        assert "rationale" in result

    @pytest.mark.asyncio
    async def test_check_known_action(self, ctx):
        from weft.mcp.tools import weft_autonomy_check, weft_autonomy_set

        await weft_autonomy_set(ctx, action="status_check", tier="always")
        result = await weft_autonomy_check(ctx, action="status_check")
        assert result["tier"] == "always"
        assert result["permitted"] is True
        assert result["requires_approval"] is False
        assert result["blocked"] is False
        assert result["policy_id"] is not None

    @pytest.mark.asyncio
    async def test_check_never_action(self, ctx):
        from weft.mcp.tools import weft_autonomy_check, weft_autonomy_set

        await weft_autonomy_set(
            ctx, action="delete_production_db", tier="never",
            description="Hard stop",
        )
        result = await weft_autonomy_check(ctx, action="delete_production_db")
        assert result["tier"] == "never"
        assert result["blocked"] is True
        assert result["permitted"] is False


# ── weft_autonomy_set ──────────────────────────────────────────────


class TestAutonomySet:
    @pytest.mark.asyncio
    async def test_set_minimal(self, ctx):
        from weft.mcp.tools import weft_autonomy_set

        result = await weft_autonomy_set(ctx, action="recall_memory")
        assert result["success"] is True
        policy = result["policy"]
        assert policy["action"] == "recall_memory"
        assert policy["tier"] == "never"
        assert policy["enabled"] is True

    @pytest.mark.asyncio
    async def test_set_with_all_fields(self, ctx):
        from weft.mcp.tools import weft_autonomy_set

        result = await weft_autonomy_set(
            ctx,
            action="deploy",
            tier="earned",
            description="Deploy to staging",
            conditions={"environment": "staging"},
            enabled=True,
        )
        assert result["success"] is True
        policy = result["policy"]
        assert policy["tier"] == "earned"
        assert policy["description"] == "Deploy to staging"
        assert policy["conditions"] == {"environment": "staging"}

    @pytest.mark.asyncio
    async def test_set_invalid_tier(self, ctx):
        from weft.mcp.tools import weft_autonomy_set

        result = await weft_autonomy_set(ctx, action="test", tier="invalid")
        assert "error" in result
        assert "invalid" in result["detail"].lower()


# ── weft_autonomy_list ─────────────────────────────────────────────


class TestAutonomyList:
    @pytest.mark.asyncio
    async def test_list_returns_policies(self, ctx):
        from weft.mcp.tools import weft_autonomy_list

        result = await weft_autonomy_list(ctx)
        assert "count" in result
        assert "policies" in result
        assert isinstance(result["policies"], list)

    @pytest.mark.asyncio
    async def test_list_includes_created(self, ctx):
        from weft.mcp.tools import weft_autonomy_list, weft_autonomy_set

        await weft_autonomy_set(ctx, action="list_test_a", tier="always")
        await weft_autonomy_set(ctx, action="list_test_b", tier="never")
        result = await weft_autonomy_list(ctx)
        actions = {p["action"] for p in result["policies"]}
        assert "list_test_a" in actions
        assert "list_test_b" in actions

    @pytest.mark.asyncio
    async def test_list_filter_by_tier(self, ctx):
        from weft.mcp.tools import weft_autonomy_list, weft_autonomy_set

        await weft_autonomy_set(ctx, action="tier_filter_a", tier="always")
        await weft_autonomy_set(ctx, action="tier_filter_n", tier="never")
        result = await weft_autonomy_list(ctx, tier="always")
        actions = {p["action"] for p in result["policies"]}
        assert "tier_filter_a" in actions
        assert "tier_filter_n" not in actions

    @pytest.mark.asyncio
    async def test_list_invalid_tier(self, ctx):
        from weft.mcp.tools import weft_autonomy_list

        result = await weft_autonomy_list(ctx, tier="bogus")
        assert "error" in result


# ── weft_autonomy_calibrate ────────────────────────────────────────


class TestAutonomyCalibrate:
    @pytest.mark.asyncio
    async def test_calibrate_earned_to_always(self, ctx):
        from weft.mcp.tools import weft_autonomy_calibrate, weft_autonomy_set

        r = await weft_autonomy_set(ctx, action="promote_me", tier="earned")
        policy_id = r["policy"]["id"]

        result = await weft_autonomy_calibrate(
            ctx,
            policy_id=policy_id,
            new_tier="always",
            reason="Approved after testing",
        )
        assert result["success"] is True
        assert result["policy"]["tier"] == "always"
        assert len(result["calibration_history"]) == 1
        assert result["calibration_history"][0]["reason"] == "Approved after testing"

    @pytest.mark.asyncio
    async def test_calibrate_never_blocked(self, ctx):
        from weft.mcp.tools import weft_autonomy_calibrate, weft_autonomy_set

        r = await weft_autonomy_set(ctx, action="hard_stop", tier="never")
        policy_id = r["policy"]["id"]

        result = await weft_autonomy_calibrate(
            ctx, policy_id=policy_id, new_tier="earned",
        )
        assert "error" in result
        assert "NEVER" in result["detail"] or "hard-stop" in result["detail"]

    @pytest.mark.asyncio
    async def test_calibrate_nonexistent_policy(self, ctx):
        from weft.mcp.tools import weft_autonomy_calibrate

        result = await weft_autonomy_calibrate(
            ctx, policy_id="weft-nonexist", new_tier="always",
        )
        assert "error" in result
        assert "not found" in result["detail"]

    @pytest.mark.asyncio
    async def test_calibrate_invalid_tier(self, ctx):
        from weft.mcp.tools import weft_autonomy_calibrate

        result = await weft_autonomy_calibrate(
            ctx, policy_id="weft-12345678", new_tier="bogus",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_calibrate_invalid_policy_id_format(self, ctx):
        from weft.mcp.tools import weft_autonomy_calibrate

        result = await weft_autonomy_calibrate(
            ctx, policy_id="bad-format", new_tier="always",
        )
        assert "error" in result


# ── Primer integration ──────────────────────────────────────────────


class TestPrimerAutonomySection:
    @pytest.mark.asyncio
    async def test_primer_includes_autonomy_section(self, pool):
        from weft.primer import build_primer

        result = await build_primer(pool, disclosure="progressive")
        assert "autonomy" in result

    @pytest.mark.asyncio
    async def test_primer_autonomy_full_disclosure(self, pool):
        from weft.autonomy import ActionPolicyCreate, AutonomyTier, create_policy
        from weft.primer import build_primer

        await create_policy(pool, ActionPolicyCreate(
            action="primer_test_action", tier=AutonomyTier.always,
            description="Test policy for primer",
        ))

        result = await build_primer(pool, disclosure="full")
        assert "autonomy" in result
        autonomy = result["autonomy"]
        assert isinstance(autonomy, list)
        actions = {p["action"] for p in autonomy}
        assert "primer_test_action" in actions

    @pytest.mark.asyncio
    async def test_primer_autonomy_progressive_deferred(self, pool):
        from weft.primer import build_primer

        result = await build_primer(pool, disclosure="progressive")
        autonomy = result["autonomy"]
        assert "deferred" in autonomy
        assert autonomy["deferred"] is True
