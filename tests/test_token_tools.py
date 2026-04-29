"""Tests for `weft_token_issue / weft_token_list / weft_token_revoke`
(Phase 2.5 / L6 — MCP surface for credential management).

The tools are supervisor-only. Every test below either runs in the
default supervisor mode or explicitly sets ``current_caller_mode`` to
``'agent'`` and asserts the supervisor gate refuses the call.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.auth import current_caller_mode
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@contextmanager
def _as_caller_mode(mode: str):
    tok = current_caller_mode.set(mode)
    try:
        yield
    finally:
        current_caller_mode.reset(tok)


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbedding(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# --- Issue ----------------------------------------------------------------


class TestIssue:
    async def test_supervisor_can_mint_supervisor_token(self, ctx):
        from weft.mcp.tools import weft_token_issue

        result = await weft_token_issue(
            ctx,
            user_id="u-mcp",
            caller_mode="supervisor",
            label="face-2026",
        )
        assert result["token"].startswith("weft-")
        assert len(result["token_hash"]) == 64
        assert result["user_id"] == "u-mcp"
        assert result["caller_mode"] == "supervisor"
        assert result["label"] == "face-2026"
        assert result["expires_at"] is None
        assert "warning" in result

    async def test_supervisor_can_mint_agent_token_with_expiry(self, ctx):
        from weft.mcp.tools import weft_token_issue

        result = await weft_token_issue(
            ctx,
            user_id="u-mcp",
            caller_mode="agent",
            label="wick-runtime",
            expires_in="7d",
        )
        assert result["caller_mode"] == "agent"
        assert result["expires_at"] is not None

    async def test_agent_caller_blocked(self, ctx):
        from weft.mcp.tools import weft_token_issue

        with _as_caller_mode("agent"):
            result = await weft_token_issue(
                ctx, user_id="u-mcp", caller_mode="supervisor",
            )
        assert "error" in result
        assert "supervisor-only" in result["error"]
        assert result["tool"] == "weft_token_issue"

    async def test_invalid_expires_in_returns_input_error(self, ctx):
        from weft.mcp.tools import weft_token_issue

        result = await weft_token_issue(
            ctx,
            user_id="u-mcp",
            caller_mode="supervisor",
            expires_in="forever",
        )
        assert result.get("error") == "Invalid input"
        assert result["tool"] == "weft_token_issue"

    async def test_invalid_caller_mode_returns_input_error(self, ctx):
        """credentials.issue_token raises ValueError on bad caller_mode —
        the tool wraps that into the standard input-error envelope so the
        client gets a structured response rather than an exception."""
        from weft.mcp.tools import weft_token_issue

        result = await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="root",  # type: ignore[arg-type]
        )
        assert result.get("error") == "Invalid input"


# --- List -----------------------------------------------------------------


class TestList:
    async def test_list_returns_issued_tokens_newest_first(self, ctx):
        from weft.mcp.tools import weft_token_issue, weft_token_list

        await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="supervisor", label="first",
        )
        await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="agent", label="second",
        )
        result = await weft_token_list(ctx, user_id="u-mcp")

        assert result["count"] == 2
        labels = [t["label"] for t in result["tokens"]]
        assert labels == ["second", "first"]
        statuses = {t["status"] for t in result["tokens"]}
        assert statuses == {"active"}
        # token_hash is full-length so it can be passed to revoke
        assert all(len(t["token_hash"]) == 64 for t in result["tokens"])

    async def test_list_default_excludes_revoked(self, ctx):
        from weft.mcp.tools import (
            weft_token_issue, weft_token_list, weft_token_revoke,
        )

        issued = await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="agent", label="ephemeral",
        )
        await weft_token_revoke(ctx, token_hash=issued["token_hash"])

        result = await weft_token_list(ctx, user_id="u-mcp")
        assert result["count"] == 0

    async def test_list_include_revoked_surfaces_revoked(self, ctx):
        from weft.mcp.tools import (
            weft_token_issue, weft_token_list, weft_token_revoke,
        )

        issued = await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="agent", label="ephemeral",
        )
        await weft_token_revoke(ctx, token_hash=issued["token_hash"])

        result = await weft_token_list(ctx, user_id="u-mcp", include_revoked=True)
        assert result["count"] == 1
        assert result["tokens"][0]["status"] == "revoked"
        assert result["tokens"][0]["revoked_at"] is not None

    async def test_list_empty_when_no_tokens(self, ctx):
        from weft.mcp.tools import weft_token_list

        result = await weft_token_list(ctx, user_id="u-mcp")
        assert result == {"count": 0, "tokens": []}

    async def test_agent_caller_blocked(self, ctx):
        from weft.mcp.tools import weft_token_list

        with _as_caller_mode("agent"):
            result = await weft_token_list(ctx, user_id="u-mcp")
        assert "error" in result
        assert "supervisor-only" in result["error"]


# --- Revoke ---------------------------------------------------------------


class TestRevoke:
    async def test_revoke_flips_status(self, ctx):
        from weft.mcp.tools import (
            weft_token_issue, weft_token_list, weft_token_revoke,
        )

        issued = await weft_token_issue(
            ctx, user_id="u-mcp", caller_mode="supervisor",
        )
        result = await weft_token_revoke(ctx, token_hash=issued["token_hash"])
        assert result == {
            "token_hash": issued["token_hash"],
            "revoked": True,
            "detail": "Token revoked.",
        }

        # Second revoke is a no-op (idempotent)
        again = await weft_token_revoke(ctx, token_hash=issued["token_hash"])
        assert again["revoked"] is False
        assert "already revoked" in again["detail"]

    async def test_revoke_unknown_hash_reports_no_match(self, ctx):
        from weft.mcp.tools import weft_token_revoke

        result = await weft_token_revoke(ctx, token_hash="0" * 64)
        assert result["revoked"] is False
        assert "No live token matched" in result["detail"]

    async def test_revoke_rejects_short_hash(self, ctx):
        from weft.mcp.tools import weft_token_revoke

        result = await weft_token_revoke(ctx, token_hash="abc123")
        assert result.get("error") == "Invalid input"
        assert "64-char" in result["detail"]

    async def test_agent_caller_blocked(self, ctx):
        from weft.mcp.tools import weft_token_revoke

        with _as_caller_mode("agent"):
            result = await weft_token_revoke(ctx, token_hash="0" * 64)
        assert "error" in result
        assert "supervisor-only" in result["error"]


# --- Cross-tool round-trip ------------------------------------------------


async def test_round_trip_issue_then_list_then_revoke(ctx):
    """The trip an operator actually takes: mint → see it in list →
    revoke by the listed hash → it disappears from the live list."""
    from weft.mcp.tools import (
        weft_token_issue, weft_token_list, weft_token_revoke,
    )

    issued = await weft_token_issue(
        ctx, user_id="u-roundtrip", caller_mode="agent", label="wick-1",
    )

    listed = await weft_token_list(ctx, user_id="u-roundtrip")
    listed_hashes = [t["token_hash"] for t in listed["tokens"]]
    assert issued["token_hash"] in listed_hashes

    revoked = await weft_token_revoke(ctx, token_hash=issued["token_hash"])
    assert revoked["revoked"] is True

    after = await weft_token_list(ctx, user_id="u-roundtrip")
    assert after["count"] == 0
