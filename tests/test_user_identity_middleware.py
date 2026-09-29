"""Tests for user identity resolution and the Phase 2.5 credential-bound
``UserIdentityMiddleware``.

The middleware now resolves Authorization headers through
:func:`weft.credentials.lookup_token`. These tests bootstrap real rows
in the test pool, then drive the middleware via Starlette's ``TestClient``
to verify identity + caller mode end up on the contextvars correctly.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import jwt as pyjwt
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from weft.auth import _reset_auth, current_caller_mode, current_user_id
from weft.credentials import bootstrap_legacy_api_key, issue_token
from weft.db.connection import _resolve_user_id
from weft.mcp.server import UserIdentityMiddleware

_SECRET = "test-supabase-jwt-secret-32chars!"
_DEFAULT_USER = "d445dd9f-8a8e-41a7-a46f-35327d978bb1"
_LEGACY_API_KEY = "test-legacy-api-key-value"


class TestResolveUserId:
    """Test _resolve_user_id which reads from the contextvar."""

    def test_returns_contextvar_when_set(self):
        token = current_user_id.set("user-from-contextvar")
        try:
            assert _resolve_user_id() == "user-from-contextvar"
        finally:
            current_user_id.reset(token)

    def test_returns_none_when_no_auth(self):
        assert current_user_id.get() is None
        assert _resolve_user_id() is None


class TestUserIdentityMiddleware:
    def test_middleware_importable(self):
        from weft.mcp.server import UserIdentityMiddleware, user_identity_middleware
        assert UserIdentityMiddleware is not None
        assert user_identity_middleware is not None


# --- Fixtures for full-stack middleware tests ---


@pytest.fixture(autouse=True)
def _jwt_secret_env():
    _reset_auth()
    with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": _SECRET}):
        yield
    _reset_auth()


def _build_app(
    *,
    pool,
    auth_required: bool = True,
    oauth_enabled: bool = False,
) -> Starlette:
    """Minimal Starlette app with the middleware wired to a /mcp echo
    route. The route returns the contextvar-resolved user_id and
    caller_mode so tests can assert end-to-end behaviour."""

    async def echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "user_id": current_user_id.get(),
                "caller_mode": current_caller_mode.get(),
            }
        )

    app = Starlette(routes=[Route("/mcp", echo)])
    app.add_middleware(
        UserIdentityMiddleware,
        pool_getter=lambda: pool,
        auth_required=auth_required,
        oauth_enabled=oauth_enabled,
    )
    return app


def _make_jwt(sub: str) -> str:
    payload = {"sub": sub, "exp": int(time.time()) + 3600}
    return pyjwt.encode(payload, _SECRET, algorithm="HS256")


def _async_client(app: Starlette) -> httpx.AsyncClient:
    """Drive the ASGI app with httpx — same event loop as the pool
    fixture, so asyncpg's protocol state stays consistent."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# --- Token-row resolution path -----------------------------------------


@pytest.mark.asyncio
async def test_legacy_bootstrap_row_resolves_to_default_user(pool):
    """The L3 bootstrap inserts a row matching WEFT_API_KEY → that
    plaintext as Bearer must resolve to default_user_id, supervisor."""
    await bootstrap_legacy_api_key(
        pool, api_key=_LEGACY_API_KEY, default_user_id=_DEFAULT_USER,
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {_LEGACY_API_KEY}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": _DEFAULT_USER, "caller_mode": "supervisor"}


@pytest.mark.asyncio
async def test_supervisor_token_with_no_header_resolves_to_supervisor(pool):
    plaintext, _ = await issue_token(
        pool, user_id="u-1", caller_mode="supervisor",
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {plaintext}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u-1", "caller_mode": "supervisor"}


@pytest.mark.asyncio
async def test_supervisor_token_with_agent_header_downgrades_to_agent(pool):
    """Supervisor credentials let the operator downgrade themselves
    to ``agent`` for testing — the header IS honoured here."""
    plaintext, _ = await issue_token(
        pool, user_id="u-2", caller_mode="supervisor",
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp",
            headers={
                "authorization": f"Bearer {plaintext}",
                "x-weft-caller-mode": "agent",
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u-2", "caller_mode": "agent"}


@pytest.mark.asyncio
async def test_agent_token_resolves_to_agent_regardless_of_header(pool):
    """The escalation path being closed: an agent token claiming
    'caller_mode: supervisor' must NOT be honoured."""
    plaintext, _ = await issue_token(
        pool, user_id="u-3", caller_mode="agent",
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp",
            headers={
                "authorization": f"Bearer {plaintext}",
                "x-weft-caller-mode": "supervisor",  # ignored
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u-3", "caller_mode": "agent"}


@pytest.mark.asyncio
async def test_agent_token_no_header_still_agent(pool):
    plaintext, _ = await issue_token(
        pool, user_id="u-4", caller_mode="agent",
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {plaintext}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u-4", "caller_mode": "agent"}


# --- 401 paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_authorization_returns_401(pool):
    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get("/mcp")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_unknown_token_returns_401(pool):
    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": "Bearer weft-not-a-real-token"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_revoked_token_returns_401(pool):
    from weft.credentials import revoke_token

    plaintext, row = await issue_token(
        pool, user_id="u-5", caller_mode="supervisor",
    )
    await revoke_token(pool, row.token_hash)

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {plaintext}"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_expired_token_returns_401(pool):
    from datetime import timedelta

    plaintext, _ = await issue_token(
        pool,
        user_id="u-6",
        caller_mode="supervisor",
        expires_in=timedelta(seconds=-1),
    )

    async with _async_client(_build_app(pool=pool)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {plaintext}"},
        )
    assert resp.status_code == 401


# --- OAuth fallback ----------------------------------------------------


@pytest.mark.asyncio
async def test_valid_jwt_resolves_when_no_token_row(pool):
    """OAuth path is the fallback: Bearer doesn't match any row,
    OAuth is enabled → JWT decode happens, sub becomes user_id,
    caller_mode is supervisor (until OAuth scope claim arrives)."""
    jwt_token = _make_jwt("jwt-user-sub")

    async with _async_client(_build_app(pool=pool, oauth_enabled=True)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {jwt_token}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "jwt-user-sub", "caller_mode": "supervisor"}


@pytest.mark.asyncio
async def test_token_row_wins_over_jwt(pool):
    """If a row matches the Bearer, the row wins — even if the
    plaintext also happens to be a valid JWT. Closes a confused-deputy
    risk: a JWT-issuing party could otherwise claim a user_id that
    differs from the row's user_id."""
    jwt_token = _make_jwt("jwt-claims-this-user")
    from weft.credentials import _hash_plaintext

    await pool.execute(
        """
        INSERT INTO weft_tokens (token_hash, user_id, caller_mode)
        VALUES ($1, $2, 'supervisor')
        """,
        _hash_plaintext(jwt_token),
        "row-claims-this-user",
    )

    async with _async_client(_build_app(pool=pool, oauth_enabled=True)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {jwt_token}"},
        )

    assert resp.status_code == 200
    assert resp.json()["user_id"] == "row-claims-this-user"


@pytest.mark.asyncio
async def test_invalid_jwt_with_oauth_returns_401(pool):
    async with _async_client(_build_app(pool=pool, oauth_enabled=True)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": "Bearer not.a.jwt"},
        )
    assert resp.status_code == 401


# --- Auth-not-required mode (local dev) --------------------------------


@pytest.mark.asyncio
async def test_unauth_mode_lets_anonymous_through(pool):
    """When neither WEFT_ENV=production nor OAuth is configured, the
    middleware doesn't enforce auth on /mcp — the legacy local-dev path."""
    async with _async_client(_build_app(pool=pool, auth_required=False)) as client:
        resp = await client.get("/mcp")

    assert resp.status_code == 200
    assert resp.json()["user_id"] is None


@pytest.mark.asyncio
async def test_unauth_mode_still_extracts_jwt_if_present(pool):
    jwt_token = _make_jwt("local-dev-user")

    async with _async_client(_build_app(pool=pool, auth_required=False)) as client:
        resp = await client.get(
            "/mcp", headers={"authorization": f"Bearer {jwt_token}"},
        )

    assert resp.status_code == 200
    assert resp.json()["user_id"] == "local-dev-user"
