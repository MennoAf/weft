"""Tests for user identity resolution and API key middleware."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from weft.auth import _reset_auth, current_user_id
from weft.db.connection import _resolve_user_id
from weft.mcp.server import UserIdentityMiddleware

_SECRET = "test-supabase-jwt-secret-32chars!"
_DEFAULT_USER = "d445dd9f-8a8e-41a7-a46f-35327d978bb1"
_API_KEY = "test-api-key-value"


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
    """Test UserIdentityMiddleware API key enforcement."""

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


def _build_app(*, api_key: str | None, default_user_id: str | None) -> Starlette:
    """Build a minimal Starlette app with the middleware wired to a /mcp echo route."""

    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"user_id": current_user_id.get()})

    app = Starlette(routes=[Route("/mcp", echo)])
    app.add_middleware(
        UserIdentityMiddleware,
        api_key=api_key,
        default_user_id=default_user_id,
    )
    return app


def _make_jwt(sub: str) -> str:
    payload = {"sub": sub, "exp": int(time.time()) + 3600}
    return pyjwt.encode(payload, _SECRET, algorithm="HS256")


class TestDefaultUserIdFallback:
    """Single-tenant WEFT_DEFAULT_USER_ID fallback behavior."""

    def test_api_key_auth_no_jwt_stamps_default(self):
        """API key matches + no JWT + default set → default stamped."""
        client = TestClient(_build_app(api_key=_API_KEY, default_user_id=_DEFAULT_USER))
        resp = client.get("/mcp", headers={"authorization": f"Bearer {_API_KEY}"})
        assert resp.status_code == 200
        assert resp.json() == {"user_id": _DEFAULT_USER}

    def test_api_key_auth_with_valid_jwt_prefers_jwt(self):
        """If Bearer is a valid JWT that also equals the API key, JWT sub wins.

        In practice the API key and a JWT are different strings, so the api_key
        hmac check would reject a JWT Bearer. This test guards the precedence
        logic for the one edge case where they happened to collide, plus
        documents that a future separate-header flow would preserve JWT-wins.
        """
        jwt_token = _make_jwt("jwt-user-sub")
        # Configure api_key == jwt_token so the hmac check passes.
        client = TestClient(_build_app(api_key=jwt_token, default_user_id=_DEFAULT_USER))
        resp = client.get("/mcp", headers={"authorization": f"Bearer {jwt_token}"})
        assert resp.status_code == 200
        assert resp.json() == {"user_id": "jwt-user-sub"}

    def test_missing_api_key_rejected_default_not_applied(self):
        """No Authorization header → 401, default never applied."""
        client = TestClient(_build_app(api_key=_API_KEY, default_user_id=_DEFAULT_USER))
        resp = client.get("/mcp")
        assert resp.status_code == 401

    def test_wrong_api_key_rejected_default_not_applied(self):
        """Bearer present but wrong → 401, default never applied."""
        client = TestClient(_build_app(api_key=_API_KEY, default_user_id=_DEFAULT_USER))
        resp = client.get("/mcp", headers={"authorization": "Bearer not-the-key"})
        assert resp.status_code == 401

    def test_no_api_key_configured_default_not_applied(self):
        """Unprotected endpoint + default set → default ignored.

        Default is coupled to the api_key gate so anonymous traffic can't
        inherit the owner's identity.
        """
        client = TestClient(_build_app(api_key=None, default_user_id=_DEFAULT_USER))
        resp = client.get("/mcp")
        assert resp.status_code == 200
        assert resp.json() == {"user_id": None}

    def test_default_unset_stamps_none(self):
        """API key matches but no default → user_id remains None (current behavior)."""
        client = TestClient(_build_app(api_key=_API_KEY, default_user_id=None))
        resp = client.get("/mcp", headers={"authorization": f"Bearer {_API_KEY}"})
        assert resp.status_code == 200
        assert resp.json() == {"user_id": None}
