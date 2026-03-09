"""Tests for UserIdentityMiddleware in weft/mcp/server.py."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from weft.auth import current_user_id
from weft.mcp.server import UserIdentityMiddleware

_SECRET = "test-supabase-jwt-secret-32chars!"


def _make_jwt(sub: str = "test-user-uuid", exp: int | None = None) -> str:
    payload: dict = {"sub": sub}
    payload["exp"] = exp if exp is not None else int(time.time()) + 3600
    return pyjwt.encode(payload, _SECRET, algorithm="HS256")


def _echo_user_id(request: Request) -> JSONResponse:
    """Handler that returns the current_user_id contextvar value."""
    uid = current_user_id.get()
    return JSONResponse({"user_id": uid})


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": _SECRET}):
        yield


@pytest.fixture
def client():
    app = Starlette(
        routes=[Route("/test", _echo_user_id)],
        middleware=[Middleware(UserIdentityMiddleware)],
    )
    return TestClient(app)


class TestUserIdentityMiddleware:
    def test_valid_bearer_sets_user_id(self, client):
        token = _make_jwt(sub="user-abc")
        resp = client.get("/test", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "user-abc"

    def test_no_auth_header_passes_through(self, client):
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json()["user_id"] is None

    def test_invalid_token_passes_through(self, client):
        resp = client.get("/test", headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] is None

    def test_expired_token_passes_through(self, client):
        token = _make_jwt(exp=int(time.time()) - 100)
        resp = client.get("/test", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] is None

    def test_wrong_scheme_passes_through(self, client):
        token = _make_jwt()
        resp = client.get("/test", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] is None

    def test_contextvar_reset_after_request(self, client):
        token = _make_jwt(sub="user-reset-test")
        client.get("/test", headers={"Authorization": f"Bearer {token}"})
        # After request completes, contextvar should be reset
        assert current_user_id.get() is None

    def test_empty_bearer_passes_through(self, client):
        resp = client.get("/test", headers={"Authorization": "Bearer "})
        assert resp.status_code == 200
        assert resp.json()["user_id"] is None
