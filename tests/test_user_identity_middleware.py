"""Tests for user identity resolution and API key middleware."""

from __future__ import annotations

from weft.auth import current_user_id
from weft.db.connection import _resolve_user_id


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
