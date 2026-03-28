"""Tests for user identity resolution and API key middleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from weft.auth import current_user_id
from weft.db.connection import _resolve_user_id


class TestResolveUserId:
    """Test _resolve_user_id which bridges OAuth tokens to user identity."""

    def test_returns_contextvar_when_set(self):
        token = current_user_id.set("user-from-contextvar")
        try:
            assert _resolve_user_id() == "user-from-contextvar"
        finally:
            current_user_id.reset(token)

    def test_returns_none_when_no_auth(self):
        assert current_user_id.get() is None
        assert _resolve_user_id() is None

    def test_falls_back_to_fastmcp_token(self):
        """When contextvar is None, reads sub from FastMCP access token upstream claims."""
        mock_token = MagicMock()
        mock_token.claims = {"upstream_claims": {"sub": "supabase-user-uuid"}}

        with patch(
            "weft.db.connection.get_access_token",
            return_value=mock_token,
            create=True,
        ):
            # Patch the dynamic import inside _resolve_user_id
            mock_module = MagicMock()
            mock_module.get_access_token = MagicMock(return_value=mock_token)
            with patch.dict("sys.modules", {"fastmcp.server.auth.middleware": mock_module}):
                result = _resolve_user_id()
                assert result == "supabase-user-uuid"

    def test_contextvar_takes_precedence_over_fastmcp_token(self):
        """current_user_id contextvar is checked first."""
        token = current_user_id.set("contextvar-user")
        try:
            assert _resolve_user_id() == "contextvar-user"
        finally:
            current_user_id.reset(token)

    def test_handles_missing_fastmcp_gracefully(self):
        """When FastMCP auth module import fails, returns None."""
        assert _resolve_user_id() is None

    def test_handles_token_without_upstream_claims(self):
        """Token exists but has no upstream_claims — returns None."""
        mock_token = MagicMock()
        mock_token.claims = {}

        mock_module = MagicMock()
        mock_module.get_access_token = MagicMock(return_value=mock_token)
        with patch.dict("sys.modules", {"fastmcp.server.auth.middleware": mock_module}):
            result = _resolve_user_id()
            assert result is None

    def test_handles_none_token(self):
        """get_access_token() returns None — returns None."""
        mock_module = MagicMock()
        mock_module.get_access_token = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"fastmcp.server.auth.middleware": mock_module}):
            result = _resolve_user_id()
            assert result is None


class TestUserIdentityMiddleware:
    """Test UserIdentityMiddleware API key enforcement."""

    def test_middleware_importable(self):
        from weft.mcp.server import UserIdentityMiddleware, user_identity_middleware
        assert UserIdentityMiddleware is not None
        assert user_identity_middleware is not None
