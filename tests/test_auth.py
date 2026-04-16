"""Tests for API key authentication and OAuth proxy dual-auth."""

from unittest.mock import AsyncMock, patch

import pytest

from fastmcp.server.auth import AccessToken
from weft.mcp.auth import ApiKeyVerifier, get_auth_provider, get_oauth_provider


class TestApiKeyVerifier:
    @pytest.fixture
    def verifier(self):
        return ApiKeyVerifier("test-api-key-123")

    async def test_valid_key(self, verifier):
        result = await verifier.verify_token("test-api-key-123")
        assert result is not None
        assert result.client_id == "weft-client"

    async def test_invalid_key(self, verifier):
        result = await verifier.verify_token("wrong-key")
        assert result is None

    async def test_empty_key(self, verifier):
        result = await verifier.verify_token("")
        assert result is None

    async def test_timing_safe_comparison(self, verifier):
        # Ensure we don't short-circuit on partial matches
        result = await verifier.verify_token("test-api-key-12")
        assert result is None


class TestGetAuthProvider:
    def test_local_mode_returns_none(self):
        # Auth disabled in local mode even with api_key set
        assert get_auth_provider("some-key", is_production=False) is None

    def test_local_mode_no_key_returns_none(self):
        assert get_auth_provider(None, is_production=False) is None

    def test_production_with_key(self):
        provider = get_auth_provider("my-key", is_production=True)
        assert provider is not None
        assert isinstance(provider, ApiKeyVerifier)

    def test_production_without_key_raises(self):
        with pytest.raises(ValueError, match="WEFT_API_KEY is required"):
            get_auth_provider(None, is_production=True)

    def test_production_empty_key_raises(self):
        with pytest.raises(ValueError, match="WEFT_API_KEY is required"):
            get_auth_provider("", is_production=True)


class TestWeftOAuthProxy:
    """Dual-auth: API key intercepted before OAuth JWT flow."""

    @pytest.fixture
    def oauth_config(self):
        """Minimal WeftConfig with OAuth configured."""
        from weft.config import OAuthConfig, WeftConfig
        config = WeftConfig(api_key="test-api-key")
        config.oauth = OAuthConfig(
            client_id="fake-client-id",
            client_secret="fake-client-secret",
            base_url="https://example.com",
        )
        return config

    @pytest.fixture
    def proxy(self, oauth_config):
        """Build a WeftOAuthProxy with a mock pool factory."""
        mock_pool_factory = lambda: AsyncMock()
        provider = get_oauth_provider(oauth_config, mock_pool_factory)
        assert provider is not None
        return provider

    async def test_api_key_accepted(self, proxy):
        """API key should be accepted without hitting OAuth JWT validation."""
        result = await proxy.load_access_token("test-api-key")
        assert result is not None
        assert result.client_id == "weft-apikey"

    async def test_wrong_key_falls_through(self, proxy):
        """Non-API-key token should fall through to OAuth JWT flow (which returns None for invalid JWTs)."""
        result = await proxy.load_access_token("not-the-api-key")
        assert result is None

    async def test_empty_token_rejected(self, proxy):
        """Empty token should not match API key."""
        result = await proxy.load_access_token("")
        assert result is None

    def test_required_scopes_set(self, proxy):
        """Proxy should advertise openid and email scopes for Google."""
        assert "openid" in proxy.required_scopes
        assert "email" in proxy.required_scopes

    def test_not_configured_returns_none(self):
        """When OAuth env vars are missing, returns None (API key mode only)."""
        from weft.config import WeftConfig
        config = WeftConfig()
        result = get_oauth_provider(config, lambda: AsyncMock())
        assert result is None
