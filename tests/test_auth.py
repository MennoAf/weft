"""Tests for API key authentication and dual-mode OAuth verifier."""

from unittest.mock import AsyncMock

import pytest

from fastmcp.server.auth import AccessToken
from weft.mcp.auth import ApiKeyVerifier, WeftTokenVerifier, get_auth_provider


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


class TestWeftTokenVerifier:
    """Dual-mode verifier: API key (fast) then Google (slow)."""

    @pytest.fixture
    def mock_google_verifier(self):
        verifier = AsyncMock()
        verifier.verify_token = AsyncMock(return_value=None)
        return verifier

    @pytest.fixture
    def dual_verifier(self, mock_google_verifier):
        return WeftTokenVerifier(
            api_key="test-api-key",
            google_verifier=mock_google_verifier,
        )

    async def test_api_key_takes_priority(self, dual_verifier, mock_google_verifier):
        """API key should be checked first — no Google call needed."""
        result = await dual_verifier.verify_token("test-api-key")
        assert result is not None
        assert result.client_id == "weft-apikey"
        mock_google_verifier.verify_token.assert_not_called()

    async def test_falls_through_to_google(self, dual_verifier, mock_google_verifier):
        """Non-API-key token should fall through to Google verifier."""
        google_token = AccessToken(
            token="google-access-token",
            client_id="google-client",
            scopes=["openid"],
            claims={"sub": "12345", "email": "user@gmail.com"},
        )
        mock_google_verifier.verify_token.return_value = google_token

        result = await dual_verifier.verify_token("google-access-token")
        assert result is not None
        assert result.client_id == "google-client"
        assert result.claims["sub"] == "12345"
        mock_google_verifier.verify_token.assert_called_once_with("google-access-token")

    async def test_both_fail_returns_none(self, dual_verifier, mock_google_verifier):
        """If API key doesn't match and Google rejects, return None."""
        mock_google_verifier.verify_token.return_value = None
        result = await dual_verifier.verify_token("unknown-token")
        assert result is None

    async def test_api_key_only_mode(self):
        """Verifier with no Google fallback — API key only."""
        verifier = WeftTokenVerifier(api_key="my-key")
        result = await verifier.verify_token("my-key")
        assert result is not None
        # Unknown token
        result = await verifier.verify_token("nope")
        assert result is None

    async def test_google_only_mode(self, mock_google_verifier):
        """Verifier with no API key — Google only."""
        verifier = WeftTokenVerifier(google_verifier=mock_google_verifier)
        google_token = AccessToken(
            token="gtoken", client_id="gc", scopes=[], claims={"sub": "u1"},
        )
        mock_google_verifier.verify_token.return_value = google_token
        result = await verifier.verify_token("gtoken")
        assert result is not None
        assert result.claims["sub"] == "u1"
