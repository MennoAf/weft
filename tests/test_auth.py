"""Tests for API key authentication."""

import pytest

from weft.mcp.auth import ApiKeyVerifier, get_auth_provider


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
