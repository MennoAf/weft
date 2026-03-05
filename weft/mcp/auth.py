"""API key authentication for the Weft MCP server."""

from __future__ import annotations

import hmac
import logging

from fastmcp.server.auth import AccessToken, TokenVerifier

logger = logging.getLogger(__name__)


class ApiKeyVerifier(TokenVerifier):
    """Verifies bearer tokens against a single API key.

    Used for production deployments where MCP clients authenticate
    with a shared secret (WEFT_API_KEY).
    """

    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self._api_key):
            logger.warning("Rejected invalid API key")
            return None
        return AccessToken(
            token=token,
            client_id="weft-client",
            scopes=[],
        )


def get_auth_provider(api_key: str | None, is_production: bool) -> TokenVerifier | None:
    """Return an auth provider based on config, or None to skip auth.

    In local mode, auth is always disabled regardless of api_key.
    In production mode, api_key is required.
    """
    if not is_production:
        return None
    if not api_key:
        raise ValueError(
            "WEFT_API_KEY is required in production mode (WEFT_ENV=production)"
        )
    return ApiKeyVerifier(api_key)
