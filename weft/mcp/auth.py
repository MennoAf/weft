"""Authentication providers for the Weft MCP server.

Two modes:
- **API key only** (default): ``ApiKeyVerifier`` checks bearer tokens against
  ``WEFT_API_KEY``. Used when OAuth is not configured.
- **OAuth + API key** (when Google OAuth is configured): ``WeftTokenVerifier``
  tries API key first (fast, no network), then falls back to Google token
  verification. ``get_oauth_provider()`` returns an ``OAuthProxy`` wired to
  Google's endpoints with persistent PostgreSQL-backed client storage.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Callable

import asyncpg
from fastmcp.server.auth import AccessToken, TokenVerifier

from weft.config import WeftConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key verifier (existing, unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dual-mode verifier: API key + Google OAuth
# ---------------------------------------------------------------------------

class WeftTokenVerifier(TokenVerifier):
    """Tries API key first (fast, no network), then Google token verification.

    This allows both Claude Code (API key) and Claude web/app (OAuth) clients
    to authenticate against the same server.
    """

    required_scopes: list[str] = ["openid", "email"]

    def __init__(
        self,
        api_key: str | None = None,
        google_verifier: TokenVerifier | None = None,
    ):
        super().__init__()
        self._api_key = api_key
        self._google_verifier = google_verifier

    async def verify_token(self, token: str) -> AccessToken | None:
        # Fast path: API key check (no network call)
        if self._api_key and hmac.compare_digest(token, self._api_key):
            return AccessToken(
                token=token,
                client_id="weft-apikey",
                scopes=[],
            )

        # Slow path: Google token verification (HTTP call to tokeninfo API)
        if self._google_verifier:
            result = await self._google_verifier.verify_token(token)
            if result is not None:
                return result

        logger.warning("Token rejected by all verifiers")
        return None


# ---------------------------------------------------------------------------
# OAuth provider factory
# ---------------------------------------------------------------------------

def get_oauth_provider(
    config: WeftConfig,
    pool_factory: Callable[[], asyncpg.Pool],
) -> object | None:
    """Create an OAuthProxy with Google upstream if OAuth is configured.

    Returns None when OAuth env vars are not set (backward compat — API key
    auth continues to work as before).

    Uses OAuthProxy directly (not GoogleProvider) so we can inject
    WeftTokenVerifier for dual-auth (API key + Google).
    """
    if not config.oauth.is_configured:
        return None

    # Lazy imports — only needed when OAuth is actually configured
    from fastmcp.server.auth.oauth_proxy import OAuthProxy
    from fastmcp.server.auth.providers.google import GoogleTokenVerifier

    from weft.mcp.oauth_storage import PostgresKeyValueStore

    store = PostgresKeyValueStore(pool_factory)

    verifier = WeftTokenVerifier(
        api_key=config.api_key,
        google_verifier=GoogleTokenVerifier(),
    )

    provider = OAuthProxy(
        # Google OAuth endpoints
        upstream_authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        upstream_token_endpoint="https://oauth2.googleapis.com/token",
        upstream_client_id=config.oauth.client_id,
        upstream_client_secret=config.oauth.client_secret,
        # Token validation
        token_verifier=verifier,
        # Server config
        base_url=config.oauth.base_url,
        # Don't set jwt_signing_key — derives from client_secret (avoids bug #2867)
        # Persistent storage (survives Fly.io deploys)
        client_storage=store,
        # Single-user system — skip consent screen
        require_authorization_consent=False,
        # Google-specific: get refresh tokens
        extra_authorize_params={
            "access_type": "offline",
            "prompt": "consent",
        },
    )

    logger.info(
        "OAuth configured: Google upstream, base_url=%s",
        config.oauth.base_url,
    )
    return provider
