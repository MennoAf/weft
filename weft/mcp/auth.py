"""Authentication providers for the Weft MCP server.

Two modes:
- **API key only** (default): ``ApiKeyVerifier`` checks bearer tokens against
  ``WEFT_API_KEY``. Used when OAuth is not configured.
- **OAuth + API key** (when Google OAuth is configured): ``WeftOAuthProxy``
  subclasses FastMCP's ``OAuthProxy`` to try API key verification first
  (fast, no network) before falling back to the standard OAuth JWT flow.
  This allows Claude Code CLI (API key) and Claude web/app (OAuth) to
  authenticate against the same server.
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
# API Key verifier
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
# OAuth provider factory
# ---------------------------------------------------------------------------

def get_oauth_provider(
    config: WeftConfig,
    pool_factory: Callable[[], asyncpg.Pool],
) -> object | None:
    """Create a WeftOAuthProxy with Google upstream if OAuth is configured.

    Returns None when OAuth env vars are not set (backward compat — API key
    auth continues to work as before).

    WeftOAuthProxy subclasses OAuthProxy to intercept bearer tokens before
    JWT validation — if the token matches the API key, it's accepted
    immediately without going through the OAuth flow.
    """
    if not config.oauth.is_configured:
        return None

    from fastmcp.server.auth.oauth_proxy import OAuthProxy
    from fastmcp.server.auth.providers.google import GoogleTokenVerifier

    from weft.mcp.oauth_storage import PostgresKeyValueStore

    store = PostgresKeyValueStore(pool_factory)

    class WeftOAuthProxy(OAuthProxy):
        """OAuthProxy that accepts API keys before trying OAuth JWT validation.

        OAuthProxy.load_access_token() validates bearer tokens as its own
        JWTs first, so plain API keys never reach the token_verifier. This
        subclass intercepts the token and checks the API key before delegating
        to the standard OAuth flow.
        """

        async def load_access_token(self, token: str) -> AccessToken | None:
            # Fast path: API key check (no network, no JWT parsing)
            if config.api_key and hmac.compare_digest(token, config.api_key):
                return AccessToken(
                    token=token,
                    client_id="weft-apikey",
                    scopes=["openid", "email"],
                )
            # Standard OAuth JWT flow
            return await super().load_access_token(token)

    # Set required_scopes on the verifier so OAuthProxy propagates them
    # to the authorization URL (scope=openid email).
    google_verifier = GoogleTokenVerifier()
    google_verifier.required_scopes = ["openid", "email"]

    provider = WeftOAuthProxy(
        # Google OAuth endpoints
        upstream_authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        upstream_token_endpoint="https://oauth2.googleapis.com/token",
        upstream_client_id=config.oauth.client_id,
        upstream_client_secret=config.oauth.client_secret,
        # Token validation (for upstream Google tokens after OAuth flow)
        token_verifier=google_verifier,
        # Server config
        base_url=config.oauth.base_url,
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
