"""API key and OAuth authentication for the Weft MCP server.

Two layers live side-by-side:

* :class:`ApiKeyVerifier` — constant-time compare of a shared secret
  (``WEFT_API_KEY``). Used for stdio Claude Code and HTTP API-key
  clients. Unchanged since the known-good ``c965d49`` snapshot.
* :class:`JWTVerifier` (imported from ``weft.mcp.oauth``) — local RS256
  verification of Weft-issued OAuth access tokens. Wired in only when
  ``WEFT_OAUTH_ENABLED=1``.

When OAuth is enabled, both mechanisms are served via
:class:`CompositeTokenVerifier`, which tries the cheap HMAC compare
first and falls back to JWT decode. API-key clients stay byte-identical
in behaviour; OAuth clients get the additional JWT path.
"""

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


class CompositeTokenVerifier(TokenVerifier):
    """Try API-key verification first, then delegate to the JWT verifier.

    Ordering is intentional and load-bearing:

    * API-key compare is a single ``hmac.compare_digest`` call — constant
      time, no parsing, no crypto. Put it first so the hot path for
      Claude Code (the only existing production client) stays as cheap
      as it is today.
    * Only if the API-key compare fails do we attempt a JWT decode,
      which is O(signature verify) ~ 0.5 ms.

    Both halves are constructor-injected so tests can drop in
    :class:`ApiKeyVerifier`-style stubs without touching module globals.
    """

    def __init__(
        self,
        *,
        api_key_verifier: ApiKeyVerifier,
        jwt_verifier: TokenVerifier,
    ):
        super().__init__()
        self._api_key_verifier = api_key_verifier
        self._jwt_verifier = jwt_verifier

    async def verify_token(self, token: str) -> AccessToken | None:
        # Fast path: API key.
        if (result := await self._api_key_verifier.verify_token(token)) is not None:
            return result
        # Slow path: JWT. Any error inside JWTVerifier already collapses
        # to None, so the composite surface stays "Optional[AccessToken]".
        return await self._jwt_verifier.verify_token(token)


def get_auth_provider(
    api_key: str | None,
    is_production: bool,
    *,
    oauth_verifier: TokenVerifier | None = None,
) -> TokenVerifier | None:
    """Return an auth provider based on config, or None to skip auth.

    Parameters
    ----------
    api_key:
        Value of ``WEFT_API_KEY``. Required in production mode.
    is_production:
        ``True`` when ``WEFT_ENV=production``. In local mode auth is
        disabled regardless of *api_key*, mirroring prior behaviour.
    oauth_verifier:
        When supplied (i.e. ``WEFT_OAUTH_ENABLED=1``), the returned
        provider is a :class:`CompositeTokenVerifier` that accepts both
        the API key *and* OAuth JWTs. When ``None``, behaviour is
        identical to the pre-OAuth state — plain :class:`ApiKeyVerifier`.
    """
    if not is_production:
        return None
    if not api_key:
        raise ValueError(
            "WEFT_API_KEY is required in production mode (WEFT_ENV=production)"
        )
    api_key_verifier = ApiKeyVerifier(api_key)
    if oauth_verifier is None:
        return api_key_verifier
    return CompositeTokenVerifier(
        api_key_verifier=api_key_verifier,
        jwt_verifier=oauth_verifier,
    )
