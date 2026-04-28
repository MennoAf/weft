"""User identity extraction from Supabase JWTs.

Orthogonal to API key auth in weft/mcp/auth.py — that authenticates the
MCP client, this identifies the *user* for Row Level Security scoping.

Flow: HTTP Authorization header → extract_user_id() → current_user_id
contextvar → connection.py reads it for SET LOCAL app.user_id.

Supports two verification modes (checked in order):
1. **JWKS (asymmetric, RS256/ES256)** — the default for Supabase projects
   created after May 2025.  Set SUPABASE_JWKS_URL or SUPABASE_URL and the
   public key is fetched automatically.
2. **Shared secret (symmetric, HS256)** — legacy mode.  Set
   SUPABASE_JWT_SECRET to the project's JWT secret.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

# Request-scoped user identity. Set by middleware, read by connection layer.
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

# Module-level JWKS client singleton — created once, reuses cached keys.
_jwk_client: PyJWKClient | None = None
_jwt_secret: str | None = None
_auth_mode: str | None = None  # "jwks", "secret", or None


def _init_auth() -> None:
    """Lazily initialize the JWT verification strategy.

    Checks environment variables once and caches the result so subsequent
    calls are fast.  Order of precedence:

    1. SUPABASE_JWKS_URL — explicit JWKS endpoint
    2. SUPABASE_URL — derive JWKS endpoint as <url>/auth/v1/.well-known/jwks.json
    3. SUPABASE_JWT_SECRET — legacy HS256 symmetric secret
    """
    global _jwk_client, _jwt_secret, _auth_mode

    if _auth_mode is not None:
        return  # already initialized

    # Option 1: Explicit JWKS URL
    jwks_url = os.environ.get("SUPABASE_JWKS_URL")

    # Option 2: Derive from SUPABASE_URL
    if not jwks_url:
        supabase_url = os.environ.get("SUPABASE_URL")
        if supabase_url:
            jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    if jwks_url:
        _jwk_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=600)
        _auth_mode = "jwks"
        logger.info("JWT auth: JWKS mode (endpoint: %s)", jwks_url)
        return

    # Option 3: Legacy symmetric secret
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if secret:
        _jwt_secret = secret
        _auth_mode = "secret"
        logger.info("JWT auth: legacy HS256 mode")
        return

    _auth_mode = "none"
    logger.info("JWT auth: disabled (no SUPABASE_JWKS_URL, SUPABASE_URL, or SUPABASE_JWT_SECRET)")


def _is_own_oauth_issuer(issuer: str) -> bool:
    """Return ``True`` if *issuer* is Weft's own OAuth AS.

    Phase-1 hardening (scope §9): once we start minting our own RS256
    JWTs the two token families need to stay disjoint, otherwise a
    confused deputy could present a Weft OAuth access token on a
    Supabase-auth path and pick up a user identity. We already diverge
    by ``kid`` (Supabase's JWKS won't list ours), but an explicit issuer
    match is a cheap second line of defence — and it's robust to both
    HS256 and unkeyed-JWKS misconfigurations.

    ``OAUTH_ISSUER`` is set for the Weft MCP process in production; when
    unset (e.g. local tests) this helper conservatively returns False so
    legacy behaviour is preserved.
    """
    own_issuer = os.environ.get("OAUTH_ISSUER")
    if not own_issuer:
        return False
    return issuer.rstrip("/") == own_issuer.rstrip("/")


def extract_user_id(token: str) -> str | None:
    """Decode a Supabase JWT and return the ``sub`` claim, or None.

    Graceful degradation: returns None for any invalid, expired, or
    malformed token rather than raising. This ensures unauthenticated
    requests proceed with global-only visibility (user_id=NULL rows).

    Parameters
    ----------
    token:
        Raw JWT string (without "Bearer " prefix).
    """
    _init_auth()

    if _auth_mode == "none":
        logger.debug("JWT auth disabled — skipping decode")
        return None

    try:
        if _auth_mode == "jwks":
            # Asymmetric verification: fetch public key by kid from JWKS endpoint
            signing_key = _jwk_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256", "EdDSA"],
                options={"require": ["sub", "exp"], "verify_aud": False},
            )
        else:
            # Legacy symmetric verification
            payload = jwt.decode(
                token,
                _jwt_secret,
                algorithms=["HS256"],
                options={"require": ["sub", "exp"], "verify_aud": False},
            )

        # Harden against confused-deputy risk between our own OAuth access
        # tokens (see weft/mcp/oauth/*) and Supabase-issued tokens. When
        # ``OAUTH_ISSUER`` is configured and the token's ``iss`` matches it,
        # the token is ours — never accept it here as a Supabase identity.
        # Scope §9.
        issuer = payload.get("iss")
        if isinstance(issuer, str) and _is_own_oauth_issuer(issuer):
            logger.warning(
                "Rejecting Weft OAuth token presented on Supabase path (iss=%s)",
                issuer,
            )
            return None
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT: %s", exc)
        return None
    except Exception as exc:
        # JWKS fetch failures, network errors, etc.
        logger.warning("JWT verification failed: %s", exc)
        return None

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        logger.debug("JWT missing or empty 'sub' claim")
        return None

    return sub


def extract_user_id_from_header(auth_header: str | None) -> str | None:
    """Extract user_id from an Authorization header value.

    Expects ``Bearer <token>``. Returns None for missing, empty, or
    non-Bearer headers.
    """
    if not auth_header:
        return None

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        return None

    token = parts[1].strip()
    if not token:
        return None

    return extract_user_id(token)


def _reset_auth() -> None:
    """Reset auth state — only for testing."""
    global _jwk_client, _jwt_secret, _auth_mode
    _jwk_client = None
    _jwt_secret = None
    _auth_mode = None
