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

# Request-scoped caller mode: 'supervisor' (Face / human / Orchestrator —
# trusted) vs 'agent' (autonomous agent inside a Wick container — untrusted
# write authority per Phase 2 / Q4 four-layer defense). Set by middleware
# from the ``X-Weft-Caller-Mode`` request header. Defaults to 'supervisor'
# so that pre-Phase-2 callers (and any path that doesn't go through the
# HTTP middleware) keep their existing trust level — agent containers
# explicitly opt in to the lower trust tier.
CallerMode = str  # narrowed at runtime to {"supervisor", "agent"}
_VALID_CALLER_MODES = {"supervisor", "agent"}
DEFAULT_CALLER_MODE: CallerMode = "supervisor"

current_caller_mode: ContextVar[CallerMode] = ContextVar(
    "current_caller_mode", default=DEFAULT_CALLER_MODE,
)


def get_caller_mode() -> CallerMode:
    """Return the current caller mode, defaulting to 'supervisor'.

    Centralized so write paths don't have to know about the contextvar.
    Anything other than the literal 'agent' falls back to 'supervisor' —
    a malformed header can't accidentally upgrade trust.
    """
    mode = current_caller_mode.get()
    if mode not in _VALID_CALLER_MODES:
        return DEFAULT_CALLER_MODE
    return mode


def is_agent_caller() -> bool:
    """Convenience predicate for Layer 1 enforcement."""
    return get_caller_mode() == "agent"


def resolve_caller_user_id() -> str:
    """Return the user_id that owns the current request.

    Precedence:

    1. ``current_user_id`` contextvar — set by the HTTP middleware from the
       authenticated credential (token-row ``user_id`` or Supabase JWT
       ``sub``). This is the authenticated *caller*, which on a hosted /
       multi-tenant server is the only correct scope for reads and writes.
    2. ``get_user_id()`` — the installation's canonical id. The fallback for
       paths that never run the middleware: local stdio/CLI usage, scripts,
       and unauthenticated local-dev where the contextvar stays unset.

    Centralized so MCP tool handlers don't re-resolve identity themselves.
    The previous ``get_user_id()``-only default silently scoped every hosted
    request to the *server's* installation id instead of the caller's,
    making the caller's own corpus invisible to recall (see weft-6b7c05a8).
    """
    from weft.config.user_identity import get_user_id

    return current_user_id.get() or get_user_id()


def resolve_canary_user_id() -> str:
    """Return the owner scope used by deployment-wide canary surfaces.

    Deployment-owned canary surfaces use the explicit ``WEFT_DEFAULT_USER_ID``
    owner, matching the background scheduler. Deployments without an owner
    configured retain the authenticated/install identity fallback.
    """
    return os.environ.get("WEFT_DEFAULT_USER_ID") or resolve_caller_user_id()


def parse_caller_mode_header(header_value: str | None) -> CallerMode:
    """Normalize an ``X-Weft-Caller-Mode`` header into a known value.

    Unknown / missing / malformed values resolve to 'supervisor'. Agent
    containers must send the literal string 'agent' to enter the lower
    trust tier — fail-closed against typos.
    """
    if not header_value:
        return DEFAULT_CALLER_MODE
    candidate = header_value.strip().lower()
    if candidate in _VALID_CALLER_MODES:
        return candidate
    return DEFAULT_CALLER_MODE

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
    non-Bearer REDACTED
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
