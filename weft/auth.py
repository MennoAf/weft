"""User identity extraction from Supabase JWTs.

Orthogonal to API key auth in weft/mcp/auth.py — that authenticates the
MCP client, this identifies the *user* for Row Level Security scoping.

Flow: HTTP Authorization header → extract_user_id() → current_user_id
contextvar → connection.py reads it for SET LOCAL app.user_id.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

import jwt

logger = logging.getLogger(__name__)

# Request-scoped user identity. Set by middleware, read by connection layer.
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)


def _get_jwt_secret() -> str | None:
    """Read the Supabase JWT secret from environment."""
    return os.environ.get("SUPABASE_JWT_SECRET")


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
    secret = _get_jwt_secret()
    if not secret:
        logger.debug("No SUPABASE_JWT_SECRET configured — skipping JWT decode")
        return None

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"], "verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT: %s", exc)
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
