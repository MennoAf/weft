"""RFC 8414 OAuth Authorization Server Metadata mirror.

In the Supabase-OAuth-Server architecture (see ``oauth_consent.py``)
Supabase hosts the entire OAuth 2.1 dance. The MCP-correct way for
clients to discover that is RFC 9728 — fetch our
``/.well-known/oauth-protected-resource`` document, follow
``authorization_servers[0]``, and fetch *that* origin's
``/.well-known/oauth-authorization-server``.

In practice (observed 2026-05-06 in production traffic logs), Claude
Code's MCP OAuth client does **not** implement RFC 9728. After a 401
on ``/mcp`` it goes straight to
``https://weft-mcp.fly.dev/.well-known/oauth-authorization-server`` —
the RFC 8414 path at the *resource* origin — and gives up on 404. It
never reads our ``WWW-Authenticate: ... resource_metadata=...`` hint.

To unblock that client without waiting on Anthropic to ship RFC 9728
support, we mirror Supabase's auth-server metadata at our origin. All
endpoint URLs in the document still point to Supabase, so the actual
OAuth dance (authorize → consent → token → refresh) goes directly
between client and Supabase exactly as before. The mirror is a
discovery convenience, nothing more.

Issuer is preserved as Supabase's GoTrue mount so JWT ``iss`` claims
match what tokens carry — token validation downstream is unaffected.
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from weft.config import WeftConfig

logger = logging.getLogger(__name__)


def build_authorization_server_metadata(cfg: WeftConfig) -> dict:
    """Construct the RFC 8414 metadata document.

    All endpoints point to Supabase's GoTrue mount
    (``<project>.supabase.co/auth/v1/...``); the issuer matches Supabase
    so JWT ``iss`` validation stays consistent. The shape mirrors what
    Supabase itself publishes at
    ``<supabase_url>/auth/v1/.well-known/oauth-authorization-server``.
    """
    supabase_root = (cfg.supabase_url or "").rstrip("/")
    if not supabase_root:
        return {}
    issuer = f"{supabase_root}/auth/v1"
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth/authorize",
        "token_endpoint": f"{issuer}/oauth/token",
        "registration_endpoint": f"{issuer}/oauth/clients/register",
        "jwks_uri": f"{issuer}/.well-known/jwks.json",
        "userinfo_endpoint": f"{issuer}/oauth/userinfo",
        "scopes_supported": ["openid", "profile", "email"],
        "response_types_supported": ["code"],
        "response_modes_supported": ["query"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256", "HS256", "ES256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
            "none",
        ],
        "code_challenge_methods_supported": ["S256", "plain"],
    }


async def handle_authorization_server_metadata(
    request: Request,
) -> JSONResponse:
    """``GET /.well-known/oauth-authorization-server`` — RFC 8414 mirror.

    Returns 501 when the deployment isn't configured for OAuth
    (``SUPABASE_URL`` missing); RFC-8414-only clients can then fall
    through to a clean error rather than a half-built document.
    """
    from weft.config import load_config

    cfg = load_config()
    payload = build_authorization_server_metadata(cfg)
    if not payload:
        return JSONResponse(
            {
                "error": "misconfigured",
                "error_description": (
                    "OAuth authorization-server metadata requires "
                    "SUPABASE_URL to be set."
                ),
            },
            status_code=501,
        )
    return JSONResponse(payload)
