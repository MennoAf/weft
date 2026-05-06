"""Tests for the RFC 8414 OAuth Authorization Server Metadata mirror.

Background: Claude Code's MCP OAuth client skips RFC 9728's
protected-resource redirection and looks for the auth-server metadata
at the resource origin directly. We mirror Supabase's metadata at
``/.well-known/oauth-authorization-server`` so RFC-8414-only clients
can find it. All endpoint URLs in the document still point at Supabase
— the dance itself is unchanged.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from weft.config import WeftConfig, WeftEnv
from weft.mcp.oauth_metadata import (
    build_authorization_server_metadata,
    handle_authorization_server_metadata,
)


def _app_for(handler) -> Starlette:
    return Starlette(
        routes=[
            Route(
                "/.well-known/oauth-authorization-server",
                handler,
                methods=["GET"],
            ),
        ],
    )


class TestBuildAuthorizationServerMetadata:
    def test_endpoints_point_at_supabase_gotrue_mount(self):
        cfg = WeftConfig(
            env=WeftEnv.production,
            supabase_url="https://example.supabase.co",
        )
        doc = build_authorization_server_metadata(cfg)
        issuer = "https://example.supabase.co/auth/v1"
        assert doc["issuer"] == issuer
        assert doc["authorization_endpoint"] == f"{issuer}/oauth/authorize"
        assert doc["token_endpoint"] == f"{issuer}/oauth/token"
        assert doc["registration_endpoint"] == (
            f"{issuer}/oauth/clients/register"
        )
        assert doc["jwks_uri"] == f"{issuer}/.well-known/jwks.json"
        assert doc["userinfo_endpoint"] == f"{issuer}/oauth/userinfo"

    def test_supports_pkce_and_refresh_grant(self):
        # Refresh-token support in particular is load-bearing — without
        # it the connection dies on access-token expiry instead of being
        # silently renewed.
        cfg = WeftConfig(supabase_url="https://x.supabase.co")
        doc = build_authorization_server_metadata(cfg)
        assert "refresh_token" in doc["grant_types_supported"]
        assert "authorization_code" in doc["grant_types_supported"]
        assert "S256" in doc["code_challenge_methods_supported"]

    def test_strips_trailing_slash_on_supabase_url(self):
        cfg = WeftConfig(supabase_url="https://example.supabase.co/")
        doc = build_authorization_server_metadata(cfg)
        assert doc["issuer"] == "https://example.supabase.co/auth/v1"
        # No double slashes anywhere in the constructed URLs.
        for key, value in doc.items():
            if isinstance(value, str) and value.startswith("https://"):
                assert "//" not in value[len("https://"):], (
                    f"{key}={value} has a doubled slash"
                )

    def test_returns_empty_dict_when_supabase_url_missing(self):
        # No supabase_url → caller (the route) should surface a 501,
        # not publish a half-built document.
        cfg = WeftConfig(supabase_url=None)
        assert build_authorization_server_metadata(cfg) == {}


class TestHandleAuthorizationServerMetadata:
    @pytest.fixture
    def env_overrides(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_test_key")
        monkeypatch.setenv("WEFT_OAUTH_ENABLED", "1")
        monkeypatch.setenv("OAUTH_ISSUER", "https://weft-mcp.example")

    async def test_returns_metadata_when_configured(self, env_overrides):
        app = _app_for(handle_authorization_server_metadata)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/.well-known/oauth-authorization-server")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["issuer"] == "https://example.supabase.co/auth/v1"
        assert body["token_endpoint"] == (
            "https://example.supabase.co/auth/v1/oauth/token"
        )

    async def test_501_when_supabase_url_missing(self, monkeypatch):
        # Strip every var that load_config would treat as Supabase config
        # so the builder returns {} and the handler 501s. We can't just
        # delenv SUPABASE_URL because the local dev fallback might still
        # populate it.
        for var in (
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "WEFT_SUPABASE_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("WEFT_OAUTH_ENABLED", "1")

        app = _app_for(handle_authorization_server_metadata)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/.well-known/oauth-authorization-server")
        assert r.status_code == 501
        body = r.json()
        assert body["error"] == "misconfigured"
