"""Tests for the Supabase-OAuth-Server consent page handler.

In the new architecture Weft hosts only the consent UI page; everything
else (authorize/token/register/JWKS) lives on Supabase. The page is
mostly JS — these tests cover the small server-side surface: requested
config gating, correct content type, and that the rendered HTML embeds
the project's URL + anon key as JSON-encoded JS literals so the SDK can
boot.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from weft.config import WeftConfig, WeftEnv
from weft.mcp.oauth_consent import handle_consent, render_consent_page


def _app_for(handler) -> Starlette:
    return Starlette(routes=[Route("/oauth/consent", handler, methods=["GET"])])


@pytest.fixture
def env_overrides(monkeypatch):
    """Configure the env so :func:`weft.config.load_config` returns the test values.

    The handler calls ``load_config()`` itself rather than receiving a
    container, so we have to set env vars rather than instantiate a
    config object. This mirrors how the real route runs in the FastMCP
    lifespan.
    """
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_test_key")
    monkeypatch.setenv("WEFT_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_ISSUER", "https://weft-mcp.example")


class TestRenderConsentPage:
    def test_includes_supabase_url_and_anon_key(self):
        cfg = WeftConfig(
            env=WeftEnv.production,
            supabase_url="https://example.supabase.co",
            supabase_anon_key="sb_publishable_xyz",
        )
        html = render_consent_page(cfg)
        # Both values must land as JSON-quoted JS literals so the SDK
        # picks them up. The leading-quote check is enough — broken
        # interpolation would have left the {placeholder} unrendered.
        assert '"https://example.supabase.co"' in html
        assert '"sb_publishable_xyz"' in html
        assert "{supabase_url_json}" not in html
        assert "{supabase_anon_key_json}" not in html

    def test_renders_signin_panel(self):
        cfg = WeftConfig(
            supabase_url="https://x.supabase.co", supabase_anon_key="k",
        )
        html = render_consent_page(cfg)
        assert 'id="signin"' in html
        assert 'id="consent"' in html
        assert 'id="check-email"' in html


class TestHandleConsent:
    async def test_renders_html_when_configured(self, env_overrides):
        app = _app_for(handle_consent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/oauth/consent?authorization_id=auth_abc")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        # The page is self-contained — Supabase JS SDK is loaded via ESM CDN.
        assert "@supabase/supabase-js" in r.text
        # The authorization_id is read client-side from window.location, so
        # there's no server-side templating of it. The script must reference
        # ``authorization_id`` for the client to know which authz to act on.
        assert "authorization_id" in r.text

    async def test_returns_501_when_supabase_url_missing(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.setenv("SUPABASE_ANON_KEY", "sb_publishable_x")
        app = _app_for(handle_consent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/oauth/consent?authorization_id=x")
        assert r.status_code == 501
        assert r.json()["error"] == "misconfigured"

    async def test_returns_501_when_anon_key_missing(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
        app = _app_for(handle_consent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/oauth/consent?authorization_id=x")
        assert r.status_code == 501
