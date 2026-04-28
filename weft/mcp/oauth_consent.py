"""OAuth 2.1 consent page — Supabase OAuth Server integration.

In the Supabase-OAuth-Server architecture, Supabase hosts the entire
OAuth 2.1 dance (``/auth/v1/oauth/{authorize,token}``, JWKS, dynamic
client registration). The MCP client (Claude) discovers Supabase as the
authorization server via the ``/.well-known/oauth-protected-resource``
document we serve. Supabase only redirects back to *us* for one thing:
**user consent.**

This module hosts that single consent page at the path Supabase is
configured to send users to (``Authentication → OAuth Server →
Authorization URL Path`` in the Supabase dashboard, set to
``/oauth/consent``). The page is mostly JS — it loads the Supabase JS
SDK and uses it to:

1. Read ``authorization_id`` from the URL.
2. Check for a Supabase session in ``localStorage``. If none, show a
   magic-link form (``signInWithOtp``) and wait for the redirected
   sign-in to complete.
3. Once signed in, call ``supabase.auth.oauth.getAuthorizationDetails``
   to fetch the requesting client and scopes.
4. Render Approve / Deny buttons. On click call
   ``approveAuthorization`` or ``denyAuthorization`` and redirect the
   browser to the URL Supabase returns.

There's no Python state to manage here — the Weft server's only job is
to ship this static HTML. Token validation for the resulting session
tokens happens in the middleware via the existing Supabase JWKS path.
"""

from __future__ import annotations

import html
import logging

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from weft.config import WeftConfig

logger = logging.getLogger(__name__)


_CONSENT_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Authorize MCP client</title>
<meta name="robots" content="noindex, nofollow">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif;
         max-width: 32rem; margin: 3rem auto; padding: 0 1rem; color: #222;
         line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
  p {{ color: #555; }}
  .panel {{ background: #f5f5f7; border-radius: 6px; padding: 0.75rem 1rem;
           margin: 1rem 0; }}
  label {{ display: block; font-size: 0.875rem; font-weight: 600;
           margin-bottom: 0.25rem; }}
  input[type=email] {{ width: 100%; padding: 0.6rem 0.75rem; font-size: 1rem;
                       border: 1px solid #ccc; border-radius: 6px;
                       box-sizing: border-box; }}
  button {{ padding: 0.6rem 1rem; font-size: 1rem;
            border: 0; border-radius: 6px; cursor: pointer;
            min-width: 6rem; }}
  .actions {{ display: flex; gap: 0.75rem; margin-top: 1rem; }}
  .approve {{ background: #0a7; color: #fff; }}
  .approve:hover {{ background: #085; }}
  .deny {{ background: #eee; color: #222; }}
  .deny:hover {{ background: #ddd; }}
  .send {{ background: #0a7; color: #fff; margin-top: 0.75rem; width: 100%; }}
  .send:hover {{ background: #085; }}
  .scopes {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.85rem; }}
  .error {{ color: #a00; margin-top: 0.5rem; font-size: 0.9rem; }}
  .meta {{ color: #666; font-size: 0.8rem; margin-top: 1.5rem; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<h1>Authorize MCP client</h1>

<div id="loading">
  <p>Loading…</p>
</div>

<div id="signin" class="hidden">
  <p>Sign in to your Weft account to approve this client.</p>
  <div class="panel">
    <label for="email">Email address</label>
    <input id="email" type="email" autocomplete="email" autofocus>
    <button class="send" id="send-link">Send sign-in link</button>
    <p class="error hidden" id="signin-error"></p>
  </div>
  <p class="meta">A one-time link will be emailed to you. Click it to
  return here and complete authorization.</p>
</div>

<div id="check-email" class="hidden">
  <p>We sent a sign-in link to <span id="check-email-addr"></span>.</p>
  <p class="meta">Click the link in your inbox. The link will bring you
  back to this page already signed in.</p>
</div>

<div id="consent" class="hidden">
  <p>The application <strong id="client-name">an unnamed client</strong>
  is requesting access to your Weft memory.</p>
  <div class="panel">
    <div>Requested scopes:</div>
    <div class="scopes" id="scopes"></div>
  </div>
  <div class="actions">
    <button class="approve" id="approve">Approve</button>
    <button class="deny" id="deny">Deny</button>
  </div>
  <p class="error hidden" id="consent-error"></p>
  <p class="meta">Signed in as <span id="user-email"></span>.
  <a href="#" id="signout">Not you?</a></p>
</div>

<div id="result" class="hidden">
  <p id="result-message"></p>
</div>

<script type="module">
  import {{ createClient }} from
    "https://esm.sh/@supabase/supabase-js@2";

  const SUPABASE_URL = {supabase_url_json};
  const SUPABASE_ANON_KEY = {supabase_anon_key_json};

  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {{
    auth: {{
      persistSession: true,
      detectSessionInUrl: true,
      flowType: "pkce",
    }},
  }});

  const $ = (id) => document.getElementById(id);
  const show = (id) => $(id).classList.remove("hidden");
  const hide = (id) => $(id).classList.add("hidden");
  const showOnly = (id) => {{
    ["loading", "signin", "check-email", "consent", "result"].forEach(hide);
    show(id);
  }};
  const showError = (id, msg) => {{
    const el = $(id);
    el.textContent = msg;
    el.classList.remove("hidden");
  }};

  function authorizationId() {{
    const params = new URLSearchParams(window.location.search);
    return params.get("authorization_id");
  }}

  async function init() {{
    const authId = authorizationId();
    if (!authId) {{
      showOnly("result");
      $("result-message").textContent =
        "Missing authorization_id — open this page from a Supabase OAuth flow.";
      return;
    }}

    const {{ data: {{ session }} }} = await supabase.auth.getSession();
    if (!session) {{
      showOnly("signin");
      return;
    }}

    showOnly("consent");
    $("user-email").textContent = session.user.email || session.user.id;

    try {{
      const {{ data, error }} =
        await supabase.auth.oauth.getAuthorizationDetails(authId);
      if (error) throw error;
      $("client-name").textContent =
        (data && data.client && data.client.name) || "an unnamed client";
      const scopes = (data && data.scopes) || [];
      $("scopes").textContent = scopes.length
        ? scopes.join(" ") : "(no scopes requested)";
    }} catch (err) {{
      showError("consent-error",
        "Couldn't load authorization details: " + (err.message || err));
    }}
  }}

  $("send-link").addEventListener("click", async () => {{
    hide("signin-error");
    const email = $("email").value.trim().toLowerCase();
    if (!email || !email.includes("@")) {{
      showError("signin-error", "Enter a valid email address.");
      return;
    }}
    const {{ error }} = await supabase.auth.signInWithOtp({{
      email,
      options: {{ emailRedirectTo: window.location.href }},
    }});
    if (error) {{
      showError("signin-error",
        error.message || "Couldn't send sign-in link.");
      return;
    }}
    $("check-email-addr").textContent = email;
    showOnly("check-email");
  }});

  function pickRedirect(data) {{
    if (!data) return null;
    // The Supabase JS OAuth SDK is still in beta and the response field
    // names have shifted between versions: redirect_to, redirect_url,
    // url, location. Accept whichever shows up so a minor SDK rev
    // doesn't break the dance.
    return (
      data.redirect_to ||
      data.redirect_url ||
      data.redirectTo ||
      data.url ||
      data.location ||
      (data.data && (
        data.data.redirect_to ||
        data.data.redirect_url ||
        data.data.url
      )) ||
      null
    );
  }}

  $("approve").addEventListener("click", async () => {{
    hide("consent-error");
    try {{
      const resp =
        await supabase.auth.oauth.approveAuthorization(authorizationId());
      if (resp && resp.error) throw resp.error;
      const target = pickRedirect(resp && resp.data);
      if (!target) {{
        showError("consent-error",
          "Approval succeeded but no redirect URL returned. " +
          "See browser console for the response shape.");
        return;
      }}
      window.location.href = target;
    }} catch (err) {{
      showError("consent-error",
        "Couldn't approve: " + (err.message || err));
    }}
  }});

  $("deny").addEventListener("click", async () => {{
    hide("consent-error");
    try {{
      const resp =
        await supabase.auth.oauth.denyAuthorization(authorizationId());
      if (resp && resp.error) throw resp.error;
      const target = pickRedirect(resp && resp.data);
      if (target) {{
        window.location.href = target;
      }} else {{
        showOnly("result");
        $("result-message").textContent = "Authorization denied.";
      }}
    }} catch (err) {{
      showError("consent-error", "Couldn't deny: " + (err.message || err));
    }}
  }});

  $("signout").addEventListener("click", async (ev) => {{
    ev.preventDefault();
    await supabase.auth.signOut();
    window.location.reload();
  }});

  // The PKCE detectSessionInUrl above auto-consumes magic-link tokens
  // when the user lands back here from email — re-init so we move from
  // "check-email" → "consent" without a manual refresh.
  supabase.auth.onAuthStateChange((event) => {{
    if (event === "SIGNED_IN" || event === "TOKEN_REFRESHED") {{
      init();
    }}
  }});

  init();
</script>
</body>
</html>
"""


def render_consent_page(config: WeftConfig) -> str:
    """Render the consent HTML page with project URL + anon key embedded.

    The Supabase URL and anon (publishable) key are injected as JSON-encoded
    JavaScript literals so a malformed URL cannot break the script tag. Both
    values are public-by-design; the anon key carries no privileged claims.
    """
    import json

    return _CONSENT_PAGE_TEMPLATE.format(
        supabase_url_json=json.dumps(config.supabase_url or ""),
        supabase_anon_key_json=json.dumps(config.supabase_anon_key or ""),
    )


async def handle_consent(request: Request) -> HTMLResponse | JSONResponse:
    """``GET /oauth/consent`` — render the Supabase-OAuth consent page.

    Supabase redirects to this path with ``?authorization_id=…`` after a
    DCR-registered MCP client requests authorization. The page is fully
    self-contained: it talks to Supabase directly via the JS SDK and
    only renders the consent UI — Weft doesn't proxy the approval call.
    """
    from weft.config import load_config

    config = load_config()
    if not config.supabase_url or not config.supabase_anon_key:
        return JSONResponse(
            {
                "error": "misconfigured",
                "error_description": (
                    "OAuth consent requires SUPABASE_URL and "
                    "SUPABASE_ANON_KEY to be set."
                ),
            },
            status_code=501,
        )

    return HTMLResponse(render_consent_page(config))
