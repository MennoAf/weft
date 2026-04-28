# Weft — Known-Good Configuration Snapshot

> **Purpose:** If Jason ever says "yes I'm a dumbfuck, here's how you get your brain back," point at this file. It captures the exact working state of Weft as of the commit pinned below, how Claude Code is configured to reach it, and the anti-patterns that have historically broken it.
>
> **Written:** 2026-04-16
> **Commit pin (known-good HEAD):** `c965d49` — _Remove OAuth 2.1 support, restore API-key-only auth (2035 tests)_
> **Verified:** Claude Code talks to `https://weft-mcp.fly.dev/mcp` over HTTPS + Bearer, `weft_prime` returns in <1s.

---

## TL;DR

Weft runs on Fly.io. Claude Code connects over **Streamable HTTP + a static Bearer token (the API key)**. That is the entire auth story. Anything more elaborate than that has historically failed, and the smacking clause exists because of it.

If Weft's connection is broken and you're trying to restore it, you do **three things** and no more:

1. Confirm the Fly.io app is running: `fly status -a weft-mcp`
2. Confirm the Bearer token in Claude Code settings matches the `WEFT_API_KEY` secret in Fly.
3. Confirm the current git HEAD is on `c965d49` (or a later commit that kept API-key-only auth).

If those three are green and Weft still doesn't respond, the problem is downstream (Supabase, embeddings provider, or Redis), not auth. Do **not** "fix" it by adding OAuth. See §5.

---

## 1. Deployment state (the part that lives on Fly.io)

| Field | Value |
|---|---|
| App | `weft-mcp` |
| Region | `iad` (us-east, co-located with Supabase) |
| Hostname | `https://weft-mcp.fly.dev` |
| MCP endpoint | `https://weft-mcp.fly.dev/mcp` |
| Health endpoint | `https://weft-mcp.fly.dev/healthz` (unauthenticated) |
| Transport | `streamable-http` (set in `fly.toml` env) |
| Deploy strategy | `bluegreen` |
| VM | 512 MB, 1 shared CPU |
| `min_machines_running` | 1 |

**Env vars set on Fly (in `fly.toml`):**
```toml
WEFT_ENV        = "production"
WEFT_TRANSPORT  = "streamable-http"
WEFT_REDIS_URL  = ""            # empty → NullCache, not a bug
WEFT_LOG_LEVEL  = "INFO"
PORT            = "8000"
```

**Fly secrets (names only — values live only in Fly):**
```
DATABASE_URL
WEFT_API_KEY                    ← the Bearer token for MCP auth
OPENAI_API_KEY                  ← embeddings
SUPABASE_URL
SLACK_BOT_TOKEN
SLACK_SIGNING_SECRET
WEFT_DAILY_BRIEF_CHANNEL
WEFT_SLACK_SYNC_INTERVAL
```

To list: `fly secrets list -a weft-mcp` (values are never displayed — only digests). To set: `fly secrets set KEY=VALUE -a weft-mcp`.

---

## 2. How Claude Code reaches Weft

The working config lives in `~/.claude/settings.json` under `mcpServers.weft`:

```json
{
  "mcpServers": {
    "weft": {
      "type": "http",
      "url": "https://weft-mcp.fly.dev/mcp",
      "headers": {
        "Authorization": "Bearer <WEFT_API_KEY value — do not commit>"
      }
    }
  }
}
```

The Bearer token must match `fly secrets list -a weft-mcp | grep WEFT_API_KEY` digest. If you rotated the secret, the value in `settings.json` has to be rotated to the new plaintext.

**Do not put the API key in this markdown file.** It's in the user's private dotfile, and that's where it belongs.

---

## 3. How authentication actually works (the short version)

Two middlewares live in `weft/mcp/server.py:44-83` — `UserIdentityMiddleware`:

1. **API-key gate on `/mcp`.** If `WEFT_ENV=production` and `WEFT_API_KEY` is set, the middleware requires `Authorization: Bearer <key>` on any `/mcp` request, HMAC-compares it against the configured key, and 401s if wrong or missing. Health checks (`/healthz`) and Slack routes (`/slack/commands`) are unauthenticated.
2. **Best-effort user identity from JWT.** After the key check passes, the same middleware looks for a Supabase JWT in the same `Authorization` header and calls `extract_user_id_from_header()` (`weft/auth.py:79-151`). If present and valid, sets `current_user_id` contextvar; if missing/invalid, sets it to `None`. The contextvar drives `SET LOCAL app.user_id` in the DB layer (`weft/db/connection.py:108-183`), which RLS uses to scope reads and writes.

**Today's single-user setup** does not rely on the JWT path — the API key alone is enough because there's one user. The JWT path is present so a future multi-user remote MCP can coexist without a rewrite.

`ApiKeyVerifier` in `weft/mcp/auth.py` is the older `fastmcp.server.auth.TokenVerifier`-style implementation. It's kept for compatibility. The live auth path is the Starlette middleware in `server.py`.

---

## 4. The prime path (what `/prime` actually does under the hood)

`weft_prime` fans out 17 sections concurrently via `asyncio.gather` in `weft/primer.py:1010-1028`. Each section issues 1–2 Supabase queries. Wall-clock is dominated by the slowest section (~30–40 ms typical). **Prime is already fast when the server is fast. If prime is slow, the bottleneck is not the primer — it is either the transport or the network to Supabase.**

If prime feels slow:
- Check `https://weft-mcp.fly.dev/healthz` — if 503, the Fly app is cold-starting.
- Check Supabase status — queries are what actually run.
- Check the Bearer token is correct; 401 loops don't always surface cleanly in Claude Code.

**Do not** attempt to "optimize" by adding caching layers, shortening the primer, or introducing auth-bypass tricks. Those introduce more failure modes than they remove.

---

## 5. Anti-patterns (read before touching auth)

These are the specific patterns that have broken Weft in the past. The commit carnage lives in git log if you want receipts. You don't need receipts — you need to not do them.

### 5a. OAuthProxy (the big one)
**Commits: `d668dbf`, `64ce63f`, `db33597`, `0499934` (all reverted by `c965d49`).**

Attempted to wrap the MCP server in `fastmcp.server.auth.OAuthProxy` bridging to Google OAuth. `OAuthProxy` validates the token **against the external IdP on every request**. Every MCP tool call paid a 50–100 ms network round trip to Google. `/prime` fans out enough tool invocations that the aggregate latency blew past Claude's request timeout. Connections worked; the server was unusably slow; every rollback attempt cost multiple hours.

**If you catch yourself about to add `OAuthProxy`, stop.** If the goal is "expose Weft to claude.ai," that is a known open problem (Anthropic bug [claude-ai-mcp#134](https://github.com/anthropics/claude-ai-mcp/issues/134)) and the correct path is `BearerAuthProvider` + a local JWT issuer, **not** a proxy to an external IdP. Route 1 scope lives at `boiler_room/weft-route1-scope.md` (if present) — check there first.

### 5b. Per-request introspection of any kind
Any auth scheme that makes a network call to validate a token per request will break `/prime`. Local verification only: HMAC compare for API keys, local JWT signature check with cached JWKS for user tokens.

### 5c. Moving the API key out of Fly secrets
The API key belongs in `fly secrets`. Not in the repo. Not in `fly.toml`. Not in a committed `.env`. If you find it in any of those places, that's the bug — fix the leak before doing anything else.

### 5d. Blowing away `~/.weft/fallback.md`
The server exports a fallback markdown snapshot of active memories every 30 minutes to `~/.weft/fallback.md` (`weft/mcp/server.py:178-195`). If Weft is completely unreachable and you need memory context in a pinch, that file is a plain-text dump. Don't delete it. It is the manual-recovery backstop.

---

## 6. Restore procedure

Run these in order. Stop at the first one that fails — that's the problem.

### Step 1 — Confirm the deploy is alive
```bash
fly status -a weft-mcp
# Expect: 1 machine, state=started, checks=1 passing
curl -sS https://weft-mcp.fly.dev/healthz
# Expect: {"status":"ok"}
```
If unhealthy: `fly logs -a weft-mcp` for cause. Common: Supabase credentials rotated, embedding provider API key expired, Postgres migration failed.

### Step 2 — Confirm the Bearer token matches
```bash
# On local machine:
grep -A5 '"weft"' ~/.claude/settings.json

# On Fly:
fly secrets list -a weft-mcp | grep WEFT_API_KEY
```
The digest on Fly won't match the plaintext in settings.json (it's a hash), but if you rotated the Fly secret recently and forgot to update `settings.json`, you will get 401s from Claude Code. To fix: pull the plaintext from wherever you saved it when setting the secret, or rotate (§7).

### Step 3 — Confirm git HEAD is a known-good commit
```bash
cd /Users/jasonbauman/Documents/code_projects/Personal/Weft
git log --oneline -1
# Expect: c965d49 (or a later commit that did NOT re-introduce OAuthProxy)
```
If HEAD is on one of the reverted OAuth commits (`d668dbf`, `64ce63f`, `db33597`, `0499934`), revert to `c965d49`:
```bash
git reset --hard c965d49
fly deploy
```

### Step 4 — Smoke test from Claude Code
Open any Claude Code session and run `/prime`. Expect output in under 2 seconds. If it hangs or errors:
- 401 → Bearer token mismatch (Step 2).
- 500 → server error, check `fly logs -a weft-mcp`.
- Timeout → server is overloaded or something slow was added. Check recent commits for introspection/round-trip patterns.

### Step 5 — Last-ditch fallback
If Weft is completely dead and you need context *right now*:
```bash
cat ~/.weft/fallback.md
```
This is a snapshot of active memories last exported by the server (up to 30 minutes stale). You can hand-copy what you need while you diagnose.

---

## 7. Key rotation

If the Bearer token ever leaks or you just want to rotate:

```bash
NEW_KEY=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-43)
fly secrets set WEFT_API_KEY="$NEW_KEY" -a weft-mcp
# Fly auto-restarts the app.

# Then update Claude Code settings.json:
# ~/.claude/settings.json → mcpServers.weft.REDACTEDAuthorization
#   = "Bearer $NEW_KEY"
# Save, restart Claude Code.
```
There is no "refresh token" concept here — the API key is a static shared secret. That's fine for single-user use. When Route 1 ships (JWT-based auth), this procedure gets replaced.

---

## 8. Adding Claude Desktop (bonus — same auth, different config file)

Same Bearer pattern works in Claude Desktop via its local config file (the UI does not expose headers; the file does).

**Config location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add to `mcpServers`:
```json
"weft": {
  "type": "http",
  "url": "https://weft-mcp.fly.dev/mcp",
  "headers": {
    "Authorization": "Bearer <same WEFT_API_KEY as Claude Code>"
  }
}
```
Restart Claude Desktop. Run `/prime` in a chat.

**Caveat — does NOT sync:** This config is local per-machine. To get Weft on a new Mac, copy the file (or symlink from iCloud / Dropbox / a dotfiles repo). claude.ai web and the mobile apps are not reachable this way; they need the OAuth Custom Connector path, which is on Route 1.

---

## 9. Key file paths (for future agents)

| What | Where |
|---|---|
| MCP server entrypoint | `weft/mcp/__main__.py` |
| Transport + middleware wiring | `weft/mcp/server.py` |
| API key verifier (legacy path) | `weft/mcp/auth.py` |
| JWT user-identity extraction | `weft/auth.py` |
| DB connection + RLS scoping | `weft/db/connection.py` |
| Primer (`weft_prime` internals) | `weft/primer.py` |
| Config loader | `weft/config.py` |
| Fly.io deploy config | `fly.toml` |
| Dockerfile | `Dockerfile` |
| Claude Code MCP config (on Jason's Mac) | `~/.claude/settings.json` |
| Fallback memory snapshot | `~/.weft/fallback.md` |

---

## 10. When this document is wrong (API-key path)

This file is a point-in-time snapshot. The following events invalidate parts of it:

- Route 1 ships (Weft becomes its own OAuth 2.1 server) → §3 and §5a
  change. The post-Phase-4 known-good for that path is captured in §11
  below; when OAuth is flipped on, update this section to reflect what
  live prod actually runs.
- Fly.io app moves region or rename → §1 table.
- Supabase project migrates → secret names in §1.
- `WEFT_API_KEY` is replaced by JWT-based auth → §2 and §7.

If you're editing Weft and this doc is stale, update the doc in the
same commit as the change. A stale recovery doc is worse than none.

---

## 11. OAuth 2.1 path (when enabled)

> **Status:** additive to §§1–10. The API-key path is canonical and
> stays working forever. OAuth is **opt-in** via `WEFT_OAUTH_ENABLED=1`.
> When the flag is off the server is byte-identical to the `c965d49`
> API-key-only baseline (no extra routes, no extra middleware).

This section captures the post-Phase-4 known-good for Route 1. It
exists so that when Jason enables OAuth on staging or prod, there's a
single place describing what "working" looks like.

### 11.1 When is OAuth "on"?

- `WEFT_OAUTH_ENABLED=1` set as a Fly env var (or secret).
- All of the following secrets set on the same app:
  - `OAUTH_ISSUER` — absolute base URL of the deploy
    (e.g. `https://weft-mcp-staging.fly.dev`).
  - `OAUTH_JWT_PRIVATE_KEY_PEM` — RSA-2048 PKCS8 PEM (generated by
    `scripts/generate_oauth_secrets.sh`).
  - `OAUTH_SESSION_SECRET` — ≥32 bytes of HMAC entropy for the
    authorize→callback cookie.
  - `SUPABASE_URL` — base URL of the Supabase project backing the
    identity hop.
- Optional:
  - `OAUTH_SOLE_USER_SUB` — single-user mode pin. If present, only
    this Supabase `sub` can complete the dance; consent page is
    skipped. If absent, multi-user mode runs and every user sees the
    consent page at `/oauth/consent`.
  - `OAUTH_JWT_PRIVATE_KEY_PEM_NEXT` — during a rotation overlap,
    JWKS exposes both keys and the verifier trusts both. Generate via
    `python -m weft.rotate_key`.
  - `WEFT_SUPABASE_AUTH_PROVIDER` — default `email` (magic link); set
    to `github` / `google` / etc. when using a third-party provider.

### 11.2 Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/.well-known/oauth-authorization-server` | GET | RFC 8414 metadata |
| `/.well-known/oauth-protected-resource`   | GET | RFC 9728 metadata |
| `/.well-known/jwks.json`                  | GET | Public key set; 2 keys during rotation overlap |
| `/oauth/register`                         | POST | RFC 7591 DCR, rate-limited 30/hr/IP |
| `/oauth/authorize`                        | GET  | PKCE first hop, redirects to Supabase |
| `/oauth/callback`                         | GET  | HTML + JS shim that parses URL fragment |
| `/oauth/callback/finalize`                | POST | Consumes Supabase JWT, rate-limited 60/hr/IP |
| `/oauth/consent`                          | GET/POST | Multi-user consent page (when `OAUTH_SOLE_USER_SUB` unset) |
| `/oauth/token`                            | POST | `authorization_code` + `refresh_token` grants |
| `/oauth/revoke`                           | POST | RFC 7009 revocation |

The API-key `/mcp` endpoint is unchanged. With OAuth on, the middleware
accepts **either** a valid API key **or** a valid RS256 OAuth access
token in the Bearer header — API-key clients (Claude Code, Claude
Desktop today) keep working with zero config changes.

### 11.3 Background loops added by OAuth

A single new scheduler loop is wired in lifespan when `WEFT_OAUTH_ENABLED=1`:

- `revocation_sweep_loop` — hourly prune of expired `oauth_access_revocations`,
  old `oauth_authorization_codes`, and stale `oauth_clients` (90-day
  `last_used_at` window per scope §16 decision 4).

### 11.4 Smoke tests (once per deploy)

```bash
curl -fsS "$OAUTH_ISSUER/.well-known/oauth-authorization-server" \
    | jq '.issuer, .authorization_endpoint, .token_endpoint'
# Expected: $OAUTH_ISSUER, $OAUTH_ISSUER/oauth/authorize, $OAUTH_ISSUER/oauth/token

curl -fsS "$OAUTH_ISSUER/.well-known/jwks.json" | jq '.keys | length'
# Expected: 1 (or 2 during rotation overlap).

# /oauth/register expects a JSON body — a GET should return 405.
curl -sS -o /dev/null -w "%{http_code}" "$OAUTH_ISSUER/oauth/register"
# Expected: 405
```

A full end-to-end test requires a browser and the Claude Desktop /
claude.ai custom connector — see `DEPLOY.md` → "Route 1 staging
rollout" for the step list.

### 11.5 Rollback

If OAuth misbehaves on a live app, flip the flag off:

```bash
fly secrets unset WEFT_OAUTH_ENABLED -a weft-mcp       # or weft-mcp-staging
fly deploy
```

The OAuth route handlers stop registering the moment the flag is
absent. The server returns to pure API-key operation — the §§1–10
baseline. Existing OAuth tokens simply stop being accepted (the
verifier isn't constructed at all); clients fall back to their
API-key credentials if they have them, or get prompted to re-auth.

### 11.6 What hasn't changed from §§1–10

- `weft-mcp` prod deploy still runs with `WEFT_OAUTH_ENABLED` **unset**.
- The only client of production today (Claude Code) still uses the
  API-key path. No change to `~/.claude/settings.json` required.
- All of §5's anti-patterns still apply — especially `OAuthProxy`
  (§5a). Route 1's own authorization server is the opposite of a
  proxy: verification is **local**, no network round-trip per request.

