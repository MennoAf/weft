# Deploying Weft to Fly.io

## Prerequisites

- [Fly CLI](https://fly.io/docs/flyctl/install/) installed and authenticated
- A Supabase project with pgvector enabled

## First-time Setup

```bash
# Create the Fly.io app
fly launch --no-deploy

# Set secrets (never put these in fly.toml)
fly secrets set DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres"
fly secrets set WEFT_API_KEY="your-bootstrap-secret"
```

`WEFT_API_KEY` is the **bootstrap-only** credential — it auto-creates a
single supervisor token row on first request so a brand-new deployment
has at least one usable credential. From there, mint per-client tokens
via `weft tokens issue` (see "Connecting MCP Clients" below) and stop
handing the bootstrap key out. The env-var fallback stays available
through Phase 5 / public Wick launch and will be removed once
deprecation logs show zero hits for two weeks.

### Supabase Connection Notes

- Use the **direct connection** URL (port 5432), not the pooler URL
- Passwords with special characters are automatically URL-encoded by Weft
- If using the pooler URL (port 6543), set `database.statement_cache_size=0` in your config — asyncpg requires this for pgbouncer compatibility

## Deploy

```bash
fly deploy                          # production (weft-mcp)
fly deploy -c fly.staging.toml      # staging (weft-mcp-staging, OAuth on)
```

Migrations run automatically on startup before the server accepts
traffic. `fly.staging.toml` is a separate Fly config with
`WEFT_OAUTH_ENABLED=1` pinned in the env block; see "Bringing up a
new environment" below for OAuth setup.

## Verify

```bash
# Health check
curl https://weft-mcp.fly.dev/healthz

# Should return: {"status": "ok"}
```

## Configuration

| Environment Variable | Required | Default | Description |
|---------------------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | Supabase Postgres connection string |
| `WEFT_API_KEY` | Yes (prod, bootstrap) | — | Auto-bootstraps one supervisor token row on first authenticated request. Use it to mint real per-client tokens via `weft tokens issue`, then stop sharing it. Coexists with OAuth and bearer-token paths. |
| `WEFT_DEFAULT_USER_ID` | No | — | Single-tenant fallback `user_id` for hosted deployments without OAuth. Most installs do not need this. |
| `WEFT_OAUTH_ENABLED` | No | `0` | Publish RFC 9728 protected-resource metadata + serve consent page. See OAuth section below. |
| `WEFT_ENV` | No | `production` | `local` or `production` |
| `WEFT_TRANSPORT` | No | `streamable-http` | `stdio`, `sse`, or `streamable-http` |
| `WEFT_REDIS_URL` | No | `""` (disabled) | Redis URL for caching (optional, uses NullCache if empty) |
| `PORT` | No | `8000` | HTTP port |
| `OAUTH_ISSUER` | Only if OAuth on | — | Absolute base URL of this deployment (e.g. `https://weft-mcp.fly.dev`). Used as the `resource` field in the protected-resource doc. |
| `SUPABASE_URL` | Only if OAuth on | — | Base URL of the Supabase project (e.g. `https://abc.supabase.co`). Used to build the `authorization_servers` URL and embedded into the consent page so the JS SDK can boot. |
| `SUPABASE_ANON_KEY` | Only if OAuth on | — | Supabase anon (publishable) key. Public-by-design; embedded in the HTML consent page. The `sb_publishable_*` and legacy `eyJ...` JWT formats both work. |

## Supabase project configuration for OAuth

In this architecture **Supabase IS the OAuth 2.1 authorization server**.
Weft just (a) publishes RFC 9728 metadata pointing the MCP client at
Supabase and (b) hosts the consent page Supabase redirects users to
during authorization. There are three dashboard settings to verify per
project (staging + prod):

1. **Authentication → OAuth Server**: enable the OAuth 2.1 server.
2. **Authentication → OAuth Server → "Authorization URL Path"**: set
   to either `/oauth/consent` (a relative path Supabase joins onto the
   project's app URL) or `https://<your-app>.fly.dev/oauth/consent`
   (full absolute URL — leave the scheme on or Supabase joins the bare
   host as a relative path and the user lands on
   `<project>.supabase.co/auth/v1/oauth/<your-app>.fly.dev/...`).
3. **Authentication → OAuth Server → "Allow dynamic client
   registration"**: on. Lets MCP clients (Claude) register themselves
   via Supabase's DCR endpoint without manual setup.

The Supabase OAuth Server feature is in beta — if you don't see the
section in the dashboard, it may need to be enabled per project.

## Connecting MCP Clients

Three paths, all supported simultaneously.

**OAuth (Claude app, claude.ai, anything with discovery)** — just point
the connector at the MCP URL. The client follows the
WWW-Authenticate header to discover Supabase, runs the OAuth dance,
and uses the resulting Supabase token:

```
https://weft-mcp.fly.dev/mcp
```

**Issued bearer tokens (recommended for Claude Code, scripts, agents,
Wick containers)** — mint a token per client via the CLI, with the
caller mode bound to the credential at issuance time:

```bash
# Supervisor token for your own Face / Claude Code:
weft tokens issue --user-id <your-uuid> --mode supervisor --label face --expires-in 365d

# Agent-mode token for a Wick container or background worker:
weft tokens issue --user-id <your-uuid> --mode agent --label wick-cleanroom --expires-in 30d
```

The plaintext token prints **once**. Configure the client with it as
the bearer:

```json
{
  "mcpServers": {
    "weft": {
      "url": "https://weft-mcp.fly.dev/mcp",
      "headers": {
        "Authorization": "Bearer wf_..."
      }
    }
  }
}
```

`caller_mode` is determined by the token row, not by the
`X-Weft-Caller-Mode` header. An agent-mode token cannot escalate to
supervisor by sending the header — at most a supervisor can
**downgrade** itself to agent for testing. Rotate by minting a new
token, redeploying, then `weft tokens revoke <id>` on the old one.

**Bootstrap key (legacy / single-shared-secret)** — `WEFT_API_KEY` is
auto-treated as a supervisor token row on first request. Useful for
the very first deployment before the CLI is reachable, but every
production setup should mint per-client tokens and stop sharing it.
The env-var fallback emits a deprecation log line on each use and
will be removed once those logs go quiet.

## Operations

```bash
# View logs
fly logs

# SSH into the machine
fly ssh console

# Scale memory (if needed)
fly scale memory 1024

# Rotate a per-client token (no Fly restart needed):
weft tokens issue --user-id <uuid> --mode <supervisor|agent> --label <client>
# Update the client config with the new token, then:
weft tokens revoke <old-token-id>

# Rotate the bootstrap key (triggers rolling restart, only useful
# if the bootstrap key itself was leaked):
fly secrets set WEFT_API_KEY="new-bootstrap-secret"
```

## OAuth flow at runtime

End-to-end with Supabase as the OAuth 2.1 authorization server:

1. MCP client (Claude) hits `POST https://weft-mcp.fly.dev/mcp` with
   no auth.
2. Weft returns `401` with `WWW-Authenticate: Bearer realm="weft",
   resource_metadata="/.well-known/oauth-protected-resource"`.
3. Client fetches `/.well-known/oauth-protected-resource` →
   discovers `authorization_servers: ["https://<project>.supabase.co/auth/v1"]`.
4. Client fetches Supabase's
   `/auth/v1/.well-known/oauth-authorization-server` for the actual
   authorize / token / registration endpoints.
5. Client does dynamic client registration at Supabase.
6. Client opens Supabase's authorize URL in a browser.
7. Supabase redirects browser to
   `https://weft-mcp.fly.dev/oauth/consent?authorization_id=…` (the
   "Authorization URL Path" set in the dashboard).
8. Consent page (pure HTML + JS, served by Weft) loads the Supabase
   JS SDK, prompts for email magic-link sign-in if needed, then calls
   `supabase.auth.oauth.{getAuthorizationDetails,approveAuthorization,
   denyAuthorization}` from the browser.
9. On approve, JS redirects browser to the URL Supabase returned →
   completes back at the client with a code.
10. Client exchanges code at Supabase's `/auth/v1/oauth/token` →
    receives a Supabase-signed access token (RS256, JWKS at
    `/auth/v1/.well-known/jwks.json`).
11. Client calls `/mcp` with the token. Weft middleware verifies the
    JWT via Supabase JWKS (existing `weft.auth` path) and pins the
    `sub` as the request identity.

Bearer-token clients (Claude Code, agents, Wick containers) use the
`weft tokens issue` flow described above instead of the OAuth dance —
the credential row carries both `user_id` and `caller_mode`, so no
discovery round trip is needed. Both paths coexist on the same
deployment.

## Bringing up a new environment

The same shape works for staging, prod, or any new Fly app.

```bash
# 1. Set Fly secrets (replace project-specific values):
fly secrets set \
  WEFT_OAUTH_ENABLED=1 \
  OAUTH_ISSUER="https://<your-app>.fly.dev" \
  SUPABASE_URL="https://<your-supabase-project>.supabase.co" \
  SUPABASE_ANON_KEY="sb_publishable_..." \
  -a <your-app>

# 2. Configure Supabase Dashboard (one-time, per project):
#    Authentication → OAuth Server → enable
#    Authentication → OAuth Server → Authorization URL Path =
#       /oauth/consent
#    Authentication → OAuth Server → Allow dynamic client registration =
#       on

# 3. Deploy:
fly deploy -a <your-app>

# 4. Smoke test the discovery doc:
curl https://<your-app>.fly.dev/.well-known/oauth-protected-resource | jq
#    Expect: resource = your URL; authorization_servers contains
#            <supabase-project>.supabase.co/auth/v1

# 5. Test the gate:
curl -i -X POST https://<your-app>.fly.dev/mcp
#    Expect: HTTP 401 with WWW-Authenticate header.

# 6. End-to-end: add the connector in Claude (or Claude Desktop), run
#    the OAuth dance, call a tool to confirm the round trip.
```

Rollback at any point: `fly secrets unset WEFT_OAUTH_ENABLED -a
<your-app> && fly deploy -a <your-app>`. The middleware drops the
401 discovery gate and the consent + protected-resource routes stop
registering — bearer-token clients (issued tokens + the legacy
`WEFT_API_KEY` bootstrap path) keep working unchanged.

## Staging environment

`fly.staging.toml` defines a parallel `weft-mcp-staging` Fly app
(separate Supabase project) for testing changes against a real
Claude connector before promoting to prod. Same code, different
secrets, same dashboard config shape.

```bash
fly deploy -c fly.staging.toml
```

## Automated Backups

A GitHub Actions workflow runs every 12 hours to export a full JSON backup (memories, embeddings, relationships) from Supabase.

### Setup

Add the Supabase pooler connection secrets (use the pooler endpoint from Supabase Dashboard → Settings → Database → Connection pooling):

```bash
echo 'aws-0-REGION.pooler.supabase.com' | gh secret set BACKUP_PGHOST
echo '6543' | gh secret set BACKUP_PGPORT
echo 'postgres.YOUR_PROJECT_REF' | gh secret set BACKUP_PGUSER
echo 'YOUR_PASSWORD' | gh secret set BACKUP_PGPASSWORD
echo 'postgres' | gh secret set BACKUP_PGDATABASE
```

Using separate secrets avoids URL-parsing issues with special characters in passwords.

### How it works

- **Schedule**: Every 4 hours (:15 past the hour)
- **Artifacts**: Each backup is stored as a GitHub Actions artifact with 90-day retention
- **Validation**: Workflow fails if the backup is empty or contains 0 memories
- **Manual trigger**: Run from the Actions tab; optionally commit to a `backups` branch for git-based durability
- **Restore**: Download the artifact, then `weft restore weft-backup.json`
