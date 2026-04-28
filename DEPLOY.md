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
fly secrets set WEFT_API_KEY="your-secret-api-key"
```

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
| `WEFT_API_KEY` | Yes (prod) | — | Bearer token for legacy MCP clients (Claude Code). Coexists with the OAuth path. |
| `WEFT_ENV` | No | `production` | `local` or `production` |
| `WEFT_TRANSPORT` | No | `streamable-http` | `stdio`, `sse`, or `streamable-http` |
| `WEFT_REDIS_URL` | No | `""` (disabled) | Redis URL for caching (optional, uses NullCache if empty) |
| `PORT` | No | `8000` | HTTP port |
| `WEFT_OAUTH_ENABLED` | No | `0` | Set to `1` to publish RFC 9728 protected-resource metadata and serve the consent page. When off, the service stays byte-identical to the API-key-only build. |
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

Two paths, both supported simultaneously.

**OAuth (Claude app, claude.ai, anything with discovery)** — just point
the connector at the MCP URL. The client follows the
WWW-Authenticate header to discover Supabase, runs the OAuth dance,
and uses the resulting Supabase token:

```
https://weft-mcp.fly.dev/mcp
```

**API key (Claude Code, scripts, anything pre-OAuth)** — set the
bearer token to your `WEFT_API_KEY`:

```json
{
  "mcpServers": {
    "weft": {
      "url": "https://weft-mcp.fly.dev/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WEFT_API_KEY"
      }
    }
  }
}
```

## Operations

```bash
# View logs
fly logs

# SSH into the machine
fly ssh console

# Scale memory (if needed)
fly scale memory 1024

# Rotate a secret (triggers rolling restart)
fly secrets set WEFT_API_KEY="new-key-value"
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

The `WEFT_API_KEY` path stays available in parallel — Claude Code and
other legacy clients keep working with their static bearer token.

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
401 gate and the consent + protected-resource routes stop registering
— legacy `WEFT_API_KEY` clients keep working unchanged.

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
