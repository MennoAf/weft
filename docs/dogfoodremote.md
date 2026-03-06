# Weft Remote MCP — Multi-Client Setup

## Architecture

- **Database:** Supabase PostgreSQL (direct connection, port 5432)
- **MCP Server:** Fly.io app `weft-mcp` (IAD region), streamable-http transport
- **URL:** `https://weft-mcp.fly.dev/mcp/`
- **Auth:** Bearer token via `WEFT_API_KEY`

All clients share the same Supabase database — memories are instantly visible across all connected Claude instances.

## API Key

Stored in your password manager. To reset:

```bash
# Generate a new key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Set it on Fly.io (triggers redeploy)
fly secrets set WEFT_API_KEY='NEW_KEY_HERE' -a weft-mcp
```

## Client Configuration

### Claude Code (any system)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "weft": {
      "type": "url",
      "url": "https://weft-mcp.fly.dev/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_WEFT_API_KEY"
      }
    }
  }
}
```

### Claude Desktop App

Settings > MCP Servers > Add remote server:
- URL: `https://weft-mcp.fly.dev/mcp/`
- Header: `Authorization: Bearer YOUR_WEFT_API_KEY`

### Claude Web App

If remote MCP is supported, same URL and auth header as above.

## Fly.io Management

```bash
# Check status
fly status -a weft-mcp

# View logs
fly logs -a weft-mcp

# Health check
curl https://weft-mcp.fly.dev/healthz

# Deploy after code changes
fly deploy -a weft-mcp

# View secrets (names only, values are hidden)
fly secrets list -a weft-mcp
```

## Auto-Stop Behavior

The Fly.io machine has `auto_stop_machines = "stop"` and `auto_start_machines = true` with `min_machines_running = 0`. This means:

- The machine stops after a period of inactivity (no requests)
- It auto-starts on the next incoming request
- First request after idle may take ~5-10 seconds (cold start: fastembed model load + DB connection)
- Subsequent requests are fast

## Troubleshooting

**502 errors on first request:** The machine is waking from auto-stop. Wait a few seconds and retry. The health check at `/healthz` will return 200 once ready.

**Fly Doctor "not listening" warning:** This can appear transiently during cold starts while the lifespan initializes (DB, embeddings). If health checks are passing in the logs, it's a stale diagnostic — ignore it.

**Auth failures (401):** Verify your API key matches what's set on Fly. Check with `fly secrets list -a weft-mcp` to confirm `WEFT_API_KEY` exists.

**Redis warnings in logs:** Expected — Redis is intentionally disabled in production (`WEFT_REDIS_URL=""`). The server uses NullCache and logs "No Redis URL configured, using NullCache" on startup.

## Key Secrets on Fly.io

| Secret | Purpose |
|--------|---------|
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `WEFT_API_KEY` | Bearer token for MCP client auth |

Note: `BACKUP_DATABASE_URL` and `DATABASE_URL` GitHub secrets were cleaned up (2026-03-06). GitHub Actions backups use separate `BACKUP_PG*` secrets pointing to the Supabase pooler (port 6543).
