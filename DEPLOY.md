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
fly secrets set ANTHROPIC_API_KEY="your-anthropic-key"
```

### Supabase Connection Notes

- Use the **direct connection** URL (port 5432), not the pooler URL
- Passwords with special characters are automatically URL-encoded by Weft
- If using the pooler URL (port 6543), set `database.statement_cache_size=0` in your config — asyncpg requires this for pgbouncer compatibility

## Deploy

```bash
fly deploy
```

Migrations run automatically on startup before the server accepts traffic.

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
| `WEFT_API_KEY` | Yes (prod) | — | Bearer token for MCP client auth |
| `ANTHROPIC_API_KEY` | Yes | — | For embedding generation |
| `WEFT_ENV` | No | `production` | `local` or `production` |
| `WEFT_TRANSPORT` | No | `streamable-http` | `stdio`, `sse`, or `streamable-http` |
| `PORT` | No | `8000` | HTTP port |

## Connecting MCP Clients

Configure your MCP client to connect via streamable HTTP with a bearer token:

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
