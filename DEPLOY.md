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
| `WEFT_ENV` | No | `production` | `local` or `production` |
| `WEFT_TRANSPORT` | No | `streamable-http` | `stdio`, `sse`, or `streamable-http` |
| `WEFT_REDIS_URL` | No | `""` (disabled) | Redis URL for caching (optional, uses NullCache if empty) |
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

## Automated Backups

A GitHub Actions workflow runs every 12 hours to export a full JSON backup (memories, embeddings, relationships) from Supabase.

### Setup

Add the `DATABASE_URL` repository secret:

```bash
gh secret set DATABASE_URL --body "postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres"
```

### How it works

- **Schedule**: Every 12 hours (00:15 and 12:15 UTC)
- **Artifacts**: Each backup is stored as a GitHub Actions artifact with 90-day retention
- **Validation**: Workflow fails if the backup is empty or contains 0 memories
- **Manual trigger**: Run from the Actions tab; optionally commit to a `backups` branch for git-based durability
- **Restore**: Download the artifact, then `weft restore weft-backup.json`
