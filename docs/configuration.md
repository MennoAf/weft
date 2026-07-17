# Configuration

Weft uses four-layer configuration with increasing precedence:

1. **Defaults** built into code
2. **TOML file** at `~/.weft/config.toml`
3. **Project YAML** at `.weft/config.yaml` in your project directory
4. **Environment variables** (highest precedence)

To inspect what's actually in effect: `weft config show` prints the resolved config plus the source of each value.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WEFT_DATABASE_URL` | PostgreSQL connection string | `postgresql://weft:weft_local@localhost:5433/weft` |
| `WEFT_REDIS_URL` | Redis connection string | `redis://localhost:6380` |
| `WEFT_EMBEDDING_PROVIDER` | Embedding provider | `fastembed` |
| `WEFT_EMBEDDING_MODEL` | Embedding model | `BAAI/bge-small-en-v1.5` |
| `WEFT_LOG_LEVEL` | Log level | `INFO` |
| `WEFT_API_KEY` | Hosted-server bearer (legacy — auto-bootstraps a token row at startup; see [authentication](user-identity.md)) | unset |
| `WEFT_DEFAULT_USER_ID` | UUID the legacy bootstrap row binds to. Required for the `weft_api_key` path and owner-scoped integrations such as Slack sync and the daily brief. The recall-canary scheduler discovers and audits owners independently. | unset |
| `WEFT_OAUTH_ENABLED` | Enable Supabase JWT fallback when the bearer doesn't match a token row | `0` |
| `WEFT_USER_ID` | Override the local user identity for this process | unset |

## Configurable keys

Set via `weft config set <key> <value>`:

```
project_name                    Project identifier
log_level                       Logging level
database.url                    PostgreSQL URL
database.pool_min_size          Connection pool minimum
database.pool_max_size          Connection pool maximum
redis.url                       Redis URL
embedding.provider              fastembed | openai | google
embedding.model                 Model name
embedding.dimensions            Vector dimensions
embedding.batch_size            Batch size for bulk embedding
retrieval.default_top_k         Default result count
retrieval.similarity_threshold  Minimum similarity for results
retrieval.context_budget_tokens Default token budget
decay.enabled                   Enable automatic decay
decay.half_life_days            Days until confidence halves
decay.floor_score               Minimum score after decay
```

## Embedding providers

| Provider | Install | Notes |
|----------|---------|-------|
| `fastembed` | included | Local inference, no API key needed. Default: `BAAI/bge-small-en-v1.5` (384d). Best for getting started. |
| `openai` | included | Requires `OPENAI_API_KEY`. Default: `text-embedding-3-small` (768d). Recommended for production. |
| `google` | `pip install google-generativeai` | Requires `GOOGLE_API_KEY`. |

Switch providers with `weft config set embedding.provider <name>`. Note that switching providers after data is ingested will leave existing embeddings in the old vector space until re-embedded — use the bundled `weft re-embed` workflow.

## Infrastructure

`docker-compose.weft.yml` (run via `weft up`) provides:

- **PostgreSQL 16** with pgvector extension on port 5433
- **Redis 7** with append-only persistence on port 6380

Both services include health checks. Data persists in named Docker volumes (`weft-postgres-data`, `weft-redis-data`) — these survive `weft down` and `docker compose down`. They're cleared only by `docker volume rm`.

If you want to run against externally-managed Postgres + Redis, set `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` and skip `weft up`. Migrations still run automatically on first MCP-server boot if the database is reachable.
