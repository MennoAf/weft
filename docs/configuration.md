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
| `WEFT_MIGRATION_MODE` | `apply` runs owner-managed DDL; `verify` performs a read-only exact migration-version check for restricted hosted runtimes | `apply` |
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
decay.enabled                   Legacy configuration field; automatic destructive decay is disabled
decay.half_life_days            Review-score recency half-life in days
decay.floor_score               Minimum review score used to propose candidates
```

## Embedding providers

| Provider | Install | Notes |
|----------|---------|-------|
| `fastembed` | included | Local inference, no API key needed. Default: `BAAI/bge-small-en-v1.5` (384d). Best for getting started. |
| `openai` | included | Requires `OPENAI_API_KEY`. Default: `text-embedding-3-small` (768d). Recommended for production. |
| `google` | `pip install google-generativeai` | Requires `GOOGLE_API_KEY`. |

### Text-generation provider routing

LLM-powered features use the provider-neutral text-generation seam in
`weft.text_generation`. Anthropic remains the default provider and existing
Haiku model behavior is unchanged. The provider and logical role models can be
configured without editing feature code:

```toml
[text_generation]
provider = "anthropic"

[text_generation.models]
ingest_classifier = "claude-haiku-4-5-20251001"
codebase_summary = "claude-haiku-4-5-20251001"
codebase_architecture = "claude-haiku-4-5-20251001"
```

Environment variables override TOML values:

- `WEFT_TEXT_PROVIDER`
- `WEFT_TEXT_MODEL_INGEST_CLASSIFIER`
- `WEFT_TEXT_MODEL_CODEBASE_SUMMARY`
- `WEFT_TEXT_MODEL_CODEBASE_ARCHITECTURE`
- `WEFT_TEXT_MODEL_QUARANTINE_REVIEW`
- `WEFT_TEXT_MODEL_BELIEF_DETECTOR`
- `WEFT_TEXT_MODEL_TOPIC_SYNTHESIS`
- `WEFT_TEXT_MODEL_REPLAY_AGGREGATE`

Provider adapters are opt-in and must preserve each feature's existing
bounded-call, parsing, and fail-closed behavior. Selecting an unregistered
provider fails explicitly; this is intentional until a provider has an
implemented adapter and role-specific quality/safety evidence.

The OpenAI text-generation adapter uses the Responses API and is available
through the optional `openai` extra. It reads `OPENAI_API_KEY`; the SDK client
uses bounded defaults of 30 seconds and two retries, configurable with
`WEFT_OPENAI_TEXT_TIMEOUT` and `WEFT_OPENAI_TEXT_RETRIES`. OpenAI's SDK owns
eligible transient retries (including 429 `slow_down` and 503 model overload),
so Weft does not add a second retry loop. Authentication, quota, billing, and
spend-limit failures are not retryable. Responses requests set `store = false`
and normalize `output_text`; structured output can be supplied through the
provider-neutral `GenerationRequest.response_format` field.

For example, to opt low-risk roles into GPT-5.6 Luna after installing the
extra:

```bash
uv sync --extra openai
export OPENAI_API_KEY=...
export WEFT_TEXT_PROVIDER=openai
export WEFT_TEXT_MODEL_INGEST_CLASSIFIER=gpt-5.6-luna
export WEFT_TEXT_MODEL_CODEBASE_SUMMARY=gpt-5.6-luna
export WEFT_TEXT_MODEL_CODEBASE_ARCHITECTURE=gpt-5.6-luna
```

Do not use this switch as production approval for quarantine review, belief
detection, topic synthesis, or replay aggregation. Those roles remain on the
validated default until provider-specific safety and quality benchmarks pass.

Switch providers with `weft config set text_generation.provider <name>`. Note that switching providers after data is ingested will leave existing embeddings in the old vector space until re-embedded — use the bundled `weft re-embed` workflow.

## Infrastructure

`docker-compose.weft.yml` (run via `weft up`) provides:

- **PostgreSQL 16** with pgvector extension on port 5433
- **Redis 7** with append-only persistence on port 6380

Both services include health checks. Data persists in named Docker volumes (`weft-postgres-data`, `weft-redis-data`) — these survive `weft down` and `docker compose down`. They're cleared only by `docker volume rm`.

If you want to run against externally-managed Postgres + Redis, set `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` and skip `weft up`. Migrations still run automatically on first MCP-server boot if the database is reachable.

## Production deployment boundary

Production Fly deploys must come from a clean checkout at exactly one approved
remote tag matching `production-YYYY-MM-DD` (an optional suffix is allowed,
for example `production-2026-09-03-hotfix`). The tag must point to `HEAD` and
must resolve to the same commit on `origin`. A production tag may intentionally
lag `main` while an RC is being reviewed; a synchronized `main` is not itself
production approval. Do not deploy from an RC, benchmark, recovery, or dirty
development worktree.

To prepare an approved deployment ref, an operator should first review and
verify the candidate commit, create the production tag in the hosting system,
push the tag to `origin`, fetch it locally, and check out the tag in detached
HEAD mode. Tag creation and the deployment itself remain explicit operator
actions; this repository does not contain credentials or automate approval.

Run the guarded wrapper from the repository root while checked out at the
approved tag:

```bash
git fetch origin --tags
git checkout --detach production-YYYY-MM-DD
./scripts/deploy_production.sh
```

The guard requires the exact `weft-mcp` Fly app configuration, an approved tag
at `HEAD` that is also present on `origin`, rejects dirty or untracked deploy-ref
content, rejects generated benchmark data/snapshots/runs/results, rejects Git
blobs over GitHub's 100 MB limit, and confirms that the Dockerfile does not
copy benchmark content into the image. Internal benchmark source and RC work
belong on separate branches or worktrees and must never be the production deploy
ref. Leave production on the known-good recovery image until the RC has passed
review and a new production tag is explicitly approved.
