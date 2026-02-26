# Weft

Persistent memory system for AI agents. Replaces flat-file memory with a queryable knowledge base supporting semantic retrieval, confidence tracking, relationship mapping, and automatic decay.

Part of the trilogy: **Loom** (orchestration) &rarr; **Warp** (builder agent) &rarr; **Weft** (memory).

## Why Weft

Current agent memory is a flat markdown file with no structure, no retrieval beyond grep, no decay, and no way to distinguish high-confidence knowledge from speculation. Weft treats memory as a first-class data system:

- **Semantic search** via pgvector embeddings
- **Memory types** with distinct lifecycles (preferences, facts, patterns, architecture decisions)
- **Confidence & decay** so stale knowledge fades and useful knowledge rises
- **Cross-project sharing** with isolation (global memories visible everywhere, project-scoped memories stay private)
- **Token-budget context assembly** so sessions start with the right 5% of knowledge
- **Feedback loop** that adjusts relevance based on whether memories were actually helpful
- **Fallback resilience** so agents still have memory access when infrastructure is down

## Quickstart

### 1. Start infrastructure

```bash
# Clone and install
git clone <repo-url> && cd weft
uv sync

# Start Postgres (pgvector) + Redis and run migrations
weft up
```

### 2. Import existing memories

```bash
# Import from a MEMORY.md file (deduplicates automatically)
weft import ~/.claude/memory/MEMORY.md

# Or scope to a project
weft import MEMORY.md --project-id my-project
```

### 3. Query memories

```bash
# Semantic search from the CLI
weft recall "database configuration patterns"

# View memory stats
weft status
```

### 4. Register as MCP server

Copy `.mcp.json.example` to your project's `.mcp.json` (or merge into an existing one), adjusting the `--directory` path:

```json
{
  "mcpServers": {
    "weft": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/weft", "python", "-m", "weft.mcp"]
    }
  }
}
```

Then use Weft tools from any MCP-compatible client (Claude Desktop, Claude Code, etc.).

## Architecture

```
┌─────────────────────────────────────────────┐
│                MCP Interface                │
│  weft_remember  weft_recall  weft_context   │
│  weft_revise    weft_forget  weft_feedback  │
│  weft_relate    weft_consolidate            │
│  weft_prime     weft_status  weft_extract   │
├─────────────────────────────────────────────┤
│            Business Logic                   │
│  store  relevance  context  primer          │
│  consolidation  importer  exporter          │
│  extract  fallback                          │
├─────────────────────────────────────────────┤
│            Infrastructure                   │
│  PostgreSQL + pgvector  │  Redis cache      │
│  fastembed / OpenAI / Google embeddings     │
└─────────────────────────────────────────────┘
```

## MCP Tool Reference

### weft_remember

Store a new memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Memory content |
| `type` | `str` | `"fact"` | One of: `preference`, `fact`, `pattern`, `relationship`, `solution`, `architecture`, `user_model` |
| `topic` | `list[str]` | `[]` | Topic tags for filtering |
| `source` | `str` | `"conversation"` | One of: `conversation`, `code`, `documentation`, `inference` |
| `confidence` | `float` | `0.7` | Confidence score (0.0-1.0) |
| `project_id` | `str` | `null` | Scope to a project (null = global) |
| `agent_id` | `str` | `null` | Originating agent identifier |
| `check_contradictions` | `bool` | `true` | Check for contradicting memories on store |

### weft_recall

Retrieve memories by semantic query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Natural language search query |
| `topic` | `str` | `null` | Filter by topic |
| `type` | `str` | `null` | Filter by memory type |
| `status` | `str` | `"active"` | Filter by status: `active`, `archived`, `decayed` |
| `project_id` | `str` | `null` | Filter by project |
| `limit` | `int` | `10` | Max results |
| `threshold` | `float` | `0.3` | Minimum similarity score |

### weft_context

Budget-aware context loading. Returns the best memories for a situation within a token budget.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Context query |
| `budget_tokens` | `int` | `4000` | Maximum tokens to return |
| `topic` | `str` | `null` | Filter by topic |
| `type` | `str` | `null` | Filter by memory type |
| `project_id` | `str` | `null` | Filter by project |
| `max_per_topic` | `int` | `3` | Maximum memories per topic |

### weft_prime

Session primer: assemble structured context for session startup.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_id` | `str` | `null` | Scope to project |
| `budget_tokens` | `int` | `4000` | Token budget |
| `recent_days` | `int` | `7` | How far back to look for recent work |

Returns `{ preferences, recent_work, relevant, total_tokens, budget_tokens, budget_remaining }`.

### weft_revise

Update a memory's content, creating a new version that supersedes the old one.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | ID of memory to revise |
| `new_content` | `str` | *required* | Updated content |
| `new_confidence` | `float` | `null` | Updated confidence |
| `new_topic` | `list[str]` | `null` | Updated topics |

### weft_forget

Archive or permanently delete a memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | ID of memory to forget |
| `hard` | `bool` | `false` | Hard-delete instead of archive |

### weft_feedback

Record whether a memory was helpful. Adjusts the usefulness score for future ranking via exponential moving average.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | Memory that was used |
| `helpful` | `bool` | *required* | Was the memory helpful? |

### weft_relate

Manage relationships between memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `str` | *required* | `add`, `get`, or `remove` |
| `memory_id` | `str` | *required* | Source memory ID |
| `target_id` | `str` | `null` | Target memory ID (for add/remove) |
| `relation` | `str` | `null` | Relation type: `supersedes`, `related_to`, `contradicts`, `derived_from` |

### weft_consolidate

Run the consolidation pipeline: decay stale memories, merge duplicates, flag contradictions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dry_run` | `bool` | `false` | Preview changes without applying |

### weft_extract

Extract memory candidates from a block of text using heuristic pattern matching. Returns proposals for review -- does NOT auto-store.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Text to extract candidates from |
| `min_confidence` | `float` | `0.5` | Minimum confidence threshold for candidates |

Returns `{ count, candidates }` where each candidate has `content`, `type`, `confidence`, `topic`, `source_line`.

### weft_status

Return memory statistics: total count, breakdown by type/topic/status, recently accessed.

*No parameters.*

## CLI Reference

```
weft mcp                 Start the MCP server (stdio transport)
weft up                  Start Postgres + Redis, run migrations
weft down                Stop containers
weft status              Show memory statistics
weft recall QUERY        Semantic search (--limit, --topic)
weft import FILE         Import MEMORY.md (--dry-run, --project-id)
weft export              Export memories (--format md|json, --type, --topic, --status, --output)
weft consolidate         Run decay/dedup/contradiction pipeline (--dry-run)
weft config show         Display current configuration
weft config set KEY VAL  Persist a config value to ~/.weft/config.toml
```

## Configuration

Weft uses four-layer configuration with increasing precedence:

1. **Defaults** (built into code)
2. **TOML file** (`~/.weft/config.toml`)
3. **Project YAML** (`.weft/config.yaml` in project directory)
4. **Environment variables** (highest precedence)

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WEFT_DATABASE_URL` | PostgreSQL connection string | `postgresql://weft:weft_local@localhost:5433/weft` |
| `WEFT_REDIS_URL` | Redis connection string | `redis://localhost:6380` |
| `WEFT_EMBEDDING_PROVIDER` | Embedding provider | `fastembed` |
| `WEFT_EMBEDDING_MODEL` | Embedding model | `BAAI/bge-small-en-v1.5` |
| `WEFT_LOG_LEVEL` | Log level | `INFO` |

### Configurable keys

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

## Infrastructure

### Docker Compose

`docker-compose.weft.yml` provides:

- **PostgreSQL 16** with pgvector extension (port 5433)
- **Redis 7** with append-only persistence (port 6380)

Both services include health checks. Data is persisted in named Docker volumes (`weft-postgres-data`, `weft-redis-data`).

### Embedding Providers

| Provider | Install | Notes |
|----------|---------|-------|
| `fastembed` | Included | Local inference, no API key needed. Default: `BAAI/bge-small-en-v1.5` (384d) |
| `openai` | `pip install openai` | Requires `OPENAI_API_KEY` |
| `google` | `pip install google-generativeai` | Requires `GOOGLE_API_KEY` |

## Development

```bash
uv sync                        # Install dependencies
uv run pytest tests/ -v        # Run all tests (249 tests)
uv run python -m weft          # Run CLI
uv run python -m weft.mcp      # Run MCP server (stdio)
```

Tests use `testcontainers` for database isolation -- each test gets a fresh Postgres+pgvector instance. No external services required for testing.

## License

MIT
