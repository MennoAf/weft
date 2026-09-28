# Weft

> Shared persistent brain for you and your agents.

Weft is the persistent brain you share with your agents. Conversations, decisions, plans, and other useful information can be saved for later retrieval by you and your agents. Queryable via semantic search (pgvector), structured memory types, confidence metadata, pinned conventions, hierarchical retrieval, and cross-session/cross-agent continuity. Lifecycle scoring is review-only: consolidation can propose stale candidates, but it does not automatically hide or delete memories.

Use Weft as a persistent memory service alongside the agent tools and workflows you already use.

## Why Weft

Default agent memory is a flat file the agent grep-reads at session start. That works for a while, then it doesn't:

- One agent can't read another agent's memories
- No retrieval beyond grep — semantic similarity, time-aware ranking, and provenance all live in the agent's head
- No structured confidence, provenance, pinning, or review lifecycle to distinguish a load-bearing convention from a one-off observation
- No cross-session continuity beyond the user re-pasting context

Weft treats memory as a first-class data system. Multiple agents and MCP clients can read and write the same brain. A handoff at session end shows up in the next session's prime — same agent, different agent, different machine, doesn't matter.

## Quickstart

Five steps, ~5 minutes.

### 1. Install

```bash
# Recommended: pipx for system-wide availability
pipx install git+https://github.com/MennoAf/weft.git

# Or as a uv tool
uv tool install git+https://github.com/MennoAf/weft.git

# Or local development install
git clone https://github.com/MennoAf/weft.git && cd weft && uv sync
```

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/). Docker Desktop is needed only to run local PostgreSQL and Redis services with `weft up` or to run the database-backed test suite; it is not required to install or import Weft.

### 2. Start infrastructure

```bash
weft up
```

This launches Postgres 16 (with pgvector) on port 5433 and Redis 7 on port 6380, then runs migrations.

### 3. Register Weft as an MCP server

Register the Weft MCP server with your agent harness — see [AGENTS.md](AGENTS.md) for a copy-paste memory protocol and [the wiring guide](docs/wiring-your-agent.md) for full setup. Claude-specific setup: [CLAUDE.md](CLAUDE.md). Use the harness's own MCP documentation for its configuration format.

### 4. Point your agent at Weft

Use the MCP registration details in [AGENTS.md](AGENTS.md), then give your agent the copy-paste memory protocol or equivalent context for your harness. Claude-specific options are in [CLAUDE.md](CLAUDE.md). See the [agent wiring guide](docs/wiring-your-agent.md) for full connection steps.

### 5. First session

Start a session with the connected harness and ask it to call `weft_prime(disclosure="progressive")`. It will load any saved context or report that none is available. To save a durable preference, ask it to use `weft_remember`, for example:

> Save: I prefer test descriptions in the form "test_<thing>_<condition>_<outcome>"

At the end of a non-trivial session, ask it to call `weft_handoff`. In a later session, `weft_prime` surfaces the saved context and handoff.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  MCP Interface                   │
│  weft_remember  weft_recall   weft_context       │
│  weft_revise    weft_forget   weft_feedback      │
│  weft_relate    weft_consolidate                 │
│  weft_prime     weft_status   weft_extract       │
│  weft_pin       weft_learn    weft_handoff       │
├──────────────────────────────────────────────────┤
│              Business Logic                      │
│  store · relevance · context · primer            │
│  consolidation · importer · exporter             │
│  extract · fallback · revise                     │
├──────────────────────────────────────────────────┤
│              Ingestion                           │
│  Obsidian vault sync · codebase ingest           │
├──────────────────────────────────────────────────┤
│              Infrastructure                      │
│  PostgreSQL + pgvector  │  Redis cache           │
│  Embeddings: fastembed (local) or OpenAI         │
└──────────────────────────────────────────────────┘
```

Full reference: [docs/tools.md](docs/tools.md).

## Benchmarks

Weft is benchmarked against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — multi-session memory evaluation across S (~40 sessions/question) and M (~500 sessions/question) haystacks.

Current numbers and reproduction harness: [docs/benchmarks.md](docs/benchmarks.md).

## What you can do

- **Query memories** — `weft recall "database configuration patterns"`
- **Import existing memories** — `weft import /path/to/MEMORY.md`
- **Sync an Obsidian vault** — `weft obsidian sync ~/Documents/MyVault` ([guide](docs/obsidian.md))
- **Ingest a codebase** — `weft ingest .` for grounded recall against your own source
- **Inspect state** — `weft status` for counts, `weft config show` for resolved config
- **Mint API tokens** — `weft tokens issue` for hosted-server auth ([guide](docs/user-identity.md))
- **Back up + restore** — `weft backup` / `weft restore` ([guide](docs/disaster-recovery.md))

Full CLI: [docs/cli.md](docs/cli.md).

## Documentation

| | |
| --- | --- |
| **[Wire your agent](docs/wiring-your-agent.md)** | Harness-neutral MCP setup and persistent-memory workflow |
| **[MCP tool reference](docs/tools.md)** | Every tool, every parameter |
| **[CLI reference](docs/cli.md)** | Every command |
| **[Configuration](docs/configuration.md)** | Environment variables, TOML keys, infrastructure |
| **[Database and schema guide](docs/database-schema.md)** | 52 public tables, ownership/RLS, migrations, vectors, and export boundaries |
| **[Authentication + identity](docs/user-identity.md)** | Tokens, caller modes, agent floor |
| **[Retrieval + scope](docs/retrieval-and-scope.md)** | How `weft_recall` decides what comes back |
| **[Benchmarks](docs/benchmarks.md)** | LongMemEval methodology + current numbers |
| **[Obsidian integration](docs/obsidian.md)** | Vault sync, frontmatter, Tasks plugin |
| **[Disaster recovery](docs/disaster-recovery.md)** | Backup, restore, schema migration |

## Development

```bash
uv sync                        # Install dependencies
uv run pytest tests/ -v        # Run all tests (~2670 tests, testcontainers-isolated)
uv run python -m weft          # Run CLI
uv run python -m weft.mcp      # Run MCP server (stdio)
```

Database-backed tests use `testcontainers` for isolation — each test gets a fresh Postgres+pgvector instance and requires Docker. Pure unit tests do not require Docker.

## License

MIT
