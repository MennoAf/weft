# Weft

> Shared persistent brain for you and your agents.

Weft is a persistent brain you share with your agents. It doesn't remember everything, and that's on purpose.

Your agent doesn't need to remember every conversation, every answer, every date. You want it to remember what you need for that moment, and nothing more. By default, Weft remembers your decisions, plans, and other things that matter for your code in a way that makes it easy for your agents to pull up later using vector-based semantic search. 

## Why Weft

When you're coding with an AI agent, your default option to have a persistent memory is a set of flat files you have your agent read and write in.

That works for awhile, until it doesn't. 

- One agent can't read another agent's memories, so your learnings stay locked to the repo. 
- Retrieval is limited to `grep` so if you're not sure where a memory is, you have to burn up your agents context reading everything.
- There's no easy way for your agent to know the difference between something that's important to every session and a temporary rule you created to deal with a bug you fixed nine months ago.
- Switching sessions means starting from zero, giving your agent the same list of flat files to read.

Weft treats memory as a first-class data system.

Multiple agent platforms can read and write the to the same brain. You can handoff a session to a fresh agent with a simple command. Same agent, different agent, *different machine*, it doesn't matter. Weft built for your projects, not for a single platform.

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

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/). Docker Desktop is needed only to run local PostgreSQL and Redis services with `weft up` or to run the database-backed test suite. I've tested the system using podman, but it does not natively support that out of the box. (yet)

### 2. Start infrastructure

```bash
weft up
```

This launches Postgres 16 (with pgvector) on port 5433 and Redis 7 on port 6380, then runs migrations.

### 3. Register Weft as an MCP server

Register the Weft MCP server with your agent harness — see [AGENTS.md](AGENTS.md) for a copy-paste memory protocol and [the wiring guide](docs/wiring-your-agent.md) for full setup. Claude-specific setup: [CLAUDE.md](CLAUDE.md). Use the harness's own MCP documentation for its configuration format.

### 4. Point your agent at Weft

Use the MCP registration details in [AGENTS.md](AGENTS.md), then give your agent the copy-paste memory protocol or equivalent context for your harness. Claude-specific options are in [CLAUDE.md](CLAUDE.md). See the [agent wiring guide](docs/wiring-your-agent.md) for full connection steps.

You'll also want to set up [handoff](https://github.com/MennoAf/weft/blob/main/templates/commands/handoff.md) and [prime](https://github.com/MennoAf/weft/blob/main/templates/commands/prime.md) as skills your agents can access. 

### 5. First session

Start a session with the connected harness and ask it to call `weft_prime(disclosure="progressive")`. It will load any saved context or report that none is available. 

Tell it to register a project in weft and then use `weft_remember`, for example:

> Save: I prefer test descriptions in the form "test_<thing>_<condition>_<outcome>"

Have it update the Agents.md with the project name it gave for that repo so it knows what to call the net time.

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

Weft doesn't benchmark well because it's not designed to remember things the way a benchmark tests. However, for transparency I did run it through [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — multi-session memory evaluation across S (~40 sessions/question) using the "turn-tier" which is the memory type that works best here.

Current numbers and reproduction harness: [docs/benchmarks.md](docs/benchmarks.md).

In the future, I plan on showing my own benchmark so you can review how the system works in more detail.

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
