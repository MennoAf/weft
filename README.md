# Weft

> Shared persistent brain for you and your agents.

Weft is the persistent brain you share with your agents. Conversations, decisions, recipes, plans — anything you'd write down so you and your agents can find it later. Queryable via semantic search (pgvector), structured memory types, confidence metadata, pinned conventions, hierarchical retrieval, and cross-session/cross-agent continuity. Lifecycle scoring is review-only: consolidation can propose stale candidates, but it does not automatically hide or delete memories.

Part of the trilogy: **Loom** (orchestration) → **Warp** (builder agent) → **Weft** (memory).

## Why Weft

Default agent memory is a flat file the agent grep-reads at session start. That works for a while, then it doesn't:

- One agent can't read another agent's memories
- No retrieval beyond grep — semantic similarity, time-aware ranking, and provenance all live in the agent's head
- No structured confidence, provenance, pinning, or review lifecycle to distinguish a load-bearing convention from a one-off observation
- No cross-session continuity beyond the user re-pasting context

Weft treats memory as a first-class data system. Multiple agents (Claude Code, custom MCP clients, Claude API apps, Warp, etc.) read and write the same brain. A handoff at session end shows up in the next session's prime — same agent, different agent, different machine, doesn't matter.

## Quickstart

Five steps, ~5 minutes.

### 1. Install

```bash
# Recommended: pipx for system-wide availability
pipx install git+https://github.com/MennoAf/weft-memory.git

# Or as a uv tool
uv tool install git+https://github.com/MennoAf/weft-memory.git

# Or local development install
git clone https://github.com/MennoAf/weft-memory.git && cd weft-memory && uv sync
```

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker Desktop.

### 2. Start infrastructure

```bash
weft up
```

This launches Postgres 16 (with pgvector) on port 5433 and Redis 7 on port 6380, then runs migrations.

### 3. Register Weft as an MCP server

Add to your project's `.mcp.json` (or Claude Desktop's MCP config):

```json
{
  "mcpServers": {
    "weft": {
      "command": "weft",
      "args": ["mcp"]
    }
  }
}
```

Use `command: "uv"` with `args: ["run", "--directory", "/path/to/weft-memory", "python", "-m", "weft.mcp"]` if you installed locally instead.

### 4. Wire your agent

The agent needs to know to use Weft instead of flat files. Drop in the templates:

```bash
# Memory protocol — append to ~/.claude/CLAUDE.md
cat templates/CLAUDE.md >> ~/.claude/CLAUDE.md

# Slash commands
mkdir -p ~/.claude/commands
cp templates/commands/prime.md ~/.claude/commands/
cp templates/commands/handoff.md ~/.claude/commands/
```

Now every Claude Code session in any project will use Weft. Type `/prime` at session start to load context, `/handoff` before ending. The full guide with verification steps is at [docs/wiring-your-agent.md](docs/wiring-your-agent.md).

### 5. First session

Start a Claude Code session and type `/prime`. The agent will call `weft_prime`, find no prior context (you're new), and ask what you're working on. Save something:

> Save: I prefer test descriptions in the form "test_<thing>_<condition>_<outcome>"

It will call `weft_remember` with type `preference`. Then `/handoff` and end the session.

Open a new session in any project, `/prime` again. The handoff lands at the top, the preference is in the prime — your agent is now working with persistent context across sessions.

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
- **Import existing memories** — `weft import ~/.claude/memory/MEMORY.md`
- **Sync an Obsidian vault** — `weft obsidian sync ~/Documents/MyVault` ([guide](docs/obsidian.md))
- **Ingest a codebase** — `weft ingest .` for grounded recall against your own source
- **Inspect state** — `weft status` for counts, `weft config show` for resolved config
- **Mint API tokens** — `weft tokens issue` for hosted-server auth ([guide](docs/user-identity.md))
- **Back up + restore** — `weft backup` / `weft restore` ([guide](docs/disaster-recovery.md))

Full CLI: [docs/cli.md](docs/cli.md).

## Documentation

| | |
| --- | --- |
| **[Wire your agent](docs/wiring-your-agent.md)** | Full CLAUDE.md template + slash command setup + verification |
| **[MCP tool reference](docs/tools.md)** | Every tool, every parameter |
| **[CLI reference](docs/cli.md)** | Every command |
| **[Configuration](docs/configuration.md)** | Environment variables, TOML keys, infrastructure |
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

Tests use `testcontainers` for database isolation — each test gets a fresh Postgres+pgvector instance. No external services required.

## License

MIT
