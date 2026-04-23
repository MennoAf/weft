# Weft — Getting Started (Brandon)

Hey Brandon — this is the short path to running Weft on your machine. You're going fully local, so no Fly.io, no Supabase. Postgres + Redis run in Docker on your laptop, and Weft installs globally so any project on the device can use it via MCP.

## What Weft is

Persistent, queryable memory for AI agents. Think of it as the replacement for flat `MEMORY.md` files — semantic search, confidence scoring, automatic decay, handoffs between sessions. It's an MCP server, so Claude Code (or any MCP client) talks to it over stdio.

Part of the trilogy: **Loom** (orchestration) → **Warp** (builder agent) → **Weft** (memory). You only need Weft to start; the other two are optional.

## Prereqs

- **Python 3.12+** (check with `python3 --version`)
- **[uv](https://docs.astral.sh/uv/)** — `brew install uv`
- **Docker Desktop** — running, because Postgres + Redis live in containers
- **pipx** (optional but recommended for global install) — `brew install pipx`

## Install (global, device-wide)

Pick one. Both put a `weft` binary on your PATH so every project on the device can invoke it.

**Option A — pipx (simplest):**
```bash
pipx install git+https://github.com/MennoAf/weft-memory.git
weft --help
```

**Option B — uv tool:**
```bash
uv tool install git+https://github.com/MennoAf/weft-memory.git
weft --help
```

If you want to hack on the code later, do a dev clone instead (`git clone ... && cd weft-memory && uv sync`) and use `uv run weft` in place of `weft`. Don't mix the two — pick global or dev.

## First-time setup

### 1. Start infrastructure

```bash
weft up
```

This boots Postgres 16 with pgvector on port **5433** and Redis 7 on port **6380** (non-default ports so they don't collide with anything else you're running), and runs migrations automatically. Data persists in named Docker volumes, so `docker restart` is safe.

`weft down` stops them. `weft status` shows memory stats once you've stored something.

### 2. Register Weft as an MCP server

Add to `~/.claude.json` (Claude Code global) or the project's `.mcp.json`:

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

Restart Claude Code. You should see `mcp__weft__*` tools available.

### 3. Drop this into your `CLAUDE.md`

So the agent actually uses Weft instead of ignoring it:

```markdown
## Weft Memory

Call `weft_prime` at the start of every session to load context.
Use `weft_remember` to store important facts, patterns, and preferences.
Use `weft_learn` after completing tasks to capture what was learned.
Use `weft_handoff` before ending a session to preserve continuity.
Rule of thumb: if you'd want to know it next session, save it now.
```

## Daily usage

Most of it happens through the agent via MCP tools, but the CLI is handy too:

```bash
weft recall "postgres connection settings"      # semantic search
weft status                                     # memory stats
weft import ~/path/to/MEMORY.md                 # one-time migration from flat-file memory
weft consolidate --dry-run                      # preview decay/dedup
weft backup --output weft-backup.json           # full snapshot
```

## Optional: Obsidian vault sync

If you keep notes in Obsidian, Weft can ingest them so the agent can semantic-search your personal notes, contacts, recipes, etc:

```bash
weft obsidian init ~/Documents/MyVault     # scaffolds folder structure + templates
weft obsidian sync ~/Documents/MyVault     # pulls notes into memory (hash-diffed, idempotent)
```

Skip this if you don't use Obsidian.

## Things that will bite you

- **Ports 5433/6380, not 5432/6379.** Intentional — don't `psql -p 5432` and get confused when it connects to something else.
- **`weft up` needs Docker Desktop running.** If `docker ps` errors, start Docker first.
- **Embeddings default to local (`fastembed`).** No API key needed. If you want OpenAI embeddings instead, `weft config set embedding.provider openai` and export `OPENAI_API_KEY`.
- **Don't write to flat-file `MEMORY.md` once Weft is running.** Use `weft_remember` / `weft_learn`. The whole point is queryable memory, not another text file.

## If something breaks

- `weft status` — does the DB respond?
- `docker ps` — are the two containers up?
- `docker logs weft-postgres` / `docker logs weft-redis` — anything complaining?
- `weft down && weft up` — nuclear option, keeps your data (volumes persist).

Ping me if you hit anything weird. — Jason
