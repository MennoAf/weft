# Weft — Persistent Agent Memory System

Part of the trilogy: Loom (orchestration) → Warp (builder agent) → Weft (memory).
Project context lives in Weft itself — `weft_prime(project_id="weft")` loads it.

## Commands
```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
uv run python -m weft            # Run MCP server locally
fly deploy                       # Deploy MCP to Fly.io (DB is Supabase, managed separately)
```

## Owner
Jason Bauman. Builder agent: Warp.
