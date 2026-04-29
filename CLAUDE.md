# Weft — Persistent Agent Memory System

Part of the trilogy: Loom (orchestration) → Warp (builder agent) → Weft (memory).
Project context lives in Weft itself — `weft_prime(project_id="weft")` loads it.

## Loom project
Active build work tracks in **`weft-wick`** (id: `6605fce2-9cde-4e56-b1f4-441db72d2687`) — the Weft V2 build driven by Wick's needs (user-scope tier, provenance, trackers, Phase 2+ poisoning defense, Phase 2.5 credential-bound caller mode). On boot, switch the Loom project context there:

```python
loom_switch_project(project_id="6605fce2-9cde-4e56-b1f4-441db72d2687")
```

The legacy `weft` project (id `0bd76172-...`) holds historical V1 tasks — leave it alone unless explicitly asked.

## Commands
```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
uv run python -m weft            # Run MCP server locally
fly deploy                       # Deploy MCP to Fly.io (DB is Supabase, managed separately)
```

## Owner
Jason Bauman. Builder agent: Warp.
