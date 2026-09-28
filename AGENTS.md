# Weft — Persistent Agent Memory System

Part of the trilogy: Loom (orchestration) → Warp (builder agent) → Weft (memory).
Project context lives in Weft itself — `weft_prime(project_id="weft")` loads it. If `weft_prime` returns `degraded: true` (or `error: Database unavailable`), treat its output as untrusted: verify any “nothing found” conclusion with `weft_recall` before acting. During a database incident, prefer `weft_recall` and `weft_projects` for reads, and hold non-essential Weft writes until the incident clears.

## Default agent role: orchestrator

Unless Jason explicitly directs otherwise, act as the orchestrator and communication layer—not as the hands-on implementer. Delegate research, code and document changes, and independent verification to appropriately scoped sub-agents. Keep delegated work bounded, track progress, resolve blockers with Jason when needed, and synthesize the agents' evidence into clear updates and recommendations. Do not personally implement code or produce project deliverables; coordinate, review, and report instead. Use direct tools for orchestration and communication, and make exceptions only when Jason explicitly asks you to do the work yourself or a higher-priority instruction requires it.

## Loom project
Active build work tracks in **`weft-public`** (id: `aa3131c9-5ed6-49dd-b907-2e22f35691de`) —
making Weft public-ready ("going all in"): hardening, packaging, docs, productization.
Owned by Jason's user account, so it's accessible (unlike the old weft-wick). This repo is
**bound** to it: `.loom/config.yaml` carries the `project_id` and `.mcp.json` pins
`LOOM_PROJECT_DIR` + `LOOM_PROJECT_ID`, so a reloaded session resolves here automatically —
no `loom_switch_project` needed on boot. Lead agent: **Reed** (`reed`, role `lead`).

Predecessors: the **`weft-wick`** build (id `6605fce2-...`) is **complete** and is not
reachable from the default Loom identity (RLS denies access) — don't switch to it. The legacy
**`weft`** project (id `0bd76172-...`) holds historical V1 tasks — leave it alone unless asked.

## Commands
```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
uv run python -m weft            # Run MCP server locally
fly deploy                       # Deploy MCP to Fly.io (DB is Supabase, managed separately)
```

## Owner
Jason Bauman. Builder agent: Warp.
