---
description: Session boot sequence — load Weft persistent memory and report status
allowed-tools: [mcp__weft__weft_prime, mcp__weft__weft_focus]
---

# Session Prime

Run at the start of every session before doing other work.

## Step 1: Load context

Call `weft_prime(disclosure="progressive")`. The response contains:

- `rules` — pinned facts about this project
- `handoff` — the most recent session handoff (the bridge from the prior session)
- `issues` — open issues to be aware of
- `anti_patterns` — things to avoid
- `behaviors`, `decisions`, `entities`, `recent_work` — returned as counts only; load specific sections later via `weft_focus(intent="<what you're doing>")`

Read the returned context carefully — don't skim. The handoff in particular sets up where to start.

## Step 2: Report

Give the user a brief status:

- What the last handoff said (summary + next steps + open questions)
- Any open issues that look load-bearing for current work
- Confirm you're ready and ask what they're working on

## Fallback

If Weft is unreachable (MCP disconnected), say so explicitly and proceed without persistent memory — don't error out.
