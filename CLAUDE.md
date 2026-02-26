# Weft — Persistent Agent Memory System

## What Is Weft

Weft is a structured, persistent memory system for AI agents. It replaces flat-file memory (like `.claude/memory/MEMORY.md`) with a queryable knowledge base that supports semantic retrieval, confidence tracking, relationship mapping, and automatic decay.

Part of the trilogy: **Loom** (orchestration) → **Warp** (builder agent) → **Weft** (memory).

## Why Weft Exists

Current agent memory is a flat markdown file truncated at 200 lines. It has no structure, no retrieval beyond grep, no decay mechanism, and no way to distinguish high-confidence knowledge from speculation. When a session starts, the entire file loads into context whether it's relevant or not.

Weft solves this by treating memory as a first-class data system rather than a text file.

## How It Relates to Loom

Weft is a **separate project** that reuses Loom's infrastructure patterns but has a fundamentally different data model:

| Aspect | Loom (tasks) | Weft (memories) |
|--------|-------------|-----------------|
| Lifecycle | pending → claimed → done (terminal) | Created → revised → decayed (evolving) |
| Relationships | DAG with crisp dependencies | Fuzzy: "related to", "supersedes", "contradicts" |
| Retrieval | Status + priority filters | Semantic similarity + topic + recency |
| Write trigger | Explicit API call (agent decides) | Semi-automatic extraction from conversation |
| Core problem | Coordinate parallel work | Decide what 5% of knowledge matters right now |

**Reusable from Loom:** Postgres + Redis architecture, MCP tool interface, event-driven updates, two-tier caching, migration system, CLI patterns.

**New in Weft:** Data model, semantic retrieval engine, memory consolidation pipeline, context budget management, confidence/decay scoring.

## Loom Task Management (IMPORTANT)

This project uses Loom for orchestration. When Loom MCP tools are available, **always use Loom instead of EnterPlanMode** for planning and decomposing work. Do NOT fall back to the built-in plan mode.

### When the user asks to plan, decompose, or break down work:
- Call `loom_decompose(goal="...", confirm=True)` to generate a task graph
- To break down an existing epic: `loom_decompose(epic_id="loom-xxx")`
- Review the proposed graph with the user, then call with `confirm=False` to write it

### Task workflow:
- `loom_ready` → see available tasks
- `loom_claim` → claim a task before starting work
- `loom_heartbeat` → extend claim TTL during long tasks
- `loom_done` → mark complete with output
- `loom_fail` → mark failed with reason
- `loom_status` → project overview or task detail

### Recovery after context compaction:
If your context was compacted and you lost track of in-progress work:
1. Call `loom_recover()` → review the dry-run classification
2. Call `loom_recover(execute=True)` → complete marker tasks, reset stale/expired
3. Call `loom_status()` → verify project state is consistent

### Subagent self-reporting:
Subagents spawned via the Task tool do NOT have MCP access. They must self-report via CLI:
```
When COMPLETE: uv run python -m loom done TASK_ID --output '{"summary": "..."}' --branch-name BRANCH
If FAIL:       uv run python -m loom fail TASK_ID --reason '...'
Every 10 min:  uv run python -m loom heartbeat TASK_ID
```

### Key rules:
- **Never use EnterPlanMode when Loom is available.** Loom decompose IS the planning step.
- Always create/switch to a project first (`loom_create_project` + `loom_switch_project`) before decomposing.

## Weft Memory Integration

Weft is this project's persistent memory system. When the Weft MCP server is available, use it instead of flat-file memory.

### Session Startup
- Call `weft_prime` at the start of every session to load relevant context (preferences, recent work, project-relevant memories)
- Review the primer output before diving into work — it contains your accumulated knowledge about the user and project

### When to Remember
Call `weft_remember` when you learn something worth persisting across sessions:

| What to store | Memory type | Confidence | Example |
|--------------|-------------|------------|---------|
| User preferences | `preference` | 0.9-1.0 | "User prefers concise responses without emojis" |
| Project facts | `fact` | 0.7-0.9 | "Project uses PostgreSQL 16 with pgvector" |
| Code patterns | `pattern` | 0.6-0.8 | "Tests use testcontainers for DB isolation" |
| Architecture decisions | `architecture` | 0.8-0.9 | "Three-tier caching: Redis L1, in-memory L2, DB L3" |
| Problem solutions | `solution` | 0.7-0.8 | "Fix asyncpg connection leak by closing pool in finally block" |
| User background | `user_model` | 0.8-0.9 | "User is a senior Python developer focused on AI tooling" |

### Feedback Loop
- After recalling memories with `weft_recall` or `weft_context`, call `weft_feedback(memory_id, helpful=true/false)` to indicate whether each memory was actually useful
- This adjusts the usefulness score so helpful memories rank higher in future sessions

### Bulk Extraction
- Use `weft_extract` on conversation chunks or documentation to identify candidate memories you may have missed
- Review the candidates before storing — `weft_extract` returns proposals, it does not auto-store

### Topic Conventions
- Use lowercase, hyphenated topics: `postgres-config`, `testing-patterns`, `api-design`
- Be specific: prefer `redis-caching` over `database`
- Reuse existing topics when possible — check with `weft_status` to see current topic distribution

### Project Scoping
- Store user preferences and general knowledge with `project_id=null` (global — visible from all projects)
- Store project-specific facts with `project_id` set to the project identifier
- When in doubt, store globally — it's better to have a memory available everywhere than to lose it

### Consolidation
- Periodically run `weft_consolidate` to decay stale memories, merge duplicates, and flag contradictions
- Use `weft_consolidate(dry_run=true)` first to preview what would change

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
uv run python -m weft            # Run MCP server (when built)
```

## Owner

Jason Bauman. Builder agent: Warp.
