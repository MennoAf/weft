# Weft — Persistent Agent Memory System

Weft is a structured, persistent memory system for AI agents. Part of the trilogy: Loom → Warp → Weft. Project details (architecture, data model, design rationale) are stored in Weft itself — call `weft_prime(project_id="weft")` for full context.

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
