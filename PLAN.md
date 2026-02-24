# Weft — Initial Plan

## The Problem

AI agents lose all learned context between sessions. Current solutions (flat files, MEMORY.md) are:
- **Unstructured** — prose blobs with no metadata
- **Unqueryable** — retrieval is "load everything" or "grep for keyword"
- **No decay** — stale knowledge clutters the space forever
- **No relationships** — can't express "X supersedes Y" or "A contradicts B"
- **No budget** — everything loads into context whether relevant or not
- **Siloed** — each project gets its own isolated memory. The same agent working across projects can't share knowledge (we experienced this firsthand: Warp's Loom knowledge doesn't transfer to Weft)

## What Weft Does

Weft gives agents a structured, persistent memory with three core capabilities:

1. **Store** — Write memories with metadata (topic, confidence, source, relationships)
2. **Retrieve** — Semantic search + topic filter + recency weighting to find relevant memories
3. **Maintain** — Automatic decay, consolidation, and conflict resolution over time

## Memory Types

Not all memories are equal. The system needs to handle different categories with different behavior:

| Type | Example | Confidence | Decay | Scope |
|------|---------|-----------|-------|-------|
| **Preference** | "always use sonnet for subagents" | High (user stated) | Never | Global |
| **Fact** | "Loom has 760 tests" | Medium | Fast (stales quickly) | Project |
| **Pattern** | "3 agents is the sweet spot for parallel work" | Medium (learned) | Slow | Global |
| **Relationship** | "Jason owns Loom and Muttr" | High | Never | Global |
| **Solution** | "testcontainers reaper causes flaky test starts — clean in conftest" | High | Slow | Project |
| **Architecture** | "store.py is the ONLY Postgres writer for tasks" | High | Slow | Project |

The `type` field drives default decay rates and scope. Agents can override.

## Data Model

### Memory Record
```
id:             weft-{8 hex}
type:           preference | fact | pattern | relationship | solution | architecture
topic:          string[]        # Multiple topic tags (e.g., ["loom", "testing", "parallel-agents"])
content:        text            # The actual knowledge
source:         string          # Where this came from: "conversation", "code", "documentation", "inference"
confidence:     float 0-1       # 1.0 = user explicitly stated, 0.7 = observed pattern, 0.5 = inferred
token_count:    int             # Estimated tokens for context budget management
created_at:     timestamp
updated_at:     timestamp
accessed_at:    timestamp       # Last time this was retrieved (for decay scoring)
access_count:   int             # Frequency of retrieval (for importance scoring)
supersedes:     [memory_id]     # This memory replaces these older ones
related_to:     [memory_id]     # Fuzzy association
contradicts:    [memory_id]     # Conflict marker — needs resolution
project_id:     string | null   # null = global memory, string = project-scoped
agent_id:       string | null   # Which agent created this (for multi-agent setups)
embedding:      vector(1536)    # For semantic retrieval (text-embedding-3-small dimension)
status:         active | archived | decayed
```

### Memory Relationships Table
```
source_id:      memory_id
target_id:      memory_id
relation:       supersedes | related_to | contradicts | derived_from
created_at:     timestamp
```

Separate table (like Loom's `task_deps`) because relationships are many-to-many and bidirectional.

### Key Differences from Loom's Task Model
- No lifecycle state machine (pending→claimed→done). Memories are **active** until they **decay** or get **archived**
- Relationships are bidirectional and typed, not a strict DAG
- `confidence`, `access_count`, and `type` drive relevance — not `priority` and `status`
- `embedding` enables semantic similarity search — Loom has nothing like this
- `supersedes` handles knowledge evolution — writing a new memory can automatically archive the old version
- `token_count` enables budget-aware context loading — pack the most relevant knowledge into N tokens
- `project_id = null` means global scope — preferences and patterns that apply everywhere

## Decisions (Not Open Questions)

### 1. Embedding provider: OpenAI text-embedding-3-small
- $0.02/1M tokens — effectively free for our scale
- 1536 dimensions, good quality for retrieval
- No torch dependency (unlike sentence-transformers)
- Can swap to Anthropic if they release an embedding API
- Abstracted behind an interface so the provider is pluggable

### 2. Start explicit, add auto-extraction later
- Phase 1: Only `weft_remember` stores memories. Agent decides what's worth remembering.
- Future: Add a consolidation step that can extract memories from conversation transcripts. This is harder and riskier — bad extraction = noise.

### 3. Both global and project-scoped memories
- `project_id = null` → global (preferences, patterns, relationships)
- `project_id = "uuid"` → project-scoped (architecture, facts, solutions)
- `weft_recall` searches both by default, with option to filter

### 4. Context budget: top-K with diversity
- `weft_context(query, budget_tokens=4000)` returns the best-fit set of memories
- Scoring: `relevance * confidence * recency_weight * frequency_bonus`
- Diversity: after picking top candidates, deduplicate by topic so you don't get 10 memories about the same thing
- Return memories in priority order with total token count

### 5. Separate MCP server
- Weft runs as its own MCP server alongside Loom
- Projects register both in `.mcp.json`: `"loom": {...}, "weft": {...}`
- Clean separation of concerns: Loom = tasks, Weft = memory
- Shared Postgres instance is fine (different tables, same migrations pattern)
- Shared Redis instance is fine (different key prefixes)

### 6. Bootstrap from MEMORY.md
- Phase 1 deliverable: `weft import` CLI command that parses existing MEMORY.md files
- Splits sections into individual memories with inferred types/topics
- Sets confidence based on heuristics (user preferences = 1.0, build status = 0.7, etc.)
- First real validation: import Warp's Loom MEMORY.md and verify recall works

## MCP Tools

| Tool | Description |
|------|-------------|
| `weft_remember` | Store a new memory with type, topics, content, confidence, source |
| `weft_recall` | Retrieve memories by semantic query, topic filter, type filter, or combination |
| `weft_forget` | Archive a memory (soft-delete) or hard-delete |
| `weft_relate` | Create/remove relationships between memories (supersedes, related, contradicts) |
| `weft_context` | Budget-aware context loading: best memories for a situation within N tokens |
| `weft_revise` | Update a memory's content — creates a new version, supersedes the old one |
| `weft_status` | Stats: total memories, by topic, by type, by confidence, recently accessed |
| `weft_consolidate` | Trigger consolidation: merge near-duplicates, archive decayed, flag contradictions |

### Tool Design Principles (Learned from Loom)
- Each tool ≤15 lines — thin coordinators, business logic in separate modules
- Tools return dicts, Pydantic models handle serialization
- Write path: Postgres first → Redis cache sync → event publish
- Read path: Redis fast → Postgres fallback on cache miss

## Architecture

```
Agent (Claude Code)
  │
  ├── MCP: weft_remember / weft_recall / weft_context
  │     │
  │     ▼
  │   Weft MCP Server (FastMCP, stdio)
  │     │
  │     ├── store.py → Postgres + pgvector
  │     │     ├── Memory CRUD
  │     │     ├── Relationship management
  │     │     └── Vector similarity search (cosine distance)
  │     │
  │     ├── cache.py → Redis
  │     │     ├── Hot memory cache (recently accessed)
  │     │     ├── Session context (current working set)
  │     │     └── Embedding cache (avoid re-computing)
  │     │
  │     ├── embeddings.py → OpenAI API
  │     │     └── text → vector, with local cache
  │     │
  │     ├── relevance.py → Scoring engine
  │     │     └── relevance * confidence * recency * frequency → budget packing
  │     │
  │     └── consolidation.py → Maintenance
  │           ├── Near-duplicate detection (embedding similarity > 0.95)
  │           ├── Decay scoring (not accessed in N days → archived)
  │           └── Contradiction flagging
  │
  └── MCP: loom_* tools (for orchestrating Weft's own development)
```

### Module Contracts (Same Discipline as Loom)
| Module | Rule |
|--------|------|
| `store.py` | ONLY writer to Postgres for memory data |
| `cache.py` | ONLY reader from Redis for memory data; falls back to store.py |
| `embeddings.py` | ONLY module that calls the embedding API |
| `mcp/tools.py` | Each tool ≤15 lines; thin coordinators only |
| `db/migrations/` | Never modify existing files; always add new numbered ones |

## CLI Commands

| Command | Description |
|---------|-------------|
| `weft up` | Start Postgres (with pgvector) + Redis, run migrations |
| `weft down` | Stop containers |
| `weft status` | Memory stats overview |
| `weft recall QUERY` | Search memories from the command line |
| `weft import FILE` | Import memories from a MEMORY.md file |
| `weft export` | Export all memories as markdown (human-readable backup) |
| `weft consolidate` | Run consolidation pass |
| `weft config show/set` | Configuration management |

## Build Phases

### Phase 1: Foundation (MVP)
- Data model + migrations (Postgres + pgvector)
- `store.py` — CRUD for memories and relationships
- `cache.py` — Redis caching layer
- `embeddings.py` — OpenAI embedding client with cache
- MCP tools: `weft_remember`, `weft_recall`, `weft_forget`, `weft_status`
- CLI: `weft up`, `weft down`, `weft status`, `weft recall`
- Tests: testcontainers, same pattern as Loom
- **Validation**: Store 10 memories, recall by query, verify semantic search works

### Phase 2: Retrieval & Context
- `relevance.py` — scoring engine with confidence/recency/frequency weights
- `weft_context` — budget-aware context loading
- `weft_revise` — version-aware memory updates
- Topic-based filtering and type-based filtering
- **Validation**: Import Warp's Loom MEMORY.md, verify `weft_context("how does Loom handle task claiming?")` returns the right memories

### Phase 3: Relationships & Maintenance
- `weft_relate` — relationship CRUD
- `weft_consolidate` — near-duplicate detection, decay, contradiction flagging
- `consolidation.py` — the maintenance pipeline
- Automatic supersedes handling on `weft_revise`
- **Validation**: Create contradicting memories, run consolidation, verify flagging works

### Phase 4: Integration & Polish
- `weft import` — MEMORY.md parser and importer
- `weft export` — human-readable markdown export
- Cross-project memory sharing (global scope)
- Docker compose template with pgvector
- `.mcp.json` registration pattern
- README and documentation
- **Validation**: Full workflow — import existing memories, use across two projects, export backup

## Build Strategy

Use Loom to orchestrate building Weft:
1. `loom_create_project` → "weft"
2. Feed this plan into `loom_decompose`
3. Multi-agent build using Loom's worktree pattern (3 agents per wave)
4. Weft's own tests validate memory operations
5. When Phase 2 is done, import Warp's MEMORY.md as the first real user
6. When Phase 4 is done, register Weft in Loom's `.mcp.json` and retire the flat MEMORY.md files
