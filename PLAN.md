# Weft — Initial Plan

## The Problem

AI agents lose all learned context between sessions. Current solutions (flat files, MEMORY.md) are:
- **Unstructured** — prose blobs with no metadata
- **Unqueryable** — retrieval is "load everything" or "grep for keyword"
- **No decay** — stale knowledge clutters the space forever
- **No relationships** — can't express "X supersedes Y" or "A contradicts B"
- **No budget** — everything loads into context whether relevant or not

## What Weft Does

Weft gives agents a structured, persistent memory with three core capabilities:

1. **Store** — Write memories with metadata (topic, confidence, source, relationships)
2. **Retrieve** — Semantic search + topic filter + recency weighting to find relevant memories
3. **Maintain** — Automatic decay, consolidation, and conflict resolution over time

## Data Model (Draft)

### Memory Record
```
id:             weft-{8 hex}
topic:          string          # Primary topic tag (e.g., "auth", "loom-architecture", "user-preferences")
content:        text            # The actual knowledge
source:         string          # Where this came from: "conversation", "code", "documentation", "inference"
confidence:     float 0-1       # How certain we are (1.0 = user explicitly stated, 0.5 = inferred)
created_at:     timestamp
updated_at:     timestamp
accessed_at:    timestamp       # Last time this was retrieved (for decay)
access_count:   int             # How often this gets retrieved (for importance)
supersedes:     [memory_id]     # This memory replaces these older ones
related_to:     [memory_id]     # Fuzzy association
project_id:     string          # Scoped to a project (or global)
embedding:      vector          # For semantic retrieval
status:         active | archived | decayed
```

### Key Differences from Loom's Task Model
- No lifecycle state machine (pending→claimed→done). Memories are **active** until they **decay** or get **archived**
- Relationships are bidirectional and fuzzy, not a strict DAG
- `confidence` and `access_count` drive relevance, not `priority`
- `embedding` enables semantic similarity search — Loom has nothing like this
- `supersedes` handles knowledge evolution — when you learn something new that replaces old knowledge

## Tech Stack

### Reuse from Loom
- **Python 3.12+, uv, hatchling** — same toolchain
- **Postgres** — structured storage for memory records. Use pgvector extension for embedding search
- **Redis** — hot memory cache, recently accessed memories, session context
- **FastMCP** — MCP tools for agent access (store, retrieve, forget, relate)
- **asyncpg + redis.asyncio** — same async patterns
- **Migration system** — same append-only numbered migrations
- **Click CLI** — same CLI patterns
- **pytest + testcontainers** — same test infrastructure

### New in Weft
- **pgvector** — Postgres extension for vector similarity search (cosine distance)
- **Embedding API** — Anthropic or OpenAI embeddings for converting text → vectors. Need to evaluate:
  - Anthropic voyage-3 (if available)
  - OpenAI text-embedding-3-small (cheap, good enough?)
  - Local model via sentence-transformers (no API dependency, but adds torch)
- **Consolidation pipeline** — periodic job that merges related memories, archives stale ones, resolves contradictions
- **Context budget manager** — given a query/situation, select the optimal set of memories that fit in N tokens

## MCP Tools (Draft)

| Tool | Description |
|------|-------------|
| `weft_remember` | Store a new memory with topic, content, confidence, source |
| `weft_recall` | Retrieve relevant memories by semantic query, topic filter, or both |
| `weft_forget` | Archive or delete a memory |
| `weft_relate` | Create a relationship between memories (related_to, supersedes, contradicts) |
| `weft_context` | Load optimal memory context for a given situation (budget-aware) |
| `weft_status` | Memory stats: total, by topic, by confidence, recently accessed |
| `weft_consolidate` | Trigger memory consolidation: merge duplicates, archive stale, resolve conflicts |

## Architecture Sketch

```
Agent (Claude Code)
  │
  ├── MCP: weft_remember / weft_recall / weft_context
  │     │
  │     ▼
  │   Weft MCP Server (FastMCP)
  │     │
  │     ├── Postgres + pgvector (durable memory store)
  │     │     └── Semantic search via vector similarity
  │     │
  │     └── Redis (hot memory cache)
  │           └── Recently accessed memories, session context
  │
  └── MCP: loom_decompose / loom_claim / loom_done
        │
        ▼
      Loom MCP Server (task orchestration for building Weft itself)
```

## Open Questions (Resolve Before Decomposing)

1. **Embedding provider** — Anthropic, OpenAI, or local? Tradeoff: API cost vs dependency vs quality
2. **Auto-extraction** — Should Weft automatically extract memories from conversation, or only store what agents explicitly `weft_remember`? Start explicit, add auto later?
3. **Scope** — Per-project memories, global memories, or both? Loom is project-scoped. Weft probably needs both (user preferences are global, codebase knowledge is project-scoped)
4. **Context budget** — How does `weft_context` decide what to load? Simple (top-K by relevance) or sophisticated (diversity sampling, topic coverage)?
5. **MCP registration** — Should Weft run as its own MCP server alongside Loom, or be a Loom plugin/extension?
6. **Bootstrap** — Can we seed Weft with existing MEMORY.md content as an initial import?

## Build Strategy

Use Loom to orchestrate building Weft. The irony is intentional and practical:
1. `loom_create_project` for Weft
2. Feed this plan into `loom_decompose`
3. Multi-agent build using Loom's worktree pattern
4. Weft's own tests validate memory operations
5. When Weft is functional, migrate Warp's MEMORY.md into it as the first real user
