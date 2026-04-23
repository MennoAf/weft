# Retrieval Mode × Scope Orthogonality

Weft's read-path tools expose three independent knobs. Each one answers a
different question, and none constrains the other two. Consumers compose them
freely.

| Parameter        | Question answered                          | Values                    |
|------------------|--------------------------------------------|---------------------------|
| `retrieval_mode` | *Which sources should surface?*            | `face`, `code`, `all`     |
| scope            | *Which visibility tier am I reading from?* | `user`, `project`, `agent`|
| `user_id`        | *Which user's rows are visible?*           | UUID or `None`            |

## Design principle

- **`retrieval_mode`** is a **source allowlist**. It decides which
  `MemorySource` values (`conversation`, `documentation`, `inference`, `seed`,
  `ingest`, `code`) are admissible in the result set. Canonical mapping lives in
  `weft/retrieval_modes.py`. `face` is the default for face-facing reads (daily
  brief, primer, human-in-the-loop recall) and deliberately excludes the
  `ingest` source so codebase embeddings don't drown personal memory.
- **Scope** is a **tier filter**. It selects *which pool* of rows is being
  queried — user-owned, project-owned, or agent-owned. Scope is expressed via
  the `user_id` / `project_id` / `agent_id` parameters on the underlying store
  functions.
- **`user_id`** is the **identity filter within the user tier**. When provided,
  the store applies OR-NULL filtering: `WHERE user_id = $1 OR user_id IS NULL`.
  That means user-owned rows AND truly-global rows (user_id IS NULL) are both
  returned — `NULL` is the canonical sentinel for "applies to every user in
  this deployment."

These three knobs compose independently. `retrieval_mode` never implies a
scope; scope never forces a source filter; `user_id` never constrains which
sources are admissible. The implementation applies them as independent WHERE
clauses joined by `AND` — no precedence, no collapsing, no fallthrough.

## Examples

```python
# Example 1 — face sources, user scope (default face-facing recall)
await weft_recall(
    query="what's the TLD I registered last month",
    retrieval_mode="face",        # excludes ingest
    user_id=get_user_id(),        # user tier, OR-NULL
)

# Example 2 — code sources, project scope (agent reading its own repo context)
await weft_recall(
    query="how does search_by_vector handle pgvector ANN",
    retrieval_mode="code",        # includes ingest + code
    project_id="weft",            # project tier
)

# Example 3 — all sources, user scope (unfiltered read over everything visible)
await weft_recall(
    query="anything about the migration freeze",
    retrieval_mode="all",         # no source filter
    user_id=get_user_id(),        # user tier, OR-NULL
)
```

All three calls target the same underlying `search_hybrid` path. The WHERE
clause composes as:

```sql
WHERE (source = ANY($1) OR $1 IS NULL)   -- retrieval_mode
  AND (user_id = $2 OR user_id IS NULL)  -- user_id
  AND (project_id = $3 OR project_id IS NULL)  -- project scope
```

## Implementation note

Every read-path MCP tool accepts `retrieval_mode` and scope parameters as
independent arguments — they are never collapsed into a single enum. The
filters apply in sequence (scope tier first, then `retrieval_mode` allowlist).
Unknown `retrieval_mode` values fall through to no-filter rather than
erroring; strict validation is the caller's responsibility. See
`weft/retrieval_modes.py::sources_for_mode` and the docstring on
`weft_recall` in `weft/mcp/tools.py` for the canonical contract.

## Non-goals

- Do not collapse `retrieval_mode` and scope into a unified "view" parameter.
  The two axes answer different questions; a unified parameter would force
  consumers into artificial combinations (e.g. "face-user view") and obscure
  that the knobs are independent.
- Do not extend `retrieval_mode` to filter by scope tier. New source buckets
  (e.g. `external`, `slack`) go into `MODE_SOURCES`; new tiers (e.g. team)
  become new scope columns.
