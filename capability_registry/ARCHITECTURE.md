# Capability Registry Architecture

## Purpose

The Capability Registry is a Weft-backed index of reusable implementation
patterns, scripts, classes, functions, and operational precedents across local
repositories. It helps agents answer "have we solved something like this
before?" without creating a new physical function repository or replacing the
source repositories where code already lives.

The registry is Python-first. JavaScript, TypeScript, and other language
scanners are future work after the Python ingestion and lookup path proves its
value.

## Memory Contract

Each capability entry is stored as a plain Weft memory. The registry uses the
existing memory substrate, not a parallel database table or new artifact store.

Every capability memory MUST have:

- `content`: a structured text block in the format defined in
  [Content Template](#content-template).
- `topics`: deterministic tags in the format defined in
  [Deterministic Topic Conventions](#deterministic-topic-conventions).
- `type`: a normal Weft memory type appropriate for implementation precedent,
  usually `solution` or `fact`, unless a caller has a stronger reason to use
  another existing type.

Capability memories MUST NOT appear in normal session boot by default. They are
discoverable through explicit topic lookup, capability lookup tooling, or
intentional pinning. Pinning is reserved for durable operating rules, not for
ordinary implementation inventory.

## Deterministic Topic Conventions

Every capability memory MUST carry all applicable tags from this set:

- `capability:<slug>`: the capability name. Slugs are lowercase, hyphenated,
  and contain no spaces.
- `repo:<slug>`: the source repository. Slugs are lowercase, hyphenated, and
  contain no spaces.
- `file:<relative-path>`: the source file path relative to the repository root,
  using forward slashes.
- `symbol:<name>`: optional symbol name for class, function, constant, command,
  or other addressable source artifact.

Example topic set:

```text
capability:bot-block-hardening
capability:crawler-escalation
repo:muttr
file:crawl/escalation.py
symbol:LazyEscalationPolicy
```

Topic slugs MUST be deterministic. The same source artifact scanned twice MUST
produce the same topic set unless the source path, symbol name, or assigned
capability classification has changed.

`file:<relative-path>` is repo-relative. It MUST NOT include an absolute local
path, username, machine-specific prefix, or repository slug unless that slug is
part of the actual repo-relative path.

## Content Template

Memory content MUST be a structured text block with one `FIELD: value` per
line. Fields SHOULD appear in this order:

```text
CAPABILITY: bot-block-hardening, crawler-escalation
REPO: muttr
FILE: crawl/escalation.py
SYMBOL: LazyEscalationPolicy
DOCSTRING: Lazy escalation policy for crawler bot-block handling.
IMPORTS: requests, time, random
REUSE_NOTES: Reuse when a crawler should delay expensive escalation until cheap retry signals fail.
FILE_HASH: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Required fields:

- `CAPABILITY`: comma-separated slugs matching the `capability:<slug>` topics,
  or `(unclassified)` when a scanner has not assigned a capability yet.
- `REPO`: slug matching the `repo:<slug>` topic.
- `FILE`: path matching the `file:<relative-path>` topic.
- `FILE_HASH`: SHA-256 hex digest of the source file content at index time.

Optional fields:

- `SYMBOL`: symbol name matching the `symbol:<name>` topic, if present.
- `DOCSTRING`: extracted docstring or concise description from the source.
- `IMPORTS`: comma-separated list of key imports from the file.
- `REUSE_NOTES`: free-text guidance on when and how to reuse the capability.

The content block is optimized for human inspection and lightweight parsing.
Downstream tools MUST treat topics as the primary identity/indexing surface and
the content block as the human-readable record plus stale-detection payload.

## Lookup Flow

The registry supports two lookup paths.

Direct topic lookup:

```python
weft_status(topic="capability:bot-block-hardening")
```

Agents use direct lookup when they already know the deterministic topic. Direct
results should return all matching memories, including repo, file, symbol,
capability topics, reuse notes, and file hash.

Tool-assisted lookup:

```python
weft_capability_lookup(query="bot blocked crawler", limit=10)
```

The MCP lookup tool accepts either natural language or explicit topic tags. The
tool infers likely capability slugs, calls the underlying topic-filtered Weft
lookup, and formats results around implementation reuse:

- repository slug
- repo-relative file path
- symbol name and kind, when present
- capability slugs
- docstring or description
- reuse notes
- stale status when available

The MCP tool MUST remain a thin coordinator. Query expansion, topic inference,
memory parsing, and result formatting belong in ordinary Python modules that can
be tested outside MCP.

## Storage Decision

Use plain Weft memories with deterministic topics for the MVP.

Do NOT add new schema columns such as `external_key`, `artifact_index`, or a
capability-specific table unless stable upsert and stale detection prove
impossible with existing Weft primitives. The registry is deliberately minimal:
topics provide identity and lookup, content provides readable payload, and
`FILE_HASH` provides stale detection.

This decision keeps the registry easy to inspect, easy to back up, and
compatible with existing Weft lifecycle behavior. If future usage shows that
plain memories cannot support safe upserts, add schema only after documenting the
specific failure mode.

## Primitive Audit (Phase 1)

This audit was performed against the current codebase after the initial
architecture contract was written. The Loom task referenced older paths
(`weft/topics.py`, `weft/recall.py`, `weft/mcp_tools.py`); the current code uses
`weft/topic_resolution.py`, `weft/topic_gather.py`, and `weft/mcp/tools.py`.

### write_memory() Contract

No function named `write_memory()` exists in the current store layer. The
canonical low-level write primitive is:

```python
async def store_memory(
    pool: asyncpg.Pool,
    create: MemoryCreate,
    embedding: list[float] | None = None,
    *,
    embed_composition_version: int = EMBED_COMPOSITION_VERSION,
) -> Memory
```

`MemoryCreate` includes `type`, `content`, `topic: list[str]`,
`source`, `confidence`, `project_id`, `agent_id`, `workspace_id`, `pinned`,
`review_after`, and `project_facets`. `store_memory()` inserts a new row and
returns a full `Memory` object with `id`, `topic`, `content`, timestamps, scope
fields, status, and lifecycle fields. It is not an idempotent write: repeated
calls with identical content and topics create distinct memories unless the
caller performs a dedup/upsert step first.

The MCP write surface is `weft_remember(content, type='fact', topic=None,
source='conversation', confidence=0.7, project_id=None, agent_id=None,
workspace_id=None, check_contradictions=True, pinned=False,
review_after=None, project_facets=None)`. It builds a `MemoryCreate`, embeds
`content + topics`, runs pre-insert dedup for unpinned memories, then calls
`store_memory()`.

There is also a store helper:

```python
async def upsert_by_topic(
    pool: asyncpg.Pool,
    *,
    topic: list[str],
    project_id: str | None,
    content: str,
    memory_type: MemoryType,
    source: MemorySource,
    confidence: float,
    review_after: datetime | None = None,
    embedding: list[float] | None = None,
) -> Memory
```

`upsert_by_topic()` matches active memories by exact whole topic array plus
`project_id`, using `topic @> $1 AND topic <@ $1`. That is useful for records
whose complete topic set is stable, but it is too strict for capability re-index
when capability slugs can be added or changed while `file:<path>` and
`symbol:<name>` remain the artifact identity.

### Topic Storage & Lookup

Topics are stored as plain strings in the `memories.topic TEXT[]` column.
`Memory.topic` and `MemoryCreate.topic` are both `list[str]`. The original
memories migration created a GIN index on `topic`, and migration 56 reinforces
the topic-digest path with `idx_memories_topic_gin ON memories USING GIN
(topic)`.

The exact topic membership predicate is already used throughout the store:
`$tag = ANY(topic)`. `list_memories()`, vector search, keyword/hybrid paths, and
topic gather all use this idiom. No topic count cap or reserved capability
prefix was found. Existing conventions include ordinary freeform tags plus
structured prefixes such as `entity:<name>` and task tags such as
`task:<loom-id>`.

The topic-first MCP read path is `weft_status(topic, synthesize=False,
budget_tokens=2000)`. It resolves the requested topic through
`resolve_topic()` and then calls `gather_topic_memories()`. Tier 1 returns the
complete active memory set for the resolved tags, with full memory content,
topic arrays, ids, types, and created timestamps. It is unbounded for primary
topic membership and sets `complete=True`; `truncated=True` only applies to the
secondary entity-linked expansion when that path hits its own cap.

`resolve_topic()` first checks the `topic_resolution_aliases` table for a
confirmed alias, then falls back to naive normalization. For explicit capability
lookups, callers should pass exact deterministic tags such as
`capability:bot-block-hardening`, `repo:muttr`, `file:crawl/escalation.py`, or
`symbol:LazyEscalationPolicy`; these are already lowercase or case-stable and
do not need alias learning to work.

### External Key / Artifact Index

No `external_key` field and no `artifact_index` table were found in the codebase
or migrations. The `memories` table primary key is the generated Weft memory
`id`; artifact identity must therefore be represented in topics and content for
the MVP.

For stale detection, the MVP will use:

- `file:<relative-path>` plus optional `symbol:<name>` topics as artifact
  identity.
- `FILE_HASH:` in memory content as the source snapshot.
- review workflow records when a file is missing, a hash changes, or a
  previously written memory lacks a parseable `FILE_HASH`.

### Stale Detection Capability

Plain memories are sufficient for first-pass stale detection. The scanner can
compute SHA-256 hashes and write them into `FILE_HASH:`. A re-indexer can gather
candidate memories by exact `file:<relative-path>` topic, optionally narrow by
`symbol:<name>`, parse `FILE_HASH:` from content, and compare it to the current
file hash.

The only caveat is upsert identity. `upsert_by_topic()` should not be used as
the sole capability upsert mechanism unless the complete topic set is treated as
immutable. Capability classification is expected to evolve, so ingestion should
perform a file/symbol lookup first, then decide whether to skip, write a new
memory, or surface a review item. That matches the non-destructive upsert
strategy above and avoids silently overwriting historical capability records.

### Existing MCP Tools

The current MCP tool module is `weft/mcp/tools.py`. Relevant tools for the
Capability Registry are:

- `weft_remember`: write a new memory with arbitrary topic strings.
- `weft_recall`: semantic/keyword/hybrid recall with optional single-topic
  filtering, limited by the caller's `limit`.
- `weft_status`: exact topic-oriented status; returns complete active memories
  for resolved tags and is the best current primitive for capability lookup.
- `weft_revise`: revise an existing memory by id, including optional content,
  type, topic, confidence, project, and review date changes.
- `weft_forget`: archive or delete memories by id.
- `weft_pin`: pin memories by id.
- `weft_learn` and `weft_extract`: higher-level memory extraction/write flows.
- `weft_prime`, `weft_focus`, `weft_context`, `weft_search_all`,
  `weft_project_status`, `weft_projects`, and recap/brief tools: read/context
  surfaces that should not be the registry write path.

Other tools in the same module cover relationships, consolidation, feedback,
boards, Slack, behaviors, episodes, entities, modes, alerts, check-ins,
autonomy, costs, calibration, degradation, triggers, workspaces, trackers,
lists, tokens, turns, and fsck. None of those provide capability-specific
artifact identity.

### Conclusion

Plain memories plus deterministic topics are sufficient for the Capability
Registry MVP. The MVP should use these topic conventions:

- `capability:<slug>`
- `repo:<slug>`
- `file:<relative-path>`
- `symbol:<name>`

No schema addition is required yet. Add a dedicated `external_key` column or
artifact index only if file/symbol topic lookup plus `FILE_HASH:` review records
prove insufficient for stable re-indexing.

## Isolation from Prime

Capability memories are inventory, not session context. Normal `weft_prime`
responses MUST exclude unpinned capability entries so boot context does not fill
with implementation catalog records.

The expected retrieval pattern is explicit:

- a user or agent asks for known precedent,
- a tool derives or receives a capability topic,
- Weft returns matching capability memories.

Capability memories MAY be pinned only when the content is a durable rule or
high-value project convention that should influence every session. Ordinary
source inventory, stale reports, and one-off reuse notes MUST remain unpinned.

## Upsert Strategy

The write path MUST be stable, reviewable, and non-destructive.

Before writing a new capability memory:

1. Build the deterministic topic set from repo slug, file path, symbol, and
   capability slugs.
2. Search for existing memories with the same `file:<relative-path>` topic and,
   when present, the same `symbol:<name>` topic.
3. Extract the prior `FILE_HASH` from matching memory content.
4. Compare the prior hash to the newly computed source file hash.

Outcomes:

- No existing memory: write a new capability memory.
- Existing memory with the same hash: do not write a duplicate.
- Existing memory with a different hash: mark the old memory for review and
  write a new memory with the updated content, topics, and hash.
- Source file missing at re-index time: surface a review record; do not delete
  or silently archive the old memory.
- Memory content has malformed or missing `FILE_HASH`: surface for review; do
  not guess.

File hash mismatch is a review workflow, not permission for silent overwrite.
Missing files are review workflow inputs, not automatic deletion signals.

## Dry-Run Output

All ingestion tools MUST support a `--dry-run` mode. Dry-run mode prints the
proposed entries without writing to Weft.

Dry-run output MUST include:

- repository slug
- repo-relative file path
- symbol name and kind, when present
- deterministic topics
- full content block or a clearly labeled preview
- computed file hash
- whether the entry would be created, skipped, updated, or surfaced for review

Dry-run output SHOULD be available as JSON for automation and as readable text
for manual review.

## Scope

In scope for the first implementation:

- Python repository scanning.
- Module, class, function, and script-level entries.
- Deterministic topic generation.
- Structured content generation.
- Dry-run ingestion.
- Non-destructive write/upsert behavior.
- Lookup by explicit topic and natural-language capability query.
- Stale detection for changed or missing source files.

Out of scope for the first implementation:

- A new physical function or snippet repository.
- New Weft schema columns or capability-specific tables.
- Automatic deletion of stale capability memories.
- Inclusion of ordinary capability memories in normal prime output.
- JavaScript, TypeScript, shell, notebook, or compiled-language scanners.
- Fully autonomous reuse or code copy operations.

Future work may add additional language scanners, richer symbol indexing,
schema-backed artifact identity, and cross-repo dependency maps after the
minimal Weft-native contract has been validated.
