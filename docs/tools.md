# MCP Tool Reference

Full reference for every tool the Weft MCP server exposes. For a tour rather than a reference, start with the [README quickstart](../README.md#quickstart).

The signatures here are stable: changes are additive-only (the test suite enforces this via `tests/test_additive_guard.py`).

## Tool lifecycle and usage

Weft records daily aggregate invocation counts for every MCP tool without
retaining request arguments. Coverage telemetry separately records the
versioned recorder heartbeat, successful writes, failures, and shutdown-drain
state. The trailing summary is included in `weft_check_health` under
`tool_usage`.

A zero count means **not observed**, not valueless. Removal recommendations
require at least 30 valid coverage days with no gaps; 30 elapsed calendar days
are insufficient. The checked-in public-tool manifest also rejects removals
without an approved deprecation record. See `inventory/` and
[`tool-profiles.json`](../inventory/tool-profiles.json) for the public discovery
profiles. Maintainer-only validation evidence is intentionally kept outside the
public candidate.

`weft_up_next` is deprecated but remains available as a compatibility alias.
Use `weft_board` for the canonical unified open-items view. It will remain
functional while usage is measured.

## Memory primitives

### `weft_remember`

Store the supplied content verbatim; this tool does not summarize or rewrite it. When composing a memory worth retaining, preserve quantitative qualifiers that materially specify it (date, duration, amount, range, unit, or period/direction); do not copy incidental numbers or retain a fact solely because it contains a number.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Memory content |
| `type` | `str` | `"fact"` | One of: `preference`, `fact`, `pattern`, `relationship`, `solution`, `architecture`, `user_model`, `decision`, `milestone`, `issue`, `anti_pattern`, `handoff` |
| `topic` | `list[str]` | `[]` | Topic tags for filtering |
| `source` | `str` | `"conversation"` | One of: `conversation`, `code`, `documentation`, `inference` |
| `confidence` | `float` | `0.7` | Confidence score (0.0–1.0) |
| `project_id` | `str` | `null` | Scope to a project (`null` = global) |
| `agent_id` | `str` | `null` | Originating agent identifier |
| `check_contradictions` | `bool` | `true` | Check for contradicting memories on store |
| `pinned` | `bool` | `false` | Pin this memory (always included in prime/context) |

### `weft_recall`

Retrieve memories by semantic query. Results include `similarity`, `confidence`, and `relevance_score` for trust assessment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Natural language search query |
| `topic` | `str` | `null` | Filter by topic |
| `type` | `str` | `null` | Filter by memory type |
| `status` | `str` | `"active"` | Filter by status: `active`, `archived`, `decayed` |
| `project_id` | `str` | `null` | Filter by project |
| `limit` | `int` | `10` | Max results |
| `threshold` | `float` | `0.3` | Minimum similarity score |
| `mode` | `str` | `"hybrid"` | `semantic` (vectors), `keyword` (BM25), or `hybrid` (RRF fusion) |
| `tier` | `str` | `"auto"` | `belief`, `turns`, or `auto` (router picks based on query shape) |
| `retrieval_mode` | `str` | `"face"` | `face` (excludes codebase ingest), `code` (includes), or `all` |

### `weft_search_all`

Cross-project brain-wide search. At least one filter (query, topic, or memory_type) is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | `null` | Natural language search query |
| `topic` | `str` | `null` | Filter by topic |
| `memory_type` | `str` | `null` | Filter by memory type |
| `days` | `int` | `null` | Restrict to memories created within N days |
| `limit` | `int` | `20` | Max results |
| `retrieval_mode` | `str` | `"face"` | Same semantics as `weft_recall` |

### `weft_context`

Budget-aware context loading. Returns the best memories for a situation within a token budget.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Context query |
| `budget_tokens` | `int` | `4000` | Maximum tokens to return |
| `topic` | `str` | `null` | Filter by topic |
| `type` | `str` | `null` | Filter by memory type |
| `project_id` | `str` | `null` | Filter by project |
| `max_per_topic` | `int` | `3` | Maximum memories per topic |

### `weft_focus`

Mid-session intent shift. Loads tier-2 sections (behaviors, decisions, recent_work) that `weft_prime` defers, ranked by relevance to the new intent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intent` | `str` | *required* | What you're now doing — drives section ranking |
| `project_id` | `str` | `null` | Scope to project. Pass explicitly when client roots are unavailable. |
| `budget_tokens` | `int` | `2400` | Token budget for the assembled context |

## Lifecycle

### `weft_prime`

Session primer: assemble structured context for session startup. Returns prioritized sections within a token budget. Handoff continuity is project-specific: if no project can be resolved, the handoff section is skipped and the response includes a project-resolution warning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_id` | `str` | `null` | Scope to project. Pass explicitly when client roots are unavailable. |
| `agent_id` | `str` | `null` | Scope to agent |
| `budget_tokens` | `int` | `2400` | Token budget for the assembled context |
| `query` | `str` | `null` | Optional intent string to bias which items are surfaced |
| `disclosure` | `str` | `"progressive"` | `progressive` (tier-1 full + tier-2 counts) or `full` (all sections) |
| `mode` | `str` | `null` | Persona name (e.g. `coding`, `research`) — adjusts section weights |

Returns `{ grounding, rules, behaviors, handoff, recent_work, issues, decisions, entities, total_tokens, budget_tokens, budget_remaining, excluded, freshness_hours, section_tokens, hints }`.

### `weft_handoff`

Session handoff: capture context for the next session before clearing. The next `weft_prime` call surfaces the most recent handoff prominently for continuity. Handoffs require a resolved project id; Weft rejects unresolved handoff writes instead of creating global handoffs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `summary` | `str` | *required* | What was accomplished this session |
| `in_progress` | `str` | `null` | What's partially done or needs follow-up |
| `next_steps` | `str` | `null` | Recommended next actions and why |
| `open_questions` | `str` | `null` | Unresolved decisions or things to investigate |
| `project_id` | `str` | `null` | Scope to a project. Required if client roots cannot resolve it. |
| `agent_id` | `str` | `null` | Originating agent identifier |

### `weft_revise`

Update a memory's content, creating a new version that supersedes the old one. Preserves the predecessor's `pinned` state and `project_id` by default — pass `new_pinned`/`new_project_id` only when explicitly changing them.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | ID of memory to revise |
| `new_content` | `str` | *required* | Updated content |
| `new_confidence` | `float` | `null` | Updated confidence |
| `new_topic` | `list[str]` | `null` | Updated topics |
| `new_type` | `str` | `null` | Updated type |
| `new_project_id` | `str` | `null` | Reassign to a different project |
| `new_pinned` | `bool` | `null` | Explicit pin override (omit to inherit) |
| `review_after` | `str` | `null` | ISO timestamp or relative (`30d`, `2w`, `3m`) for next review |

### `weft_pin`

Pin or unpin a memory. Pinned memories are always included in `weft_prime` and `weft_context` results, and are protected from decay and deduplication.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | Memory to pin/unpin |
| `pinned` | `bool` | `true` | Pin (`true`) or unpin (`false`) |

### `weft_forget`

Archive or permanently delete a memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | ID of memory to forget |
| `hard` | `bool` | `false` | Hard-delete instead of archive |

## Learning + feedback

### `weft_learn`

Capture lessons learned from completed work. Extracts memory candidates from free-text notes (gotchas, fixes, patterns) and auto-stores them.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Free-text notes about what was learned |
| `task_id` | `str` | `null` | Associated task ID (tagged as `task:<id>` topic) |
| `project_id` | `str` | `null` | Scope to a project |
| `agent_id` | `str` | `null` | Originating agent |
| `min_confidence` | `float` | `0.7` | Minimum confidence to auto-store |

### `weft_feedback`

Record whether a memory was helpful. Adjusts the usefulness score for future ranking via exponential moving average.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | *required* | Memory that was used |
| `helpful` | `bool` | *required* | Was the memory helpful? |

### `weft_feedback_general`

Submit general product feedback about Weft itself — friction points, feature requests, or praise. Stored as a memory tagged `weft-feedback` for review.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feedback` | `str` | *required* | The feedback content |
| `category` | `str` | `"suggestion"` | One of: `suggestion`, `friction`, `praise`, `bug` |
| `agent_id` | `str` | `null` | Originating agent identifier |

## Relationships + maintenance

### `weft_relate`

Manage relationships between memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `str` | *required* | `add`, `get`, or `remove` |
| `memory_id` | `str` | *required* | Source memory ID |
| `target_id` | `str` | `null` | Target memory ID (for add/remove) |
| `relation` | `str` | `null` | Relation type: `supersedes`, `related_to`, `contradicts`, `derived_from` |

### `weft_consolidate`

Run the consolidation pipeline: propose stale-memory review candidates, merge duplicates, flag contradictions, and process other maintenance passes. Decay scoring is **review-only**: neither scheduled consolidation nor this MCP tool changes a candidate's status. Confidence is write-time/revision metadata and one input to the proposal score; pinned memories and `preference`, `user_model`, and `decision` types are protected.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dry_run` | `bool` | `false` | Preview all consolidation changes; decay candidates are non-mutating in both modes |

### `weft_extract`

Extract memory candidates from a block of text using heuristic pattern matching. Returns proposals for review — does NOT auto-store. Use `weft_learn` for auto-storing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Text to extract candidates from |
| `min_confidence` | `float` | `0.5` | Minimum confidence threshold for candidates |

### `weft_status`

Return memory statistics: total count, breakdown by type/topic/status, recently accessed.

*No parameters.*

## Behaviors

### `weft_behavior_add`

Store a persistent behavioral rule that the agent applies whenever the trigger pattern matches the current context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trigger_pattern` | `str` | *required* | Describes WHEN this behavior should activate (embedded for semantic matching) |
| `action` | `str` | *required* | Describes WHAT the agent should do |
| `scope` | `str` | `"global"` | `global`, `project`, or `agent` |
| `project_id` | `str` | `null` | Required if `scope="project"` |
| `agent_id` | `str` | `null` | Required if `scope="agent"` |
| `priority` | `int` | `0` | Higher overrides lower |
| `confidence` | `float` | `0.7` | Confidence in the rule |

### `weft_behavior_list` / `weft_behavior_match` / `weft_behavior_delete`

List, semantically match, and archive behavioral rules. See the tool docstrings for parameter details.

## Beyond the core

Other tools cover entities, episodes, trackers, triggers, alerts, costs, calibration, autonomy, workspaces, and tokens. They follow the same shape as the primitives above and are documented in the MCP server's introspected tool descriptions. Run `weft.mcp.tools` introspection or check the source for the full catalog.
