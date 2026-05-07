# Primer Refactor Plan

> Design document for decomposing `weft/primer.py:build_primer()` (881 lines)
> into `weft/primer_sections/` — a package of independent section builders
> coordinated by an orchestrator.
>
> Line references are as-of commit **2955aed**.  The canonical boundary markers
> are the `SECTION:` annotation comment blocks in `primer.py` itself.

---

## Architecture Overview

### Current State

`build_primer()` is a single 715-line async function that:

1. **Pre-flight** (L203-211): timestamps, resolves mode weights, builds scope dict
2. **Parallel fetch** (L223-311): one `asyncio.gather` fetching all raw data
3. **Sequential budget packing** (L326-682): iterates sections in priority order, fitting items within per-section and global token budgets
4. **Post-sections** (L684-736): changes_since and wellness_snapshot (independent of budget)
5. **Post-processing** (L738-855): onboarding/hints, RLS diagnostic, progressive disclosure, result assembly

### Target State

```
weft/primer_sections/
├── __init__.py          # Re-exports only
├── context.py           # PrimerContext, SectionResult, SECTION_BUDGETS
├── grounding.py         # build_grounding_section
├── rules.py             # build_rules_section
├── behaviors.py         # build_behaviors_section
├── handoff.py           # build_handoff_section
├── recent_work.py       # build_recent_work_section
├── issues.py            # build_issues_section
├── anti_patterns.py     # build_anti_patterns_section
├── decisions.py         # build_decisions_section
├── entities.py          # build_entities_section
├── changes_since.py     # build_changes_since_section
├── wellness.py          # build_wellness_section
├── onboarding.py        # build_onboarding_section
└── disclosure.py        # apply_progressive_disclosure
```

The **orchestrator** (NOT created in this task) will live in `primer.py` itself,
replacing the monolithic function with a thin loop that calls section builders.

---

## Key Design Questions (Answered)

### Q1: Single DB fetch or per-section queries?

**Single parallel fetch.**  Lines 223-311 use one `asyncio.gather()` that runs
9 coroutines in parallel.  Each coroutine is a `list_memories()` or
`search_by_vector()` call for a specific section.  The results are then unpacked
and processed sequentially during budget packing.

**Implication for refactor:** Each section builder should own its own fetch
coroutine.  The orchestrator can still gather them in parallel — it collects
coroutines from section builders and runs `asyncio.gather()` — but sections
don't share raw fetch results.

### Q2: Token counting — per-section or global?

**Both.** Each section has a per-section cap (e.g., `_CAP_RULES = 100`) AND
checks the global `budget_tokens` ceiling.  The tracking variables are:
- `used_tokens` (int) — running total across all sections
- `section_tokens` (dict) — per-section usage, recorded after each section
- `excluded` (int) — count of items that didn't fit

These are mutable shared state on `PrimerContext`.

### Q3: Query-biased search — once or per-section?

**Per-section, at fetch time.**  The `biased` flag (L204) determines whether
each section uses `search_by_vector()` (with `query_vec`) or plain
`list_memories()`.  There is no single "recalled memories" cache — each biased
section runs its own vector search.

**Implication:** `query_vec` and `biased` belong on `PrimerContext`.  Sections
that support bias check `ctx.biased` and choose their fetch strategy accordingly.

### Q4: Onboarding detection?

**Computed after all sections are packed** (L760-766):
```python
total_items = sum(len(s) for s in [rules, behaviors, handoff, recent_work, issues, decisions, entities])
is_cold_start = not handoff_section and total_items <= _COLD_START_THRESHOLD
```

**Implication:** Onboarding is a post-processing step, not a peer section.
It receives the packed results from all sections and adds hints/onboarding text.

---

## Section Map

### Section 0: Grounding

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 227-234 (fetch), 332-345 (pack) |
| **Proposed function** | `build_grounding_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `project_id` |
| **Additional inputs** | None |
| **Output shape** | Single string (not list), set as `grounding_line` |
| **Token budget** | 50 |
| **Dependencies** | `memories` table (topic=`project-grounding`) |
| **Notes** | Skipped when `project_id` is None. Returns one memory max. |

### Section 1: Rules

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 236-238 (fetch), 347-373 (pack) |
| **Proposed function** | `build_rules_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `now`, `used_tokens`, `seen_ids` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{id, type, content, confidence, pinned, created_at, review_after, review_due}` |
| **Token budget** | 100 |
| **Dependencies** | `memories` table (pinned=True, status=active) |
| **Tier** | 1 (always included) |
| **Notes** | Sorted by (confidence, usefulness_score, created_at) desc. `_annotate_review_after` applied. Never query-biased. |

### Section 2: Behaviors

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 241-249 (fetch), 375-406 (pack) |
| **Proposed function** | `build_behaviors_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `biased`, `query_vec`, `behavior_boost` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{trigger, action, confidence, priority, scope, id}` |
| **Token budget** | 150 (scaled by `behavior_boost`) |
| **Max items** | 5 |
| **Dependencies** | `behaviors` table via `list_behaviors()` / `match_behaviors()` |
| **Tier** | 2 (deferred in progressive) |
| **Notes** | Uses `match_behaviors` (vector) when biased, `list_behaviors` (priority) when not. Imports `BehaviorMatch` model. |

### Section 3: Handoff

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 251-254 (fetch), 408-453 (pack) |
| **Proposed function** | `build_handoff_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `now`, `seen_ids`, `used_tokens` |
| **Additional inputs** | None |
| **Output shape** | List of dicts (0 or 1): `{id, type, content, confidence, created_at, age_hours}` |
| **Token budget** | 800 |
| **Dependencies** | `memories` table (type=handoff), fallback: topic=`session-handoff` |
| **Tier** | 1 (always included) |
| **Notes** | Only the most recent handoff. Truncated (not dropped) when oversized via `truncate_to_token_budget`. Deprecated topic fallback logs a warning. |

### Section 4: Recent Work

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 258-261 (fetch), 455-511 (pack) |
| **Proposed function** | `build_recent_work_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `biased`, `query_vec`, `now`, `recency_bias`, `seen_ids`, `project_id` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{summary, age_hours, refs, id}` |
| **Token budget** | 150 |
| **Max items** | 3 |
| **Dependencies** | `memories` table (type=milestone, 72h cutoff) |
| **Tier** | 2 (deferred in progressive) |
| **Notes** | Filters out unscoped ingested memories. When biased, uses blended similarity+recency ranking with `effective_sim_weight = _SIMILARITY_WEIGHT * (1 - recency_bias)`. |

### Section 5: Issues

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 263-266 (fetch), 513-557 (pack) |
| **Proposed function** | `build_issues_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `biased`, `query_vec`, `seen_ids`, `project_id` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{id, type, content, confidence, created_at}` |
| **Token budget** | 200 |
| **Dependencies** | `memories` table (type=issue) |
| **Tier** | 1 (always included) |
| **Notes** | Filters unscoped ingested memories. When biased, blends similarity + usefulness. |

### Section 5b: Anti-patterns

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 273-277 (fetch), 559-602 (pack) |
| **Proposed function** | `build_anti_patterns_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `biased`, `query_vec`, `seen_ids`, `project_id` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{id, type, content, confidence, created_at}` |
| **Token budget** | 150 |
| **Max items** | 3 |
| **Dependencies** | `memories` table (type=anti_pattern) |
| **Tier** | 1 (always included) |
| **Notes** | Same ranking/filtering as issues. Early break at max items (remaining counted as excluded). |

### Section 6: Decisions

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 268-272 (fetch), 604-657 (pack) |
| **Proposed function** | `build_decisions_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `biased`, `query_vec`, `seen_ids`, `project_id`, `now` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{id, type, content, confidence, project_id, created_at, review_after, review_due}` |
| **Token budget** | 250 |
| **Max items** | 5 |
| **Dependencies** | `memories` table (type=decision) |
| **Tier** | 2 (deferred in progressive) |
| **Notes** | Project-scoped decisions sort before global. `_annotate_review_after` applied. |

### Section 7: Entities

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 296 (fetch), 659-682 (pack) |
| **Proposed function** | `build_entities_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `scope`, `entity_boost` |
| **Additional inputs** | None |
| **Output shape** | List of dicts: `{name, type, description, mention_count, id}` |
| **Token budget** | 150 (scaled by `entity_boost`) |
| **Max items** | 10 |
| **Dependencies** | `entities` table via `list_entities()` |
| **Tier** | 2 (deferred in progressive) |
| **Notes** | Not query-biased. Uses `estimate_tokens` on name+description. |

### Post-section: Changes Since

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 684-699 |
| **Proposed function** | `build_changes_since_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool`, `project_id` |
| **Additional inputs** | None |
| **Output shape** | Dict: `{memories_created, memories_archived, memories_revised, since, recent_commits}` |
| **Token budget** | Not token-budgeted |
| **Dependencies** | `get_last_handoff_timestamp`, `get_memory_changes_since`, `get_recent_commits` |
| **Notes** | Independent of budget packing. Commits capped at 20. Failure-tolerant. |

### Post-section: Wellness Snapshot

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 701-736 |
| **Proposed function** | `build_wellness_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs from PrimerContext** | `pool` |
| **Additional inputs** | None |
| **Output shape** | Dict: `{trends, logging_streak, good_mood_streak, low_mood_streak, current_averages}` |
| **Token budget** | Not token-budgeted |
| **Dependencies** | `check_in_patterns.analyze_all`, `check_ins.list_check_ins` |
| **Known issue** | Uses bare `except Exception: pass` — should use explicit logging in refactored version. |
| **Notes** | Independent of budget packing. This is the canonical "tack onto the end" anti-pattern. |

### Post-processing: Onboarding & Hints

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 744-792 |
| **Proposed function** | `build_onboarding_section(ctx: PrimerContext) -> SectionResult` |
| **Inputs** | All packed section results (counts/emptiness) |
| **Output shape** | Dict: `{hints: dict, onboarding: str | None}` |
| **Dependencies** | All section results (reads counts), `pool` (RLS diagnostic) |
| **Notes** | Post-processing — depends on all other sections being complete. Cold-start threshold: no handoff AND total_items <= 2. RLS diagnostic checks `pg_class` for row count mismatch. |

### Post-processing: Progressive Disclosure

| Field | Value |
|-------|-------|
| **Lines in primer.py** | 794-855 |
| **Proposed function** | `apply_progressive_disclosure(ctx: PrimerContext) -> SectionResult` |
| **Inputs** | Full result dict, section results |
| **Output shape** | Transforms result dict in-place: tier 2 sections → `{count, deferred, hint}` |
| **Tier 1 sections** | grounding, rules, handoff, issues, anti_patterns |
| **Tier 2 sections** | behaviors, recent_work, decisions, entities |
| **Notes** | NOT a peer section — wraps/transforms the result dict. Recalculates used_tokens to reflect only tier 1. Conditionally applied when `ctx.disclosure == "progressive"`. |

---

## Shared State Analysis

Variables computed once in `build_primer` and consumed by multiple sections.
These become `PrimerContext` fields or are computed in a pre-flight step.

| Variable | Computed at | Used by | PrimerContext field? |
|----------|-----------|---------|---------------------|
| `now` | L203 | All sections (age_hours, review_after, cutoff) | Yes |
| `biased` | L204 | behaviors, recent_work, issues, anti_patterns, decisions | Yes (property) |
| `weights` | L208 | behaviors (behavior_boost), entities (entity_boost), recent_work (recency_bias) | Yes (3 floats) |
| `_scope` | L217-221 | All fetch calls | Yes (`scope` dict) |
| `used_tokens` | L327 | All budget-packed sections (read+write) | Yes (mutable) |
| `seen_ids` | L328 | All sections (prevent duplicates) | Yes (mutable set) |
| `excluded` | L329 | All sections (increment on skip) | Yes (mutable int) |
| `section_tokens` | L330 | Orchestrator (summary), disclosure | Yes (mutable dict) |
| `effective_sim_weight` | L467 | recent_work | Computed locally (from `recency_bias`) |
| `_DICT_OVERHEAD_TOKENS` | L37 | rules, issues, anti_patterns, decisions, handoff | Constant in context.py |

### Config / Environment Keys

`build_primer` does NOT read from a config dict or environment variables.
All configuration is via function parameters (`budget_tokens`, `disclosure`,
`mode`, `query_vec`) and module-level constants.

---

## Execution Order & Constraints

### Phase 1: Pre-flight (orchestrator)
1. Resolve `now`, `biased`, mode weights
2. Construct `PrimerContext`

### Phase 2: Parallel fetch (orchestrator gathers section coroutines)
All sections can fetch in parallel — no ordering dependencies between fetches.

### Phase 3: Sequential budget packing
Order matters because `used_tokens` is a running total:

```
grounding → rules → behaviors → handoff → recent_work → issues → anti_patterns → decisions → entities
```

Each section reads `ctx.used_tokens` and `ctx.seen_ids` before packing,
then updates them after.

### Phase 4: Independent post-sections (can run in parallel)
- `changes_since` — queries DB independently
- `wellness` — queries check_ins independently

### Phase 5: Post-processing (sequential, depends on Phase 3)
1. `onboarding` — reads all section results to compute hints and cold-start
2. `disclosure` — conditionally transforms tier 2 sections into deferred summaries

### Ordering constraints
- **Handoff before onboarding**: cold-start detection checks `not handoff_section`
- **All sections before onboarding**: total_items counts all section lengths
- **All sections before disclosure**: disclosure reads all section results
- **Budget packing is sequential**: each section depends on `used_tokens` from previous sections

---

## Progressive Disclosure: Tier Classification

| Tier | Sections | Behavior |
|------|----------|----------|
| 1 (always shown) | grounding, rules, handoff, issues, anti_patterns | Full content in primer |
| 2 (deferred) | behaviors, recent_work, decisions, entities | Count + hint, loadable via `weft_focus` |
| Independent | changes_since, wellness | Not token-budgeted, always included |
| Post-processing | onboarding, disclosure | Computed from other sections |

---

## Query Overlap Analysis

Sections that hit the same tables — relevant for deciding whether to batch
queries in a pre-flight loader or accept redundant queries.

| Table | Sections querying it |
|-------|---------------------|
| `memories` | rules, handoff, recent_work, issues, anti_patterns, decisions, changes_since |
| `behaviors` | behaviors |
| `entities` | entities |
| `check_ins` | wellness |

The `memories` table is queried 6 times with different filters (type, pinned,
status, topic).  These are distinct index paths in PostgreSQL, so batching
into a single query would likely be slower than parallel queries.  Keep separate.

---

## Open Design Decisions (for implementation task)

1. **Orchestrator location**: The orchestrator stays in `primer.py` — it
   replaces the monolithic function with a thin loop.  No `orchestrator.py`.

2. **Section builder protocol**: Each builder is `async def build_X(ctx) -> SectionResult`.
   The orchestrator calls `await section.build_X(ctx)` and reads `result.items`
   and `result.tokens_used` to update `ctx.used_tokens`.

3. **Fetch vs pack split**: Each section builder owns both its fetch AND its
   pack logic.  This keeps sections self-contained.  The orchestrator only
   manages the budget and ordering.

4. **Dynamic budgets**: Handoff uses `min(_CAP_HANDOFF, budget_tokens - used_tokens)`
   — a dynamic cap.  This stays inside `build_handoff_section`, not in
   `SECTION_BUDGETS`.  The constant in `SECTION_BUDGETS` is the max cap.

5. **`_DICT_OVERHEAD_TOKENS`**: Move to `context.py` as a module-level constant
   so all sections can import it.

6. **Helper functions** (`_is_unscoped_ingest`, `_annotate_review_after`,
   `_newest_created_at`): Move to `context.py` as package-level utilities.
   They are used by multiple sections.
