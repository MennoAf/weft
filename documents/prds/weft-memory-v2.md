# Weft Memory v2 — Never-Miss via Reconciliation

## Summary

Weft v2 reframes "never lose anything" from a **read-side ranking** problem into a **write-time + reconciliation** property. The north-star miss — "I know I told it that and it didn't surface" — is the one event where *nothing* surfaced and *no failed query was logged*; it is invisible by construction in any pull-only pipeline, and no tidier filing scheme organizes its way out of a probabilistic top-k cutoff. So v2 **builds the meter before the machine**: a reconciliation layer (canary + `weft_fsck`) that can observe its own misses ships first, in parallel with the cheap correctness fixes every later claim rides on.

On top of the meter, v2 splits memory into a **catalog half** (code, files, projects — things with an intrinsic, machine-derivable identity) and an **associative half** (beliefs, decisions, brain-dumps — no intrinsic key). The catalog half gets a deterministic AST/path-derived natural key (`loc_key`) — explicitly **not** Johnny-Decimal numbers, which renumber on sibling insert and rot every reference. The associative half gets *shapes, not addresses*: an enumeration-intent router that routes "list all / every / how many" to the deterministic complete gather, collections as rule-based saved predicates, and an orthogonal temporal axis. Work is phased: **Phase 0** (the meter, independent, starts now), **Phase 1** (cheap wins, depends on the in-flight Topic-Digest Recall substrate), **Phase 2** (research-shaped, measure-first).

## Goals

- **G1 — Build the reconciliation meter (Phase 0).** Stand up miss-*detection* (recall canary + `weft_fsck`) plus the cheap correctness fixes (two-tier entity threshold, surfaced truncation, deterministic vector tie-break, known-membership eval fixtures) so every later never-miss claim is *checkable*, not aspirational. Validation §V1–V6.
- **G2 — Ship the enumeration router + catalog keys (Phase 1).** Route enumeration-intent queries through the deterministic complete gather; add nullable `loc_key` + `loc_registry`; AST-derive normalized `requires` with `weft_portfolio_query`; ship collections as rule/confirmed saved predicates with the full contract. Validation §V7–V10.
- **G3 — Add the temporal + taxonomic axes (Phase 2, measure-first).** Bi-temporal `valid_from`/`valid_to` with `since`/`until`/`as_of` filters; taxonomic `is_a` layer for hypernym enumeration; keep `created_at` as the always-correct fallback. Validation §V11–V12.
- **G4 — Operationalize the north star.** Drive `min recall@membership → 1.0` over seeded known-membership sets and the rolling re-ask-miss rate → 0; keep RLS visibility invariant under any address/facet change. Validation §V4, §V13.
- **G5 — Hold the write path free.** No LLM classifier on the `weft_remember` hot path; associative area assignment by cosine over precomputed centroids, model escalation deferred to the nightly pass. Validation §Vcost.

## Non-Goals

- **Numeric Johnny-Decimal codes (`21.11.11`) as the primary address.** Self-defeating: decimals are insertion-order- and corpus-relative, so a sibling insert renumbers neighbors and the same object written months later lands at a different number — manufacturing the exact reference-rot the design exists to kill. Replaced by deterministic `loc_key`. (Supersedes the original JD sketch, Technical Decisions.)
- **Addresses for associative memories.** Beliefs have no intrinsic `key(object)→slug`; forcing one is the drift trap. The associative half navigates by facets + collections + the entity graph, not addresses.
- **The "one primitive" node collapse** (merging entities/episodes/collections/areas/code-blocks/trackers into one node table). Too risky to migrate now; new shapes ride the shared substrate instead, and the collapse is revisited after dogfooding v2 (deferred, Weft `weft-96b067f3`, review ~2026-08). Deferred, *not* superseded.
- **`weft_fingerprint` / project-map (area 30) this cycle.** It is the largest net-new build, greenfield (CONFIRMED: does not exist — `weft/identity.py` is federation hashing, unrelated), and its payoff (token savings) is real but its risk (confidently-stale map) is the sharpest. Deferred behind the cheaper, measurable wins until the ingest path can support it.
- **`belief_claims`-based enumeration (the multi-turn belief-aggregation detector, `loom-dcfaf656`).** Superseded: `belief_claims` is empty on real data (Ground Truth) — v2 routes enumeration through the populated `memories.topic[]` substrate instead. (Supersedes EPIC 2, Technical Decisions.)
- **Tenancy-as-prefix.** "Who can see it" is never "where in the tree." `workspace_id` stays the sole tenancy axis, fully orthogonal to any facet/address; asserted invariant under any `loc` change (§V13). Defended against the numbered-tree temptation.
- **LLM-extracted or hand-typed `requires`.** A copied/frozen dependency list lies (a `pandas<2` CVE scan returns a confidently-wrong set). `requires` is machine-derived from the AST and normalized, or it is not built.

## The reframe: never-miss is a write + reconcile property (behavior)

The category-defining bet underneath v2 is a truth the field largely misses: **every system that achieves never-miss at scale solves it structurally, not with better search.** Double-entry bookkeeping has the *trial balance* — an imbalance proves a missing entry exists *before you know which one*. Git has `fsck` — a dangling object is one unreachable from any ref. Amazon runs cycle counts — a bin scan reconciled against the ledger. Weft today has none of these, so it can honestly promise "pretty good recall," not "never miss."

A pull-only retrieval pipeline cannot observe the miss that matters. When a user asks and the right memory ranks below the cutoff, *something* still returns, so no error fires; when the user never re-asks, even the re-ask signal stays silent. The meter must therefore be **active**: it re-issues known-answer probes against frozen materialization and treats a failure-to-surface as a first-class logged defect, and it walks the reachability graph to flag memories that *only* a stochastic vector hop can reach. This is unglamorous, produces no demo, and is for exactly that reason the thing most likely to get cut — so it goes first.

## Phase 0 — The reconciliation meter + correctness floor (behavior)

Phase 0 has two faces: the **meter** (net-new reconciliation) and the **floor** (cheap fixes the meter and everything above it depend on). It is independent of the in-flight Topic-Digest Recall work and starts immediately.

The **floor** removes silent correctness bugs. Entity resolution today merges on a single 0.6 cosine threshold (CONFIRMED `weft/ingest_pipeline.py:359`), which both false-merges ("basil" + a different basil) and false-splits ("basil" vs "basil plant") — and *everything joins through entities*, so any collection or enumeration built on it is building on sand. v2 moves to two-tier: ≥0.85 auto-merge, 0.6–0.85 becomes a review candidate, never a silent merge. Truncation is currently *computed* but only partly surfaced — `gather_topic_memories` returns a `truncated` flag (CONFIRMED `weft/topic_gather.py:213`) but the `_ENTITY_MEMORIES_LIMIT = 100` cap (CONFIRMED `weft/topic_gather.py:34`) and the MCP surface must guarantee it reaches the agent, never swallowed. And the vector search orders by raw cosine distance with no secondary key (CONFIRMED `weft/store.py:371`, `:694`), so two memories at equal distance can swap places run-to-run; a deterministic tie-break makes a single materialized run reproducible, which is the precondition for measuring anything.

The **meter** is the net-new long pole. Every `weft_remember` enrolls the fact as a known-answer probe (the originating context → the memory that should satisfy it). A daily job re-issues those probes against *fixed* materialization (local FastEmbed is deterministic; the §V3 tie-break closes the last gap) and logs any probe that fails to surface its memory as a first-class miss event — converting an unmeasurable aspiration into a tracked defect rate. `weft_fsck` complements it from the structural side: it lists memories reachable *only* by vector cosine (no tag, entity, episode, or collection edge) — orphans, the leading indicator of a future miss — and is shippable as a user-facing CLI that prints them, an external trust artifact, not just an internal invariant.

The eval substrate lands here too: seeded known-membership fixtures (synthetic "Jim Boblaw": 12 plants, 8 medications, etc., per the synthetic-persona rule) under `benchmarks/enumeration_eval/`, reporting `min/median/max recall@membership`. The deterministic gather is the *oracle* (asserted to return all M in a single run); the *candidate* is the NL query a real agent phrases through `weft_recall`, run k≥5 because that path is stochastic — and the spread is itself the never-miss signal.

## Phase 1 — Enumeration router, catalog keys, collections (behavior)

Phase 1 delivers the visible wins. It **depends on the in-flight Topic-Digest Recall epics (Program A)** landing first, because the enumeration router and collections both ride the deterministic `gather_topic_memories` path and the `topic_digests` cache that Program A finishes.

**The enumeration router** is the single change most likely to move never-miss. Today a natural-language "list all the plants" flows through the limit-bounded hybrid search and members below the cutoff silently drop, while the deterministic complete gather sits unused behind the topic-digest surface. v2 adds a cheap enumeration-intent classifier to `weft_recall` (regex/keyword: "list all | every | how many | enumerate" + a small gate), resolves the query noun to a topic/entity/collection, and fires `gather_topic_memories` **in parallel** with the top-k search as a fallback contract — returning a reconciliation header: *"similarity surfaced 7; membership knows 12; 5 not shown: [ids]."* The deterministic path is the oracle, the similarity path is the convenience, and the gap between them is now *visible* rather than silent.

**Catalog keys** give code and files an immutable identity. A new nullable `loc_key` column (`code:repo/mod.py#fully.qualified.symbol`, `file:path`) is filled synchronously for code via AST extraction (free, deterministic) and left NULL for associative memories. A `loc_registry(loc_key → …)` append-mostly table is the anti-drift source of truth — a compiler symbol table / git object store — so an identical object written months later exact-matches its prior key instead of minting a new one. Identity keys on the stable symbol path; the **content hash is carried as a version, not the identity**, so a typo-fix does not reset a function's links or reuse count, and refactors emit `rename` edges so identity survives. Crucially, `loc_key` is a *hint, never a gate*: a NULL-`loc_key` memory is fully recallable via embeddings exactly as today.

**Dependency-by-reference** falls out once identity is a natural key. `requires` is machine-derived from AST imports (tree-sitter) and normalized to canonical package names — never hand-typed, never LLM-extracted — which makes `weft_portfolio_query(import='pandas')` an exact containment scan ("everywhere I use pandas," "which repos break if I drop 3.10," "which repos are exposed to a `requests<2.31` CVE"), not a similarity gamble. "Use this lego" becomes a *reference* to a `loc_key` node, not a copy of its bytes, so a fix to the canonical node is seen by every reference; pinned references surface "N refs behind" rather than auto-upgrading, and a content-hash snapshot fallback keeps a referenced block alive if its source repo moves.

**Collections** ship as first-class sets — but *only* with the full contract, because a bare join table relocates the miss to write-time where it is harder to see. Membership is **rule-based or confirmed (candidate → member), never silent similarity-attach**. Coverage telemetry (`members_by_rule` vs `members_by_explicit_link`) makes under-population observable; async backfill (reusing `consolidation.py`, CONFIRMED present) lets a collection created today retro-claim historical members via its predicate; and `truncated=true` is plumbed end-to-end so a user with 200 plant memories never gets a confident 100.

## Phase 2 — Temporal + taxonomic axes (behavior, measure-first)

Phase 2 is research-shaped: each item is gated on the Phase-0 eval and **never committed on a single run** (per the 37%-flip anti-pattern, Technical Decisions).

The **temporal axis** is a genuine second, orthogonal axis — neither relevance nor validity subsumes the other. v2 generalizes the bitemporal model already shipped in `belief_claims` (CONFIRMED v48: `occurred_at` valid-time vs `created_at` transaction-time + `superseded_by` chains) to all memories: nullable `valid_from`/`valid_to` (cheap ALTER; NULL = always-valid) plus `since`/`until`/`as_of` filters plumbed through the gather and search paths. This answers "where did Jason live in March" and "plants mentioned before the move." The *risk* is valid-time extraction from natural language ("since the divorce," "last spring"): a wrong window silently hides a fact, so valid-time is populated only when an explicit date is present and **`created_at` stays the always-correct fallback**.

The **taxonomic `is_a` layer** (`entity:basil --is_a--> entity:plant`, on the existing relationship graph) closes the enumeration gap where extraction names "basil" but never "plant," so "list all plants" has an entity to enumerate. Hypernym assignment is a cheap async batch pass over the entity table, cacheable, measured for over-/under-enumeration on the eval set.

`weft_fingerprint` (area 30, the project-map) is named here only to be **deferred** (Non-Goals): build the `loc_key` + registry framework now, build the map view once ingestion supports it and its token-savings-vs-staleness-risk has been measured.

## Sequencing & dependencies (behavior)

The work interleaves with two in-flight Loom programs in the `weft-public` project, and the relationship is load-bearing:

**Program A — Topic-Digest Recall** (`loom-1c010dec` Tier-1, `loom-d57e3010` Tier-2, `loom-7cef149a` + `loom-54d9a1e5` `weft_status`/acceptance; PRD `documents/prds/topic-digest-recall.md`) is the **deterministic enumeration substrate Phase 1 depends on**. It finishes the complete `gather_topic_memories` path, the `topic_digests` cache, and the L1 Resolution Ratchet. **Decision: finish Program A first;** Phase 1's enumeration router and collections `depend_on` it. Phase 0 runs *in parallel* with Program A because the meter and the correctness floor touch independent code (entity resolution, vector ordering, a new canary table, `fsck`).

**Program B — belief_claims enumeration** (`loom-dcfaf656` EPIC 2 multi-turn aggregation detector) is **superseded** by the `memories.topic[]` enumeration path (`belief_claims` is empty on real data). The companion **LongMemEval validation gate** (`loom-88799e2c`, EPIC 3) is **retained** as v2's measurement milestone — run multi-run (n≥5), never single-run, per the anti-pattern.

## Ground Truth

CONFIRMED (source greps from the repository root; local absolute paths are intentionally omitted from public documentation):
- `resolve_entities` merges on a **single** 0.6 cosine threshold. Source: `weft/ingest_pipeline.py:307` (def), `:359` (`threshold=0.6`).
- `gather_topic_memories` does an unbounded `= ANY(topic)` gather, `ORDER BY created_at ASC`, and returns a `truncated` flag. Source: `weft/topic_gather.py:43` (def), `:134` (ORDER BY), `:198-199` (sets `truncated`), `:213` (returns it).
- `_ENTITY_MEMORIES_LIMIT = 100`. Source: `weft/topic_gather.py:34`.
- Vector search orders by raw cosine distance with **no** secondary tie-break (`ORDER BY embedding <=> $1::vector`). Source: `weft/store.py:371`, `:694`.
- `topic_digests` table + `topic_digest_cache.py` exist; the GIN index on `memories.topic[]` and the alias table shipped. Source: migrations `v57_topic_digests.py`, `v56_topic_digest_gin_and_aliases.py`; `weft/topic_digest_cache.py`.
- `belief_claims` is bitemporal (`occurred_at` valid-time, `created_at` transaction-time, `superseded_by` chain). Source: `weft/db/migrations/v48_belief_claims.py:62,65,66`.
- `belief_claims` (and `episode_turns`) are **empty on real data** (0 rows; distillation layer has only run on benchmark haystacks). Source: prod `count(*)` read 2026-06-24 recorded in `documents/prds/topic-digest-recall.md` Ground Truth.
- Re-ask-miss signal exists (`is_reask_miss`, `reask_satisfying_memory_id`) with `weft/reask.py` + `weft/replay.py`. Source: `weft/db/migrations/v52_reask_miss_signal.py:46-50`.
- `weft_fingerprint` / commit-diffing project-map **does not exist**; `weft/identity.py` is federation install-identity (ed25519), unrelated. Source: grep (no hits) + `weft/identity.py:1-134`.
- `consolidation.py` (decay/dedup/contradiction maintenance) and `benchmarks/longmemeval/materialize.py` (per-question belief-view materialization, deterministic `occurred_at ASC`) both exist. Source: `weft/consolidation.py`, `benchmarks/longmemeval/materialize.py`.
- Highest migration present is **v62** (`v62_calibration_record_origin.py`). Source: migrations directory listing.
- `entity_mentions` (PK `entity_id, memory_id`) and `episode_memories` (PK `episode_id, memory_id`) membership join tables exist. Source: `weft/db/migrations/v12_entities_tables.py:31-36`, `v11_episodes_tables.py:26-32`.
- The `weft_remember` write path issues **no LLM call** — content validation, one local FastEmbed embed, dedup check, insert. Source: `weft/mcp/tools.py:228-327`, embed at `:294`.

CONFIRMED (measurement baselines, from project memory):
- LongMemEval honest baseline is **43.6% / 42.0%** overall at n=500 post-fence; single-session-assistant 16%, -preference 27%. Source: Weft `project_longmemeval_baseline` (2026-05-03). This is the `~43%` the gate measures against.
- The `60%+` multi-session+temporal target is the Lodestar falsifiable claim. Source: Weft `weft-fce61031` (referenced by `loom-88799e2c`/EPIC 3).

ASSUMED (unverified — promote to a Research Item before any `done_when` derives from it):
- The five "area" centroids for cost-routed associative placement are separable enough that argmax-cosine lands the right area outside a narrow ambiguity band. Unverified — gates §Vcost's no-LLM claim; Research Item RI-1.
- AST `requires` extraction (tree-sitter) covers the portfolio's languages well enough that `weft_portfolio_query` is *exact*, not merely *mostly*. Unverified for non-Python repos; Research Item RI-2.
- Valid-time NL extraction is reliable enough to populate `valid_from`/`valid_to` without silently hiding facts. Disfavored by prior temporal-reasoning failures; Research Item RI-3.
- The recall-canary's "originating context" is a faithful enough probe that a canary pass failing-to-surface genuinely indicates a recall miss (vs. a probe-construction artifact). Unverified; Research Item RI-4.

## Constraints Touched

Enforced constraints within one hop of a modified component, each with a scope decision:

- **`threshold=0.6`** at `weft/ingest_pipeline.py:359` — **IN SCOPE** (replace with two-tier ≥0.85 / 0.6–0.85, §V1).
- **`_ENTITY_MEMORIES_LIMIT = 100`** at `weft/topic_gather.py:34` — **IN SCOPE** (retain the cap, but set `truncated=true` when it bites, §V2).
- **Vector `ORDER BY embedding <=> $1::vector`** at `weft/store.py:371`, `:694` — **IN SCOPE** (add deterministic `id` tie-break, §V3).
- **Search `limit` defaults (`limit=10`/`50`)** in `weft/store.py` (catalogued in `documents/prds/topic-digest-recall.md` Constraints Touched) — **IN SCOPE for the router seam**: the enumeration router (§V7) fires the *unbounded* `gather_topic_memories` in parallel; these top-k caps are exactly what silently drop members. Decision: the gather path bypasses them; the top-k similarity path **retains** them by design (it is the convenience arm, the gather is the oracle), and the reconciliation header surfaces the gap.
- **`MAX_COST_PER_CALL_USD = 0.0034`** at `weft/views/belief_detector.py:50` — **IN SCOPE as precedent** for §Vcost: the deferred nightly area-escalation and any canary-side model call inherit a named cost cap, never an inline literal; no model is ever called on the `weft_remember` hot path.
- **`replay_queue` status CHECK** (`v55`) — **OUT OF SCOPE.** v2 does not enqueue replay work; the superseded EPIC 2 owned that path.

## Validation

Effective state must satisfy:

**Phase 0 — meter + floor:**
- **V1 (two-tier entity resolution):** `resolve_entities` auto-merges only at cosine ≥ 0.85; the 0.6–0.85 band produces a review *candidate*, never a silent merge. Replaces the single 0.6 threshold at `weft/ingest_pipeline.py:359`.
- **V2 (truncation surfaced):** any gather or entity path that caps results (`_ENTITY_MEMORIES_LIMIT` or a budget) sets `truncated = true`, and that flag reaches the agent through the MCP response. No cap is ever swallowed.
- **V3 (deterministic vector ranking):** every vector `ORDER BY embedding <=> …` carries a deterministic secondary sort key (e.g. `id`), so identical materialization yields identical ordering across runs. Applies at `weft/store.py:371`, `:694`.
- **V4 (enumeration eval):** `benchmarks/enumeration_eval/` reports `min/median/max recall@membership` over seeded Jim-Boblaw known-membership fixtures. The deterministic gather (oracle) returns all M in one run; the NL `weft_recall` candidate is run k≥5. North star: `min recall@membership → 1.0`.
- **V5 (recall canary):** every `weft_remember` enrolls a known-answer probe; a daily fixed-materialization audit re-issues probes and records each failure-to-surface as a first-class miss event with a rolling rate.
- **V6 (`weft_fsck`):** `weft_fsck` returns the set of active memories reachable *only* by vector similarity (no tag/entity/episode/collection edge); exposed as a CLI that prints them.

**Phase 1 — router + catalog + collections:**
- **V7 (enumeration router):** `weft_recall` detects enumeration intent and fires `gather_topic_memories` in parallel with top-k, returning a reconciliation header stating similarity count, membership count, and the ids known-but-not-shown.
- **V8 (`loc_key` is a hint, not a gate):** `loc_key` (nullable TEXT) + `loc_registry` exist; code memories get an AST-derived key synchronously, associative memories keep NULL, and a NULL-`loc_key` memory is fully recallable via embeddings. The same object re-ingested exact-matches its registry key (no re-mint).
- **V9 (AST-derived `requires`):** `requires` is populated from AST imports normalized to canonical package names; `weft_portfolio_query(import=X)` returns the exact set of `loc_key`s whose `requires` contains X — a containment scan, never a similarity draw.
- **V10 (collections contract):** collection membership is rule-based or confirmed (no silent similarity-attach); coverage telemetry exposes `members_by_rule` vs `members_by_explicit_link`; backfill is async; a capped enumeration sets `truncated = true` (V2).

**Phase 2 — temporal + taxonomy:**
- **V11 (bi-temporal validity):** nullable `valid_from`/`valid_to` (NULL = always-valid) + `since`/`until`/`as_of` filters on gather/search; `valid_from` defaults to `created_at`; `created_at` remains the always-correct fallback and is never overwritten by a failed extraction.
- **V12 (`is_a` enumeration):** with `entity:basil --is_a--> entity:plant` present, an enumeration of "plants" includes basil; over-/under-enumeration is measured on the V4 eval set.

**Cross-cutting:**
- **V13 (tenancy invariant):** `workspace_id` is the sole visibility axis; a test asserts returned-row visibility is invariant under any change to `loc`/`loc_key`/facet.
- **Vcost (free write path):** `weft_remember` issues zero LLM calls; associative area assignment is argmax-cosine over 5 precomputed area centroids using the FastEmbed vector already computed at write time; model escalation fires only inside a measured ambiguity band (top-2 within ~0.05) and is deferred to the nightly consolidation pass.

## Interfaces / Schema

All changes are **additive ALTERs / new tables**, consistent with Weft's boot-time migration model (no destructive migration).

**Phase 0:**
- `resolve_entities(...)` gains a two-tier outcome (`merged` | `candidate` | `new`); a candidate-review surface (table or column) records 0.6–0.85 pairs. (V1)
- New: `recall_canary(probe_id, memory_id, origin_context, user_id, last_checked_at, last_status)` + a daily audit job + a `canary_miss` counter. (V5)
- New MCP/CLI tool `weft_fsck() → [{memory_id, reason: "vector-only"}]`. (V6)
- `benchmarks/enumeration_eval/` harness + Jim-Boblaw fixtures. (V4)

**Phase 1:**
- `weft_recall` gains enumeration-intent detection + a `reconciliation` block in its response (`{similarity_count, membership_count, not_shown: [ids]}`). (V7)
- `memories` gains nullable `loc_key TEXT`; new `loc_registry(loc_key PK, kind, content_hash, first_seen_at, …)`; `rename` edges on the relationship graph. (V8)
- `memories` (or a code-block subtype) gains `requires TEXT[]`; new MCP tool `weft_portfolio_query(import: str) → [loc_key]`. (V9)
- New `collections` + membership tables with `membership_source ∈ {rule, confirmed}` and coverage counters. (V10)

**Phase 2:**
- `memories` gains nullable `valid_from TIMESTAMPTZ`, `valid_to TIMESTAMPTZ`; `since`/`until`/`as_of` params on gather/search. (V11)
- `is_a` edge type on `memory_relationships`; async hypernym batch job. (V12)

## Testing

### Unit Tests
- `resolve_entities`: a pair at cosine 0.90 auto-merges; a pair at 0.70 produces a candidate and does **not** merge; a pair at 0.55 stays separate (V1).
- A gather capped at `_ENTITY_MEMORIES_LIMIT` returns `truncated=true`; an uncapped gather returns `truncated=false` (V2).
- Vector ranking: two rows at identical `embedding <=> $1` distance return in a stable `id`-tie-broken order across repeated calls (V3).
- Enumeration-intent classifier: "list all my plants" / "how many meds" → enumeration=true; "what did I think about X" → enumeration=false (V7).
- `loc_key`: AST extraction of a known function yields the expected `code:repo/mod.py#symbol`; re-extracting the same symbol returns the same key; a typo-fix changes `content_hash` but not `loc_key` (V8).
- `requires` normalization maps an `import pandas as pd` to canonical `pandas` (V9).
- Collection membership rejects a silent similarity-attach; a rule-match and an explicit confirm both land, tagged by source (V10).
- `valid_from` defaults to `created_at` when no date is extracted; a failed extraction never nulls/overwrites `created_at` (V11).

### Integration Tests
- **Canary loop:** enroll N memories via `weft_remember`; force one to rank below cutoff under fixed materialization; the daily audit logs exactly that one as a `canary_miss` (V5).
- **`weft_fsck`:** seed a memory with an embedding but no tag/entity/episode/collection edge; `weft_fsck` returns it; add a tag edge; it no longer appears (V6).
- **Enumeration router e2e:** seed 12 plant memories where top-k surfaces 7; `weft_recall("list all my plants")` returns a reconciliation header `similarity=7, membership=12, not_shown=[5 ids]` and the complete set via the gather path (V7).
- **Portfolio query e2e:** ingest two repos importing `pandas` and one importing `polars`; `weft_portfolio_query("pandas")` returns exactly the two `pandas` `loc_key`s (V9).
- **Collections backfill:** create a collection with a rule predicate after members exist; async backfill retro-claims the historical members; coverage telemetry reports them under `members_by_rule` (V10).
- **Temporal filter:** a memory with `valid_from <= D < valid_to` is returned for `as_of=D` and excluded for an `as_of` outside `[valid_from, valid_to)`; a NULL-`valid_to` memory is returned for every `as_of >= valid_from` (V11).
- **Candidate-split entities (V1 downstream):** when a near-duplicate pair lands in the 0.6–0.85 band as two candidate entities (not merged), an enumeration over a topic spanning both returns members of BOTH halves — the two-tier change must not silently drop a member by splitting its entity (V1 downstream survival).
- **Tenancy invariant:** mutate a memory's `loc_key`; an other-workspace caller's result set is unchanged (V13).
- **Write path cost:** `weft_remember` over a corpus issues zero LLM calls; area assignment resolves via centroid cosine (Vcost).

### Acceptance Criteria
- On Weft's own real memory, the `benchmarks/enumeration_eval/` harness reports `min recall@membership` for the deterministic gather == 1.0 over every seeded fixture, and prints the k≥5 `min/median/max` spread for the NL candidate path (V4) — a single mechanically-runnable command.
- `weft_fsck` runs against the live store and prints a finite, inspectable orphan list; the count is recorded as the Phase-0 baseline (V6).
- The recall-canary daily audit runs one full cycle with `probes_checked > 0`, and a deliberately-planted below-cutoff probe is recorded as a `canary_miss` while a known-surfacing probe is not — proving the meter detects a real miss, not just that it ran (V5). (Non-degeneracy: a no-op meter that logs `rate=0` fails this gate.)
- After Program A + Phase 1 land, `weft_recall("list all …")` over a known-membership topic returns membership-complete results with the reconciliation header, proven by an integration test, not by inspection (V7).
- **LongMemEval gate (`loom-88799e2c`):** a multi-run (n≥5) LongMemEval pass after Phase 1 shows multi-session + temporal accuracy moving toward the 60%+ band vs the ~43% honest baseline **without** regressing overall abstention; the per-question spread is reported (never a single-run delta). If it does not move, the dead-tell fired — record it and route back to the router/linkage.

## Technical Decisions

- **Johnny-Decimal numeric addresses as primary identity** (original v2 sketch, Weft `weft-6ec496ec`, archived) — **superseded** by this PRD. Decimals demoted entirely; catalog identity is the deterministic `loc_key`. Canonical resolved design: Weft `weft-5276bf05` (pinned).
- **Multi-turn belief-aggregation enumeration over `belief_claims`** (EPIC 2, `loom-dcfaf656`) — **superseded.** `belief_claims` is empty on real data (Ground Truth); enumeration routes through `memories.topic[]` via the gather path. EPIC 2 to be marked superseded in Loom.
- **"Make the replay A/B show signal" as a standalone goal** — **superseded in part.** Replay/belief tiers are retained as benchmark apparatus only; not on v2's critical path. Per anti-pattern `weft-b015d16a` (37% single-run flip) and the topic-digest PRD lineage.
- **Topic-Digest Recall** (`documents/prds/topic-digest-recall.md`, Program A) — **retained, depended-upon.** Phase 1's enumeration router + collections build on its `gather_topic_memories` + `topic_digests` substrate; it must land first.
- **LongMemEval validation gate** (`loom-88799e2c`, EPIC 3) — **retained** as v2's measurement milestone; run multi-run per `weft-b015d16a`.
- **`belief_claims` bitemporal model** (v48) — **retained and generalized** to all memories in Phase 2 (V11).
- **The "one primitive" node collapse** (Weft `weft-96b067f3`) — **deferred, not superseded;** revisit after dogfooding v2 (~2026-08). New shapes ride the shared substrate; no seventh parallel table is built speculatively.

## Compounding Loops

> Seeded from the resolved design; `/groundhog` may refine or add blueprints before audit.

The reconciliation meter is itself the system's primary compounding loop, and the design carries two more. Each is `Signal → Store → Feedback → Proof`.

### CL1 — Canary-minted eval set (BUILD, Phase 0)
- **Signal:** every production recall miss the canary detects (and every re-ask-miss already logged via `is_reask_miss`, CONFIRMED `v52`) is a real query that should have surfaced a known memory.
- **Store:** each miss auto-mints a known-answer eval case `(query → satisfying_memory_id)` into the `benchmarks/enumeration_eval/` set.
- **Feedback:** the eval set grows from real usage, not just synthetic personas; every later phase is regression-tested against it.
- **Proof — done_when:** force one canary miss; assert a new eval case appears in the harness referencing the missed `memory_id`, and that re-running the harness exercises it. Dead-tell: canary fires but the eval-case count is flat.

### CL2 — Self-curating package index (DESIGNED, Phase 1+, Pinch-gated)
- **Signal:** the existing `access_count`/usefulness machinery on a `loc_key` node — a snippet referenced across N projects.
- **Store:** the `loc_registry` reuse count per `loc_key`.
- **Feedback:** a block reused above a threshold auto-nominates itself for promotion from associative scratch → catalog reference; Weft learns the user's actual personal stdlib from build behavior.
- **Proof — done_when:** reference one `loc_key` from ≥3 projects; assert it appears in a promotion-nomination query while a once-referenced block does not. Deferred behind Phase 1's `loc_key` landing.

### CL3 — Expected-cardinality watchdog (DESIGNED, Phase 1+)
- **Signal:** a collection or memory carrying an expected count ("I take 3 medications") vs. the live enumeration count.
- **Store:** the collection's declared cardinality vs. `members_by_rule + members_by_explicit_link`.
- **Feedback:** when enumeration falls below expectation, Weft proactively flags "you mentioned 3 meds, I have 2" — the strongest expression of never-miss (the system notices its *own* gap).
- **Proof — done_when:** declare expected=3 on a collection with 2 members; assert the watchdog raises a gap alert; add the 3rd; assert it clears. Builds on V10 coverage telemetry.

**Verdict: SEEDED.** CL1 builds in Phase 0; CL2/CL3 are designed and gated behind Phase-1 substrate.

## Research Items

- **RI-1 — Area-centroid separability.** Do 5 precomputed area centroids cleanly route associative placement by argmax-cosine, or is the ambiguity band wide enough to force frequent LLM escalation? Audit on the real corpus before trusting §Vcost's no-LLM claim. (Gates Vcost; ASSUMED-1.)
- **RI-2 — AST `requires` language coverage.** tree-sitter `requires` extraction is exact for Python; is it exact for the rest of the portfolio's languages, or does `weft_portfolio_query` need a "best-effort, N languages unparsed" disclosure? (Gates V9 exactness; ASSUMED-2.)
- **RI-3 — Valid-time NL extraction reliability.** How often does "since the divorce"-class phrasing extract a *correct* `valid_from`? Measure on the eval set; a wrong window silently hides a fact, so the bar is high. Until met, populate valid-time only on explicit dates and keep `created_at` authoritative. (Gates V11; ASSUMED-3.)
- **RI-4 — Canary probe fidelity.** Is the enrolled "originating context" a faithful probe, or does it produce false misses (probe-construction artifacts) / false passes? Calibrate before trusting the `canary_miss` rate as a defect metric. (Gates V5; ASSUMED-4.)
- **RI-5 — Collection membership predicate language.** What predicate grammar do rule-based collections use (tag-set algebra? entity + `is_a`? a saved query)? Resolve at Phase-1 Epic time; leans toward saved topic/entity predicates reusing the gather path.
- **RI-6 — `loc_key` for non-symbol files.** Symbol-bearing code keys cleanly; what is the `loc_key` for a config/asset/markdown file — `file:path` only, or path + content-hash? Resolve at Phase-1 Epic time.
