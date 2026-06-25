# Topic-Digest Recall — "Ask Weft about X, get an answer, not a fragment dump"

## Summary

Weft today answers "tell me about X" with a top-k semantic best-match list — a pile of loosely-related
fragments, not an answer. This PRD adds a **topic-anchored synthesis read path**: ask Weft about a topic
and get back the *complete* set of what it knows, ordered, optionally synthesized into a narrative
"here's where X stands."

The load-bearing concept is that a **topic digest is a materialized view over the memory neighborhood of
a topic** — the third instance of a pattern Weft already runs (one turn → a belief claim; a turn-set →
a replay enumeration claim; the **memory neighborhood → a topic digest**). The read path has two tiers:
Tier-1 is a deterministic, no-LLM *complete* gather of the active memories under a topic (via the
populated `memories.topic[]` tag set, with the entity graph as a secondary key); Tier-2 is an on-demand
Haiku synthesis pass that produces a cached digest, invalidated when a relevant memory is written.
Efficiency comes from gathering over an indexed topic tag, caching, and only paying the LLM tax on demand.

> **Substrate note (closes Research Item #1, prod read 2026-06-24):** the originally-specced substrate,
> `belief_claims` keyed by attribute prefix, is **empty on real data** (0 rows; the belief/turn/replay
> distillation layer has only ever run on benchmark haystacks). The populated substrate is `memories`
> (4,781 active, 88% topic-tagged) plus the entity graph. This PRD targets `memories`. Gathering over raw
> memories (rather than distilled claims) also aligns with the pinned "raw beats honest extraction"
> LongMemEval baseline. The `belief_claims` path is deferred to a future enhancement contingent on that
> layer ever being populated on real data (a separate, prior-evidence-disfavored decision).

## Goals

- Add a topic-anchored read surface whose Tier-1 read satisfies Validation §V1 (complete over the topic's
  memories, not top-k, temporally ordered) and §V2 (zero LLM calls).
- Add an on-demand **Tier-2 synthesis** that turns the Tier-1 memory set into a narrative status answer
  with per-assertion provenance, materialized as a cached **topic digest**.
- Keep digests fresh by **write-invalidation**: a new memory tagged with a topic marks that topic's digest
  stale. Refresh is signal-fed, never scheduled.
- Reuse the existing materialize-from-evidence shape (gather → detect → write a derived memory, cached and
  refreshed on signal) rather than inventing a new mechanism.
- Validate on real personal/project memory first — "what's the status of weft?" against the actual
  `memories` store — with LongMemEval multi-session as a secondary scoreboard, not the acceptance gate.

## Non-Goals

- **Not replacing `weft_recall`'s belief/turns/both tiers.** Top-k best-match recall is the right tool for
  "find the memory that says X"; this is an additive surface for "what's the status of X." Conflating them
  is the original design error this PRD corrects.
- **Not building Tier-1 over `belief_claims`.** That table is empty on real data (Ground Truth) — a Tier-1
  gather over it returns nothing. Deferred until/unless the distillation layer is populated on real
  memory, which prior evidence ("raw beats honest extraction") disfavors.
- **Not populating the belief/turn/replay distillation layer on real data as part of this work.** That is a
  separate, larger strategic decision; this feature deliberately stands on the already-populated `memories`
  substrate so it does not depend on resolving it.
- **Not adding a semantic embedding-only synthesis path in v1.** Tier-1 addressing rides the populated
  `topic[]` tag set (+ entity graph); semantic vector recall already exists in `weft_recall` and is the
  best-match tool, not the complete-gather tool. Mixing them is deferred.
- **Not building Branch C "weft-lang" (the natural-language query planner).** This is one concrete query
  path; the planner needs several tiers to exist before it is useful. This PRD is the tier it will later
  route to, not the planner itself.
- **Not a scheduled / cron digest refresh.** A calendar-driven "re-summarize every topic nightly" is the
  compounding-loop anti-pattern (work decoupled from signal). Refresh is triggered only by a relevant
  memory write.

## The two-tier read path (behavior)

A caller asks about a topic. Tier-1 resolves the topic to one or more `memories.topic[]` tags — including
the namespaced `entity:<Name>` form — and gathers **every** active memory carrying any of those tags for
the calling user. This is a complete membership traversal over an indexed array column, not a similarity
ranking, so it returns the same set every time and never silently drops a tagged memory. The memories come
back ordered by `created_at` so the caller sees the arc of how the topic evolved. The entity graph
(`entity_mentions → memories`) is a secondary gather merged in for the entity-rich subset. For many
"status" questions, this ordered, complete set *is* the answer, and it costs one indexed query.

Tier-2 is requested when the caller wants prose — a synthesized "here's where X stands" rather than a memory
list. It feeds the Tier-1 set to a Haiku synthesis pass that returns a narrative answer in which every
assertion is traceable to the memory `id`s it rests on. The result is written to a digest cache keyed by
topic, so a subsequent ask for the same topic returns the cached narrative with no LLM call.

## The digest lifecycle (behavior)

A topic digest is a derived memory, not a primary one. It is born when Tier-2 first synthesizes a topic,
lives in a cache keyed by `(user_id, topic, scope)`, and carries a `stale` flag and a `generated_at`
timestamp. When a memory is written carrying a topic tag, the write path marks that topic's digest stale.
A stale digest is never served as a fresh answer: the next Tier-2 ask for that topic re-synthesizes (lazy
refresh) or is re-materialized by an enqueued job (eager refresh); which ships in v1 is a Research Item.
Because invalidation is driven by the memory write, the digest tracks reality without any scheduled job,
and a topic nobody asks about costs nothing to keep "current."

This is deliberately the same shape as Weft's other derived layers: an evidence set plus a detector
produces a higher-order memory, materialized once and refreshed on signal. Here the evidence set is the
memory neighborhood and the detector is an LLM synthesis pass.

## Ground Truth

CONFIRMED (prod read 2026-06-24 under Jason's user_id, + source greps):
- `belief_claims` and `episode_turns` are **empty on real data** (0 rows). The belief/turn/replay
  distillation layer runs only on benchmark haystacks, never on real memory. Source: direct prod
  `count(*)` reads, this session.
- `memories` holds **4,781 active rows; 88% (4,221) carry ≥1 `topic` tag.** The tag set is a real topic
  tree: top tags `ci`(431), `loom`(285), `muttr`(200), `weft`(180), `delphi`(72), plus namespaced
  `intent:*`, `reaction:*`, `channel:*`, `entity:*` (e.g. `entity:Windward` 93, `entity:weft` 84). Source:
  prod reads, this session.
- The entity graph reaches **627/4,781 memories (13%)** via `entity_mentions` — rich where present
  (Windward 80, Mimic 55, Delphi 44) but sparse, so it is a *secondary* key, not primary. Source: prod
  reads, this session.
- `store.py` already filters memories by tag membership (`$N = ANY(topic)`) at `weft/store.py:186`, `:303`,
  `:414` — the complete-gather predicate exists. But the search functions default to `limit: int = 10`
  (`store.py:242,:367,:487`) / `= 50` (`store.py:142`), so a naive reuse would silently truncate. Source:
  `weft/store.py`.
- There is **no GIN index on `memories.topic[]`** today (no topic-index migration exists), so a complete
  `= ANY(topic)` gather seqscans without one. Source: migrations grep, this session.
- `weft_entity_context` is a complete entity→memories gather (`get_entity_memories`, `LIMIT 100`,
  token-pack — not top-k) — the precedent for the secondary path. Source: `weft/entities.py:275`,
  `weft/mcp/tools.py` (`weft_entity_context`).
- No LLM synthesis layer exists today: `weft_daily_brief`, `weft_weekly_recap`, `weft_project_status` are
  deterministic gather-group-rank, no model call. Source: `weft/skills.py`, `weft/daily_brief.py`.
- `weft_prime` implements a progressive/full disclosure split (Tier-1 sections with content; Tier-2 as
  `{count, deferred, hint}`, re-fetched via `weft_focus`) — the reusable Tier-1/Tier-2 pattern. Source:
  `weft/primer.py`.
- The belief detector runs on a Haiku-class model under an explicit per-call cost cap
  (`MAX_COST_PER_CALL_USD = 0.0034`, `weft/views/belief_detector.py:50`) — the precedent for V5. Source:
  `weft/views/belief_detector.py`.

ASSUMED:
- A free-text topic ask maps to the right `topic[]` tag(s) well enough for a complete gather (e.g. "weft" →
  `{'weft','entity:weft'}`). 88% of memories are tagged, but the *ask→tag* resolution is imperfect and is
  exactly what Compounding Loop L1 hardens. Promote to Research Item — V1's usefulness (not its mechanical
  completeness) derives from it.
- Synthesizing from raw memory content is good enough for "status" answers. Lower risk than the original
  claims-based assumption — the "raw beats honest extraction" baseline favors raw. Research Item (quality
  spike).
- A Haiku-class model is sufficient for multi-memory narrative synthesis. Unverified for this task.
  Research Item.

## Constraints Touched

- **memories search `limit` defaults** — `limit: int = 10` at `weft/store.py:242,:367,:487` and `= 50` at
  `:142`. **IN SCOPE.** Tier-1's complete gather must issue its own *unbounded* `WHERE <tag> = ANY(topic)
  AND status='active'` query (reusing the existing `= ANY(topic)` predicate at `store.py:186`), and must
  **not** route through the limit-capped search functions. Gated by the V1 bypass test.
- **No GIN index on `memories.topic[]`** — **IN SCOPE.** v1 adds one additive migration:
  `CREATE INDEX ... ON memories USING GIN (topic)`, so the complete topic-gather is index-backed rather
  than a seqscan. This is the corrected analog of the (empty) `belief_claims` prefix index; it is a small
  additive index, not a schema change.
- **`get_entity_memories ... LIMIT 100`** at `weft/entities.py:275` — **IN SCOPE (secondary path).** The
  entity-complement gather inherits this 100-cap; if a topic's entity-linked set exceeds 100 it must report
  `truncated = true` per the V1 discipline rather than silently dropping rows.
- **`MAX_COST_PER_CALL_USD = 0.0034`** at `weft/views/belief_detector.py:50` — **IN SCOPE (precedent).**
  Tier-2 defines its own named constant `MAX_SYNTH_COST_PER_CALL_USD` (initial $0.01, higher because
  narrative output > single-claim output); V5 is the gate. Named constant, not an inline literal.
- **`replay_queue` status CHECK `('pending','done','failed')`** at `weft/db/migrations/v55_...py:28` —
  **OUT OF SCOPE for v1** (v1 refresh is lazy and enqueues nothing). The deferred L2 (Usage-Tuned
  Materialization) reuses `replay_queue` and inherits this contract; named here so L2's Epic doesn't
  rediscover it.

## Validation

The implementation must satisfy:

- **V1 (completeness):** Tier-1 returns *every* active memory carrying any resolved topic tag for the
  calling `(user_id)`, ordered by `created_at`. It does **not** route through the `limit=10/50`-capped
  search functions — it is unbounded by default. Any budget cap that drops matching rows must set
  `truncated = true`; the default path does not truncate.
- **V2 (no-LLM Tier-1):** Tier-1 issues zero model calls and is deterministic — identical inputs return an
  identical memory set and ordering.
- **V3 (write-invalidation):** Writing an active memory tagged with topic T marks T's cached digest `stale`
  (or deletes it). No digest generated before that write is served as fresh afterward.
- **V4 (provenance):** Every assertion in a Tier-2 narrative cites at least one memory `id` it derives
  from; the response carries the provenance map. No uncited claims.
- **V5 (synthesis cost cap):** A Tier-2 synthesis call costs ≤ the named constant `MAX_SYNTH_COST_PER_CALL_USD`
  (initial $0.01, Haiku), mirroring `MAX_COST_PER_CALL_USD = 0.0034` (`weft/views/belief_detector.py:50`).
  Defined once, referenced by both the runtime guard and the V5 test — no inline literal.
- **V6 (cache hit is free):** Serving a fresh (non-stale) cached digest issues zero model calls.
- **V7 (isolation):** Tier-1 and Tier-2 only ever return the calling user's rows — RLS holds across the
  gather and the digest cache.

## Interfaces / Schema

**New read surface (MCP).** Working name `weft_status`:

```
weft_status(topic: str, *, synthesize=False, budget_tokens=2000)
  → {
      topic, resolved_tags: [str, ...],
      memories: [ {id, type, content, topic, created_at}, ... ],  # COMPLETE, ordered by created_at
      complete: bool, truncated: bool,
      digest: { content, provenance: {memory_id: [...spans]}, generated_at, stale } | null,
    }
```

`synthesize=False` returns Tier-1 only. `synthesize=True` returns a fresh or cached digest. The
natural-language `weft_ask("...")` front door (free-text → topic) is a Research Item, not v1.

**Topic → tag resolution (v1).** The caller supplies a topic string; v1 resolves it to a tag set by
lowercase normalization plus the `entity:<Name>` variant (`"weft"` → `{'weft','entity:weft'}`), matched via
`= ANY(topic)`. Correction-learned aliases (Compounding Loop L1) extend this set.

**New migration.** `CREATE INDEX ... ON memories USING GIN (topic)` (see Constraints Touched).

**Digest store.** A new cache keyed by `(user_id, topic, scope)` holding `content`, `provenance` (JSONB),
`generated_at`, `stale`, `detector_version`. Dedicated `topic_digests` table vs `MemoryType.digest` row is
a Research Item.

**Invalidation hook.** The memory write path (`store.write_memory` / `weft_remember`) gains a post-write
step that marks digests for the written memory's topic tags stale.

## Testing

### Unit Tests
- Tier-1 tag gather returns all active memories carrying a resolved tag; excludes `archived`/`superseded`.
- Tier-1 excludes other users' memories (RLS) (asserts V7).
- Tier-1 ordering is by `created_at`; identical inputs → identical output (asserts V2).
- Invalidation: writing a memory tagged T flips T's digest `stale` true; an unrelated topic's digest is untouched (asserts V3).
- Cache-hit path: with a fresh digest present and a mocked detector, `synthesize=True` calls the detector zero times (asserts V6).
- Cost guard: a synthesis whose projected cost exceeds `MAX_SYNTH_COST_PER_CALL_USD` is rejected/abstained, never silently overspent (asserts V5).

### Integration Tests
- End-to-end: write memories tagged `topic` → `weft_status(topic, synthesize=False)` returns the complete, ordered set matching a baseline `SELECT count(*) FROM memories WHERE status='active' AND 'topic' = ANY(topic)`.
- **`limit` bypass (Pattern 3 gate):** seed >50 (e.g. 60) active memories under one tag; `weft_status(topic, synthesize=False)` returns all 60 — proving Tier-1 does not inherit the `limit=10/50` cap.
- End-to-end **live synthesis** (substrate named — real Haiku call): `synthesize=True` returns a narrative that (a) is non-empty, (b) cites only real memory `id`s present in the Tier-1 set and ≥1 of them (asserts V4), and (c) records a per-call cost `> 0` and `≤ MAX_SYNTH_COST_PER_CALL_USD` (asserts V5 + non-degeneracy — proves the model actually ran).
- Re-materialization: after invalidation, the next `synthesize=True` produces a digest with a newer `generated_at` reflecting the new memory.
- Index use: `EXPLAIN` on the Tier-1 gather query shows the `memories` topic GIN index is used, not a seqscan.

### Acceptance Criteria
- On Weft's own real memory, `weft_status("weft", synthesize=False)` returns a memory set whose count
  equals AND is `> 0` relative to the baseline `SELECT count(*) FROM memories WHERE user_id=<me> AND
  status='active' AND ('weft' = ANY(topic) OR 'entity:weft' = ANY(topic))` (a known-populated topic,
  ~180 rows — non-degenerate by construction), with zero LLM calls (asserts V1, V2).
- `weft_status("weft", synthesize=True)` against a **live Haiku call** satisfies the live-synthesis
  integration gate (non-empty + ≥1 cited real memory + cost ∈ (0, cap]) — the binding mechanical gate.
  Separately, as a non-binding quality check, a human rates the digest against today's
  `weft_recall("status of weft")` fragment dump as answering the question where recall does not.
- A memory write tagged with a topic provably staleness-flips that topic's digest within the same
  transaction boundary (V3), shown by an integration test, not by inspection.
- No Tier-2 synthesis exceeds `MAX_SYNTH_COST_PER_CALL_USD` across the acceptance corpus (V5).

## Technical Decisions

- **Tier-1 over `belief_claims` (attribute-prefix)** — superseded by this PRD. `belief_claims` is empty on
  real data (Ground Truth); Tier-1 gathers over `memories` keyed on `topic[]` + entity graph. Ref the
  substrate finding (Weft fact, 2026-06-24).
- **"Make the replay A/B show signal" as a standalone goal** — superseded. Replay is retained as a
  benchmark apparatus and a future enumeration consumer; it is not on this feature's path. Ref handoff
  `weft-d06faa05`, anti-pattern `weft-b015d16a`.
- **Belief-tier top-k recall as the answer for synthesis/status queries** — superseded in part. Top-k
  retained for best-match lookups; status/synthesis routes to complete topic-anchored gather over
  `memories`. Ref `project_recall_completeness_diagnosis`, `project_memory_shapes_framework` (shape 4).
- **Roadmap Branch B (collections) / Branch C (weft-lang)** — retained, not affected. This PRD is a concrete
  first materialized tier the planner can later route to.

## Compounding Loops

This feature's quality should ratchet upward as it is used. Two loops are designed; one is built in v1,
one is deferred behind a Research Item. A third candidate was considered and rejected.

### L1 — Resolution Ratchet (BUILD in v1)

The feature's load-bearing risk (Ground Truth ASSUMED #1 / Research Item "ask→tag resolution") is that a
free-text topic does not always resolve to the right `topic[]` tag(s). When it misses, the ask returns
empty. That miss is exhaust; this loop turns it into a permanent correction so the same topic never misses
twice.

```
LOOP BLUEPRINT — Resolution Ratchet
══════════════════════════════
Family:   ops-ratchet / correction-pattern
SIGNAL:   Each weft_status ask logs (user_id, topic_string, resolved_tags, memory_count, was_empty). A
          "resolution correction" = a later ask whose topic_string normalizes to a prior was_empty token
          but resolves (via operator/agent-supplied tag) to a NON-empty set. Partially EXISTS
          (weft_recall_queries v50 retrieval log); NEEDS a small extension to log resolved_tags + was_empty
          on the new surface (size S).
STORE:    topic_resolution_aliases (user_id, topic_token, resolved_tags TEXT[], hit_count, source ∈
          {learned, manual}, updated_at). One row per corrected token.
FEEDBACK: The resolver consults topic_resolution_aliases BEFORE naive normalization. A topic that once
          returned empty and was corrected now resolves to the working tag set automatically — human is
          judge of the right tag once, system re-applies forever. Automatic on every subsequent ask;
          human-as-judge, never human-as-pump.
PROOF:    Alive: empty-result rate for repeated tokens declines; a token with an alias row never logs
          was_empty=true again. Dead tell: topic_resolution_aliases grows but per-token empty-rate is flat
          — the resolver isn't consulting the map.
Payback:  Second ask of any given topic. Effectively immediate.
Cost:     S. One table + a resolver lookup + extending the existing query log. Deterministic, no LLM, no
          Pinch flag.
done_when: Seed 5 topics whose naive normalization misses (the memory is tagged with a different stem than
          the topic word). Issue each as weft_status → all 5 return was_empty=true. Log the corrected tag
          set for each (source='manual'). Re-issue the same 5 asks → all 5 return the non-empty memory set
          via the alias path, and counter `topic_resolution.alias_hits` == 5. A 6th control topic with no
          alias still resolves via naive normalization (alias path not consulted for it).
══════════════════════════════
```

### L2 — Usage-Tuned Materialization (DESIGNED, deferred behind the lazy-vs-eager Research Item)

```
LOOP BLUEPRINT — Usage-Tuned Materialization
══════════════════════════════
Family:   data-flywheel
SIGNAL:   Per-topic ask frequency (count + recency), from the L1 query log. EXISTS once L1 logging ships.
STORE:    Ask-frequency rollup per (user_id, topic). Reuses the L1 log; no new store.
FEEDBACK: "Hot" topics are eagerly re-materialized on memory write via an enqueued job; "cold" topics stay
          lazy (re-synthesize on next read after stale). Materialization effort follows demand.
PROOF:    Alive: cache-hit rate on the top-N most-asked topics rises and their p95 read latency falls,
          while total synthesis spend stays bounded. Dead tell: synthesis spend scales with total topics,
          not ask volume.
Payback:  ~Tens of asks across a stable hot set.
Cost:     M if it adds a background re-materialization worker — PINCH FLAG: route through /pinch before
          building. Deferred until the "materialization mode: lazy vs eager" Research Item resolves
          (eager refresh is its precondition).
done_when: With L1 logging live and a seeded hot/cold split (1 topic asked 20×, 5 asked once), after a
          memory write under each, the hot topic's digest is re-materialized within one drain cycle while
          the cold topics remain stale-until-asked; counter `digest.eager_rematerialize` counts only the
          hot topic.
══════════════════════════════
```

### Rejected

- **Digest self-calibration** (grade each digest against the next memory that contradicts/extends it).
  Killed: "ground truth" for a status narrative is ill-defined; the signal is a churn metric, not
  behavior-changing feedback — a dashboard, not a loop. Not built.

**Verdict: SEEDED.** L1 is a real loop with a runnable PROOF, built in v1; L2 is designed and gated behind
its precondition. Not COMPOUNDING (nothing fires until shipped), not FLAT.

## Research Items

- **CLOSED — Substrate audit (was "Topic → attribute-prefix fidelity").** Prod read 2026-06-24:
  `belief_claims`/`episode_turns` empty; `memories` (88% topic-tagged) + entity graph are the substrate.
  Resolved into the re-pointed design above. No longer open.
- **Ask→tag resolution fidelity.** How often does a free-text topic resolve to the right `topic[]` tag(s)
  without a learned alias? Audit by sampling real asks; this sizes how much work L1 (Resolution Ratchet)
  must absorb before resolution is reliable. Gates V1's *usefulness* (not its mechanical completeness).
- **Synthesis fidelity from raw memories.** Spike: synthesize a known topic ("weft") from the raw memory
  set, human-rate coherence and provenance accuracy. Favored by the "raw beats extraction" baseline but
  unverified for the synthesis task specifically.
- **Materialization mode: lazy vs eager.** Lazy (re-synthesize on next read after stale) vs eager (enqueue
  a job on write). Trade cost (lazy wins for cold topics) vs read latency (eager wins for hot). Pick one
  for v1; gates L2.
- **Digest store shape.** Dedicated `topic_digests` table vs `MemoryType.digest` row. Resolve at Epic time
  (schema-now/logic-later lean toward a dedicated table).
- **Surface naming + the `weft_ask` front door.** v1 ships `weft_status(topic)`; whether to also add a
  natural-language `weft_ask(query)` with intent→topic resolution (closer to Branch C, heavier) is open.
  Default: `weft_status` only in v1.
