# Weft Recall Completeness — Fix Hybrid Retrieval So It Surfaces Its Own Memories

## Summary

Weft's hybrid recall silently fails to surface high-value memories — demonstrated live when `weft_recall`/`weft_search_all` could not return the canonical Memory-v2 decision (`weft-5276bf05`) for a query built from that memory's *own* vocabulary. Root cause is three compounding retrieval failures: a conjunctive keyword channel that's effectively dead, content-only embeddings that dilute long memories, and a top-k cliff that drops anything the first two demote. This PRD fixes the retrieval path so a memory is reliably findable by its own terms, and wires the fix into the existing recall-canary so the property is *measured*, not hoped for.

## Goals

- Make `search_by_keyword` recover documents that match *any* salient query term, ranked by overlap — restoring the keyword half of hybrid RRF fusion (validates per Validation §V1, §V2).
- Make memory embeddings include curated `topic` tags (and optionally `type`) so the highest-signal terms are not diluted out of the vector (validates per §V3).
- Guarantee that a memory is retrievable by a query composed of its own topics/keywords — the regression property that currently fails (validates per §V4).
- Keep the write path free (no LLM on the `weft_remember` hot path) and keep the fix safe against arbitrary query input (validates per §V5).
- Instrument the fix with the existing recall canary so a future regression is caught automatically, not by a human happening to remember a specific memory.

## Non-Goals

- **No re-architecture of the hybrid/RRF fusion algorithm.** RRF is rank-based and sound; the bug is in the inputs it fuses, not the fusion. Touching RRF would widen blast radius without addressing root cause.
- **No new embedding provider or model change.** The provider (`OpenAI text-embedding-3-small @768d`) is fine; the defect is *what text* we embed, not which model. Swapping models is a separate, larger decision.
- **No multi-vector / chunked embeddings in this PRD.** Chunking long memories is a plausible deeper fix for RC2 dilution, but it's a schema and write-path change of much larger scope; include topics first (cheap, high-leverage) and measure before committing to chunking. Tracked as a Research Item.
- **No change to the `loc_key` catalog / Memory-v2 build arc.** That's a separate initiative (`weft-858c6283`); this PRD only restores the recall the rest of the system already assumes works.
- **No raw `to_tsquery(user_text)` passthrough.** `to_tsquery` throws on unsanitized punctuation; passing raw query text would crash recall. The OR-query must be built from sanitized lexemes (see Critical behavior, §Keyword Channel).

## Keyword Channel — restore disjunctive matching (RC1)

Today `search_by_keyword` filters and ranks with `plainto_tsquery('english', query)`, which ANDs every lexeme: a document must contain *every* query term to match at all. For a 7-term query this matched exactly **1 document in a 2000+ memory corpus**, so the keyword arm of hybrid fusion contributes essentially nothing — and cannot rescue a memory that vector search has demoted.

The fix replaces the conjunctive query with a **disjunctive** one: tokenize the query into lexemes, sanitize them to safe alphanumeric terms, and build an OR-joined `to_tsquery` (`term1 | term2 | ...`). Ranking stays `ts_rank(search_tsv, <or_query>)`, so a document matching more (and rarer) terms still ranks above one matching a single common term. Precision is preserved by *ordering*, not by *exclusion* — which is exactly what rank-based RRF fusion downstream expects. Both the `@@` filter (`store.py:446`) and the `ts_rank` expression (`store.py:514`) must switch together; changing only one re-introduces the contradiction.

The sanitization step is load-bearing, not cosmetic: `to_tsquery` parses tsquery operator syntax and raises on stray punctuation, quotes, or operators in raw input. The query must be split in Python to alphanumeric lexemes, empty tokens dropped, and only then OR-joined — never interpolated raw. An empty post-sanitization token list must short-circuit to "no keyword matches" rather than emit an invalid query.

## Vector Channel — embed topics, not just content (RC2)

Memory embeddings are currently generated from `content` alone (`weft/mcp/tools.py:304`). A long, broad memory's single averaged 768-d vector is dominated by generic prose, so its specific, curated `topic` tags — the very terms a future query is most likely to use — contribute *nothing* to the vector. Measured: the canonical decision scored only **0.4458 cosine** against a query of its own keywords and ranked #17 in pure vector search.

The fix composes the embedded text as `content` plus the memory's `topic` tags (and optionally `type`), via a **single shared helper** so every write site and the re-embed backfill produce identical text. The composition must be defined once and reused, because any divergence between write-time and re-embed-time text produces vectors that disagree with each other — a silent drift bug. Applying this requires a one-time re-embed of existing memories; the `re-embed` CLI and the dimension self-heal path already exist and already support composite embed-text (episodes embed `title + summary`), so this follows an established pattern rather than inventing one.

## Top-K Cliff (RC3)

No independent fix. With the keyword channel dead and the vector channel diluted, the target landed at hybrid rank #17 while recall returns ~top-10 — so it fell off the edge. Once RC1 and RC2 lift its rank into the returned window, the cliff stops mattering. This PRD does not change the default `limit`; the Validation gates assert the target now lands *within* the existing window, which is the real proof.

## Ground Truth

CONFIRMED:
- `search_by_keyword` filters with `plainto_tsquery('english', $idx)`. Source: `weft/store.py:446`.
- `search_by_keyword` ranks with `ts_rank(search_tsv, plainto_tsquery('english', $1))`. Source: `weft/store.py:514`.
- `search_by_keyword` is the single keyword entry point for memories — used by `search_hybrid` (`store.py:603`), `turn_recall` (`turn_recall.py:313`), and the `weft_recall` keyword fallback (`mcp/tools.py:837`). Source: grep 2026-06-29.
- `search_tsv` already includes topics: `to_tsvector('english', content || ' ' || array_to_string(topic,' '))`, maintained by an INSERT/UPDATE trigger. Source: `weft/db/migrations/v19_tsvector_fts.py`.
- Memory write path embeds `content` only. Source: `weft/mcp/tools.py:304` (`embedding = await app.embedding.embed(content)`).
- Empirical fix proof: switching the 7-term query from AND (`plainto_tsquery`) to OR (`to_tsquery` joined with `|`) changed corpus matches from **1 → 2192** and moved `weft-5276bf05` from **absent → keyword rank #1 in-project**. Source: probe run 2026-06-29 (memory `weft-45029c15`).
- True cosine(query, `weft-5276bf05`) = **0.4458**; pure-vector rank **#17/40**; hybrid rank **#17/30**. Source: same probe.
- Provider is `OpenAI text-embedding-3-small`, 768 dims (Matryoshka), no query prefix required. Source: `weft/config/__init__.py:84-86`, `weft/embeddings/openai.py`.
- Hybrid `similarity` field is normalized RRF score, not cosine. Source: `weft/store.py:657`.
- A recall canary (per-memory known-answer probe + daily fixed-materialization audit) and an `enumeration_eval` harness already exist as the regression substrate. Source: migration `v63_recall_canary.py`; memory `weft-45029c15`.
- The repo gates real-LLM/real-API tests behind `WEFT_RUN_LLM_EVAL=1` + a real key (mirrors `weft.views.belief_detector` live tests). Source: memory `weft-9e85a6f2` (topic-digest acceptance pattern); AC1's real-embedder gate uses this convention.
- Canary active probing is gated by `active_probing_enabled: bool = False` (RI-4 calibration gate); active-probe query is the answer memory's content truncated to `PROBE_TEXT_MAX_CHARS=512`; hit window is `DEFAULT_AUDIT_TOP_K=10`. Source: `weft/canary.py:314,64,68`.

ASSUMED:
- `episode_turns.py:516` (`websearch_to_tsquery`) may share the same AND-death, since `websearch_to_tsquery` also ANDs unquoted terms. Unverified — promote to Research Item RI-1; no `done_when` derives from it.
- Including `topic` (and `type`) in embed text raises the target's cosine enough to clear the top-k window on its own. Unverified — RC1 alone already surfaces the target, so no acceptance gate depends on RC2's standalone effect (RI-2).

## Constraints Touched

Enforced constants within one hop of the modified components, with scope decisions:

- `DEFAULT_AUDIT_TOP_K = 10` at `weft/canary.py:68` — **IN SCOPE (binding, unchanged).** This is the canary "hit" window and the same value as the default recall `limit`; it is what makes "top-10" the pass condition shared by V4, AC1, and AC2. Named so the binding is explicit, not coincidental.
- `PROBE_TEXT_MAX_CHARS = 512` at `weft/canary.py:64` — **IN SCOPE (binding, unchanged).** An active probe's query is the answer memory's content truncated to this length — a *different* query than AC1's hand-crafted topic string. AC2 gates this query specifically; the two must not be conflated.
- default recall `limit = 10` at `weft/store.py:265` (vector) and `:414` (keyword) — **IN SCOPE (unchanged).** AC3 asserts no change; AC1 asserts the target lands within this window.
- `_RRF_K = 60` at `weft/store.py:543` — **OUT OF SCOPE.** RRF fusion is explicitly retained (Non-Goals, Technical Decisions); not touched.
- `embed_composition_version` (new, see Interfaces) — **IN SCOPE (added).** The completeness marker AC5 checks.

## Validation

Effective retrieval must satisfy:
- **V1:** `search_by_keyword(q)` returns a document that shares *any* salient lexeme with `q`, not only documents containing every lexeme. A multi-term query returns > 1 result whenever > 1 document shares a term.
- **V2:** Keyword results are ordered by `ts_rank` so higher term-overlap ranks higher; the single-term-only matches do not outrank multi-term matches.
- **V3:** A memory's embedding is generated from text that includes its `topic` tags, produced by the same shared helper at both write time and re-embed time (identical bytes for identical input).
- **V4 (the regression property):** A query composed of a memory's own topics/keywords returns that memory within the default recall window. Concretely: `weft_recall("memory v2 redesign loc_key catalog code library dependency")` returns `weft-5276bf05` in the top-10 (face mode, weft project).
- **V5:** Recall never raises on arbitrary query input (punctuation, operators, quotes, empty-after-sanitization); such input degrades gracefully to "no keyword matches," and hybrid still returns vector results.

## Interfaces / Schema

- **RC1 needs no schema change.** `search_tsv` already indexes topics; the keyword fix is query-construction only.
- **RC2 adds one small column:** `memories.embed_composition_version SMALLINT NOT NULL DEFAULT <current>` (a one-line migration), set on every write and bumped by the backfill. This is the marker AC5 checks for completeness — without it, "did the re-embed finish?" is unanswerable. This is the *only* schema change in the PRD; it does not alter the embedding column or `search_tsv`.
- New internal helper (single source of truth for embed text), e.g. `embed_text_for_memory(content, topic, type) -> str`, called by every memory write site **and** by the re-embed backfill. The helper and the `<current>` composition-version constant move together: bumping the composition is what bumps the version.
- Internal query-builder for the disjunctive tsquery (tokenize → sanitize → OR-join), used inside `search_by_keyword`.
- One-time operational step: re-embed existing memories using the new composition (existing `re-embed` CLI), which sets `embed_composition_version` to `<current>` per row.

## Testing

### Unit Tests
- `search_by_keyword`: a 3-term query where each of three distinct documents contains exactly one of the terms returns all three (proves OR, not AND).
- `search_by_keyword`: a document containing two query terms ranks above a document containing one (proves §V2 ordering).
- Query sanitizer: inputs with `&`, `|`, `!`, `:`, `()`, quotes, and unicode punctuation produce a valid OR `to_tsquery` and never raise.
- Query sanitizer: a query that is all punctuation (empty after sanitization) returns no keyword matches and does not emit an invalid query (proves §V5).
- `embed_text_for_memory`: identical `(content, topic, type)` yields byte-identical text across two calls; topics appear in the output (proves §V3).

### Integration Tests
- Seed a long, broad memory with topics `[loc-key, code-library]` plus N distractor memories; assert a query of those topics returns the seeded memory in `search_by_keyword` top-1 and in `search_hybrid` top-k (real DB, real tsvector trigger; keyword/fusion path needs no API key).
- Write a memory via the real write path, then assert its stored embedding equals `embed(embed_text_for_memory(...))` (proves write path uses the helper). **Real-API test, gated `WEFT_RUN_LLM_EVAL=1`** (calls the real embedder).
- Re-embed a memory via the CLI and assert the resulting vector equals the write-path vector for the same row (proves write/re-embed parity, §V3). **Real-API test, gated `WEFT_RUN_LLM_EVAL=1`.**

### Acceptance Criteria
- **AC1 (real-query gate):** Against the live schema with the real embedder (real-API test, gated `WEFT_RUN_LLM_EVAL=1` per the repo convention), `weft_recall("memory v2 redesign loc_key catalog code library dependency")` (face mode, weft project) returns `weft-5276bf05` within the top-10 **AND** a designated irrelevant control memory is **not** in the top-10 **AND** the result set is bounded to the default `limit` (no widening). If a fixture substitutes for the live memory, it must carry long, dilution-representative content (≥ the real memory's character length) and a real embedding — a short fixture is a stub-pass and does not satisfy AC1. This is the hand-crafted-topic query; the canary's content-truncated probe is gated separately in AC2.
- **AC2 (canary-probe gate):** An active recall-canary probe whose answer is `weft-5276bf05` — `probe_text` = the memory's own content truncated to `PROBE_TEXT_MAX_CHARS` (512, `weft/canary.py:64`) — returns that memory within `DEFAULT_AUDIT_TOP_K` (10, `weft/canary.py:68`) on the fixed-materialization audit, i.e. `canary.miss` does not increment for it. This is a **different query** than AC1 (truncated content, not topics); both must pass, because the loop's `done_when` relies on the probe query, not AC1's.
- **AC3:** Full suite green (currently ~3174+ passing) and ruff clean; no change to default recall `limit` (`store.py:265/414`, =10).
- **AC4:** Recall does **not** raise across a fuzz set of punctuation/operator/quote/empty-after-sanitization queries (§V5); each such query returns vector results (or a bounded empty result), asserted by a test in which `pytest.raises` is *not* triggered. (Corrects the earlier inverted wording — §V5 requires graceful degradation, never a raise.)
- **AC5 (re-embed completeness gate):** After the backfill, `SELECT count(*) FROM memories WHERE embed_composition_version < <current>` returns 0 — every memory's embedding was generated from the `content + topics` composition (see Interfaces). Non-degeneracy: the count of rows whose embedding actually changed during backfill is `>= ` the number of memories with non-empty `topic` (a no-op backfill that changed nothing fails this gate).

## Compounding Loops

*Groundhog verdict: SEEDED. The substrate already exists — this fix's job is to **activate** a loop that's built but switched off, and to **move the needle** on a loop that's already live. No new instrumentation of significant size; both loops consume exhaust the system already emits.*

The pivotal finding: Weft's active recall canary (per-memory known-answer probe + daily fixed-materialization audit) is **fully built but disabled** (`active_probing_enabled: bool = False`, `weft/canary.py:314`), gated behind RI-4 calibration. It is the exact mechanism that would have caught "the v2 decision stopped surfacing for its own keywords" — and it's off precisely *because* recall is currently bad enough that synthetic probes false-positive too often to trust. Fixing recall (RC1+RC2) is the calibration event that earns flipping it on. That is the loop.

```
LOOP BLUEPRINT — Recall Regression Ratchet
══════════════════════════════
Family:   Ops Ratchet (a recall regression cannot silently recur)
SIGNAL:   active-probe miss — a memory's known-answer probe fails to return that
          memory in top-k. EXISTS at weft/canary.py (run_canary_audit, probe_types
          gate L408); currently emits nothing because active_probing_enabled=False.
STORE:    recall_canary table (probes) + canary.miss counter. EXISTS.
FEEDBACK: a miss (a) auto-mints an enumeration_eval case (_try_mint_eval_case,
          canary.py:89 — EXISTS) and (b) trips the audit defect log. The NEW
          feedback this PRD closes: with recall fixed, active_probing_enabled is
          flipped TRUE, so the loop actually fires — future regressions (someone
          reverts the tsquery to AND, or a re-embed drops topics) are caught by
          the daily audit instead of by a human happening to remember one memory.
PROOF:    recall_canary active-probe miss-rate over enrolled memories. Tell-it's-dead:
          active_probing_enabled still False after the fix (loop built, never switched
          on) — OR miss-rate computed but no probe exists whose answer is weft-5276bf05.
Payback:  first run after enablement (every enrolled memory audited daily). Not
          run-count-gated — payback is immediate once the switch flips.
Cost:     S. Reuses existing canary infra; the change is a calibrated flag flip plus
          one seeded probe. No Pinch flag.
done_when: After RC1+RC2 land, run_canary_audit with active_probing_enabled=True over
          >= N enrolled probes (N pinned by RI-4 calibration so a low miss-rate can't
          come from a near-empty probe set) reports miss-rate < T, where T is the
          RI-4-calibrated threshold (cite the resolved value here once RI-4 closes),
          AND a probe with answer memory_id=weft-5276bf05 returns that memory
          (canary.miss does not increment for it) across two consecutive daily audits.
          The active_probing_enabled default flip is committed, not left to a human.
          NOTE: T and N derive from RI-4 — this loop's enablement is BLOCKED on RI-4
          closing, so the flag flip is the last step, not a precondition.
══════════════════════════════
```

```
LOOP BLUEPRINT — Re-ask Correction (already live; this fix must move it)
══════════════════════════════
Family:   Correction Pattern / Self-Calibration
SIGNAL:   is_reask_miss — Jason or an agent re-issues a query because the first
          recall missed, and a satisfying memory is later recorded. EXISTS
          (weft_recall_queries.is_reask_miss; compute_reask_rate at reask.py:133).
STORE:    weft_recall_queries rows; surfaced as reask_rate in weft_check_health
          (mcp/tools.py:3487/3533). EXISTS — this loop already runs.
FEEDBACK: re-ask misses auto-enroll as reask-bootstrap canary probes (PROVEN
          known-answer cases) AND feed usefulness/calibration (loom-c5ddc68f).
          This PRD does not add feedback here — it is the INTERVENTION the loop
          measures: a real recall fix should reduce real re-ask misses.
PROOF:    reask_rate over the window after the fix vs before. Tell-it's-dead:
          reask_rate flat across the fix → the fix moved a synthetic probe but
          not real-query behavior; the diagnosis was incomplete. This is the
          honest falsifiable check that the fix helped *Jason*, not just the test.
Payback:  ~14 days of normal usage (enough re-ask events to compare windows).
Cost:     none — pure measurement against an existing metric.
done_when: reask_rate measured over the 14 days following the fix is strictly
          lower than the 14 days preceding it (same weft_check_health metric);
          if not strictly lower, a follow-up investigation task is filed rather
          than declaring the fix successful.
══════════════════════════════
```

**Killed candidate — LongMemEval re-run (RI-3).** A post-fix LME re-run validates the benchmark hypothesis, but it is a **one-shot measurement, not a loop** — LME is not run as exhaust of normal operation, nothing accumulates per-run, and no behavior auto-changes from it. It stays an Acceptance/Research item (RI-3), not a compounding loop. Naming it honestly so it isn't dressed up as a flywheel.

## Technical Decisions

- Original `plainto_tsquery` choice in `search_by_keyword` (`store.py:446/514`) — **superseded by this PRD** (replaced with sanitized disjunctive `to_tsquery`).
- Content-only embedding at `tools.py:304` — **superseded by this PRD** (replaced with shared `content + topics` composition).
- RRF fusion (`search_hybrid`, `store.py:546`) — **retained, not affected**.
- `search_tsv` composition (v19, already includes topics) — **retained**; confirms the keyword fix needs no schema change.

## Research Items

- **RI-1:** Does `episode_turns.py:516` (`websearch_to_tsquery`) suffer the same AND-death? Investigate whether the turn-recall keyword path needs the same disjunctive fix and whether the two paths should share one query-builder. Blocks nothing in this PRD; determines whether scope should widen.
- **RI-2:** Measure RC2's standalone lift — does adding topics (and type) to embed text raise the target's cosine enough to clear top-k *without* RC1? Needed to decide whether chunked/multi-vector embeddings (the deferred deeper fix) are warranted. Run after RC1 lands so the two effects are separable.
- **RI-3:** Validate the LongMemEval hypothesis — re-run the multi-session/temporal subset after RC1 to confirm whether the AND-semantics keyword death was suppressing benchmark recall. This is the bridge between fixing production recall and moving the held LME gate (`loom-88799e2c`).
- **RI-4:** Should `search_hybrid` surface the true cosine alongside the normalized RRF score, so future diagnostics aren't misled by saturated ~1.0 values? Cosmetic but diagnostic-relevant.
