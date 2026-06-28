**LLM GROUNDING:** Implements `documents/prds/weft-memory-v2.md` (Phase 0 only) for the Weft project. Downstream Tasks must be merge-ready: each Task leaves the repo compiling, `uv run pytest tests/` green, and safe to merge independently. Open Questions must not survive into implementation. This Epic pins: (1) the meter ships before the machine — Phase 0 is the reconciliation layer + correctness floor, no Phase-1 enumeration-router or catalog-key work leaks in; (2) all changes are additive ALTERs / new tables (boot-time migration model, current head v62); (3) the write path stays LLM-free; (4) `loc_key`, collections, temporal, and `weft_fingerprint` are explicitly NOT in this Epic. Phase 0 is independent of the in-flight Topic-Digest Recall epics (Program A) and runs in parallel with them.

## Summary

Six deliverables that make every later never-miss claim *checkable*: a two-tier entity-resolution fix, end-to-end truncation surfacing, a deterministic vector tie-break, a known-membership enumeration eval harness, the recall-canary reconciliation loop, and `weft_fsck`. Four are cheap correctness fixes (the "floor"); two are the net-new reconciliation "meter." None produces a demo — that is why this Epic ships first.

## Core Decisions

- Entity resolution moves from single 0.6 threshold to two-tier: ≥0.85 auto-merge, 0.6–0.85 → review candidate (no silent merge). [CONFIRMED: single threshold today at `weft/ingest_pipeline.py:359`]
- Truncation is surfaced end-to-end, never swallowed: the cap at `_ENTITY_MEMORIES_LIMIT = 100` and any budget cap set `truncated=true`, propagated through the MCP response. [CONFIRMED: flag already returned by `weft/topic_gather.py:213`; cap at `:34`]
- Every vector `ORDER BY embedding <=> $1::vector` gains a deterministic `id` tie-break. [CONFIRMED: no tie-break today at `weft/store.py:371`, `:694`; FastEmbed is local/deterministic so this closes the last nondeterminism source]
- The recall canary is the headline reconciliation loop; it bootstraps from the already-shipped re-ask-miss signal before adding active probing. [CONFIRMED: `is_reask_miss`/`reask_satisfying_memory_id` exist via `v52`, `weft/reask.py`, `weft/replay.py`]
- `weft_fsck` defines an orphan as a memory reachable ONLY by vector cosine — no tag, `entity_mentions`, `episode_memories`, or collection edge. [CONFIRMED join tables: `entity_mentions` `v12_entities_tables.py:31`, `episode_memories` `v11_episodes_tables.py:26`; collections do not exist yet so that edge is vacuously absent this phase]
- Canary probe fidelity is unproven (PRD RI-4 / ASSUMED-4) — the "originating context" may yield false misses. [ASSUMED — also a Critical Implementation Note; do not trust the `canary_miss` rate as a defect metric until calibrated]
- Eval fixtures use the synthetic persona "Jim Boblaw," never real names. [CONFIRMED: Weft `feedback_synthetic_test_personas`]

## The floor vs the meter

The four floor fixes (entity threshold, truncation, tie-break, eval harness) are independent of each other and of the meter — they can land in any order and parallelize cleanly. The two meter deliverables (canary, `weft_fsck`) depend on the floor: the canary's daily audit needs the deterministic tie-break (§V3) to make "fixed materialization" actually fixed, and the eval harness (§V4) is where canary-detected misses are minted into regression cases (Compounding Loop CL1). So the build order is floor-first, meter-second, with the tie-break on the critical path to the canary.

## Critical Implementation Notes

- **Two-tier resolution must not break the existing single-match selection.** `resolve_entities` today calls `search_entities(..., limit=3, threshold=0.6)` and picks the top match (`weft/ingest_pipeline.py:307-360`). The candidate tier needs a place to land 0.6–0.85 pairs (a `status='candidate'` column or a review table) without auto-linking the mention — verify the ingest path still produces a usable entity link for the ≥0.85 case and a *new* entity (not a wrong merge) for the sub-0.6 case.
- **Truncation is partly plumbed already — fix the whole chain, not just `topic_gather`.** The flag is returned at `weft/topic_gather.py:213`, but the `_ENTITY_MEMORIES_LIMIT=100` entity path and the MCP tool response must both carry it to the agent. Grep every caller of the gather and the entity-context path; a swallowed flag anywhere fails §V2.
- **Both vector ORDER BYs need the tie-break.** `weft/store.py:371` AND `:694`. A tie-break on one leaves a nondeterministic path that will silently corrupt canary "fixed materialization."
- **Canary probe fidelity (ASSUMED-4 / RI-4).** Build the canary to first consume the *proven* re-ask-miss signal (`is_reask_miss`), then add active known-answer probing behind a flag, and calibrate false-miss rate before exposing `canary_miss` as a metric. Do not let an uncalibrated probe rate become a `done_when` (PRD hard rule on ASSUMED premises).
- **`weft_fsck` reachability must check ALL edge types or it over-reports.** Orphan = NOT in `topic[]` (non-empty) AND no `entity_mentions` row AND no `episode_memories` row. Missing any join check floods the orphan list with false positives and kills trust in the tool.
- **Write path stays LLM-free.** Canary enrollment happens at `weft_remember` (`weft/mcp/tools.py:228-327`); enrollment must be a cheap insert, not a model call (Vcost).

## Merge & Validation

Migrations apply on app boot via `run_migrations()` (current head `v62`); new tables/columns are additive `IF NOT EXISTS` migrations (`v63+`). Each Task: implement → `uv run pytest tests/ -v` green → merge. The eval harness (`benchmarks/enumeration_eval/`) and the canary daily job are runnable independently of the MCP server. Order: floor fixes (parallel) → tie-break confirmed → canary + fsck.

## Task Plan

1. Two-tier `resolve_entities` (≥0.85 auto / 0.6–0.85 candidate) + candidate-review surface; preserve ingest link behavior. (§V1)
2. Surface `truncated=true` end-to-end from `topic_gather` + the `_ENTITY_MEMORIES_LIMIT` entity path through the MCP response. (§V2)
3. Add deterministic `id` tie-break to both vector `ORDER BY` sites in `store.py`. (§V3)
4. `benchmarks/enumeration_eval/` harness + Jim-Boblaw known-membership fixtures; report min/median/max `recall@membership`, oracle (gather) single-run, NL candidate k≥5. (§V4)
5. Recall-canary enrollment at `weft_remember` + daily fixed-materialization audit + `canary_miss` counter; bootstrap from `is_reask_miss`. (§V5)
6. CL1 wiring: a canary/re-ask miss auto-mints an eval case `(query → satisfying_memory_id)` into the harness. (PRD Compounding Loops CL1)
7. `weft_fsck` (CLI + MCP) returning vector-only-reachable memories across all edge types. (§V6)

## Testing Standard

"Green" = `uv run pytest tests/` passes including the new cases below. Acceptance gates are the §V1–§V6 tests in the PRD's Testing section, restated nowhere here — they are the single source of truth. The two binding mechanical gates for Phase 0 closure:
- `benchmarks/enumeration_eval/` reports `min recall@membership == 1.0` for the deterministic gather over every fixture, in a single runnable command (§V4).
- The canary cycle runs with `probes_checked > 0` and detects a deliberately-planted below-cutoff probe as a `canary_miss` while a surfacing probe is not (§V5, non-degenerate).

## Technical Decisions

- Single 0.6 entity threshold (`weft/ingest_pipeline.py:359`) — superseded by this Epic (two-tier).
- Raw-distance vector ordering with no tie-break (`weft/store.py:371,:694`) — superseded by this Epic.
- `belief_claims`-based enumeration / EPIC 2 (`loom-dcfaf656`) — superseded at the PRD level; out of scope here.

## Out Of Scope

- The enumeration-intent router, `loc_key`/`loc_registry`, AST `requires`/`weft_portfolio_query`, and collections — all Phase 1 (depend on Program A landing). Not in this Epic.
- Bi-temporal `valid_from`/`valid_to`, the `is_a` taxonomy, and `weft_fingerprint` — Phase 2 / deferred.
- Active model-escalation for area assignment (Vcost) — deferred to the nightly pass, not Phase 0.
- The LongMemEval gate (`loom-88799e2c`) — retained as a post-Phase-1 measurement milestone, not Phase 0.
