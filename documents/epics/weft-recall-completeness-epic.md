**LLM GROUNDING:** Implements `documents/prds/weft-recall-completeness.md` for the Weft memory system. Downstream Tasks must be merge-ready: each Task leaves the repo compiling, full suite green, ruff clean, and safe to merge independently. Open Questions must not survive into implementation. This Epic pins: (1) RC1 is a *sanitized disjunctive* `to_tsquery` — never raw passthrough; (2) RC2 routes ALL memory-content embedding through one shared helper, write-path and re-embed identical, fronted by an `embed_composition_version` marker; (3) AC1 (topic query) and AC2 (content-truncated canary probe) are TWO distinct gates, both required; (4) activating the canary loop (`active_probing_enabled`) is BLOCKED on RI-4 calibration and is NOT part of this build.

## Summary

Restore Weft hybrid-recall completeness in two independent code changes plus a one-time backfill: RC1 switches `search_by_keyword` from conjunctive `plainto_tsquery` to a sanitized disjunctive `to_tsquery` (no migration, no re-embed — ship first); RC2 embeds `content + topics` via a single shared helper, gated by a new completeness-marker column and a re-embed backfill. RC1 and RC2 are separable; RC1 alone is the empirically-verified fix that surfaces the failing memory.

## Core Decisions

- RC1: replace `plainto_tsquery('english', $)` at BOTH `weft/store.py:446` (filter) and `:514` (rank) with a disjunctive `to_tsquery` built from sanitized lexemes. Both sites change together or not at all. [CONFIRMED: PRD Ground Truth, `store.py:446/514`]
- RC1 query construction lives in one internal builder (tokenize → sanitize to alphanumeric lexemes → drop empties → OR-join with `|`); empty post-sanitization → short-circuit to no keyword matches, never an emitted query. [CONFIRMED: PRD §Keyword Channel — also a Critical Implementation Note below]
- RC2: all memory-content embeddings flow through `embed_text_for_memory(content, topic, type) -> str` (single source of truth), called at every write site AND the re-embed backfill; the function and the `<current>` composition-version constant move together. [CONFIRMED: PRD §Vector Channel + Interfaces]
- RC2 adds `memories.embed_composition_version SMALLINT NOT NULL DEFAULT <current>` — the only schema change; set on write, bumped by backfill, checked by AC5. Does not touch the embedding column or `search_tsv`. [CONFIRMED: PRD Interfaces]
- AC1 (hand-crafted topic query via `weft_recall`) and AC2 (active canary probe = content truncated to `PROBE_TEXT_MAX_CHARS=512`) are different queries with different pass paths; both gated, both required. [CONFIRMED: PRD Constraints Touched, `canary.py:64/68`]
- `active_probing_enabled` flag flip is BLOCKED on RI-4 calibration (threshold T, probe-count N) and is explicitly out of this build. [CONFIRMED: PRD Compounding Loop 1 + RI-4]
- Provider unchanged (`OpenAI text-embedding-3-small @768d`); real-API tests gated `WEFT_RUN_LLM_EVAL=1`. [CONFIRMED: PRD Ground Truth, memory `weft-9e85a6f2`]

## Critical Implementation Notes

- **THE foot-gun — `to_tsquery` sanitization.** `to_tsquery('english', raw_text)` parses tsquery operator syntax and RAISES on stray punctuation, quotes, `&|!():*`, or apostrophes. `plainto_tsquery`/`websearch_to_tsquery` are safe precisely because they sanitize. The builder MUST split in Python to alphanumeric lexemes and OR-join — never interpolate raw query text. A fuzz test over punctuation/operator/empty inputs (AC4) is the guard; a single unsanitized passthrough is a production crash on real queries.
- **Both query sites must change together.** `:446` is the `@@` membership filter, `:514` is the `ts_rank` ordering. Changing only one re-introduces the AND/OR contradiction (e.g., OR-filter but AND-rank, or vice-versa). Verify both in the same task.
- **Write/re-embed composition parity is load-bearing.** If the write path and the re-embed script compose embed text even slightly differently, the corpus splits into two disagreeing vector representations — a silent recall-quality bug invisible to any single test. The shared helper is the mechanism; the parity integration test (PRD line 100) is the proof.
- **Transitional mixed-embedding state.** During the backfill, some rows are `content+topics`, some still `content`-only. This is acceptable mid-migration but must converge: `embed_composition_version` is how AC5 proves convergence (`count WHERE version < current = 0`). Do not declare RC2 done before the backfill completes.
- **Do NOT silently widen scope to `episode_turns.py:516`.** It uses `websearch_to_tsquery` and likely shares the AND-death (RI-1), but it is a separate recall path with its own tests. Investigate under RI-1 and, if confirmed, fix as a sibling task sharing the RC1 builder — not as an unscoped edit inside this Epic's tasks.
- **Backfill cost is negligible but real.** Re-embedding the full memory corpus (~2000+ rows) via `text-embedding-3-small` is ~1M tokens ≈ a few cents — no Pinch gate needed, but the backfill runs against the live Supabase DB; run it deliberately (off-hours, idempotent, resumable via the version marker), not as a test side-effect.

## Merge & Validation

Order of operations, each task independently merge-ready:

1. **RC1 first, standalone.** Disjunctive tsquery builder + `search_by_keyword` switch + unit/fuzz tests. No migration, no re-embed, no dependency on RC2. This is the proven high-leverage fix; ship and verify it can surface `weft-5276bf05` before touching embeddings.
2. **RC2 chain:** migration (`embed_composition_version`) → shared `embed_text_for_memory` helper + version constant → wire helper into all memory write sites → wire helper into re-embed script → run backfill → AC5 completeness check.
3. **Acceptance:** seed the `weft-5276bf05` canary probe; AC1 (real-embedder, `WEFT_RUN_LLM_EVAL=1`) and AC2 (canary content-probe) both pass.
4. **Deferred / blocked (NOT this build):** RI-4 calibration → flip `active_probing_enabled` default. Tracked separately.

Validation rules and acceptance gates are the PRD's §Validation (V1–V5) and §Acceptance (AC1–AC5) — single source of truth; tasks reference them, do not restate.

## Task Plan

1. RC1 — add the sanitized disjunctive tsquery builder and switch `search_by_keyword` (`store.py:446` filter + `:514` rank) to it; unit tests for OR-not-AND (V1), rank ordering (V2), and the sanitizer fuzz set incl. empty-after-sanitization (V5/AC4). Files: `weft/store.py`, `tests/`.
2. RC2a — add `embed_text_for_memory(content, topic, type)` helper + `<current>` composition-version constant in one module; unit test for byte-identical output + topics present (V3). Files: new helper module (or `weft/store.py`), `tests/`.
3. RC2b — migration adding `memories.embed_composition_version SMALLINT NOT NULL DEFAULT <current>`. Files: `weft/db/migrations/`.
4. RC2c — wire `embed_text_for_memory` into every memory write site (`weft/mcp/tools.py:304` and the other memory-content `embed(...)` call sites: revise, batch, feedback, consolidation/extract as applicable); set `embed_composition_version` on write; integration test asserting stored embedding == `embed(embed_text_for_memory(...))` (real-API gated). Files: `weft/mcp/tools.py` (+ any other write sites), `tests/`.
5. RC2d — wire the helper into the re-embed script (`weft/db/reembed.py`) so the memories tier uses `embed_text_for_memory` and stamps `embed_composition_version`; parity integration test (re-embed vector == write vector, real-API gated). Files: `weft/db/reembed.py`, `tests/`.
6. RC2e — run the backfill against the corpus; AC5 verification (`count WHERE version < current = 0` + rows-changed non-degeneracy). Operational task; verifiable, idempotent, resumable.
7. Acceptance — seed a canary probe with answer `weft-5276bf05`; AC1 real-query test (target in top-10, control out, bounded set; `WEFT_RUN_LLM_EVAL=1`) and AC2 canary content-probe test (within `DEFAULT_AUDIT_TOP_K`, no `canary.miss`). Files: `tests/acceptance/`.

## Testing Standard

"Green" = full suite (~3174+) passing and ruff clean after every task. Pure-logic tests (RC1 builder, sanitizer, `embed_text_for_memory`, keyword/fusion ranking on a real DB) run without an API key. Tests that call the real embedder (write-path parity, re-embed parity, AC1) are gated behind `WEFT_RUN_LLM_EVAL=1` per the repo convention and are clearly marked. No task closes on "code landed" — each closes on its referenced V/AC gate.

## Technical Decisions

- `plainto_tsquery` in `search_by_keyword` (`store.py:446/514`) — **superseded by this Epic** (sanitized disjunctive `to_tsquery`).
- Content-only embedding (`tools.py:304`) — **superseded by this Epic** (shared `content + topics` helper).
- RRF fusion (`store.py:546`), `search_tsv` composition (v19) — **retained, not affected**.

## Out Of Scope

- Chunked / multi-vector embeddings for long memories (PRD RI-2; revisit only if RC2's standalone lift proves insufficient).
- The `episode_turns.py:516` `websearch_to_tsquery` path (PRD RI-1; investigate separately, fix as a sibling sharing the RC1 builder if confirmed).
- Flipping `active_probing_enabled` / RI-4 calibration (blocked; separate work).
- Surfacing true cosine alongside RRF score (PRD RI-4 cosmetic).
- Embedding provider/model change, and any change to RRF fusion or the `loc_key` catalog arc.
