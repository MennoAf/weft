# V2 Backlog Review

**Date:** 2026-03-25
**Reviewer:** warp (agent), Jason Bauman (owner)
**V1 state:** Feature-complete, 1722 tests passing, deployed to Fly.io
**Purpose:** Structured disposition of three V2 ideas parked during V1 development. Each gets CANCEL, REVISE, or PROMOTE.

---

## Idea 1: DB Tables for Patterns

### Current State

Pattern detection lives in `weft/check_in_patterns.py`, which explicitly declares its approach on line 6:

> No new DB tables. Functions accept lists of CheckIn objects (fetched via list_check_ins) and return structured dicts.

All analysis — day-of-week stats, sleep/energy correlation, mood streaks, rolling averages, trend detection — is pure Python over in-memory `CheckIn` lists. Results are computed fresh every call via `analyze_all()` (line 291). Nothing is persisted.

The `check_ins` table (migration 17, `weft/db/migrations.py:582`) stores raw check-in data only. No `patterns` table or similar exists in the migration history.

The alert system (`check_in_patterns.py:321`, `evaluate_check_in_alerts`) evaluates patterns and creates alerts, but alert records themselves are the only DB-persisted output — they track *that* a threshold was crossed, not the analysis that found it.

**What gets lost across restarts:** Nothing critical. Check-in data is in the DB. Pattern analysis is deterministic over that data — restarting doesn't lose state, it just recomputes. The only "state" is alert dedup timestamps, which are already persisted in the alerts table.

### Disposition: CANCEL

### Rationale

The original assumption was that pattern detection would need persistent state to avoid redundant computation or to track patterns over time. V1 proved this wrong:

1. **Recomputation is cheap.** `analyze_all()` runs over at most a few hundred `CheckIn` rows. There's no expensive model inference or heavy aggregation — it's basic Python math (Pearson correlation, linear regression, streak detection).

2. **No state loss on restart.** Since patterns are derived deterministically from persisted check-in data, restarting produces identical results. The wellness_snapshot in the primer (`weft/primer_sections/wellness.py`) calls the same functions each session — it doesn't need cached results.

3. **Alert dedup already handles the "don't re-fire" concern.** The 24h dedup window (`_ALERT_DEDUP_HOURS = 24`, `check_in_patterns.py:318`) prevents duplicate alerts without needing a patterns table.

4. **Adding a patterns table would create a cache invalidation problem** — every new check-in would require re-running analysis and updating cached results, adding complexity for no measurable benefit at current scale.

**Scale projection:** 6 years of daily check-ins = ~2,190 rows, but analysis functions already use bounded time windows (`trend_direction` uses 90 days, `rolling_averages` uses 30 days). Even with additional data sources (meal tracking, exercise, Google Health sync) at 5-10 entries/day, a 90-day window is ~900 rows — trivially cheap. Time alone does not make recomputation expensive. New features requiring ML-backed cross-correlation or multi-source analysis would be the trigger for persistence, and at that point you'd design the storage around the new feature's needs, not retrofit a generic patterns table.

---

## Idea 2: Configurable Thresholds

### Current State

There are **19 hardcoded numeric constants** across 4 files that control alert triggering, pattern detection, and consolidation behavior:

**`weft/check_in_patterns.py`** (7 values):
- `_MIN_CORRELATION_POINTS = 5` (line 34)
- `_MIN_TREND_POINTS = 5` (line 35)
- `_MIN_STREAK_LENGTH = 3` (line 36)
- `_ALERT_LOW_MOOD_STREAK = 3` (line 315)
- `_ALERT_LOW_SLEEP_HOURS = 6.0` (line 316)
- `_ALERT_LOW_SLEEP_DAYS = 5` (line 317)
- `_ALERT_DEDUP_HOURS = 24` (line 318)

**`weft/loom_alerts.py`** (3 values):
- `_STALE_CLAIM_HOURS = 48` (line 31)
- `_BLOCKED_PILE_UP_THRESHOLD = 5` (line 32)
- `_DEDUP_HOURS = 24` (line 33)

**`weft/memory_hygiene_alerts.py`** (5 values):
- `_STALE_DECISION_DAYS = 90` (line 26)
- `_STALE_DECISION_CONFIDENCE = 0.6` (line 27)
- `_CONSOLIDATION_OVERDUE_HOURS = 72` (line 28)
- `_MEMORY_COUNT_THRESHOLD = 1000` (line 29)
- `_DEDUP_HOURS = 24` (line 30)

**`weft/consolidation.py`** (via `DecayConfig`/`ConsolidationConfig` dataclasses):
- `half_life_days = 30.0` (line 46)
- `floor_score = 0.1` (line 47)
- `min_confidence = 0.3` (line 48)
- `duplicate_threshold = 0.95` (line 56)
- `contradiction_similarity_min = 0.7` (line 57)
- `contradiction_similarity_max = 0.99` (line 58)
- `_DEFAULT_INTERVAL_HOURS = 24` (line 682)
- `_MIN_CONTENT_LENGTH_FOR_CONTRADICTION = 50` (line 605)

**Operator-tunable vs. implementation constants:**

Likely tunable (an operator might reasonably want to change without a code deploy):
- Alert thresholds: mood streak length, sleep hours, stale claim hours, decision age, memory count limit
- Dedup windows (all the `_DEDUP_HOURS` values)
- Consolidation interval and decay half-life

Unlikely to need tuning:
- `_MIN_CORRELATION_POINTS`, `_MIN_TREND_POINTS` (statistical minimums)
- `_MIN_CONTENT_LENGTH_FOR_CONTRADICTION` (implementation guard)
- Contradiction similarity bounds (algorithmic internals)

**Existing configurability:** `consolidation.py` already uses dataclass-based config (`DecayConfig`, `ConsolidationConfig`) with sane defaults. The other three files use bare module-level constants.

### Disposition: REVISE

### What Changed

The original idea assumed all thresholds would need a DB-backed config UI. V1 showed that:

1. **The consolidation module already solved this well** with `DecayConfig`/`ConsolidationConfig` dataclasses. Callers can override any default without a DB round-trip. This pattern works.

2. **Most "tunable" thresholds are only tuned during development**, not at runtime by operators. The current hardcoded values were set once and haven't needed changing.

3. **Only ~10 of 19 values are genuinely operator-tunable.** The rest are implementation constants that would confuse rather than help if exposed.

### Revised Scope

Instead of a DB-backed config system, extend the existing dataclass pattern:

1. Create `AlertConfig` dataclasses in `check_in_patterns.py`, `loom_alerts.py`, and `memory_hygiene_alerts.py` mirroring the `ConsolidationConfig` pattern
2. Thread them through `evaluate_*` functions as optional parameters with current defaults
3. Optionally: load overrides from environment variables or a `weft_config` metadata key (already have `get_metadata`/`set_metadata` in `store.py`)

This is a small refactor (S-sized), not an epic. No new DB tables, no config UI. Just better structure that makes thresholds injectable for testing and overridable in production.

---

## Idea 3: Formal Dedup

### Current State

Dedup exists as a subsystem of consolidation (`consolidation.py:200`, `find_duplicates()`). It is a **post-hoc batch process**:

1. Fetches all active memories with embeddings (line 217-218)
2. For each memory, searches for similar memories above a cosine similarity threshold (default 0.95, line 232)
3. Keeps the higher-confidence memory, archives the other, creates a `supersedes` relationship (lines 240-286)
4. Runs as part of the `consolidate()` pipeline, gated by a Postgres advisory lock (line 524)

**There is no pre-insert duplicate check.** `store_memory()` in `store.py` inserts unconditionally — no unique constraints on `content` or `embedding`, no similarity check before write. The `ON CONFLICT` clauses in store.py are only on `memory_relationships` (composite PK) and `metadata` (key column), not on memories themselves.

**Auto-consolidation** (`consolidate_if_due`, line 720) runs at most once per 24 hours, triggered by tool calls. Between runs, duplicates can accumulate.

**Production risk assessment:** The Fly.io deployment uses MCP over SSE — requests are session-based, not at-least-once delivery. Concurrent duplicate writes from retries are unlikely but possible if a client retries after a timeout. The `weft_remember` tool has no idempotency key.

### Disposition: REVISE

### What Changed

The original idea assumed dedup needed to be a dedicated, separate system. V1 showed that:

1. **Batch dedup via consolidation works well enough for the common case.** The 0.95 similarity threshold catches near-exact duplicates.

2. **The real gap is pre-insert, not post-hoc.** When an agent calls `weft_remember` with content very similar to an existing memory, the right behavior is to *revise* the existing memory rather than create a duplicate — and then tell the agent what happened. Currently, consolidation silently merges them later.

3. **`check_contradictions_on_store()` (consolidation.py:608) already does a pre-insert similarity search** for contradiction detection. Adding a pre-insert dedup check would follow the same pattern — search by vector before committing, and either merge or warn.

### Revised Scope

Add a pre-insert similarity check to `weft_remember`:

1. Before storing, search for active memories with similarity > 0.90 (lower than the 0.95 batch threshold — catch more at write time)
2. If a near-duplicate is found:
   - If the new content is substantively different (longer, more detailed): revise the existing memory's content, bump its confidence and timestamp
   - If effectively identical: return the existing memory ID with a `"deduplicated": true` flag
3. Keep the batch `find_duplicates()` as a safety net for anything that slips through

This is an M-sized task — similar in shape to how `check_contradictions_on_store` was added alongside the batch contradiction detection.

---

## Summary

| Idea | Disposition | Rationale |
|------|-------------|-----------|
| DB tables for patterns | **CANCEL** | Compute-on-read is cheap and stateless; no benefit to persistence at current scale |
| Configurable thresholds | **REVISE** | Extend existing dataclass config pattern to alert modules; skip DB-backed config UI |
| Formal dedup | **REVISE** | Add pre-insert similarity check to `weft_remember`; keep batch dedup as safety net |
