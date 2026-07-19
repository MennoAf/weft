# Decision: Memory decay is review-only

**Date:** 2026-07-18
**Status:** accepted for this validation program
**Baseline:** `d7a1f3575868172add934402e4889c8bfb129eec`

## Decision

Weft will not automatically transition memory status based on the current decay score. `run_decay()` reports candidate IDs by default; mutation requires an explicit operator-controlled `apply=True` call. Scheduled consolidation, `consolidate()`, and the public `weft_consolidate` MCP tool never pass that gate.

This selects the approved **review-only branch**. The existing `decayed` status and scoring code are retained for compatibility and future experimentation, but public behavior is candidate review—not automatic archival.

## Evidence

The score is:

```text
max(floor,
    0.5 * 0.5 ** (age_days / half_life_days)
  + 0.2 * min(1, log1p(access_count) / log1p(20))
  + 0.3 * confidence)
```

Default candidate eligibility additionally requires `score <= 0.1` and `confidence < 0.3`. Representative default-config scores:

| confidence | age days | accesses | score | candidate? |
|---:|---:|---:|---:|---|
| 0.70 | 0 | 0 | 0.7100 | no |
| 0.70 | 365 | 0 | 0.2101 | no; default confidence can never reach floor |
| 0.29 | 120 | 0 | 0.1183 | no |
| 0.29 | 180 | 0 | 0.1000 | yes |
| 0.29 | 180 | 1 | ~0.1405 | no |
| 0.10 | 90 | 0 | 0.1000 | yes |
| 0.10 | 180 | 1 | 0.1000 | yes |
| 0.10 | 365 | 5 | ~0.1478 | no |

Pinned records and `preference`, `user_model`, and `decision` types always score `1.0`.

The tests in `tests/test_consolidation.py` pin this matrix, custom configuration, protected types, the default non-mutating path, and the explicit apply gate. Automatic-consolidation and Phase 3 end-to-end tests assert candidates remain active.

## Why not automatic archival

- Confidence is primarily write-time/revision metadata, not a continuously calibrated truth or value estimate.
- Access count can be inflated by automated reads and is not equivalent to human usefulness.
- The threshold has sharp discontinuities: confidence `0.30` is permanently excluded while `0.29` may qualify; one access can indefinitely prevent a borderline record from reaching the floor.
- Automatic status transition removes records from default active retrieval.
- The current candidate scan is limited to 1,000 rows and was not designed as a complete, auditable archival executor.
- A clear public restore/review workflow and per-transition audit reason are absent.
- Destructive lifecycle changes require stronger tenant-scoped proof and operator review than the current formula provides.

## Operational contract

- `compute_decay_score()` remains a deterministic review signal.
- `run_decay()` defaults to preview and returns candidate IDs without mutation.
- `run_decay(apply=True)` is an internal explicit operator gate; it is not used by scheduled or MCP paths.
- `dry_run=True, apply=True` is rejected.
- Public consolidation documentation describes stale candidates, not automatic decay.
- The 1,000-row scan is not claimed to be complete archival processing. Pagination becomes mandatory only if an archival executor is proposed again.

## Reconsideration requirements

An archival-enabled branch requires a new reviewed decision with: deterministic tenant-scoped pagination beyond 1,000 rows, candidate reasons and audit metadata, production-like dry-run output, reviewed thresholds, restore semantics, isolation tests, and a separately exposed operator approval action. It must not be enabled by configuration drift or a scheduled worker alone.
