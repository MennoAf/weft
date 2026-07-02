"""Migration 66: recall_canary_audit — per-run outcome log for a WINDOWED miss rate.

## Why this table exists

The ``recall_canary`` table (v63) carries only *monotonic* counters
(``audit_count`` / ``miss_count`` — see v63 lines 61-62). ``canary_health``
summed those over all enabled probes, so its miss_rate was a **lifetime**
aggregate that could never fall: a burst of historical misses pinned the rate
forever, even after the underlying recall recovered.

This bit us concretely. PR #31 fixed the *audit selection* so pending_review /
agent-provenance probes are skipped (a guaranteed-miss artifact), but those
probes are skipped, NOT disabled — they stay ``enabled = TRUE`` with their old
artifact misses still banked in ``miss_count``. PR #32 then made an arm
``trustworthy`` at >=30 checks, which turned the still-inflated ~12.6% lifetime
rate into a FIRING drift tripwire — a false alarm driven by stale banked data
rather than a live regression.

## What this table is

One row per probe per audit run: ``(probe_id, user_id, audited_at, hit)``.
``canary_health`` computes its miss rate over a trailing *time window* (see
``_CANARY_HEALTH_WINDOW_DAYS`` in canary.py) instead of all-time counters, so
stale outcomes age out and the meter reflects *recent* recall health. The audit
loop prunes rows older than ``_CANARY_AUDIT_RETENTION_DAYS`` each run so the log
stays bounded.

The monotonic counters on ``recall_canary`` are retained (they still record
lifetime totals and drive nothing load-bearing after this change); the windowed
event log is the new source of truth for the health/tripwire surface.

## Schema notes

* ``id`` — ``BIGSERIAL`` surrogate PK; events are append-only.
* ``probe_id`` — FK to ``recall_canary(probe_id)`` ``ON DELETE CASCADE`` so a
  deleted probe's history cannot outlive it.
* ``user_id`` — ``NOT NULL DEFAULT GUC`` (post-v36 convention). The audit loop
  inserts it EXPLICITLY from the probe row — the raw scheduler pool leaves
  ``app.user_id`` unset, so the column default resolves to NULL and would trip
  NOT NULL (the same trap v63's reaREDACTED enroll hit; see canary.py).
* ``hit`` — ``TRUE`` the probe's memory surfaced in top-K, ``FALSE`` = a miss.

## RLS

Mirrors the ``recall_canary`` policy set (v63): SELECT admits the system
sentinel (``__system_global_zathras__``) for service-side callers; INSERT /
UPDATE / DELETE gate strictly on
``user_id = nullif(current_setting('app.user_id', true), '')``.

Spec: canary health windowed-rate fix (follow-up to PR #31/#32).
Depends on: v63 (recall_canary).
"""

from __future__ import annotations

VERSION = 66
DESCRIPTION = "recall_canary_audit: per-run outcome log for a windowed canary miss rate"
SQL = r"""
CREATE TABLE IF NOT EXISTS recall_canary_audit (
    id          BIGSERIAL PRIMARY KEY,
    probe_id    TEXT NOT NULL
                    REFERENCES recall_canary (probe_id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    audited_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    hit         BOOLEAN NOT NULL
);

-- Windowed rate query: recent events for the current user (audited_at DESC scan).
CREATE INDEX IF NOT EXISTS idx_recall_canary_audit_window
    ON recall_canary_audit (user_id, audited_at);

-- Join back to recall_canary per probe (arm grouping + enabled/universe filter).
CREATE INDEX IF NOT EXISTS idx_recall_canary_audit_probe
    ON recall_canary_audit (probe_id);

ALTER TABLE recall_canary_audit ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS recall_canary_audit_select ON recall_canary_audit;
CREATE POLICY recall_canary_audit_select ON recall_canary_audit FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_audit_insert ON recall_canary_audit;
CREATE POLICY recall_canary_audit_insert ON recall_canary_audit FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_audit_update ON recall_canary_audit;
CREATE POLICY recall_canary_audit_update ON recall_canary_audit FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_audit_delete ON recall_canary_audit;
CREATE POLICY recall_canary_audit_delete ON recall_canary_audit FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
