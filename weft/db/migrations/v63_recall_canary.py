"""Migration 63: recall_canary — enrollment + audit table for retrieval health monitoring.

The ``recall_canary`` table is the persistence layer for Phase 0.5 of the recall
health monitoring system (PRD §V5, loom-c27ab1d2).

Each row is a *known-answer probe*: a ``(memory_id, probe_text)`` pair.  During the
daily audit, ``probe_text`` is embedded and fed to the deterministic vector search
(local FastEmbed + the tie-break ORDER BY introduced in task 0.3). If ``memory_id``
does NOT appear in the top-K results, the probe is a *canary miss* — a real ranking
miss, not run-to-run noise.

## Two probe types

* ``active`` — synthetic, derived from the memory's own content at write time.
  Enrolled by the ``weft_remember`` write path (O(1) INSERT, no LLM call).
  Gated behind ``active_probing_enabled`` in the audit until the miss rate is
  calibrated and trusted.  Default: collected but *not yet audited*.

* ``reask-bootstrap`` — high-confidence, derived from real re-ask events.
  Auto-enrolled at audit time from ``weft_recall_queries`` rows where
  ``is_reask_miss = TRUE``.  The original query text → satisfying memory pair is
  a proven known-answer case.  Always audited (not gated by any flag).

## Schema notes

* ``probe_id`` — caller-supplied ``TEXT PRIMARY KEY`` (``cp-{shortid}``).
* ``user_id`` — ``NOT NULL DEFAULT GUC`` following post-v36 convention.
* ``probe_type`` — ``'active' | 'reask-bootstrap'``; CHECK constraint enforces the set.
* ``enabled`` — soft-disable a probe without deleting it (e.g., after the
  underlying memory is archived).
* ``audit_count / miss_count`` — monotonic counters; a rising ``miss_count`` while
  ``audit_count`` grows is the degradation tell.
* ``last_audit_at`` — timestamps the most recent audit pass.

## RLS

Follows v57 ``topic_digests`` pattern: SELECT admits the system sentinel
(``__system_global_zathras__``) for service-side callers;
INSERT / UPDATE / DELETE gate strictly on
``user_id = nullif(current_setting('app.user_id', true), '')``.

Spec: loom-c27ab1d2 (Phase 0.5 canary).
Depends on: loom-3acb7cf8 (tie-break, v0.3 DONE), loom-f52e8ea8 (truncation, DONE).
"""

from __future__ import annotations

VERSION = 63
DESCRIPTION = "recall_canary: enrollment + audit table for retrieval health monitoring"
SQL = r"""
CREATE TABLE IF NOT EXISTS recall_canary (
    probe_id      TEXT PRIMARY KEY,
    memory_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    probe_text    TEXT NOT NULL,
    probe_type    TEXT NOT NULL DEFAULT 'active'
                      CHECK (probe_type IN ('active', 'reask-bootstrap')),
    enrolled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    last_audit_at TIMESTAMPTZ,
    audit_count   INT NOT NULL DEFAULT 0,
    miss_count    INT NOT NULL DEFAULT 0
);

-- Efficient per-user audit query: enabled probes for the current user.
CREATE INDEX IF NOT EXISTS idx_recall_canary_user_enabled
    ON recall_canary (user_id, enabled);

-- Lookup probes by memory_id (e.g., to disable when memory is archived).
CREATE INDEX IF NOT EXISTS idx_recall_canary_memory_id
    ON recall_canary (memory_id);

ALTER TABLE recall_canary ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS recall_canary_select ON recall_canary;
CREATE POLICY recall_canary_select ON recall_canary FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_insert ON recall_canary;
CREATE POLICY recall_canary_insert ON recall_canary FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_update ON recall_canary;
CREATE POLICY recall_canary_update ON recall_canary FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS recall_canary_delete ON recall_canary;
CREATE POLICY recall_canary_delete ON recall_canary FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
