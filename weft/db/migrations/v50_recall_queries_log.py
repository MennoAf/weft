"""Migration 50: weft_recall_queries log table — Step 1.5 of the compounding loop.

The Anvil-reframed Step 1 (v49) bumps per-memory retrieval telemetry, but
the three baseline metrics the 2-week observation window is supposed to
produce are all *query-level*, not memory-level:

* (a) total weft_recall + weft_search_all calls per week
* (b) % of queries on projects with prior queries within 7 days
* (c) average semantic similarity between consecutive same-project queries

None of those are cleanly computable from existing tables. ``memory_access_log``
records one row per (session, memory) pair — useless for counting distinct
queries or attributing project_id to a query. This migration adds the
minimal query log needed to make metric (a)/(b) trivial and metric (c)
computable post-hoc by re-embedding the stored query text at analysis time.

Design choices:

* No embedding column. The query text is immutable; re-embedding it later
  with whichever model is current at analysis time is cheaper than
  committing to a model now AND avoids a backfill if the embedding model
  changes mid-window. Trade-off: ~50ms of embedding work per query at
  analysis time × ~thousands of queries = a few minutes of one-shot
  compute, totally fine.

* ``project_id`` nullable. weft_recall and weft_search_all both run
  without a resolved project_id in plenty of code paths (cross-project
  search, ad-hoc lookups). NULL here means "no project context" — same
  semantics as ``memories.project_id``.

* RLS mirrors v48 belief_claims: SELECT allows the system sentinel
  (``__system_global_zathras__``); INSERT/UPDATE/DELETE gate on
  ``user_id = current_setting('app.user_id')``. The query log is
  caller-scoped — one user should not see another user's query history.

* ``query_text`` is stored verbatim, not normalized. Metric (c) wants
  the actual semantic content, and analysis-time tools can do their own
  normalization (lowercase, strip, etc.) without losing information.

* Single composite index on (user_id, project_id, created_at DESC) — the
  shape every baseline-metric query will run.

Decision provenance: handoff weft-bb582eb3 (Step 1 ship) flagged metric
(c) as ungettable from v49 alone. This migration closes that gap before
traffic accumulates, so the 2-week window has all three signals
available at re-decision time.
"""

from __future__ import annotations

VERSION = 50
DESCRIPTION = "weft_recall_queries: per-call query log for the compounding-loop observation window"
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_recall_queries (
    query_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    project_id      TEXT,
    tool_name       TEXT NOT NULL CHECK (tool_name IN ('recall', 'search_all')),
    query_text      TEXT NOT NULL,
    tier            TEXT,
    mode            TEXT,
    retrieval_mode  TEXT,
    result_count    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recall_queries_user_project_time
    ON weft_recall_queries (user_id, project_id, created_at DESC);

ALTER TABLE weft_recall_queries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_recall_queries_select ON weft_recall_queries;
CREATE POLICY weft_recall_queries_select ON weft_recall_queries FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recall_queries_insert ON weft_recall_queries;
CREATE POLICY weft_recall_queries_insert ON weft_recall_queries FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recall_queries_update ON weft_recall_queries;
CREATE POLICY weft_recall_queries_update ON weft_recall_queries FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recall_queries_delete ON weft_recall_queries;
CREATE POLICY weft_recall_queries_delete ON weft_recall_queries FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
