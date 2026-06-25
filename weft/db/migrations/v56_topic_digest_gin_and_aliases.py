"""Migration 56: GIN index on memories(topic[]) + topic_resolution_aliases table.

Two additive objects for the topic-digest recall feature (PRD:
``documents/prds/topic-digest-recall.md``).

1. **GIN index on ``memories.topic[]``.**  The complete topic-gather
   (``WHERE '<tag>' = ANY(topic) AND status='active'``) currently seqscans
   the full ``memories`` table (~5k rows today, growing).  A GIN index turns
   the containment/membership predicate into an index scan without any schema
   change — the existing ``topic TEXT[]`` column is already populated.

2. **``topic_resolution_aliases`` table.**  Persistence layer for the L1
   Resolution Ratchet (PRD §Compounding Loops, L1).  Each row records a
   correction: a free-text ``topic_token`` that naive normalization failed to
   resolve, paired with the ``resolved_tags`` array that a user or agent
   confirmed does return the right memories.  The resolver consults this table
   first so the same empty-result token never misses twice.

Design notes:
- ``user_id`` follows the post-v36 convention: ``NOT NULL`` with a GUC
  default so an unset ``app.user_id`` fails loudly rather than leaking into a
  "global" sentinel row.
- ``source`` CHECK mirrors similar enum guards in the codebase
  (``source_provenance`` in ``belief_claims``, ``status`` in multiple tables).
- ``hit_count INT DEFAULT 0`` is a write-lightweight counter; callers increment
  on each alias hit to feed the L2 Usage-Tuned Materialization signal.
- ``updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`` tracks the last time the
  alias was written or corrected so stale aliases can be audited.
- RLS mirrors v48 ``belief_claims_select``: SELECT admits the system sentinel
  (``__system_global_zathras__``) for service-side callers; INSERT/UPDATE/DELETE
  gate strictly on ``user_id = current_setting('app.user_id', true)``.

Idempotency: ``CREATE INDEX IF NOT EXISTS``, ``CREATE TABLE IF NOT EXISTS``,
``DROP POLICY IF EXISTS`` before each ``CREATE POLICY``,
``ALTER TABLE ... ENABLE ROW LEVEL SECURITY``.

Spec: ``documents/prds/topic-digest-recall.md`` §Constraints Touched,
      §Interfaces/Schema, §Compounding Loops L1.
Loom task: loom-2aba1cf5.
"""

from __future__ import annotations

VERSION = 56
DESCRIPTION = "memories GIN index on topic[] + topic_resolution_aliases table"
SQL = r"""
-- ── 1. GIN index on memories.topic[] ──────────────────────────────────────
-- Backs the complete topic-gather: WHERE '<tag>' = ANY(topic) AND status='active'.
-- Naming convention mirrors the belief_claims GIN index (v48).
CREATE INDEX IF NOT EXISTS idx_memories_topic_gin
    ON memories USING GIN (topic);

-- ── 2. topic_resolution_aliases table ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS topic_resolution_aliases (
    user_id         TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    topic_token     TEXT NOT NULL,
    resolved_tags   TEXT[] NOT NULL,
    hit_count       INT NOT NULL DEFAULT 0,
    source          TEXT NOT NULL
                    CHECK (source IN ('learned', 'manual')),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, topic_token)
);

ALTER TABLE topic_resolution_aliases ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS topic_resolution_aliases_select ON topic_resolution_aliases;
CREATE POLICY topic_resolution_aliases_select ON topic_resolution_aliases FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_resolution_aliases_insert ON topic_resolution_aliases;
CREATE POLICY topic_resolution_aliases_insert ON topic_resolution_aliases FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_resolution_aliases_update ON topic_resolution_aliases;
CREATE POLICY topic_resolution_aliases_update ON topic_resolution_aliases FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_resolution_aliases_delete ON topic_resolution_aliases;
CREATE POLICY topic_resolution_aliases_delete ON topic_resolution_aliases FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
