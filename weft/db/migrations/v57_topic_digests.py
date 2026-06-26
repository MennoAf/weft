"""Migration 57: topic_digests cache table.

The ``topic_digests`` table is the digest-cache persistence layer for the
topic-digest recall feature (PRD: ``documents/prds/topic-digest-recall.md``).

Each row holds a materialized narrative synthesis (Tier-2) for a single
``(user_id, topic, scope)`` triple — the result of feeding the Tier-1 complete
memory-neighborhood gather to a Haiku synthesis pass.  The cache is keyed by
``(user_id, topic, scope)`` via a UNIQUE index; ``scope`` defaults to
``'global'`` for user-level digests with no workspace qualifier.

Design notes:

* ``digest_id`` uses ``TEXT PRIMARY KEY`` (caller-supplied), following the
  codebase convention (belief_claims, episode_turns, replay_queue, etc.).
  The writer supplies a prefixed short-id (e.g. ``td-{shortid}``) for
  unambiguous log correlation.

* ``user_id`` follows the post-v36 convention: ``NOT NULL`` with a GUC
  default so an unset ``app.user_id`` fails loudly rather than silently
  leaking into a sentinel row.

* ``provenance`` is ``JSONB`` to carry the per-assertion memory-id map
  (``{memory_id: [spans, ...]}``) emitted by the Haiku synthesis pass (V4
  provenance requirement).

* ``generated_at`` records when the synthesis was materialised so the caller
  can surface staleness age to the user.

* ``stale`` defaults to ``false``; the memory write path flips it to ``true``
  when a new memory tagged with this topic is written (V3 write-invalidation).

* ``detector_version`` pins the Haiku prompt version so stale-on-version
  upgrades are detectable without re-running.

* The UNIQUE index on ``(user_id, topic, scope)`` enforces the single-cached-
  digest invariant and backs the UPDATE-on-refresh and EXISTS-on-staleness-flip
  hot paths.

* RLS mirrors v48 ``belief_claims`` and v56 ``topic_resolution_aliases``:
  SELECT admits the system sentinel (``__system_global_zathras__``) for
  service-side callers; INSERT/UPDATE/DELETE gate strictly on
  ``user_id = nullif(current_setting('app.user_id', true), '')``.

Idempotency: ``CREATE TABLE IF NOT EXISTS``,
``CREATE UNIQUE INDEX IF NOT EXISTS``, ``DROP POLICY IF EXISTS`` before each
``CREATE POLICY``, ``ALTER TABLE ... ENABLE ROW LEVEL SECURITY``.

Spec: ``documents/prds/topic-digest-recall.md`` §Interfaces/Schema — Digest store.
Loom task: loom-43b8aa18.
"""

from __future__ import annotations

VERSION = 57
DESCRIPTION = "topic_digests cache table"
SQL = r"""
CREATE TABLE IF NOT EXISTS topic_digests (
    digest_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    topic            TEXT NOT NULL,
    scope            TEXT NOT NULL DEFAULT 'global',
    content          TEXT,
    provenance       JSONB,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    stale            BOOLEAN NOT NULL DEFAULT false,
    detector_version TEXT NOT NULL
);

-- Single-digest invariant: one cached entry per (user_id, topic, scope).
-- Also backs the UPDATE-on-refresh and EXISTS-on-staleness-flip hot paths.
CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_digests_user_topic_scope
    ON topic_digests (user_id, topic, scope);

ALTER TABLE topic_digests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS topic_digests_select ON topic_digests;
CREATE POLICY topic_digests_select ON topic_digests FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_digests_insert ON topic_digests;
CREATE POLICY topic_digests_insert ON topic_digests FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_digests_update ON topic_digests;
CREATE POLICY topic_digests_update ON topic_digests FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS topic_digests_delete ON topic_digests;
CREATE POLICY topic_digests_delete ON topic_digests FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
