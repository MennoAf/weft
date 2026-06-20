"""Migration 53: replay_queue table — episode replay cache with RLS.

The ``replay_queue`` table holds pending replay requests for episodes:
stored turn lists to be replayed later, tagged with a user and episode.

Design choices:

* ``id`` uses an ``rq-{shortid}`` prefix — tier-specific prefixes make log
  correlation unambiguous without a type column, same convention as
  ``episode_turns.id = et-{shortid}`` and ``belief_claims.id = belief-{shortid}``.

* ``user_id`` follows the post-migration-36 convention: ``NOT NULL`` with a
  GUC default so unset ``app.user_id`` is a fail-loud constraint violation,
  not a silent global-leak. See v45 line 46 for the exact pattern.

* ``turn_ids TEXT[]`` holds an ordered list of turn IDs to replay.

* ``reason TEXT`` describes why the replay is queued (e.g., "correction",
  "analysis", "revision").

* ``status TEXT DEFAULT 'pending'`` tracks replay lifecycle:
  - 'pending': queued, not yet replayed
  - 'done': replay completed

* ``created_at`` records when the replay was queued.

* ``episode_id`` references the episode being replayed.

* RLS mirrors v45 ``episode_turns_*`` policies: SELECT allows the system
  sentinel (``__system_global_zathras__``) for service-side enumeration;
  INSERT/UPDATE/DELETE gate strictly on ``user_id = current_setting('app.user_id')``.

* Two indexes: one on (status) for finding pending replays, one on
  (episode_id) for episode-scoped lookups.

Spec: Loom task loom-a6fa27eb.
"""

from __future__ import annotations

VERSION = 53
DESCRIPTION = "replay_queue: episode replay cache + RLS"
SQL = r"""
CREATE TABLE IF NOT EXISTS replay_queue (
    id               TEXT PRIMARY KEY,
    episode_id       TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    turn_ids         TEXT[] NOT NULL,
    reason           TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'done')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id          TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), '')
);

CREATE INDEX IF NOT EXISTS idx_replay_queue_status
    ON replay_queue (status);
CREATE INDEX IF NOT EXISTS idx_replay_queue_episode
    ON replay_queue (episode_id);

ALTER TABLE replay_queue ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS replay_queue_select ON replay_queue;
CREATE POLICY replay_queue_select ON replay_queue FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS replay_queue_insert ON replay_queue;
CREATE POLICY replay_queue_insert ON replay_queue FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS replay_queue_update ON replay_queue;
CREATE POLICY replay_queue_update ON replay_queue FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS replay_queue_delete ON replay_queue;
CREATE POLICY replay_queue_delete ON replay_queue FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
