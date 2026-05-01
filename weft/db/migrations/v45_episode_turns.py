"""Migration 45: episode_turns — turn-tier dialogue trace under episodes.

Third relation under the episodes umbrella. Existing tables (episodes,
episode_memories) stay untouched. Each row is one conversational turn
inside an episode, with its own embedding for retrieval and its own
occurred_at for temporal queries.

Schema highlights:

* ``user_id`` follows the post-migration-36 convention: ``NOT NULL`` with a
  GUC default so unset ``app.user_id`` is a fail-loud constraint violation,
  not a silent global-leak.
* ``trace_id`` is a single-valued nullable column that maps to Wick's
  ``run_id``. Multi-system nesting (e.g. a tool-call turn participating in
  both a Wick trace and a Loom trace) is carried in the federation envelope
  via ``originator.parent_run_id`` rather than overloading this column.
* ``importance_score`` is the Face hook: populated async post-hoc; gates
  retention at graduation time, not ingest. Stays NULL until Face is online.
* ``UNIQUE(episode_id, turn_index)`` guarantees per-episode ordering and
  enables race-safe append via ``ON CONFLICT (episode_id, turn_index) DO ...``.

Driver: LongMemEval oracle run showed 32–37% empty/uncertain recall on
multi-anchor temporal-reasoning questions. Belief-tier extraction collapses
dialogue dates and loses conversational anchoring. Turn-tier is the
load-bearing fix for Wick's harness use case.

Spec: weft-d3a2ef78. Loom epic: loom-52ffc3a2.
"""

from __future__ import annotations

VERSION = 45
DESCRIPTION = "episode_turns: turn-tier dialogue trace + RLS + HNSW"
SQL = r"""
CREATE TABLE IF NOT EXISTS episode_turns (
    id               TEXT PRIMARY KEY,
    episode_id       TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    turn_index       INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding        vector(768),
    trace_id         TEXT,
    importance_score REAL,
    token_count      INTEGER NOT NULL DEFAULT 0,
    user_id          TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (episode_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_episode_turns_episode_index
    ON episode_turns (episode_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_episode_turns_occurred
    ON episode_turns (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_episode_turns_trace
    ON episode_turns (trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episode_turns_user
    ON episode_turns (user_id);
CREATE INDEX IF NOT EXISTS idx_episode_turns_embedding_hnsw
    ON episode_turns USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

ALTER TABLE episode_turns ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS episode_turns_select ON episode_turns;
CREATE POLICY episode_turns_select ON episode_turns FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_turns_insert ON episode_turns;
CREATE POLICY episode_turns_insert ON episode_turns FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_turns_update ON episode_turns;
CREATE POLICY episode_turns_update ON episode_turns FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS episode_turns_delete ON episode_turns;
CREATE POLICY episode_turns_delete ON episode_turns FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
