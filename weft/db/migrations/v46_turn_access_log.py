"""Migration 46: turn_access_log + episode_turns usefulness columns.

Schema substrate for the turn-tier boost loop (Phase 1 Track A). Mirrors
the belief-tier pair (memory_access_log + memories.usefulness_*) one-for-one
so the boost loop has a turn-level analog to write to.

``turn_access_log`` is a system table — no ``user_id`` column. Session-scoped
access tracking is keyed by ``(session_id, turn_id)``, identical in shape to
``memory_access_log``. RLS follows the v23 system-table pattern: service-role
only, ``USING (true) WITH CHECK (true)``. The cascading FK on ``turn_id``
keeps the log clean when turns are pruned (graduation TTL, importance gate).

The three new ``episode_turns`` columns mirror the belief-tier scoring
fields:

* ``usefulness_score`` (REAL, default 0.7) — boostable retrieval prior.
* ``usefulness_count`` (INTEGER, default 0) — how many boost events have
  landed; used for confidence weighting and saturation detection.
* ``last_boosted_at`` (TIMESTAMPTZ, nullable) — most recent boost event,
  feeds the time-decay term identical to v17 on memories.

Out of scope (deliberately): the v41 read-side audit columns
(``reader_user_id``, ``reader_caller_mode``, ``retrieval_mode``). The boost
loop only needs session/turn/tool/timestamp; richer audit comes later when
the turn-tier surface itself is credential-bound.

Driver: P1.A1 — boost loop needs a place to record turn accesses and a
score column to mutate. Spec lives in the weft-wick Loom epic.
"""

from __future__ import annotations

VERSION = 46
DESCRIPTION = (
    "turn_access_log table + episode_turns usefulness columns for boost loop"
)
SQL = r"""
CREATE TABLE IF NOT EXISTS turn_access_log (
    session_id  TEXT NOT NULL,
    turn_id     TEXT NOT NULL REFERENCES episode_turns(id) ON DELETE CASCADE,
    tool_name   TEXT NOT NULL,
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_turn_access_log_session
    ON turn_access_log (session_id);

CREATE INDEX IF NOT EXISTS idx_turn_access_log_accessed_at
    ON turn_access_log (accessed_at);

-- System table: service-role only, session tracking (mirrors v23 policy
-- on memory_access_log).
ALTER TABLE turn_access_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS turn_access_log_service ON turn_access_log;
CREATE POLICY turn_access_log_service ON turn_access_log
    USING (true) WITH CHECK (true);

ALTER TABLE episode_turns ADD COLUMN IF NOT EXISTS
    usefulness_score REAL DEFAULT 0.7;
ALTER TABLE episode_turns ADD COLUMN IF NOT EXISTS
    usefulness_count INTEGER DEFAULT 0;
ALTER TABLE episode_turns ADD COLUMN IF NOT EXISTS
    last_boosted_at TIMESTAMPTZ DEFAULT NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
