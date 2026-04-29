"""Migration 41: Phase 2 follow-on: read-side audit columns on memory_access_log"""

from __future__ import annotations

VERSION = 41
DESCRIPTION = 'Phase 2 follow-on: read-side audit columns on memory_access_log'
SQL = r"""
-- Phase 2 / 2.5 stamps WRITE provenance (write_provenance on memories,
-- behaviors, triggers; caller_mode bound to weft_tokens row). Reads
-- are still logged only via the session-tracking path
-- (memory_access_log: session_id, memory_id, tool_name, accessed_at)
-- which is sufficient for the usefulness-boost feature but useless
-- for incident reconstruction: if a poisoned agent-provenance
-- memory slips past Layer 3 and reaches an agent's context, we
-- can't say which user_id / caller_mode read it.
--
-- This migration extends the existing table rather than forking a
-- v2: the (session_id, memory_id) PK already provides
-- session-deduplicated "first access" semantics, which is exactly
-- what an audit trail wants — every additional access in the same
-- session by the same caller adds no forensic information. New
-- columns are nullable because backfilling old rows with the
-- current contextvars would lie about history.

ALTER TABLE memory_access_log
    ADD COLUMN IF NOT EXISTS reader_user_id     TEXT,
    ADD COLUMN IF NOT EXISTS reader_caller_mode TEXT,
    ADD COLUMN IF NOT EXISTS retrieval_mode     TEXT;

ALTER TABLE memory_access_log
    DROP CONSTRAINT IF EXISTS memory_access_log_caller_mode_check;
ALTER TABLE memory_access_log
    ADD CONSTRAINT memory_access_log_caller_mode_check
    CHECK (
        reader_caller_mode IS NULL
        OR reader_caller_mode IN ('supervisor', 'agent')
    );

-- "What has user X read lately?" — supervisor incident-response query.
CREATE INDEX IF NOT EXISTS idx_access_log_user_recent
    ON memory_access_log (reader_user_id, accessed_at DESC)
    WHERE reader_user_id IS NOT NULL;

-- "Who has read memory M, and when?" — given a confirmed-poisoned
-- row, find every reader. The existing PK is (session_id,
-- memory_id) which can't satisfy this query efficiently because
-- session_id is the lead column.
CREATE INDEX IF NOT EXISTS idx_access_log_memory_recent
    ON memory_access_log (memory_id, accessed_at DESC);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
