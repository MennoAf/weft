"""Migration 16: Add memory_access_log for session-scoped access tracking"""

from __future__ import annotations

VERSION = 16
DESCRIPTION = 'Add memory_access_log for session-scoped access tracking'
SQL = r"""
CREATE TABLE IF NOT EXISTS memory_access_log (
    session_id  TEXT NOT NULL,
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tool_name   TEXT NOT NULL,
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_access_log_session
ON memory_access_log (session_id);

CREATE INDEX IF NOT EXISTS idx_access_log_accessed
ON memory_access_log (accessed_at);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
