"""Migration 69: versioned tool-usage recorder coverage and failures."""

from __future__ import annotations

VERSION = 69
DESCRIPTION = "tool usage recorder coverage, failures, and shutdown drain"
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_tool_usage_coverage (
    coverage_date      DATE PRIMARY KEY,
    recorder_version   TEXT NOT NULL,
    first_heartbeat_at TIMESTAMPTZ NOT NULL,
    last_heartbeat_at  TIMESTAMPTZ NOT NULL,
    successful_writes  BIGINT NOT NULL DEFAULT 0 CHECK (successful_writes >= 0),
    failure_count      BIGINT NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    shutdown_drained   BOOLEAN
);

-- System telemetry only: no payloads, tool arguments, or user identifiers.
ALTER TABLE weft_tool_usage_coverage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_tool_usage_coverage_service
    ON weft_tool_usage_coverage;
CREATE POLICY weft_tool_usage_coverage_service
    ON weft_tool_usage_coverage
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
