"""Migration 68: daily aggregate MCP tool usage telemetry."""

from __future__ import annotations

VERSION = 68
DESCRIPTION = "daily aggregate MCP tool usage telemetry"
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_tool_usage_daily (
    usage_date      DATE NOT NULL,
    tool_name       TEXT NOT NULL,
    call_count      BIGINT NOT NULL DEFAULT 0,
    first_called_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_called_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (usage_date, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_weft_tool_usage_daily_tool_date
    ON weft_tool_usage_daily (tool_name, usage_date DESC);

-- System telemetry: no user payloads or per-user data are stored here.
ALTER TABLE weft_tool_usage_daily ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_tool_usage_daily_service ON weft_tool_usage_daily;
CREATE POLICY weft_tool_usage_daily_service ON weft_tool_usage_daily
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
