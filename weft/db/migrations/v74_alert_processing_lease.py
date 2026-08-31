"""Migration 74: durable alert polling reservations with lease recovery."""

from __future__ import annotations

VERSION = 74
DESCRIPTION = "alerts: add processing_at for recoverable scheduler reservations"
SQL = r"""
ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS processing_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_alerts_processing_lease
    ON alerts (processing_at)
    WHERE status = 'processing';
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
