"""Migration 76: durable single-worker ownership and fencing state."""

from __future__ import annotations

VERSION = 76
DESCRIPTION = "durable single-worker ownership and fencing leases"
SQL = r"""
-- This is operational state, not user data.  It deliberately has no user_id,
-- RLS policy, or runtime grant: the restricted role must fail closed unless an
-- owner explicitly provisions the narrowly reviewed lease privilege.
CREATE TABLE IF NOT EXISTS worker_leases (
    lease_key TEXT PRIMARY KEY CHECK (char_length(lease_key) BETWEEN 1 AND 256),
    owner_token TEXT NULL CHECK (owner_token IS NULL OR char_length(owner_token) BETWEEN 1 AND 512),
    generation BIGINT NOT NULL DEFAULT 0 CHECK (generation >= 0),
    acquired_at TIMESTAMPTZ NULL,
    renewed_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NULL,
    CHECK (
        (owner_token IS NULL AND acquired_at IS NULL AND renewed_at IS NULL)
        OR (owner_token IS NOT NULL AND acquired_at IS NOT NULL AND renewed_at IS NOT NULL)
    )
);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
