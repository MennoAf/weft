"""Migration 40: Phase 2.5: weft_tokens — credential-bound caller mode"""

from __future__ import annotations

VERSION = 40
DESCRIPTION = 'Phase 2.5: weft_tokens — credential-bound caller mode'
SQL = r"""
-- Wick Phase 2.5 — close the X-Weft-Caller-Mode header escalation.
-- See weft_v2_spec.md §Q4 Layer 1 (the trust signal that gates
-- layers 1-3 of the Phase 2 poisoning defense).
--
-- Phase 2 trusts the X-Weft-Caller-Mode header at the middleware
-- boundary, but an agent holding a bearer token can still send
-- caller_mode=supervisor and we believe it. Phase 2.5 binds caller
-- mode to the credential at issuance time: every bearer token
-- (legacy API key, OAuth access token, future federation key) gets
-- a row here, and middleware resolves user_id + caller_mode from
-- the row instead of trusting the header.
--
-- Service-role only by design: auth lookup happens at middleware
-- time, BEFORE app.user_id is set on the connection, so RLS
-- scoping can't apply to this table. Any future per-user listing
-- of tokens has to filter by user_id at the application layer.
--
-- Column notes:
--   token_hash   sha256 of the plaintext token (hex). Plaintext
--                is shown to the caller exactly once at issuance.
--   user_id      TEXT, not UUID — accommodates non-UUID auth
--                subjects (legacy bootstrap, federation IDs).
--   caller_mode  matches the existing 'supervisor'/'agent' enum
--                used by the trackers.provenance and Phase 2
--                write_provenance columns.
--   label        operator-supplied free text (e.g. "wick-runtime",
--                "warp-supervisor"). Optional.
--   expires_at / revoked_at  null means active forever / not
--                revoked. Lookup helper filters both.
--   last_used_at  bumped on successful lookup; lets operators
--                spot abandoned credentials.

CREATE TABLE IF NOT EXISTS weft_tokens (
    token_hash    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    caller_mode   TEXT NOT NULL,
    label         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    expires_at    TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);

ALTER TABLE weft_tokens
    DROP CONSTRAINT IF EXISTS weft_tokens_caller_mode_check;
ALTER TABLE weft_tokens
    ADD CONSTRAINT weft_tokens_caller_mode_check
    CHECK (caller_mode IN ('supervisor', 'agent'));

CREATE INDEX IF NOT EXISTS idx_weft_tokens_user_active
    ON weft_tokens (user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_weft_tokens_expires
    ON weft_tokens (expires_at) WHERE expires_at IS NOT NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
