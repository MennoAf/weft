"""Migration 33: Create OAuth 2.1 tables (clients, codes, refresh tokens, revocations)"""

from __future__ import annotations

VERSION = 33
DESCRIPTION = 'Create OAuth 2.1 tables (clients, codes, refresh tokens, revocations)'
SQL = r"""
-- OAuth dynamic-client registrations (RFC 7591)
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id                  TEXT PRIMARY KEY,
    client_name                TEXT,
    redirect_uris              TEXT[] NOT NULL,
    grant_types                TEXT[] NOT NULL DEFAULT '{authorization_code,refresh_token}',
    response_types             TEXT[] NOT NULL DEFAULT '{code}',
    token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',
    scope                      TEXT NOT NULL DEFAULT 'mcp.read mcp.write',
    software_id                TEXT,
    software_version           TEXT,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at               TIMESTAMPTZ
);

-- Pending authorization codes (short-lived, ~10 min)
CREATE TABLE IF NOT EXISTS oauth_authorization_codes (
    code                  TEXT PRIMARY KEY,
    client_id             TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    user_sub              TEXT NOT NULL,
    redirect_uri          TEXT NOT NULL,
    scope                 TEXT NOT NULL,
    code_challenge        TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL,
    expires_at            TIMESTAMPTZ NOT NULL,
    consumed_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_oauth_codes_expires
    ON oauth_authorization_codes (expires_at);

-- Refresh tokens (longer-lived, rotated on use)
CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
    jti         TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    user_sub    TEXT NOT NULL,
    scope       TEXT NOT NULL,
    token_hash  TEXT NOT NULL,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked_at  TIMESTAMPTZ,
    rotated_to  TEXT
);
CREATE INDEX IF NOT EXISTS idx_oauth_refresh_user
    ON oauth_refresh_tokens (user_sub);
CREATE INDEX IF NOT EXISTS idx_oauth_refresh_expires
    ON oauth_refresh_tokens (expires_at);

-- Access-token revocation blocklist (rare; most access tokens expire
-- before revoke)
CREATE TABLE IF NOT EXISTS oauth_access_revocations (
    jti         TEXT PRIMARY KEY,
    revoked_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oauth_access_rev_expires
    ON oauth_access_revocations (expires_at);

-- OAuth tables are service-role only — not user-scoped via RLS.
-- Pattern mirrors migration 23 (schema_migrations, weft_metadata,
-- memory_access_log): USING (true) WITH CHECK (true) means RLS is
-- enforced structurally by restricting which connections can reach
-- these tables (service role, app.user_id='').
ALTER TABLE oauth_clients ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS oauth_clients_service ON oauth_clients;
CREATE POLICY oauth_clients_service ON oauth_clients
    USING (true) WITH CHECK (true);

ALTER TABLE oauth_authorization_codes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS oauth_authorization_codes_service ON oauth_authorization_codes;
CREATE POLICY oauth_authorization_codes_service ON oauth_authorization_codes
    USING (true) WITH CHECK (true);

ALTER TABLE oauth_refresh_tokens ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS oauth_refresh_tokens_service ON oauth_refresh_tokens;
CREATE POLICY oauth_refresh_tokens_service ON oauth_refresh_tokens
    USING (true) WITH CHECK (true);

ALTER TABLE oauth_access_revocations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS oauth_access_revocations_service ON oauth_access_revocations;
CREATE POLICY oauth_access_revocations_service ON oauth_access_revocations
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
