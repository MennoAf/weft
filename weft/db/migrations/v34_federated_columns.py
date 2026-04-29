"""Migration 34: Schema v1: add federated-future columns to memories (additive)"""

from __future__ import annotations

VERSION = 34
DESCRIPTION = 'Schema v1: add federated-future columns to memories (additive)'
SQL = r"""
-- Federated-Future Schema v1: additive only. No behavior change.
-- See 2026-04-26-weft-federated-future-schema-v1.md.
--
-- schema_version  : dispatch field for future upgrade chain
-- author_identity : who wrote this row (survives sharing)
-- visibility      : private | global  ('shared' rejected by CHECK
--                   until workspace logic ships in v2)
-- provenance      : audit trail; lives ALONGSIDE the existing
--                   `source` column (do not unify — see opinion)
-- sharing_metadata: escape hatch for federation/ACL/expiry; empty in v1
-- workspace_id    : FK target for migration 35's workspaces table

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS schema_version  INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS author_identity JSONB   NOT NULL DEFAULT '{"kind":"unknown"}'::jsonb,
    ADD COLUMN IF NOT EXISTS visibility      TEXT    NOT NULL DEFAULT 'private',
    ADD COLUMN IF NOT EXISTS provenance      JSONB   NOT NULL DEFAULT '{"source":"self"}'::jsonb,
    ADD COLUMN IF NOT EXISTS sharing_metadata JSONB  NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS workspace_id    TEXT;

-- Backfill author_identity + visibility from existing user_id state.
-- Rows with user_id IS NULL = current "global by convention" rows
-- (system seeds, shared modes, etc.). Migration 36 will replace
-- the NULL convention with a SYSTEM_GLOBAL sentinel.
UPDATE memories
SET
    author_identity = CASE
        WHEN user_id IS NULL THEN '{"kind":"system","component":"seed"}'::jsonb
        ELSE jsonb_build_object('kind', 'local_user', 'user_id', user_id)
    END,
    visibility = CASE
        WHEN user_id IS NULL THEN 'global'
        ELSE 'private'
    END
WHERE author_identity = '{"kind":"unknown"}'::jsonb;

-- Fail-loud constraint: 'shared' is reserved but not honored by v1
-- logic. Reject writes until workspace primitive ships, so we never
-- silently treat would-be-shared rows as private.
ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_visibility_check;
ALTER TABLE memories
    ADD CONSTRAINT memories_visibility_check
    CHECK (visibility IN ('private', 'global'));

CREATE INDEX IF NOT EXISTS idx_memories_workspace
    ON memories (workspace_id) WHERE workspace_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_visibility
    ON memories (visibility);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
