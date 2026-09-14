"""Migration 75: embedding profile identity and resumable re-embed state.

The existing ``embedding`` column remains the authoritative read materialization.
A maintenance run writes to the additive target columns and promotes them only
when every requested row has been verified.  This keeps interrupted runs
invisible to normal vector reads.
"""

from __future__ import annotations

VERSION = 75
DESCRIPTION = "embedding profiles and resumable re-embed run state"
SQL = r"""
CREATE TABLE IF NOT EXISTS embedding_profiles (
    profile_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL CHECK (char_length(provider) BETWEEN 1 AND 128),
    model TEXT NOT NULL CHECK (char_length(model) BETWEEN 1 AND 512),
    dimensions INTEGER NOT NULL CHECK (dimensions > 0 AND dimensions <= 65535),
    composition JSONB NOT NULL DEFAULT '{}'::jsonb,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'active', 'retired', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at TIMESTAMPTZ NULL
);

CREATE TABLE IF NOT EXISTS embedding_profile_state (
    state_key TEXT PRIMARY KEY DEFAULT 'default' CHECK (state_key = 'default'),
    active_profile_id TEXT NULL REFERENCES embedding_profiles(profile_id),
    target_profile_id TEXT NULL REFERENCES embedding_profiles(profile_id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO embedding_profile_state (state_key)
VALUES ('default') ON CONFLICT (state_key) DO NOTHING;

CREATE TABLE IF NOT EXISTS embedding_reembed_runs (
    run_id TEXT PRIMARY KEY,
    target_profile_id TEXT NOT NULL REFERENCES embedding_profiles(profile_id),
    tables TEXT[] NOT NULL CHECK (cardinality(tables) > 0),
    batch_size INTEGER NOT NULL CHECK (batch_size > 0 AND batch_size <= 10000),
    cursor JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'interrupted', 'failed', 'promoted')),
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    total_rows INTEGER NOT NULL DEFAULT 0 CHECK (total_rows >= 0),
    embedded_rows INTEGER NOT NULL DEFAULT 0 CHECK (embedded_rows >= 0),
    error TEXT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_reembed_runs_status
    ON embedding_reembed_runs (status, updated_at DESC);

CREATE TABLE IF NOT EXISTS embedding_profile_vectors (
    profile_id TEXT NOT NULL REFERENCES embedding_profiles(profile_id) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    row_id TEXT NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (profile_id, table_name, row_id)
);

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS embedding_target vector,
    ADD COLUMN IF NOT EXISTS embedding_target_profile_id TEXT,
    ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT;
ALTER TABLE behaviors
    ADD COLUMN IF NOT EXISTS embedding_target vector,
    ADD COLUMN IF NOT EXISTS embedding_target_profile_id TEXT,
    ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT;
ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS embedding_target vector,
    ADD COLUMN IF NOT EXISTS embedding_target_profile_id TEXT,
    ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT;
ALTER TABLE episodes
    ADD COLUMN IF NOT EXISTS embedding_target vector,
    ADD COLUMN IF NOT EXISTS embedding_target_profile_id TEXT,
    ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT;

-- Preserve an identity for the pre-v75 materialization.  It is deliberately
-- explicit that the provider/model are unknown rather than guessing history.
INSERT INTO embedding_profiles (profile_id, provider, model, dimensions, composition, state)
VALUES ('legacy', 'unknown', 'unknown', 1, '{"version":0,"text":"legacy"}', 'active')
ON CONFLICT (profile_id) DO NOTHING;
UPDATE embedding_profile_state
SET active_profile_id = COALESCE(active_profile_id, 'legacy'), updated_at = now()
WHERE state_key = 'default';
UPDATE memories SET embedding_profile_id = 'legacy'
WHERE embedding IS NOT NULL AND embedding_profile_id IS NULL;
UPDATE behaviors SET embedding_profile_id = 'legacy'
WHERE embedding IS NOT NULL AND embedding_profile_id IS NULL;
UPDATE entities SET embedding_profile_id = 'legacy'
WHERE embedding IS NOT NULL AND embedding_profile_id IS NULL;
UPDATE episodes SET embedding_profile_id = 'legacy'
WHERE embedding IS NOT NULL AND embedding_profile_id IS NULL;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
