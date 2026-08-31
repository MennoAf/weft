"""Migration 72: bounded retrieval-recovery attempt telemetry.

Recovery diagnostics are additive and append-only.  The table stores only
bounded query hashes/labels and redacted stable result identifiers; it never
stores a provider prompt, raw evidence summary, or answer text.
"""
from __future__ import annotations

VERSION = 72
DESCRIPTION = "weft_recovery_attempts: bounded retrieval-recovery stage telemetry"
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_recovery_attempts (
    attempt_id TEXT PRIMARY KEY CHECK (char_length(attempt_id) BETWEEN 1 AND 80),
    user_id TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    parent_query_id TEXT NULL REFERENCES weft_recall_queries(query_id) ON DELETE SET NULL,
    recovery_version TEXT NOT NULL CHECK (char_length(recovery_version) BETWEEN 1 AND 64),
    stage TEXT NOT NULL CHECK (stage IN ('primary','deterministic_reformulation','alternate_tier','model_planner')),
    trigger TEXT NOT NULL CHECK (char_length(trigger) BETWEEN 1 AND 96),
    answerability TEXT NOT NULL CHECK (answerability IN ('sufficient','insufficient_evidence','conflicting_evidence','abstained')),
    query_hash TEXT NOT NULL CHECK (query_hash ~ '^[0-9a-f]{64}$'),
    query_label TEXT NOT NULL CHECK (char_length(query_label) BETWEEN 1 AND 160),
    result_ids JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_array_length(result_ids) <= 24),
    coverage JSONB NOT NULL DEFAULT '{}' CHECK (octet_length(coverage::text) <= 4096),
    requested_project_id TEXT NULL CHECK (requested_project_id IS NULL OR char_length(requested_project_id) <= 256),
    resolved_project_id TEXT NULL CHECK (resolved_project_id IS NULL OR char_length(resolved_project_id) <= 256),
    project_policy TEXT NOT NULL CHECK (project_policy IN ('facet_boost','hard_wall','baseline')),
    retrieval_mode TEXT NOT NULL CHECK (retrieval_mode IN ('face','code','all')),
    source_allowlist JSONB NOT NULL DEFAULT '[]' CHECK (octet_length(source_allowlist::text) <= 2048),
    provider_calls INTEGER NOT NULL DEFAULT 0 CHECK (provider_calls BETWEEN 0 AND 1),
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens BETWEEN 0 AND 20000),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens BETWEEN 0 AND 8000),
    latency_ms INTEGER NOT NULL CHECK (latency_ms BETWEEN 0 AND 10000),
    error_category TEXT NULL CHECK (error_category IS NULL OR char_length(error_category) <= 96),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recovery_attempts_user_time
    ON weft_recovery_attempts (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recovery_attempts_parent_time
    ON weft_recovery_attempts (parent_query_id, created_at);

ALTER TABLE weft_recovery_attempts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_recovery_attempts_select ON weft_recovery_attempts;
CREATE POLICY weft_recovery_attempts_select ON weft_recovery_attempts FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recovery_attempts_insert ON weft_recovery_attempts;
CREATE POLICY weft_recovery_attempts_insert ON weft_recovery_attempts FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recovery_attempts_update ON weft_recovery_attempts;
CREATE POLICY weft_recovery_attempts_update ON weft_recovery_attempts FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS weft_recovery_attempts_delete ON weft_recovery_attempts;
CREATE POLICY weft_recovery_attempts_delete ON weft_recovery_attempts FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
