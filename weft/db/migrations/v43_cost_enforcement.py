"""Migration 43: autonomy_overrides + cost_enforcement_state tables.

Splits *intent* (autonomy_policies — durable, calibration-driven) from
*situational circuit breakers* (autonomy_overrides — TTL'd, source-tagged).
get_effective_tier() consults overrides first, then policies, then the
EARNED default. Federation-friendly: foreign agents can read a single
resolution stack and understand both the normal tier and what's currently
suppressing it, without inspecting the firing agent's config.

cost_enforcement_state holds one row per (state_date, user_id) and is
the daily idempotency anchor for the cost enforcement loop. Without it,
every tick past a threshold would re-fire and re-create overrides.

Both tables follow the post-mig-36 contract: user_id NOT NULL with the
session-GUC default, and SELECT/INSERT/UPDATE/DELETE policies that pin
to the GUC (no legacy ``user_id IS NULL`` global path). Pinned by
tests/test_rls_invariants.py.
"""

from __future__ import annotations

VERSION = 43
DESCRIPTION = (
    "autonomy_overrides + cost_enforcement_state tables for the "
    "cost-to-autonomy/degradation enforcement feedback loop"
)
SQL = r"""
-- ---------------------------------------------------------------
-- autonomy_overrides — TTL'd circuit breakers consulted by
-- get_effective_tier() before falling back to autonomy_policies.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS autonomy_overrides (
    id              TEXT PRIMARY KEY,
    action          TEXT NOT NULL,
    effective_tier  TEXT NOT NULL,
    source          TEXT NOT NULL,
    reason          TEXT,
    expires_at      TIMESTAMPTZ NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    project_id      TEXT,
    agent_id        TEXT,
    user_id         TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- now() isn't IMMUTABLE so we can't use it in a partial-index predicate.
-- Composite (action, expires_at) lets the resolver scan a small slice.
CREATE INDEX IF NOT EXISTS idx_autonomy_overrides_action_expires
    ON autonomy_overrides (action, expires_at);
CREATE INDEX IF NOT EXISTS idx_autonomy_overrides_source
    ON autonomy_overrides (source);
CREATE INDEX IF NOT EXISTS idx_autonomy_overrides_expires
    ON autonomy_overrides (expires_at);
CREATE INDEX IF NOT EXISTS idx_autonomy_overrides_user
    ON autonomy_overrides (user_id);

ALTER TABLE autonomy_overrides ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS autonomy_overrides_select ON autonomy_overrides;
CREATE POLICY autonomy_overrides_select ON autonomy_overrides FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_overrides_insert ON autonomy_overrides;
CREATE POLICY autonomy_overrides_insert ON autonomy_overrides FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_overrides_update ON autonomy_overrides;
CREATE POLICY autonomy_overrides_update ON autonomy_overrides FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS autonomy_overrides_delete ON autonomy_overrides;
CREATE POLICY autonomy_overrides_delete ON autonomy_overrides FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

-- ---------------------------------------------------------------
-- cost_enforcement_state — one row per (state_date, user_id).
-- max_threshold_fired_pct prevents same-band re-fires within a day.
-- actions_taken is a jsonb log of what we did, for audit/primer.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cost_enforcement_state (
    id                       TEXT PRIMARY KEY,
    state_date               DATE NOT NULL,
    user_id                  TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    max_threshold_fired_pct  DOUBLE PRECISION NOT NULL DEFAULT 0,
    daily_limit_usd          DOUBLE PRECISION NOT NULL,
    last_pct_used            DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    actions_taken            JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_cost_enforcement_state_date_user
    ON cost_enforcement_state (state_date, user_id);

CREATE INDEX IF NOT EXISTS idx_cost_enforcement_state_date
    ON cost_enforcement_state (state_date DESC);

ALTER TABLE cost_enforcement_state ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS cost_enforcement_state_select ON cost_enforcement_state;
CREATE POLICY cost_enforcement_state_select ON cost_enforcement_state FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_enforcement_state_insert ON cost_enforcement_state;
CREATE POLICY cost_enforcement_state_insert ON cost_enforcement_state FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_enforcement_state_update ON cost_enforcement_state;
CREATE POLICY cost_enforcement_state_update ON cost_enforcement_state FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS cost_enforcement_state_delete ON cost_enforcement_state;
CREATE POLICY cost_enforcement_state_delete ON cost_enforcement_state FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
