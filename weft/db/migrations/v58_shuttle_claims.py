"""Migration 58: shuttle_claims table — Shuttle's current-value blackboard.

Shuttle (the "loop of loops" personal-assistant layer) needs a versioned
current-value store with freshness and supersession — the same shape as
``belief_claims`` — but written by *scheduler-driven observers*, not the
LLM detector pipeline. ``belief_claims`` cannot serve that role: its
``evidence_turn_ids CHECK (cardinality > 0)``, ``source_provenance CHECK
IN (...)`` and ``detector_version NOT NULL`` invariants are detector-contract
constraints a pure observer write structurally cannot satisfy. So Shuttle
gets its own isolated table; the detector's ``belief_claims`` is untouched.

Design choices (deliberately mirror ``belief_claims`` where the semantics
match, v48 line refs in parens):

* ``claim_id`` uses an ``sc-{shortid}`` prefix — tier-specific prefixes make
  log correlation unambiguous (same convention as ``belief-`` / ``et-``).

* ``user_id NOT NULL`` with the post-v36 GUC default — unset ``app.user_id``
  is a fail-loud constraint violation, not a silent global leak (v48:14-16).

* ``superseded_by`` + ``status`` are redundant on purpose: ``status`` drives
  the partial-unique index so the read path never walks the chain to find the
  current value; ``superseded_by`` is the forward pointer for reconstruction
  (v48:23-26).

* ``occurred_at`` is separate from ``created_at``: supersession ordering uses
  ``occurred_at`` (when the observed event happened) as the authoritative axis
  (v48:28-32).

* ``error_at`` (nullable, NET-NEW vs belief_claims) supports the freshness
  gate's ERRORED-before-age check: a failing-but-last-known-good leaf is
  ERRORED, not merely STALE. The evaluator checks ``error_at > occurred_at``
  *before* age. Used in v1 by the dispatcher's observer-error path.

* ``inputs_hash`` (nullable, NET-NEW, reserved) is the synthesis content-hash
  dedup gate. Unused in v1 (observe-only); present now so the v2 synthesis
  path lands with no migration. Always NULL for leaf/observer claims.

* RLS mirrors ``belief_claims`` (v48:95-112): SELECT allows the system
  sentinel so service-side callers can enumerate; INSERT/UPDATE/DELETE gate
  strictly on ``user_id = current_setting('app.user_id')``.

Canonical design: Weft memory weft-8a20222a (amended by weft-705d822a).
Shuttle DESIGN.md §2 (one owner per concern), §5 (net-new surface).
"""

from __future__ import annotations

VERSION = 58
DESCRIPTION = "shuttle_claims table: Shuttle observer/synthesis current-value blackboard"
SQL = r"""
CREATE TABLE IF NOT EXISTS shuttle_claims (
    claim_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    loop_id        TEXT NOT NULL,
    attribute      TEXT NOT NULL,
    value          JSONB NOT NULL,
    scope          TEXT NOT NULL DEFAULT 'global',
    superseded_by  TEXT REFERENCES shuttle_claims(claim_id),
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'superseded', 'retracted')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    occurred_at    TIMESTAMPTZ NOT NULL,
    error_at       TIMESTAMPTZ,
    inputs_hash    TEXT
);

-- Current-value lookup: "what is the loop's answer for attribute X right now?"
-- Partial unique enforces the single-active-claim invariant per (user, attribute, scope).
CREATE UNIQUE INDEX IF NOT EXISTS idx_shuttle_claims_current
    ON shuttle_claims (user_id, attribute, scope)
    WHERE status = 'active';

-- Chain reconstruction: walk history for a given (user, attribute, scope).
CREATE INDEX IF NOT EXISTS idx_shuttle_claims_chain
    ON shuttle_claims (user_id, attribute, scope, occurred_at DESC);

-- Per-loop observability: "what has loop X produced?" (drift/health surface, §5.7).
CREATE INDEX IF NOT EXISTS idx_shuttle_claims_loop
    ON shuttle_claims (user_id, loop_id, occurred_at DESC);

ALTER TABLE shuttle_claims ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS shuttle_claims_select ON shuttle_claims;
CREATE POLICY shuttle_claims_select ON shuttle_claims FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS shuttle_claims_insert ON shuttle_claims;
CREATE POLICY shuttle_claims_insert ON shuttle_claims FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS shuttle_claims_update ON shuttle_claims;
CREATE POLICY shuttle_claims_update ON shuttle_claims FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS shuttle_claims_delete ON shuttle_claims;
CREATE POLICY shuttle_claims_delete ON shuttle_claims FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
