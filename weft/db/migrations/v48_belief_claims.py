"""Migration 48: belief_claims table — FK-provenance anchor for belief-view.

The ``belief_claims`` table is the persistence layer for shape #1 of the
four-shape memory framework (see ``docs/architecture/belief_view.md``).
Each row represents one extracted belief about a user, anchored to the
dialogue turn(s) from which it was derived.

Design choices:

* ``claim_id`` uses a ``belief-{shortid}`` prefix — tier-specific prefixes
  make log correlation unambiguous without a type column, same convention as
  ``episode_turns.id = et-{shortid}``.

* ``user_id`` follows the post-migration-36 convention: ``NOT NULL`` with a
  GUC default so unset ``app.user_id`` is a fail-loud constraint violation,
  not a silent global-leak. See v45 line 46 for the exact pattern.

* ``evidence_turn_ids TEXT[]`` carries a hard ``cardinality > 0`` check.
  The non-empty invariant is enforced at the DB level here and at the
  detector contract level in the writer. This is the provenance anchor:
  a claim without a source turn cannot be verified and must not exist.

* ``superseded_by`` and ``status`` are redundant on purpose. ``status``
  drives the partial unique index so the query path never needs to walk the
  chain to find the current value. ``superseded_by`` is the forward pointer
  for chain reconstruction. Neither alone is sufficient.

* ``occurred_at`` is separate from ``created_at`` because batch-ingested
  claims (delayed Slack ingest, historical fixtures, offline session replay)
  arrive with a lag between when the turn happened and when the claim was
  materialized. Supersession ordering uses ``occurred_at`` as the
  authoritative "when did this event occur" axis.

* RLS mirrors v45 ``episode_turns_*`` policies: SELECT allows the system
  sentinel (``__system_global_zathras__``) so service-side callers that set
  that sentinel can enumerate all users' claims; INSERT/UPDATE/DELETE gate
  strictly on ``user_id = current_setting('app.user_id')``.

* An additional GIN index on ``evidence_turn_ids`` (filtered to
  active/superseded claims) lets the prune-guard EXISTS subquery in
  ``weft.episode_turns`` hit an index rather than seqscan. The filter
  ``WHERE status IN ('active', 'superseded')`` keeps the GIN index lean —
  retracted claims are excluded because retracted claims do not anchor turns.

Spec: ``docs/architecture/belief_view.md`` §1 and §6.4.
Loom task: loom-040b0ca4.
"""

from __future__ import annotations

VERSION = 48
DESCRIPTION = "belief_claims table: FK-provenance anchor for belief-view"
SQL = r"""
CREATE TABLE IF NOT EXISTS belief_claims (
    claim_id           TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), ''),
    attribute          TEXT NOT NULL,
    value              JSONB NOT NULL,
    scope              TEXT NOT NULL DEFAULT 'global',
    evidence_turn_ids  TEXT[] NOT NULL
                       CHECK (cardinality(evidence_turn_ids) > 0),
    superseded_by      TEXT REFERENCES belief_claims(claim_id),
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'superseded', 'retracted')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    occurred_at        TIMESTAMPTZ NOT NULL,
    source_provenance  TEXT NOT NULL
                       CHECK (source_provenance IN ('user_stated', 'agent_suggested', 'joint_decision')),
    detector_confidence REAL NOT NULL DEFAULT 1.0,
    detector_version   TEXT NOT NULL
);

-- Current-value lookup: "what does the user believe about X right now?"
-- Partial unique enforces the single-active-claim invariant per (user, attribute, scope).
CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_claims_current
    ON belief_claims (user_id, attribute, scope)
    WHERE status = 'active';

-- Chain reconstruction: walk history for a given (user, attribute, scope).
CREATE INDEX IF NOT EXISTS idx_belief_claims_chain
    ON belief_claims (user_id, attribute, scope, occurred_at DESC);

-- Named-artifact retrieval by attribute prefix (text_pattern_ops enables LIKE 'prefix%').
CREATE INDEX IF NOT EXISTS idx_belief_claims_attribute_prefix
    ON belief_claims (user_id, attribute text_pattern_ops);

-- Prune-guard index: lets the NOT EXISTS subquery in episode_turns prune
-- functions use a GIN index instead of seqscan.  Filtered to only the
-- statuses that anchor turns (active + superseded); retracted claims are
-- excluded so they don't bloat the index.
CREATE INDEX IF NOT EXISTS idx_belief_claims_evidence_gin
    ON belief_claims USING GIN (evidence_turn_ids)
    WHERE status IN ('active', 'superseded');

ALTER TABLE belief_claims ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS belief_claims_select ON belief_claims;
CREATE POLICY belief_claims_select ON belief_claims FOR SELECT
    USING (user_id = '__system_global_zathras__'
           OR user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS belief_claims_insert ON belief_claims;
CREATE POLICY belief_claims_insert ON belief_claims FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS belief_claims_update ON belief_claims;
CREATE POLICY belief_claims_update ON belief_claims FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

DROP POLICY IF EXISTS belief_claims_delete ON belief_claims;
CREATE POLICY belief_claims_delete ON belief_claims FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
