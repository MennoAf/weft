"""Migration 39: Phase 2: provenance + review_status (poisoning-defense Layer 1/3)"""

from __future__ import annotations

VERSION = 39
DESCRIPTION = 'Phase 2: provenance + review_status (poisoning-defense Layer 1/3)'
SQL = r"""
-- Wick Phase 2 — agent-mode write-authority defense.
-- See weft_v2_spec.md "Phase 2 — Provenance + write-authority defense"
-- and the Q4 four-layer model.
--
-- Layer 1 — write-time tagging:
--   memories / behaviors / triggers grow a ``provenance`` enum that
--   matches the trackers column from migration 38: 'supervisor' for
--   trusted writes (Face / human / Orchestrator), 'agent' for writes
--   originating inside an agent container. App layer stamps the
--   value from the request's caller-mode contextvar.
--
-- Layer 3 — instruction-shape quarantine on memories:
--   ``review_status`` distinguishes ordinary writes ('active') from
--   agent-provenance writes the heuristic flagged ('pending_review').
--   Pending rows are filtered out of retrieval until promoted via
--   weft_quarantine_review. Kept orthogonal to MemoryStatus
--   (active/archived) so the soft-delete axis stays untouched.
--
-- All columns are additive with safe defaults; existing rows are
-- treated as supervisor-provenance / active-review by definition
-- (they were written before Phase 2 enforcement existed).

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS write_provenance TEXT NOT NULL DEFAULT 'supervisor',
    ADD COLUMN IF NOT EXISTS review_status    TEXT NOT NULL DEFAULT 'active';

ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_write_provenance_check;
ALTER TABLE memories
    ADD CONSTRAINT memories_write_provenance_check
    CHECK (write_provenance IN ('supervisor', 'agent'));

ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_review_status_check;
ALTER TABLE memories
    ADD CONSTRAINT memories_review_status_check
    CHECK (review_status IN ('active', 'pending_review'));

CREATE INDEX IF NOT EXISTS idx_memories_write_provenance
    ON memories (write_provenance) WHERE write_provenance = 'agent';
CREATE INDEX IF NOT EXISTS idx_memories_review_status
    ON memories (review_status) WHERE review_status = 'pending_review';

ALTER TABLE behaviors
    ADD COLUMN IF NOT EXISTS write_provenance TEXT NOT NULL DEFAULT 'supervisor';
ALTER TABLE behaviors
    DROP CONSTRAINT IF EXISTS behaviors_write_provenance_check;
ALTER TABLE behaviors
    ADD CONSTRAINT behaviors_write_provenance_check
    CHECK (write_provenance IN ('supervisor', 'agent'));
CREATE INDEX IF NOT EXISTS idx_behaviors_write_provenance
    ON behaviors (write_provenance) WHERE write_provenance = 'agent';

ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS write_provenance TEXT NOT NULL DEFAULT 'supervisor';
ALTER TABLE triggers
    DROP CONSTRAINT IF EXISTS triggers_write_provenance_check;
ALTER TABLE triggers
    ADD CONSTRAINT triggers_write_provenance_check
    CHECK (write_provenance IN ('supervisor', 'agent'));
CREATE INDEX IF NOT EXISTS idx_triggers_write_provenance
    ON triggers (write_provenance) WHERE write_provenance = 'agent';
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
