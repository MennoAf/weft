"""Migration 54: weft_counters — global named failure/telemetry counters.

A small key/value counter table for aggregate operational telemetry —
specifically the silently-swallowed failure sites that log-and-continue
with no durable signal (see loom-3c4a0be3). Without an aggregate counter,
a persistently-broken enqueue or auto-promotion is invisible: it logs a
warning per occurrence and the surrounding path proceeds as if nothing
happened.

Design choices:

* **Global, not user-scoped** — these are system health counts (e.g.
  "replay.enqueue.failed"), not user data. They are incremented from
  swallow paths that may run outside a user context (the scheduler-driven
  auto-promotion, the future replay executor), so a NOT-NULL user_id with a
  GUC default would fail-loud exactly where we most need the counter to
  record. Mirrors the ``weft_metadata`` system-table pattern (v15 + v23):
  RLS enabled with a permissive service policy, no ``user_id`` column.
  (Deliberately NOT user-scoped — revisit if multi-tenant per-user failure
  attribution is ever needed; see issue weft-ec347009.)

* **Atomic increment** via ``INSERT ... ON CONFLICT (name) DO UPDATE
  SET count = weft_counters.count + EXCLUDED.count`` — race-safe under
  concurrent swallow paths, unlike a read-modify-write on a weft_metadata
  JSONB value.

* ``count BIGINT`` — failure counters are monotonic and may run for a long
  time before anyone resets them.

Spec: Loom task loom-3c4a0be3.
"""

from __future__ import annotations

VERSION = 54
DESCRIPTION = "weft_counters: global named telemetry counters + service RLS"
SQL = r"""
CREATE TABLE IF NOT EXISTS weft_counters (
    name        TEXT PRIMARY KEY,
    count       BIGINT NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- System table, no user data: service-role only (mirrors weft_metadata, v23).
ALTER TABLE weft_counters ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_counters_service ON weft_counters;
CREATE POLICY weft_counters_service ON weft_counters
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
