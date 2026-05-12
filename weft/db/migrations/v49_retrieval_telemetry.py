"""Migration 49: retrieval telemetry columns — Step 1 of the compounding loop.

Adds two columns to ``memories`` so every successful recall/search_all return
is recorded at the memory level:

* ``last_retrieved_at`` (TIMESTAMPTZ, nullable) — wall-clock of most recent
  bump. NULL for any row that has not yet been surfaced through the
  retrieval surfaces wired in this migration.

* ``retrieval_count`` (INT, default 0) — running counter of returns.

These are intentionally separate from the existing ``accessed_at`` /
``access_count`` pair (v01) and from the ``memory_access_log`` table (v16).
The existing pair is bumped by ``touch_memory`` which also nudges
``usefulness_score`` via an EMA — coupling retrieval telemetry to that
side-effecting path would muddy the signal. The new columns are pure
counters with no downstream effects; they exist only to be observed.

Decision provenance: weft-23fbb7c8 (handoff), weft-81fe480b (Anvil
reframe), weft-8dbe6343 (Lodestar 10x-GAP-on-Compounding verdict). The
companion build steps (re-ask detection, weft_health tool, ranking
weight adjustments) are deliberately NOT in this migration — Anvil's
reframe gated them on two-week baseline metrics.
"""

from __future__ import annotations

VERSION = 49
DESCRIPTION = "Retrieval telemetry columns on memories (compounding-loop Step 1)"
SQL = r"""
ALTER TABLE memories ADD COLUMN IF NOT EXISTS
    last_retrieved_at TIMESTAMPTZ DEFAULT NULL;

ALTER TABLE memories ADD COLUMN IF NOT EXISTS
    retrieval_count INTEGER NOT NULL DEFAULT 0;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
