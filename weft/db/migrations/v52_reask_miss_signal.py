"""Migration 52: re-ask miss signal on weft_recall_queries.

Adds a boolean column ``is_reask_miss`` to ``weft_recall_queries`` to record
when a query was identified as the *original* (missed) query in a re-ask pair.

== Why a column, not a new table? ==

The miss signal is a property of the original query row: it means "this query
was re-asked, so the retrieval that satisfied it the first time must have
failed". Stamping the row in place avoids a join, keeps the signal close to
the data it annotates, and requires no new FK references. The satisfying
memory_id (the memory that answered the *second* / successful query) is stored
alongside as ``reask_satisfying_memory_id`` — a nullable text FK-ish field (no
FK constraint because the memory may be deleted).

== Column semantics ==

* ``is_reask_miss = TRUE`` — this query was detected as the original miss in a
  re-ask pair. The satisfying memory for the follow-up re-ask is recorded in
  ``reask_satisfying_memory_id``.
* ``is_reask_miss = FALSE`` (default) — not a detected miss.
* ``reask_satisfying_memory_id`` — the memory.id that answered the second
  (successful) query, allowing usefulness-boost attribution. Nullable so
  existing rows need no update.

== What runs at re-ask detection time? ==

When ``apply_reask_feedback()`` (weft/store.py) detects a re-ask pair it:
1. Calls ``record_feedback(pool, satisfying_memory_id, helpful=True)`` — EMA
   boost (alpha=0.3) reusing the existing usefulness path. (DO NOT reimplement
   the EMA here — call the function.)
2. Stamps the original query row with ``is_reask_miss=TRUE`` and
   ``reask_satisfying_memory_id=<satisfying_memory_id>``.

== Migration provenance ==

Loom task: loom-c5a6e9e3 (L2: Wire re-ask miss into usefulness feedback).
Parent epic: loom-c5ddc68f. Depends on loom-438f4781 (L1: reask.py exists).
"""

from __future__ import annotations

VERSION = 52
DESCRIPTION = "weft_recall_queries: is_reask_miss + reask_satisfying_memory_id columns"
SQL = r"""
ALTER TABLE weft_recall_queries
    ADD COLUMN IF NOT EXISTS is_reask_miss BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE weft_recall_queries
    ADD COLUMN IF NOT EXISTS reask_satisfying_memory_id TEXT;

CREATE INDEX IF NOT EXISTS idx_recall_queries_reask_miss
    ON weft_recall_queries (is_reask_miss)
    WHERE is_reask_miss = TRUE;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
