"""Migration 60: canonicalize entity tags in memories.topic[] to lowercase.

Backfill half of the entity-tag casing fix (latent bug weft-6318d198). Entity
tags were written as ``entity:{e.name}`` preserving the entity's mixed-case
display name (``entity:Windward``, ``entity:weft``, ``entity:R0.1``), while the
``weft_status`` topic resolver naive-normalized to a Title-cased ``entity:Weft``
and the gather did a case-sensitive ``= ANY(topic)`` exact match. So the entity
branch only matched when the stored name happened to be Title-case and silently
missed every lowercase one.

The forward fix is canonical lowercase: the resolver now emits ``entity:{lower}``
(weft/topic_resolution.py) and ingest now writes ``entity:{e.name.lower()}``
(weft/ingest_pipeline.py). This migration brings pre-existing rows into line so
the exact match works uniformly across historical data.

What it does: for every ``memories`` row whose ``topic[]`` contains at least one
mixed-case ``entity:*`` element, rebuild the array lowercasing ONLY the
``entity:*`` elements (every other tag — ``intent:*``, ``custom:*``, free tags —
is left byte-for-byte unchanged). Array order is preserved via WITH ORDINALITY.
Idempotent: the WHERE guard skips rows already fully lowercased, so a second run
is a no-op. The v56 GIN index on ``topic`` keeps exact-match lookups fast; this
only changes element casing, not the column type.
"""

from __future__ import annotations

VERSION = 60
DESCRIPTION = "canonicalize entity:* tags in memories.topic[] to lowercase"
SQL = r"""
UPDATE memories AS m
SET topic = (
    SELECT array_agg(
               CASE
                   WHEN t LIKE 'entity:%' THEN lower(t)
                   ELSE t
               END
               ORDER BY ord
           )
    FROM unnest(m.topic) WITH ORDINALITY AS u(t, ord)
)
WHERE EXISTS (
    SELECT 1
    FROM unnest(m.topic) AS e(t)
    WHERE e.t LIKE 'entity:%'
      AND e.t <> lower(e.t)
);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
