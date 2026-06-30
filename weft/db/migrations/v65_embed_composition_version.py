"""Migration 65: Add embed_composition_version to memories (RC2).

Memory embeddings were generated from ``content`` alone, which diluted the
high-signal ``topic`` tags out of long memories' vectors and made them hard to
recall by their own keywords (root cause weft-45029c15). RC2 composes embed
text as ``content + topics`` via a single shared helper
(``weft.store.embed_text_for_memory``).

This column marks which composition version produced a row's embedding so the
re-embed backfill can find stale rows (``embed_composition_version < current``):
  - 0 = legacy content-only embeddings (all existing rows backfill to this).
  - 1 = content + topics (new writes; rows after the re-embed backfill).

Non-destructive: schema only. The re-embed backfill (weft/db/reembed.py) bumps
existing rows to the current version as it re-embeds them with the new
composition. AC5 of the recall-completeness PRD gates on
``count(*) WHERE embed_composition_version < 1 = 0`` after backfill.

Spec: documents/prds/weft-recall-completeness.md (RC2, Interfaces, AC5)
"""

from __future__ import annotations

VERSION = 65
DESCRIPTION = "Add embed_composition_version SMALLINT to memories (RC2)"

SQL = r"""
-- Existing rows are content-only embeddings → version 0 (legacy).
-- New inserts pass the current version explicitly (store_memory); the DEFAULT
-- only applies to these pre-existing rows and any insert that omits the column.
ALTER TABLE memories
ADD COLUMN embed_composition_version SMALLINT NOT NULL DEFAULT 0;
"""


def up() -> str:
    return SQL


def down() -> str:
    return "ALTER TABLE memories DROP COLUMN IF EXISTS embed_composition_version;"


MIGRATION = (VERSION, DESCRIPTION, SQL)
