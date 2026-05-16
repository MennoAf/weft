"""Migration 51 (GATED — NOT auto-applied): episode_turns FTS index.

Filename intentionally does NOT match the auto-discovery pattern
``v[0-9]+_*.py`` used by ``weft/db/migrations/__init__.py:_discover``.
This file lives next to the other migrations as a draft. The migration
runner will not pick it up while named ``pending_v51_episode_turns_fts.py``.

== Promotion (apply this migration) ==

When the gate condition fires:

    git mv weft/db/migrations/pending_v51_episode_turns_fts.py \\
           weft/db/migrations/v51_episode_turns_fts.py

After the rename, ``run_migrations(pool)`` will discover and apply it
on next startup. At that point — and only then — update
``weft.episode_turns.recall_turns`` so the keyword half references the
stored ``search_tsv`` column instead of computing ``to_tsvector('english',
content)`` inline (currently at episode_turns.py:472-477).

== Gate condition ==

Apply when a single installation crosses ~25 000 turns in
``episode_turns``. Below that threshold, the inline ``to_tsvector`` call
is acceptable; above it, the un-indexed sequential scan becomes a real
recall-latency ceiling. The 25k threshold is conservative; the inline
code path's docstring (episode_turns.py:419-425) names ~100k as the
absolute ceiling. The lower number here gives headroom to ship the
change before users feel it.

== Design choice: GENERATED column, not trigger ==

v19 used a BEFORE INSERT/UPDATE trigger to maintain ``memories.search_tsv``
because memory content can be edited. ``episode_turns`` is effectively
append-only — the content of a recorded conversational turn does not
change after ingest. A ``GENERATED ALWAYS AS (...) STORED`` column gives
us the same behavior with one moving part instead of three (column +
function + trigger), and Postgres maintains it atomically with the row.

Loom: loom-336fae29 (parent epic loom-e8a5c2c0, Phase 4.1).
"""

from __future__ import annotations

VERSION = 51
DESCRIPTION = "episode_turns FTS: stored search_tsv + GIN index (gated, applied on rename)"
SQL = r"""
ALTER TABLE episode_turns
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_episode_turns_search_tsv
    ON episode_turns USING GIN (search_tsv);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
