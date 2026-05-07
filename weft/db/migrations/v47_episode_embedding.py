"""Migration 47: episodes.embedding column + HNSW index.

Phase 2 Track 1 — give the ``episodes`` table its own embedding column so
semantic search can hit the episode tier directly. Belief-tier (memories)
and turn-tier (episode_turns) already carry vector(768) columns; episodes
were the missing middle.

Schema additions (SQL-only — idempotent):

* ``embedding vector(768) NULL`` on ``episodes`` — populated post-hoc by
  the backfill helper in ``weft.db.reembed`` so the migration itself
  never has to call out to a network embedding provider.
* HNSW index ``idx_episodes_embedding_hnsw`` matching the project's
  established ``(m=16, ef_construction=64)`` parameters used on
  memories / behaviors / entities / episode_turns. Wrapped in
  ``IF NOT EXISTS`` so re-runs are no-ops.

Backfill is **not** in this SQL block. It runs at server startup via
``weft.db.reembed.auto_reembed``, which now includes ``episodes`` in its
table map. The text fed to the embedder is
``title || ' ' || COALESCE(summary, '')`` — see ``TABLE_TEXT_EXPRESSIONS``
in ``weft.db.reembed``. The backfill is idempotent (skips rows whose
``embedding`` column is already set) and chunked (batch_size=100).

Embedder selection for the backfill is delegated to
``weft.db.reembed.resolve_episode_embedder`` so future episode-tier
providers can diverge from the memory-tier provider via the
``WEFT_EPISODE_EMBEDDER`` env var without touching this file.

Spec: P2.1 (Loom task loom-8a7036c5).
"""

from __future__ import annotations

VERSION = 47
DESCRIPTION = "episodes.embedding vector(768) column + HNSW index"
SQL = r"""
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS embedding vector(768);

CREATE INDEX IF NOT EXISTS idx_episodes_embedding_hnsw
    ON episodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
