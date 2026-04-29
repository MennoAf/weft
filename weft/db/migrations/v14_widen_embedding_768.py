"""Migration 14: Widen embedding columns from vector(384) to vector(768) for OpenAI provider"""

from __future__ import annotations

VERSION = 14
DESCRIPTION = 'Widen embedding columns from vector(384) to vector(768) for OpenAI provider'
SQL = r"""
-- Drop existing HNSW indexes (cannot ALTER type with index present)
DROP INDEX IF EXISTS idx_memories_embedding_hnsw;
DROP INDEX IF EXISTS idx_behaviors_embedding_hnsw;
DROP INDEX IF EXISTS idx_entities_embedding_hnsw;

-- Null out existing embeddings (384-dim vectors can't be cast to 768-dim;
-- run `weft re-embed` after this migration to regenerate)
UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL;
UPDATE behaviors SET embedding = NULL WHERE embedding IS NOT NULL;
UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL;

-- Widen columns: 384 → 768 (Matryoshka-truncated OpenAI embeddings)
ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);
ALTER TABLE behaviors ALTER COLUMN embedding TYPE vector(768);
ALTER TABLE entities ALTER COLUMN embedding TYPE vector(768);

-- Recreate HNSW indexes at new dimension
CREATE INDEX idx_memories_embedding_hnsw
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_behaviors_embedding_hnsw
ON behaviors USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_entities_embedding_hnsw
ON entities USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
