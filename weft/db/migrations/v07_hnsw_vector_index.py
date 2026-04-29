"""Migration 7: Set embedding dimension and add HNSW index for vector search"""

from __future__ import annotations

VERSION = 7
DESCRIPTION = 'Set embedding dimension and add HNSW index for vector search'
SQL = r"""
ALTER TABLE memories ALTER COLUMN embedding TYPE vector(384);

CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
