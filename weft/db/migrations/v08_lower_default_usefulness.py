"""Migration 8: Lower default usefulness_score to 0.7 and backfill existing rows"""

from __future__ import annotations

VERSION = 8
DESCRIPTION = 'Lower default usefulness_score to 0.7 and backfill existing rows'
SQL = r"""
ALTER TABLE memories ALTER COLUMN usefulness_score SET DEFAULT 0.7;

UPDATE memories
SET usefulness_score = 0.7
WHERE usefulness_score >= 0.9999
  AND usefulness_score <= 1.0001
  AND (usefulness_count = 0 OR usefulness_count IS NULL);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
