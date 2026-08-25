"""Migration 73: restrict credential-token RLS to the hosted app role."""

from __future__ import annotations

VERSION = 73
DESCRIPTION = "Restrict weft_tokens RLS policy to hosted application role"
SQL = r"""
-- Token hashes and their bound user IDs are credential material. Hosted Weft
-- connects as the explicitly provisioned, non-owner ``weft_app`` role; owner
-- migrations run out of band. The explicit current_user predicate keeps the
-- policy present on local databases too, while denying every non-hosted role.
ALTER TABLE weft_tokens ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS weft_tokens_service ON weft_tokens;
CREATE POLICY weft_tokens_service ON weft_tokens
    TO PUBLIC
    USING (current_user = 'weft_app')
    WITH CHECK (current_user = 'weft_app');
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
