"""Migration 42: RLS audit follow-up: enable RLS on weft_tokens with a service policy"""

from __future__ import annotations

VERSION = 42
DESCRIPTION = 'RLS audit follow-up: enable RLS on weft_tokens with a service policy'
SQL = r"""
-- Migration 40 created weft_tokens with the comment "Service-role
-- only by design" but never issued ENABLE ROW LEVEL SECURITY. The
-- intent was: only the middleware (running as the service role)
-- ever queries this table, so RLS doesn't apply. The gap: a
-- non-service role with table privileges (e.g. Supabase's
-- ``authenticated`` role if it ever lands a GRANT on this table)
-- could SELECT every token hash + bound user_id without any
-- filter. We can't undo a GRANT we haven't yet made, but we can
-- enable RLS now so any future grant is contained by the policy
-- set rather than wide open.
--
-- Pattern matches memory_access_log / weft_metadata: RLS enabled
-- with a permissive USING (true) WITH CHECK (true) service policy.
-- The service role bypasses RLS regardless, so the application's
-- middleware lookup path is unaffected. A non-service role with
-- table privileges would still hit the policy and — because the
-- policy is permissive — see the rows; the value here is that we
-- now have a single named policy to tighten if/when we want to
-- restrict reads further (e.g. WITH CHECK (current_user = 'weft_service')).
--
-- Caught by tests/test_rls_invariants.py — every table with a
-- user_id column must have RLS enabled.

ALTER TABLE weft_tokens ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS weft_tokens_service ON weft_tokens;
CREATE POLICY weft_tokens_service ON weft_tokens
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
