"""Migration 37: Schema v1: smart default for memories.author_identity"""

from __future__ import annotations

VERSION = 37
DESCRIPTION = 'Schema v1: smart default for memories.author_identity'
SQL = r"""
-- Migration 34 added author_identity with a placeholder default of
-- ``{"kind":"unknown"}``. Now that user_id is NOT NULL with a
-- session-derived default (migration 36), we can compute a real
-- author_identity at INSERT time from the same session value.
--
-- Rows the SYSTEM_GLOBAL sentinel writes get ``{"kind":"system",...}``;
-- everything else gets ``{"kind":"local_user","user_id":"..."}``.
-- Call sites that want to override (e.g. to record an agent acting
-- on a user's behalf, ``{"kind":"agent","on_behalf_of":"..."}``)
-- can still pass author_identity explicitly.

ALTER TABLE memories
    ALTER COLUMN author_identity SET DEFAULT
    CASE
        WHEN nullif(current_setting('app.user_id', true), '')
             = '__system_global_zathras__'
        THEN '{"kind":"system","component":"runtime"}'::jsonb
        WHEN nullif(current_setting('app.user_id', true), '') IS NOT NULL
        THEN jsonb_build_object(
            'kind', 'local_user',
            'user_id', current_setting('app.user_id', true)
        )
        ELSE '{"kind":"unknown"}'::jsonb
    END;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
