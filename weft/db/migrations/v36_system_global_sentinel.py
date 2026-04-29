"""Migration 36: Schema v1: SYSTEM_GLOBAL sentinel + NOT NULL user_id + RLS rewrite"""

from __future__ import annotations

VERSION = 36
DESCRIPTION = 'Schema v1: SYSTEM_GLOBAL sentinel + NOT NULL user_id + RLS rewrite'
SQL = r"""
-- Kill the implicit-global path. Before this migration, ``user_id IS
-- NULL`` was the convention for "global / seed / system-owned" rows,
-- which meant any agent that forgot to set ``app.user_id`` silently
-- wrote a globally readable row. After: every row has a non-null
-- ``user_id``; the literal string ``__system_global_zathras__`` is
-- the sentinel for system-owned rows. Forgetting to set
-- ``app.user_id`` becomes a NOT NULL constraint violation — fail
-- loud, not silent leak. Named-string (not UUID) so an agent cannot
-- accidentally land on it via "just generate a UUID."

-- Step 1: backfill every NULL across all user-scoped tables.
UPDATE memories                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE memory_relationships      SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE behaviors                 SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE entities                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE entity_mentions           SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE episodes                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE episode_memories          SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE modes                     SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE alerts                    SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE check_ins                 SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE autonomy_policies         SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE policy_calibration_events SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE cost_entries              SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE triggers                  SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE calibration_records       SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;
UPDATE degradation_policies      SET user_id = '__system_global_zathras__' WHERE user_id IS NULL;

-- Step 2: NOT NULL on every user-scoped table, plus a column DEFAULT
-- so INSERTs that don't explicitly specify user_id pick up the
-- session's app.user_id. (When app.user_id is unset, the default
-- evaluates to NULL → NOT NULL violation → fail loud.) This keeps
-- the safety property while removing boilerplate from the call sites.
ALTER TABLE memories                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE memory_relationships      ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE behaviors                 ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE entities                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE entity_mentions           ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE episodes                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE episode_memories          ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE modes                     ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE alerts                    ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE check_ins                 ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE autonomy_policies         ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE policy_calibration_events ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE cost_entries              ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE triggers                  ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE calibration_records       ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE degradation_policies      ALTER COLUMN user_id SET DEFAULT nullif(current_setting('app.user_id', true), ''), ALTER COLUMN user_id SET NOT NULL;

-- Step 3: rewrite RLS policies. Old form was
--   user_id IS NULL OR user_id = current_setting
-- which silently passed for unauthenticated writes. New form is
--   user_id = current_setting OR user_id = SYSTEM_GLOBAL
-- where SYSTEM_GLOBAL is a literal string only writable when an
-- operator explicitly sets ``app.user_id`` to it. (current_setting
-- returns the empty string when unset, which matches nothing.)

-- memories: SELECT keeps the workspace-membership branch from migration 35.
DROP POLICY IF EXISTS memories_select ON memories;
CREATE POLICY memories_select ON memories FOR SELECT
    USING (
        user_id = '__system_global_zathras__'
        OR user_id = nullif(current_setting('app.user_id', true), '')
        OR (
            workspace_id IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM workspace_members wm
                WHERE wm.workspace_id = memories.workspace_id
                  AND wm.member_identity->>'user_id'
                      = nullif(current_setting('app.user_id', true), '')
            )
        )
    );

DROP POLICY IF EXISTS memories_insert ON memories;
CREATE POLICY memories_insert ON memories FOR INSERT
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''));
DROP POLICY IF EXISTS memories_update ON memories;
CREATE POLICY memories_update ON memories FOR UPDATE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));
DROP POLICY IF EXISTS memories_delete ON memories;
CREATE POLICY memories_delete ON memories FOR DELETE
    USING (user_id = nullif(current_setting('app.user_id', true), ''));

-- All other tables: simple sentinel-or-self policy. Generated via
-- a DO block to keep the migration short.
DO $rls$
DECLARE
    t TEXT;
    tables TEXT[] := ARRAY[
        'memory_relationships',
        'behaviors',
        'entities',
        'entity_mentions',
        'episodes',
        'episode_memories',
        'modes',
        'alerts',
        'check_ins',
        'autonomy_policies',
        'policy_calibration_events',
        'cost_entries',
        'triggers',
        'calibration_records',
        'degradation_policies'
    ];
BEGIN
    FOREACH t IN ARRAY tables LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I_select ON %I', t, t);
        EXECUTE format('DROP POLICY IF EXISTS %I_insert ON %I', t, t);
        EXECUTE format('DROP POLICY IF EXISTS %I_update ON %I', t, t);
        EXECUTE format('DROP POLICY IF EXISTS %I_delete ON %I', t, t);
        -- Some legacy migrations used different policy names.
        EXECUTE format('DROP POLICY IF EXISTS calibration_events_select ON %I', t);
        EXECUTE format('DROP POLICY IF EXISTS calibration_events_insert ON %I', t);
        EXECUTE format('DROP POLICY IF EXISTS calibration_events_update ON %I', t);
        EXECUTE format('DROP POLICY IF EXISTS calibration_events_delete ON %I', t);

        EXECUTE format($p$
            CREATE POLICY %I_select ON %I FOR SELECT
            USING (user_id = '__system_global_zathras__'
                   OR user_id = nullif(current_setting('app.user_id', true), ''))
        $p$, t, t);
        EXECUTE format($p$
            CREATE POLICY %I_insert ON %I FOR INSERT
            WITH CHECK (user_id = nullif(current_setting('app.user_id', true), ''))
        $p$, t, t);
        EXECUTE format($p$
            CREATE POLICY %I_update ON %I FOR UPDATE
            USING (user_id = nullif(current_setting('app.user_id', true), ''))
        $p$, t, t);
        EXECUTE format($p$
            CREATE POLICY %I_delete ON %I FOR DELETE
            USING (user_id = nullif(current_setting('app.user_id', true), ''))
        $p$, t, t);
    END LOOP;
END
$rls$;

-- Drop the now-stale partial unique index on modes (was scoped to
-- ``WHERE user_id IS NULL``; after backfill nothing matches).
-- Replace with one keyed to the sentinel.
DROP INDEX IF EXISTS uq_modes_null_user_name;
CREATE UNIQUE INDEX IF NOT EXISTS uq_modes_global_user_name
    ON modes (name) WHERE user_id = '__system_global_zathras__';
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
