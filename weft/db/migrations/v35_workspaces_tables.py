"""Migration 35: Schema v1: workspaces + workspace_members tables"""

from __future__ import annotations

VERSION = 35
DESCRIPTION = 'Schema v1: workspaces + workspace_members tables'
SQL = r"""
-- Minimum workspace primitive. Two tables; no role hierarchy beyond
-- 'member'; no invitation flow; no expiry. Owner inserts members
-- directly. v1 use case: AIO Cleanroom shared brain with Brandon.
--
-- member_identity is JSONB so it can carry remote-install members
-- once federation lands ({"kind":"remote_install","install_pubkey":
-- "...","member_uuid":"..."}). For local users it's
-- {"kind":"local_user","user_id":"<uuid>"}.

CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    created_by      TEXT NOT NULL,
    install_pubkey  TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_workspaces_created_by
    ON workspaces (created_by);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    member_identity JSONB NOT NULL,
    role            TEXT NOT NULL DEFAULT 'member',
    added_by        TEXT NOT NULL,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, member_identity)
);
CREATE INDEX IF NOT EXISTS idx_workspace_members_user
    ON workspace_members ((member_identity->>'user_id'))
    WHERE member_identity->>'user_id' IS NOT NULL;

-- FK from memories.workspace_id (added in migration 34) to workspaces.
-- ON DELETE SET NULL: deleting a workspace orphans the memories back
-- to private/global scope rather than destroying them.
ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_workspace_fk;
ALTER TABLE memories
    ADD CONSTRAINT memories_workspace_fk
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE SET NULL;

-- RLS: service-role for now. App layer enforces "who can read
-- workspace metadata" via tool-level checks. A v2 tightening pass
-- can scope these to "members only" via subquery policies.
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS workspaces_service ON workspaces;
CREATE POLICY workspaces_service ON workspaces
    USING (true) WITH CHECK (true);

ALTER TABLE workspace_members ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS workspace_members_service ON workspace_members;
CREATE POLICY workspace_members_service ON workspace_members
    USING (true) WITH CHECK (true);

-- Extend memories RLS to include workspace membership. A row is
-- visible if (a) globally scoped, (b) user-owned, or (c) lives in a
-- workspace the current user is a member of. The workspace_members
-- subquery hits the service policy above (USING true) so it works
-- under the normal user connection.
DROP POLICY IF EXISTS memories_select ON memories;
CREATE POLICY memories_select ON memories FOR SELECT
    USING (
        user_id IS NULL
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

-- INSERT/UPDATE/DELETE policies stay strict: only the row owner can
-- write. Workspace members can read but not directly mutate other
-- members' rows. Cross-member writes happen via app-level tools
-- that act as the row author.
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
