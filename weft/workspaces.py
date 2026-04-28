"""Workspace store — single writer for the workspaces and workspace_members tables.

Workspaces are the shared-brain primitive: a named bucket of memories that
multiple user_ids can read. Membership lives in ``workspace_members`` keyed
on ``(workspace_id, member_identity)`` where ``member_identity`` is JSONB so
remote-install members can be added later without a schema change.

Permission model (v1, intentionally minimal):
- Anyone authenticated can call ``create_workspace`` — they become ``created_by``
  and are auto-inserted as a member with role ``admin``.
- Only the workspace owner (``created_by``) can ``add_member`` or
  ``remove_member`` in v1. A v2 pass can promote any ``admin`` member.
- ``list_workspaces_for_user`` returns workspaces the caller is a member of.

The server-role RLS policies on these two tables are ``USING true`` (see
migration 35), so we enforce permission at the app layer here.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from weft.db.connection import get_db
from weft.models import Workspace, WorkspaceMember

logger = logging.getLogger(__name__)


def _workspace_id() -> str:
    return f"ws-{uuid.uuid4().hex[:10]}"


async def create_workspace(
    pool: asyncpg.Pool,
    *,
    name: str,
    created_by: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Workspace:
    """Create a workspace and auto-add the creator as an admin member."""
    ws_id = _workspace_id()
    now = datetime.now(timezone.utc)
    md = metadata or {}

    db = get_db(pool)
    await db.execute(
        """
        INSERT INTO workspaces (
            id, name, description, created_by, metadata,
            created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $6)
        """,
        ws_id, name, description, created_by, json.dumps(md), now,
    )

    await db.execute(
        """
        INSERT INTO workspace_members (
            workspace_id, member_identity, role, added_by, added_at
        ) VALUES ($1, $2::jsonb, 'admin', $3, $4)
        """,
        ws_id,
        json.dumps({"kind": "local_user", "user_id": created_by}),
        created_by,
        now,
    )

    return Workspace(
        id=ws_id,
        name=name,
        description=description,
        created_by=created_by,
        metadata=md,
        created_at=now,
        updated_at=now,
    )


async def get_workspace(pool: asyncpg.Pool, workspace_id: str) -> Workspace | None:
    row = await get_db(pool).fetchrow(
        "SELECT * FROM workspaces WHERE id = $1", workspace_id,
    )
    return _row_to_workspace(row) if row else None


async def is_member(
    pool: asyncpg.Pool, workspace_id: str, user_id: str,
) -> bool:
    """True if ``user_id`` is a local_user member of ``workspace_id``."""
    val = await get_db(pool).fetchval(
        """
        SELECT 1 FROM workspace_members
        WHERE workspace_id = $1
          AND member_identity->>'user_id' = $2
        LIMIT 1
        """,
        workspace_id, user_id,
    )
    return val is not None


async def add_member(
    pool: asyncpg.Pool,
    *,
    workspace_id: str,
    user_id: str,
    added_by: str,
    role: str = "member",
) -> WorkspaceMember:
    """Add a local_user member. Caller (added_by) must own the workspace.

    Raises ``PermissionError`` if added_by is not the workspace owner.
    Raises ``LookupError`` if the workspace doesn't exist.
    """
    db = get_db(pool)
    owner = await db.fetchval(
        "SELECT created_by FROM workspaces WHERE id = $1", workspace_id,
    )
    if owner is None:
        raise LookupError(f"workspace not found: {workspace_id}")
    if owner != added_by:
        raise PermissionError(
            f"only the workspace owner can add members "
            f"(workspace={workspace_id}, owner={owner}, caller={added_by})"
        )

    now = datetime.now(timezone.utc)
    identity = {"kind": "local_user", "user_id": user_id}
    await db.execute(
        """
        INSERT INTO workspace_members (
            workspace_id, member_identity, role, added_by, added_at
        ) VALUES ($1, $2::jsonb, $3, $4, $5)
        ON CONFLICT (workspace_id, member_identity) DO UPDATE
            SET role = EXCLUDED.role
        """,
        workspace_id, json.dumps(identity), role, added_by, now,
    )

    return WorkspaceMember(
        workspace_id=workspace_id,
        member_identity=identity,
        role=role,
        added_by=added_by,
        added_at=now,
    )


async def remove_member(
    pool: asyncpg.Pool,
    *,
    workspace_id: str,
    user_id: str,
    removed_by: str,
) -> bool:
    """Remove a local_user member. Caller must own the workspace.

    Returns True if a row was removed, False if no membership existed.
    Refuses to remove the owner — workspaces always have an owner-member.
    """
    db = get_db(pool)
    owner = await db.fetchval(
        "SELECT created_by FROM workspaces WHERE id = $1", workspace_id,
    )
    if owner is None:
        raise LookupError(f"workspace not found: {workspace_id}")
    if owner != removed_by:
        raise PermissionError(
            f"only the workspace owner can remove members "
            f"(workspace={workspace_id}, owner={owner}, caller={removed_by})"
        )
    if user_id == owner:
        raise PermissionError(
            "cannot remove the workspace owner — delete the workspace instead"
        )

    result = await db.execute(
        """
        DELETE FROM workspace_members
        WHERE workspace_id = $1
          AND member_identity->>'user_id' = $2
        """,
        workspace_id, user_id,
    )
    # asyncpg returns a status string like 'DELETE 1' or 'DELETE 0'
    return result.endswith(" 1")


async def list_workspaces_for_user(
    pool: asyncpg.Pool, user_id: str,
) -> list[Workspace]:
    """Return all workspaces ``user_id`` is a member of, newest first."""
    rows = await get_db(pool).fetch(
        """
        SELECT w.*
        FROM workspaces w
        JOIN workspace_members wm ON wm.workspace_id = w.id
        WHERE wm.member_identity->>'user_id' = $1
        ORDER BY w.created_at DESC
        """,
        user_id,
    )
    return [_row_to_workspace(r) for r in rows]


async def list_members(
    pool: asyncpg.Pool, workspace_id: str,
) -> list[WorkspaceMember]:
    """Return all members of a workspace."""
    rows = await get_db(pool).fetch(
        """
        SELECT * FROM workspace_members
        WHERE workspace_id = $1
        ORDER BY added_at ASC
        """,
        workspace_id,
    )
    return [_row_to_member(r) for r in rows]


def _row_to_workspace(row: asyncpg.Record) -> Workspace:
    md = row["metadata"]
    if isinstance(md, str):
        md = json.loads(md)
    return Workspace(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_by=row["created_by"],
        install_pubkey=row["install_pubkey"],
        metadata=md or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_member(row: asyncpg.Record) -> WorkspaceMember:
    identity = row["member_identity"]
    if isinstance(identity, str):
        identity = json.loads(identity)
    return WorkspaceMember(
        workspace_id=row["workspace_id"],
        member_identity=identity,
        role=row["role"],
        added_by=row["added_by"],
        added_at=row["added_at"],
    )
