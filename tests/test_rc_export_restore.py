"""Focused RC-FL-10 portable export/restore contract tests."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.backup import (
    BACKUP_VERSION,
    restore_all,
    verify_backup,
    verify_backup_preconditions,
)


def _valid_payload(**overrides):
    memories = [{"id": "m1", "type": "fact", "content": "portable"}]
    payload = {
        "version": BACKUP_VERSION,
        "schema_version": 74,
        "exported_at": "2026-01-01T00:00:00+00:00",
        "checksum": hashlib.sha256(
            json.dumps(["m1portable"], sort_keys=True).encode()
        ).hexdigest(),
        "memory_count": 1,
        "relationship_count": 0,
        "memories": memories,
        "relationships": [],
        "workspaces": [],
        "workspace_members": [],
        "behaviors": [],
        "entities": [],
        "entity_mentions": [],
        "episodes": [],
        "episode_memories": [],
        "modes": [],
        "trackers": [],
        "counts": {},
    }
    payload.update(overrides)
    return payload


def test_verify_rejects_credentials_and_nonportable_sections_without_echoing_values():
    payload = _valid_payload(
        weft_tokens=[{"token_hash": "do-not-emit-this-secret"}],
        audit_log=[{"payload": "private-operational-state"}],
    )

    report = verify_backup(payload)

    assert report["valid"] is False
    assert any("non-portable" in issue.lower() for issue in report["issues"])
    assert "do-not-emit-this-secret" not in " ".join(report["issues"])
    assert "private-operational-state" not in " ".join(report["issues"])


def test_verify_rejects_malformed_rows_without_key_error_or_secret_echo():
    payload = _valid_payload(memories=["malformed-row"])

    report = verify_backup(payload)

    assert report["valid"] is False
    assert any("memories[0]" in issue for issue in report["issues"])


def test_verify_accepts_legacy_10_and_11_shapes():
    for version in ("1.0", "1.1"):
        payload = _valid_payload(version=version)
        for section in (
            "workspaces",
            "workspace_members",
            "behaviors",
            "entities",
            "entity_mentions",
            "episodes",
            "episode_memories",
            "modes",
            "trackers",
            "counts",
        ):
            payload.pop(section, None)
        assert verify_backup(payload)["valid"] is True


@pytest.mark.asyncio
async def test_preconditions_reject_restricted_runtime_role_without_migration_write():
    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            return_value={
                "role": "weft_app",
                "rolsuper": False,
                "rolbypassrls": False,
            }
        ),
        fetchval=AsyncMock(),
        fetch=AsyncMock(),
    )

    with pytest.raises(PermissionError, match="owner-capable"):
        await verify_backup_preconditions(pool)

    assert not any("INSERT" in call.args[0] or "UPDATE" in call.args[0]
                   for call in pool.fetchval.await_args_list)


class _CollisionConn:
    def __init__(self):
        self.executed: list[str] = []

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if sql.startswith("SET LOCAL"):
            return "SET"
        return "INSERT 0 0"

    async def fetch(self, sql, *args):
        return [{"column_name": "id", "data_type": "text", "udt_name": "text"}]

    async def fetchrow(self, sql, *args):
        return {"role": "postgres", "rolsuper": True, "rolbypassrls": True}

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _CollisionPool:
    def __init__(self):
        self.conn = _CollisionConn()

    def acquire(self):
        return self

    async def fetchrow(self, sql, *args):
        return await self.conn.fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        if "information_schema.tables" in sql:
            return True
        if "schema_migrations" in sql:
            from weft.db.migrations import MIGRATIONS
            return [v for v, _, _ in MIGRATIONS]
        return None

    async def fetch(self, sql, *args):
        if "schema_migrations" in sql:
            from weft.db.migrations import MIGRATIONS
            return [{"version": v} for v, _, _ in MIGRATIONS]
        return await self.conn.fetch(sql, *args)

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_strict_collision_is_explicit_and_no_replace_mode_exists(monkeypatch):
    pool = _CollisionPool()
    monkeypatch.setattr("weft.backup._introspect_columns", AsyncMock(return_value={"id": "primitive"}))
    payload = _valid_payload(
        memories=[{"id": "m1", "type": "fact", "content": "portable"}],
        checksum=hashlib.sha256(
            json.dumps(["m1portable"], sort_keys=True).encode()
        ).hexdigest(),
    )

    with pytest.raises(ValueError, match="collision"):
        await restore_all(pool, payload, skip_duplicates=False)

    assert not any("REPLACE" in call.upper() for call in pool.conn.executed)
