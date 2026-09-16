"""Provider-free execution tests for the local runtime-role verifier."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


bootstrap = importlib.import_module("weft.local_bootstrap")


class _FakeConnection:
    """Minimal asyncpg-shaped connection for verifier control-flow tests."""

    def __init__(self, *, ledger_write: bool) -> None:
        self.ledger_write = ledger_write
        self.privilege_calls: list[tuple[str, str]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "current_database()" in query:
            return {
                "current_user": "weft_app",
                "database_name": "weft",
                "schema_name": "public",
            }
        if "FROM pg_roles" in query:
            return {
                "current_user": "weft_app",
                "rolcanlogin": True,
                "rolsuper": False,
                "rolbypassrls": False,
                "rolcreatedb": False,
                "rolcreaterole": False,
                "rolreplication": False,
                "rolinherit": False,
                "rolconfig": None,
            }
        if "relrowsecurity" in query:
            return {
                "relrowsecurity": True,
                "relforcerowsecurity": False,
                "owner_name": "weft",
            }
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "c.relname" in query or "pg_auth_members" in query:
            return []
        raise AssertionError(f"unexpected fetch query: {query}")

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "has_table_privilege" in query:
            table, operation = str(args[0]), str(args[1])
            self.privilege_calls.append((table, operation))
            if table == "public.schema_migrations" and operation == "INSERT":
                return self.ledger_write
            return table == "public.schema_migrations" and operation == "SELECT"
        if "count(*)" in query:
            return 0
        raise AssertionError(f"unexpected fetchval query: {query}")


async def _empty_role_configuration(connection: _FakeConnection, database: str) -> list[str]:
    return []


async def _empty_default_acls(connection: _FakeConnection, owner: str) -> list[str]:
    return []


async def _empty_public_acls(connection: _FakeConnection) -> list[str]:
    return []


async def _empty_operations(connection: _FakeConnection) -> list[str]:
    return []


async def _empty_sequences(connection: _FakeConnection) -> list[str]:
    return []


async def _empty_denials(connection: _FakeConnection) -> tuple[list[str], dict[str, str]]:
    return [], {"schema_ddl_probe": "denied_and_rollback_verified"}


@pytest.fixture
def isolated_verifier(monkeypatch: pytest.MonkeyPatch):
    """Keep this test focused on verify_runtime_role's async ledger branch."""
    monkeypatch.setattr(bootstrap, "_verify_role_configuration", _empty_role_configuration)
    monkeypatch.setattr(bootstrap, "_verify_default_acls", _empty_default_acls)
    monkeypatch.setattr(bootstrap, "_verify_public_acls", _empty_public_acls)
    monkeypatch.setattr(bootstrap, "_verify_operation_allowlist", _empty_operations)
    monkeypatch.setattr(bootstrap, "_verify_sequence_allowlist", _empty_sequences)
    monkeypatch.setattr(bootstrap, "_probe_denials", _empty_denials)


@pytest.mark.asyncio
async def test_verifier_executes_clean_ledger_path(isolated_verifier) -> None:
    """A clean runtime role reaches the normal verifier return path."""
    connection = _FakeConnection(ledger_write=False)

    result = await bootstrap.verify_runtime_role(connection)

    assert result["current_user"] == "weft_app"
    assert result["schema_migrations_write"] is False
    assert result["schema_migrations_read"] is True
    assert ("public.schema_migrations", "INSERT") in connection.privilege_calls
    assert ("public.schema_migrations", "UPDATE") in connection.privilege_calls
    assert ("public.schema_migrations", "DELETE") in connection.privilege_calls
    assert ("public.schema_migrations", "TRUNCATE") in connection.privilege_calls


@pytest.mark.asyncio
async def test_verifier_rejects_ledger_write_capability(isolated_verifier) -> None:
    """A schema-ledger write privilege is rejected after async evaluation."""
    connection = _FakeConnection(ledger_write=True)

    with pytest.raises(RuntimeError, match="can modify schema_migrations"):
        await bootstrap.verify_runtime_role(connection)

    # INSERT is true, so short-circuiting is intentional; no async generator
    # is passed to synchronous any(), and the branch reports the real violation.
    assert ("public.schema_migrations", "INSERT") in connection.privilege_calls
