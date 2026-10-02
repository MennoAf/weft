"""Provision and verify the restricted local Compose database role.

The local Compose stack deliberately separates schema ownership from the app
connection. ``weft`` is used only by the one-shot migration service; the app
runs as ``weft_app`` so PostgreSQL's ordinary RLS policies are effective.

This module is intentionally local-only.  Its grant map is an explicit contract
for the current runtime callers, not a promise that arbitrary future public
relations are reachable.  Provisioning revokes all existing relation grants
and PostgreSQL default grants before applying this map; an unlisted relation,
including dormant OAuth relations, remains inaccessible.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any

import asyncpg


_MIGRATION_ROLE = "weft"
_RUNTIME_ROLE = "weft_app"
_FUTURE_PROBE_TABLE = "weft_runtime_future_probe"
_FUTURE_PROBE_SEQUENCE = "weft_runtime_future_probe_seq"
_TAINT_ROLE = "weft_runtime_taint"
_LOCAL_DATABASE = "weft"
_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

# Operations are grounded in current callers under ``weft/``.  User-scoped
# relations need CRUD because the MCP tools implement their stores directly.
# The service exceptions are deliberately narrower and documented below:
# credentials/auth (weft_tokens), identity/watermark metadata, and telemetry /
# feedback logs/counters used by trusted runtime paths.  OAuth tables are
# dormant in this architecture (Supabase owns OAuth) and therefore omitted.
_RUNTIME_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    "memories": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "memory_relationships": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "behaviors": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "episodes": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "episode_memories": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    # Workspace management is a direct MCP runtime caller; its policies are
    # intentionally service-shaped, so keep the exception explicit.
    "workspaces": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "workspace_members": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "entities": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "entity_mentions": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "modes": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "alerts": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "check_ins": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "autonomy_policies": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "policy_calibration_events": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "cost_entries": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    # Calibration records are written/read by the calibration MCP tools and
    # feed the self-calibration loop; they are not dormant telemetry.
    "calibration_records": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "triggers": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "degradation_policies": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "trackers": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "autonomy_overrides": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "cost_enforcement_state": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "alert_state": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "episode_turns": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "belief_claims": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "weft_recall_queries": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "replay_queue": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "topic_resolution_aliases": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "topic_digests": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "shuttle_claims": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "recall_canary": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "recall_canary_audit": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "board_triage_events": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "board_feedback_proposals": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "weft_recovery_attempts": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    # Trusted service-runtime exceptions.  These tables intentionally have
    # unconditional service policies, so access is granted only explicitly.
    "weft_metadata": ("SELECT", "INSERT", "UPDATE"),
    "memory_access_log": ("SELECT", "INSERT", "DELETE"),
    "turn_access_log": ("SELECT", "INSERT", "DELETE"),
    "weft_counters": ("SELECT", "INSERT", "UPDATE"),
    "weft_tool_usage_daily": ("SELECT", "INSERT", "UPDATE"),
    "weft_tool_usage_coverage": ("SELECT", "INSERT", "UPDATE"),
    # Middleware must resolve/update credentials; RLS v73 restricts rows to
    # this exact role.  No DELETE is needed by the app runtime.
    "weft_tokens": ("SELECT", "INSERT", "UPDATE"),
}

_RUNTIME_TABLES = frozenset(_RUNTIME_TABLE_PRIVILEGES)


def _quote_identifier(value: str) -> str:
    """Quote a fixed database identifier for a utility statement."""
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe database identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """Quote a password literal for PostgreSQL utility statements."""
    return "'" + value.replace("'", "''") + "'"


def _database_url() -> str:
    """Read the explicit owner/runtime DSN supplied by Compose."""
    value = os.environ.get("WEFT_DATABASE_URL")
    if not value:
        raise RuntimeError("WEFT_DATABASE_URL must be supplied")
    return value


async def _owner_scope(connection: asyncpg.Connection) -> tuple[str, str]:
    """Return and validate the dedicated local DB and migration object creator."""
    scope = await connection.fetchrow(
        "SELECT current_user AS object_creator, current_database() AS database_name, "
        "current_schema() AS schema_name"
    )
    if scope is None:
        raise RuntimeError("unable to inspect local database scope")
    object_creator = str(scope["object_creator"])
    database_name = str(scope["database_name"])
    if object_creator != _MIGRATION_ROLE:
        raise RuntimeError(
            f"local bootstrap must run as object creator {_MIGRATION_ROLE!r}, got {object_creator!r}"
        )
    if database_name != _LOCAL_DATABASE or scope["schema_name"] != "public":
        raise RuntimeError(
            "local bootstrap is limited to database/schema "
            f"{_LOCAL_DATABASE!r}/'public', got {database_name!r}/{scope['schema_name']!r}"
        )
    return object_creator, database_name


async def provision_runtime_role(
    connection: asyncpg.Connection,
    *,
    password: str,
) -> None:
    """Create/reset the non-owner runtime role and apply the allowlist.

    Migrations have completed before this function runs.  Existing role
    capabilities and grants are reset first, including the persistent-volume
    case where an operator previously created ``weft_app`` with extra power.
    Both ``pg_authid.rolconfig`` and the database-specific
    ``pg_db_role_setting`` row are reset: a global RESET ALL does not clear the
    latter.  Default ACLs are revoked for the actual migration object creator,
    both globally and for ``public``, so future owner-created relations do not
    acquire PUBLIC or runtime-role access.
    """
    object_creator, database_name = await _owner_scope(connection)
    role = _quote_identifier(_RUNTIME_ROLE)
    database = _quote_identifier(database_name)
    owner = _quote_identifier(object_creator)
    literal_password = _quote_literal(password)
    await connection.execute(
        f"""
        DO $bootstrap$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_RUNTIME_ROLE}') THEN
                CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB
                    NOCREATEROLE NOREPLICATION NOINHERIT PASSWORD {literal_password};
            ELSE
                ALTER ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB
                    NOCREATEROLE NOREPLICATION NOINHERIT PASSWORD {literal_password};
            END IF;
        END
        $bootstrap$;
        """
    )
    # Role configuration is privilege-bearing (search_path, session GUCs,
    # preload-related settings). RESET ALL is idempotent and clears stale
    # pg_authid.rolconfig values from a pre-provisioned role. The IN DATABASE
    # form is separate state in pg_db_role_setting and must also be reset.
    await connection.execute(f"ALTER ROLE {role} RESET ALL")
    await connection.execute(f"ALTER ROLE {role} IN DATABASE {database} RESET ALL")

    memberships = await connection.fetch(
        """
        SELECT parent.rolname
        FROM pg_auth_members membership
        JOIN pg_roles member ON member.oid = membership.member
        JOIN pg_roles parent ON parent.oid = membership.roleid
        WHERE member.rolname = $1
        """,
        _RUNTIME_ROLE,
    )
    for membership in memberships:
        await connection.execute(
            f"REVOKE {_quote_identifier(membership['rolname'])} FROM {role}"
        )

    # Revoke current and future grants. Unknown future tables/sequences then
    # fail closed until this reviewed allowlist is deliberately updated. PUBLIC
    # is audited too: PostgreSQL effective privileges include PUBLIC, and a
    # default ACL can otherwise re-grant a future owner-created relation after
    # this one-time verifier has run.
    await connection.execute(f"REVOKE ALL ON SCHEMA public FROM {role}, PUBLIC")
    await connection.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
    await connection.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}, PUBLIC")
    await connection.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {role}, PUBLIC")
    for target in (role, "PUBLIC"):
        await connection.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} REVOKE ALL ON TABLES FROM {target}"
        )
        await connection.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} REVOKE ALL ON SEQUENCES FROM {target}"
        )
        await connection.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
            f"REVOKE ALL ON TABLES FROM {target}"
        )
        await connection.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
            f"REVOKE ALL ON SEQUENCES FROM {target}"
        )

    existing = {
        row["table_name"]
        for row in await connection.fetch(
            """
            SELECT tablename AS table_name
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
            """
        )
    }
    for table, privileges in _RUNTIME_TABLE_PRIVILEGES.items():
        if table not in existing:
            continue
        relation = f"public.{_quote_identifier(table)}"
        await connection.execute(
            f"GRANT {', '.join(privileges)} ON TABLE {relation} TO {role}"
        )

    # Ledger is intentionally SELECT-only; it is read by verify-mode startup.
    if "schema_migrations" in existing:
        await connection.execute(
            f"GRANT SELECT ON TABLE public.schema_migrations TO {role}"
        )

    # Only nextval/currval/sequence reads are needed for identity columns.
    # UPDATE would permit setval and is intentionally never granted.  Both
    # dependency types are required: ``a`` is serial/sequence ownership and
    # ``i`` is PostgreSQL identity-column ownership.
    sequences = await connection.fetch(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'S'
          AND EXISTS (
              SELECT 1
              FROM pg_depend d
              JOIN pg_class owner_table ON owner_table.oid = d.refobjid
              JOIN pg_namespace owner_schema ON owner_schema.oid = owner_table.relnamespace
              WHERE d.objid = c.oid AND d.deptype IN ('a', 'i')
                AND owner_schema.nspname = 'public'
                AND owner_table.relname = ANY($1::text[])
          )
        """,
        list(_RUNTIME_TABLES),
    )
    for sequence in sequences:
        await connection.execute(
            f"GRANT USAGE, SELECT ON SEQUENCE public.{_quote_identifier(sequence['relname'])} TO {role}"
        )


async def _privilege(connection: asyncpg.Connection, table: str, operation: str) -> bool:
    """Return an effective table privilege without trusting catalog ACL text."""
    return bool(
        await connection.fetchval(
            "SELECT has_table_privilege(current_user, $1, $2)",
            f"public.{table}",
            operation,
        )
    )


async def _verify_default_acls(
    connection: asyncpg.Connection,
    owner_name: str,
) -> list[str]:
    """Reject runtime/PUBLIC grants in global or ``public`` default ACLs."""
    runtime_oid = await connection.fetchval(
        "SELECT oid FROM pg_roles WHERE rolname = $1", _RUNTIME_ROLE
    )
    rows = await connection.fetch(
        """
        SELECT COALESCE(namespace.nspname, '*') AS schema_name,
               defaults.defaclobjtype,
               privileges.grantee,
               privileges.privilege_type
        FROM pg_default_acl defaults
        JOIN pg_roles owner ON owner.oid = defaults.defaclrole
        LEFT JOIN pg_namespace namespace ON namespace.oid = defaults.defaclnamespace
        CROSS JOIN LATERAL aclexplode(defaults.defaclacl) privileges
        WHERE owner.rolname = $1
          AND (defaults.defaclnamespace = 0 OR namespace.nspname = 'public')
        ORDER BY schema_name, defaults.defaclobjtype, privileges.grantee,
                 privileges.privilege_type
        """,
        owner_name,
    )
    violations: list[str] = []
    for row in rows:
        if row["grantee"] == 0:
            violations.append(
                f"default ACL grants PUBLIC {row['privilege_type']} "
                f"for {row['defaclobjtype']} in {row['schema_name']}"
            )
        elif runtime_oid is not None and row["grantee"] == runtime_oid:
            violations.append(
                f"default ACL grants runtime role {row['privilege_type']} "
                f"for {row['defaclobjtype']} in {row['schema_name']}"
            )
    return violations


async def _verify_role_configuration(
    connection: asyncpg.Connection,
    database_name: str,
) -> list[str]:
    """Prove global and current-database runtime role settings are empty."""
    row = await connection.fetchrow(
        """
        SELECT r.rolconfig,
               COALESCE(setting.setconfig, ARRAY[]::text[]) AS database_config
        FROM pg_roles r
        LEFT JOIN pg_database database ON database.datname = $1
        LEFT JOIN pg_db_role_setting setting
          ON setting.setrole = r.oid AND setting.setdatabase = database.oid
        WHERE r.rolname = $2
        """,
        database_name,
        _RUNTIME_ROLE,
    )
    if row is None:
        return ["runtime role configuration metadata missing"]
    violations: list[str] = []
    if row["rolconfig"]:
        violations.append(f"runtime role has global role configuration={row['rolconfig']!r}")
    if row["database_config"]:
        violations.append(
            "runtime role has database-specific configuration="
            f"{row['database_config']!r}"
        )
    return violations


async def _verify_public_acls(connection: asyncpg.Connection) -> list[str]:
    """Reject effective/direct PUBLIC privileges on local schema relations."""
    violations: list[str] = []
    schema_rows = await connection.fetch(
        """
        SELECT privileges.privilege_type
        FROM pg_namespace namespace
        CROSS JOIN LATERAL aclexplode(namespace.nspacl) privileges
        WHERE namespace.nspname = 'public' AND privileges.grantee = 0
        """
    )
    for row in schema_rows:
        violations.append(f"PUBLIC schema privilege={row['privilege_type']}")

    relation_rows = await connection.fetch(
        """
        SELECT c.relname,
               CASE WHEN c.relkind IN ('r', 'p') THEN 'table' ELSE 'sequence' END AS kind,
               privileges.privilege_type
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        CROSS JOIN LATERAL aclexplode(c.relacl) privileges
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'S')
          AND privileges.grantee = 0
        ORDER BY c.relname, privileges.privilege_type
        """
    )
    for row in relation_rows:
        violations.append(
            f"PUBLIC {row['kind']} privilege={row['relname']}:{row['privilege_type']}"
        )
    return violations


async def _sequence_privilege(
    connection: asyncpg.Connection,
    sequence: str,
    operation: str,
) -> bool:
    """Return an effective sequence privilege for the current role."""
    return bool(
        await connection.fetchval(
            "SELECT has_sequence_privilege(current_user, $1, $2)",
            f"public.{sequence}",
            operation,
        )
    )


async def _verify_operation_allowlist(connection: asyncpg.Connection) -> list[str]:
    """Check every public relation against the explicit operation contract."""
    violations: list[str] = []
    relations = await connection.fetch(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
        ORDER BY c.relname
        """
    )
    for row in relations:
        table = row["relname"]
        expected = set(_RUNTIME_TABLE_PRIVILEGES.get(table, ()))
        if table == "schema_migrations":
            expected = {"SELECT"}
        for operation in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"):
            actual = await _privilege(connection, table, operation)
            if (operation in expected) != actual:
                state = "missing" if operation in expected else "unexpected"
                violations.append(f"{table}:{operation}={state}")
    return violations


async def _verify_sequence_allowlist(connection: asyncpg.Connection) -> list[str]:
    """Check exact USAGE/SELECT grants for allowlisted identity/serial sequences."""
    rows = await connection.fetch(
        """
        SELECT c.relname,
               EXISTS (
                   SELECT 1
                   FROM pg_depend d
                   JOIN pg_class owner_table ON owner_table.oid = d.refobjid
                   JOIN pg_namespace owner_schema ON owner_schema.oid = owner_table.relnamespace
                   WHERE d.objid = c.oid AND d.deptype IN ('a', 'i')
                     AND owner_schema.nspname = 'public'
                     AND owner_table.relname = ANY($1::text[])
               ) AS allowlisted
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'S'
        ORDER BY c.relname
        """,
        list(_RUNTIME_TABLES),
    )
    violations: list[str] = []
    for row in rows:
        sequence = row["relname"]
        expected = bool(row["allowlisted"])
        for operation in ("USAGE", "SELECT", "UPDATE"):
            actual = await _sequence_privilege(connection, sequence, operation)
            should_have = expected and operation in {"USAGE", "SELECT"}
            if actual != should_have:
                state = "missing" if should_have else "unexpected"
                violations.append(f"{sequence}:{operation}={state}")
    return violations


async def _connection_is_usable(connection: asyncpg.Connection) -> bool:
    """Confirm an expected denial was rolled back without poisoning the session."""
    try:
        return await connection.fetchval("SELECT 1") == 1
    except asyncpg.PostgresError:
        return False


async def _probe_denials(
    connection: asyncpg.Connection,
) -> tuple[list[str], dict[str, str]]:
    """Execute representative denials with explicit rollback assertions."""
    violations: list[str] = []
    details: dict[str, str] = {}
    # Schema CREATE is the direct DDL boundary. A denied command aborts the
    # explicit transaction; rollback is required before the same connection can
    # be used again. Rolling back a successful CREATE also leaves no artifact.
    probe_table = "weft_runtime_ddl_probe"
    transaction = connection.transaction()
    await transaction.start()
    try:
        await connection.execute(f"CREATE TABLE public.{probe_table} (id integer)")
    except asyncpg.InsufficientPrivilegeError:
        await transaction.rollback()
        details["schema_ddl_probe"] = "denied_and_rollback_verified"
        if not await _connection_is_usable(connection):
            violations.append("DDL denial left connection transaction-aborted")
    except asyncpg.PostgresError as exc:
        await transaction.rollback()
        violations.append(f"DDL probe raised unexpected {type(exc).__name__}")
    else:
        await transaction.rollback()
        violations.append("runtime role can CREATE TABLE in public schema")

    # A sequence UPDATE privilege enables setval. Assert it is absent for every
    # sequence, then target only an allowlisted identity/serial sequence.
    sequence = await connection.fetchval(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'S'
          AND has_sequence_privilege(current_user, c.oid, 'UPDATE')
        LIMIT 1
        """
    )
    if sequence is not None:
        violations.append(f"runtime role has sequence UPDATE/setval privilege on {sequence}")
    probe_sequence = await connection.fetchval(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'S'
          AND EXISTS (
              SELECT 1
              FROM pg_depend d
              JOIN pg_class owner_table ON owner_table.oid = d.refobjid
              JOIN pg_namespace owner_schema ON owner_schema.oid = owner_table.relnamespace
              WHERE d.objid = c.oid AND d.deptype IN ('a', 'i')
                AND owner_schema.nspname = 'public'
                AND owner_table.relname = ANY($1::text[])
          )
        ORDER BY c.relname LIMIT 1
        """,
        list(_RUNTIME_TABLES),
    )
    if probe_sequence is None:
        details["sequence_setval_probe"] = "skipped_no_allowlisted_sequence"
    else:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.fetchval(
                "SELECT setval($1::regclass, 1)", f"public.{probe_sequence}"
            )
        except asyncpg.InsufficientPrivilegeError:
            await transaction.rollback()
            details["sequence_setval_probe"] = "denied_and_rollback_verified"
            if not await _connection_is_usable(connection):
                violations.append("setval denial left connection transaction-aborted")
        except asyncpg.PostgresError as exc:
            await transaction.rollback()
            violations.append(f"setval probe raised unexpected {type(exc).__name__}")
        else:
            await transaction.rollback()
            violations.append(f"runtime role can setval sequence {probe_sequence}")

    # Unknown owner-created relations must remain inaccessible even after the
    # one-time provisioning step. These names are created by the owner-only
    # ``create-future-probes`` mode after reprovisioning.
    for statement, detail, message in (
        (
            f"SELECT 1 FROM public.{_FUTURE_PROBE_TABLE} LIMIT 1",
            "future_table_select_probe",
            "runtime role can SELECT owner-created future table",
        ),
        (
            "SELECT setval($1::regclass, 1)",
            "future_sequence_setval_probe",
            "runtime role can setval owner-created future sequence",
        ),
    ):
        transaction = connection.transaction()
        await transaction.start()
        try:
            if "$1" in statement:
                await connection.fetchval(statement, f"public.{_FUTURE_PROBE_SEQUENCE}")
            else:
                await connection.fetchval(statement)
        except asyncpg.InsufficientPrivilegeError:
            await transaction.rollback()
            details[detail] = "denied_and_rollback_verified"
            if not await _connection_is_usable(connection):
                violations.append(f"{detail} left connection transaction-aborted")
        except asyncpg.UndefinedTableError:
            await transaction.rollback()
            violations.append(f"{detail} relation missing")
        except asyncpg.PostgresError as exc:
            await transaction.rollback()
            violations.append(f"{detail} raised unexpected {type(exc).__name__}")
        else:
            await transaction.rollback()
            violations.append(message)

    # Service tables are explicit exceptions, not an accidental side effect of
    # the user-table grant. OAuth relations are checked through the generic
    # allowlist audit above and must have no privileges.
    for table in ("weft_metadata", "memory_access_log", "turn_access_log", "weft_tokens"):
        if table in _RUNTIME_TABLE_PRIVILEGES and not await _privilege(connection, table, "SELECT"):
            violations.append(f"runtime exception table {table} lacks SELECT")
    for table in ("oauth_clients", "oauth_authorization_codes", "oauth_refresh_tokens", "oauth_access_revocations"):
        exists = await connection.fetchval(
            "SELECT to_regclass($1) IS NOT NULL", f"public.{table}"
        )
        if exists:
            oauth_privileges = []
            for operation in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                oauth_privileges.append(await _privilege(connection, table, operation))
            if any(oauth_privileges):
                violations.append(f"dormant OAuth table {table} is reachable")
    return violations, details


async def verify_runtime_role(connection: asyncpg.Connection) -> dict[str, Any]:
    """Verify effective app connection role, RLS, grants, and denials."""
    scope = await connection.fetchrow(
        "SELECT current_user AS current_user, current_database() AS database_name, "
        "current_schema() AS schema_name"
    )
    if scope is None:
        raise RuntimeError("effective database scope metadata missing")
    database_name = str(scope["database_name"])
    role = await connection.fetchrow(
        """
        SELECT current_user AS current_user, r.rolcanlogin, r.rolsuper,
               r.rolbypassrls, r.rolcreatedb, r.rolcreaterole,
               r.rolreplication, r.rolinherit, r.rolconfig
        FROM pg_roles r
        WHERE r.rolname = current_user
        """
    )
    memories = await connection.fetchrow(
        """
        SELECT c.relrowsecurity, c.relforcerowsecurity,
               owner.rolname AS owner_name
        FROM pg_class c
        JOIN pg_roles owner ON owner.oid = c.relowner
        WHERE c.oid = 'public.memories'::regclass
        """
    )
    owned_objects = await connection.fetch(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = ANY($1::char[])
          AND c.relowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)
        """,
        ["r", "p", "S", "v", "m", "f"],
    )
    memberships = await connection.fetchval(
        """
        SELECT count(*)
        FROM pg_auth_members membership
        JOIN pg_roles member ON member.oid = membership.member
        WHERE member.rolname = current_user
        """
    )
    ledger_write = await _privilege(connection, "schema_migrations", "INSERT")
    if not ledger_write:
        for operation in ("UPDATE", "DELETE", "TRUNCATE"):
            if await _privilege(connection, "schema_migrations", operation):
                ledger_write = True
                break
    ledger_read = await _privilege(connection, "schema_migrations", "SELECT")
    violations: list[str] = []
    if scope["schema_name"] != "public":
        violations.append(f"effective schema is {scope['schema_name']!r}")
    if database_name != _LOCAL_DATABASE:
        violations.append(f"effective database is {database_name!r}")
    if role is None:
        violations.append("effective role metadata missing")
    else:
        if role["current_user"] != _RUNTIME_ROLE:
            violations.append(f"effective role is {role['current_user']!r}")
        for field, label in (
            ("rolcanlogin", "LOGIN"),
        ):
            if not role[field]:
                violations.append(f"runtime role lacks {label}")
        for field, label in (
            ("rolsuper", "SUPERUSER"),
            ("rolbypassrls", "BYPASSRLS"),
            ("rolcreatedb", "CREATEDB"),
            ("rolcreaterole", "CREATEROLE"),
            ("rolreplication", "REPLICATION"),
            ("rolinherit", "INHERIT"),
        ):
            if role[field]:
                violations.append(f"runtime role has {label}")
        if role["rolconfig"]:
            violations.append(f"runtime role has role configuration={role['rolconfig']!r}")
    violations.extend(await _verify_role_configuration(connection, database_name))
    violations.extend(await _verify_default_acls(connection, _MIGRATION_ROLE))
    violations.extend(await _verify_public_acls(connection))
    if memories is None:
        violations.append("memories metadata missing")
    else:
        if not memories["relrowsecurity"]:
            violations.append("memories RLS is disabled")
        if memories["owner_name"] == _RUNTIME_ROLE and not memories["relforcerowsecurity"]:
            violations.append("runtime role owns memories while FORCE RLS is disabled")
    if owned_objects:
        violations.append("runtime role owns public database objects")
    if memberships:
        violations.append("runtime role has role memberships")
    if ledger_write:
        violations.append("runtime role can modify schema_migrations")
    if not ledger_read:
        violations.append("runtime role cannot read schema_migrations")
    violations.extend(await _verify_operation_allowlist(connection))
    violations.extend(await _verify_sequence_allowlist(connection))
    denial_violations, denial_details = await _probe_denials(connection)
    violations.extend(denial_violations)
    if violations:
        raise RuntimeError("Runtime role verification failed: " + "; ".join(violations))
    return {
        "current_user": _RUNTIME_ROLE,
        "rolsuper": False,
        "rolbypassrls": False,
        "rolcreatedb": False,
        "rolcreaterole": False,
        "rolreplication": False,
        "rolinherit": False,
        "rolconfig": None,
        "memories_rls": True,
        "memories_owner": memories["owner_name"],
        "schema_migrations_write": False,
        "schema_migrations_read": True,
        "explicit_runtime_tables": sorted(_RUNTIME_TABLES),
        "dormant_oauth_tables_denied": True,
        "sequence_setval": False,
        "schema_ddl": False,
        "database_name": database_name,
        "schema_name": scope["schema_name"],
        "role_configuration_reset": True,
        "public_acl_reset": True,
        "probe_details": denial_details,
    }


async def taint_runtime_role(connection: asyncpg.Connection) -> None:
    """Taint global/database role settings and PUBLIC defaults for acceptance."""
    _, database_name = await _owner_scope(connection)
    role = _quote_identifier(_RUNTIME_ROLE)
    database = _quote_identifier(database_name)
    owner = _quote_identifier(_MIGRATION_ROLE)
    parent = _quote_identifier(_TAINT_ROLE)
    await connection.execute(
        f"DO $taint$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_TAINT_ROLE}') THEN CREATE ROLE {parent} NOLOGIN; END IF; END $taint$;"
    )
    # Keep role attributes and configuration in separate ALTER ROLE statements;
    # PostgreSQL parses RESET/SET configuration options separately from the
    # capability option list on this server version.
    await connection.execute(
        f"ALTER ROLE {role} CREATEDB CREATEROLE REPLICATION INHERIT"
    )
    await connection.execute(f"ALTER ROLE {role} SET search_path TO pg_catalog")
    await connection.execute(
        f"ALTER ROLE {role} IN DATABASE {database} SET search_path TO pg_catalog"
    )
    await connection.execute(f"GRANT {parent} TO {role}")
    # These are deliberate taints. Provisioning must remove both global and
    # public-schema defaults, otherwise a future owner-created relation could
    # become reachable through PUBLIC after startup verification.
    await connection.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} GRANT SELECT ON TABLES TO PUBLIC"
    )
    await connection.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} GRANT USAGE ON SEQUENCES TO PUBLIC"
    )
    await connection.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
        f"GRANT SELECT ON TABLES TO PUBLIC"
    )
    await connection.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner} IN SCHEMA public "
        f"GRANT USAGE ON SEQUENCES TO PUBLIC"
    )


async def create_future_probes(connection: asyncpg.Connection) -> None:
    """Create owner-owned relations after provisioning for future-ACL denial tests."""
    await _owner_scope(connection)
    await connection.execute(
        f"DROP TABLE IF EXISTS public.{_FUTURE_PROBE_TABLE} CASCADE"
    )
    await connection.execute(f"DROP SEQUENCE IF EXISTS public.{_FUTURE_PROBE_SEQUENCE} CASCADE")
    await connection.execute(
        f"CREATE TABLE public.{_FUTURE_PROBE_TABLE} (id integer NOT NULL)"
    )
    await connection.execute(f"CREATE SEQUENCE public.{_FUTURE_PROBE_SEQUENCE}")


async def _run(mode: str) -> dict[str, Any] | None:
    """Run provisioning as owner or verification as the app role."""
    connection = await asyncpg.connect(_database_url())
    try:
        if mode == "provision":
            password = os.environ.get("WEFT_LOCAL_RUNTIME_DB_PASSWORD")
            if not password:
                raise RuntimeError("WEFT_LOCAL_RUNTIME_DB_PASSWORD must be supplied")
            await provision_runtime_role(connection, password=password)
            return {
                "runtime_role": _RUNTIME_ROLE,
                "provisioned": True,
                "explicit_runtime_tables": sorted(_RUNTIME_TABLES),
            }
        if mode == "taint-runtime":
            await taint_runtime_role(connection)
            return {"runtime_role": _RUNTIME_ROLE, "tainted": True}
        if mode == "create-future-probes":
            await create_future_probes(connection)
            return {"future_probe_table": _FUTURE_PROBE_TABLE, "future_probe_sequence": _FUTURE_PROBE_SEQUENCE}
        if mode == "verify-runtime":
            return await verify_runtime_role(connection)
        raise ValueError(f"unknown bootstrap mode: {mode}")
    finally:
        await connection.close()


def main(argv: list[str] | None = None) -> int:
    """Provision or verify the local runtime role."""
    mode = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "provision"
    try:
        result = asyncio.run(_run(mode))
    except Exception as exc:
        print(f"local database bootstrap failed: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
