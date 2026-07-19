"""Migration runner: applies pending migrations under an advisory lock.

Migration tuples themselves live in sibling ``vNN_*.py`` modules and are
aggregated by ``__init__.py``. This module is the only place that
executes SQL.
"""

from __future__ import annotations

import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

# Advisory lock ID for serializing migrations across processes
_MIGRATION_LOCK_ID = 839271  # arbitrary unique int

# Migrations run on the shared pool, which carries a short request-path
# command_timeout (~30s). DDL like an HNSW index build over a large table, and
# the advisory-lock wait during a concurrent deploy, can legitimately exceed
# that — so migration statements pass an explicit, generous per-call timeout
# that OVERRIDES the pool default (asyncpg: a positive per-call timeout wins;
# None would inherit the 30s ceiling and cancel the build mid-flight).
# Overridable for very large tables via WEFT_MIGRATION_TIMEOUT.
_MIGRATION_TIMEOUT = float(os.environ.get("WEFT_MIGRATION_TIMEOUT", 600.0))


async def _get_applied_versions(pool: asyncpg.Pool) -> set[int]:
    """Get set of already-applied migration versions."""
    # Supabase ships its own ``auth.schema_migrations`` and
    # ``storage.schema_migrations`` tables, so filter by current schema —
    # otherwise a fresh Supabase DB reports the table as existing and the
    # SELECT below blows up with UndefinedTableError on public.schema_migrations.
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'schema_migrations'
              AND table_schema = 'public'
        )
        """
    )
    if not exists:
        return set()
    rows = await pool.fetch("SELECT version FROM public.schema_migrations")
    return {r["version"] for r in rows}


_REQUIRED_RUNTIME_TABLES = (
    "memories",
    "weft_tokens",
    "episode_turns",
    "replay_queue",
    "weft_tool_usage_coverage",
)


async def verify_runtime_invariants(pool: asyncpg.Pool) -> None:
    """Read-only privilege, RLS, policy, and head-schema checks."""
    role = await pool.fetchrow(
        "SELECT r.rolsuper, r.rolbypassrls, r.rolcanlogin "
        "FROM pg_roles r WHERE r.rolname = current_user"
    )
    table = await pool.fetchrow(
        "SELECT c.relrowsecurity, c.relforcerowsecurity, "
        "current_user = owner.rolname AS app_is_owner "
        "FROM pg_class c JOIN pg_roles owner ON owner.oid = c.relowner "
        "WHERE c.oid = 'public.memories'::regclass"
    )
    if role is None or table is None:
        raise RuntimeError("Runtime schema verification failed: role/table metadata missing")
    violations: list[str] = []
    if not role["rolcanlogin"]:
        violations.append("runtime role lacks LOGIN")
    if role["rolsuper"]:
        violations.append("runtime role is SUPERUSER")
    if role["rolbypassrls"]:
        violations.append("runtime role has BYPASSRLS")
    if table["app_is_owner"] and not table["relforcerowsecurity"]:
        violations.append("runtime role owns memories while FORCE RLS is disabled")
    owned_objects = await pool.fetch(
        "SELECT c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' "
        "AND c.relkind = ANY($1::char[]) "
        "AND c.relowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)",
        ["r", "p", "S", "v", "m", "f"],
    )
    if owned_objects:
        violations.append("runtime role owns public database objects")
    if not table["relrowsecurity"]:
        violations.append("memories RLS is disabled")

    memberships = await pool.fetch(
        "SELECT parent.rolname FROM pg_auth_members m "
        "JOIN pg_roles member ON member.oid = m.member "
        "JOIN pg_roles parent ON parent.oid = m.roleid "
        "WHERE member.rolname = current_user"
    )
    if memberships:
        violations.append("runtime role has role memberships")

    ledger_write_privileges = await pool.fetchval(
        "SELECT has_table_privilege(current_user, "
        "'public.schema_migrations', 'INSERT,UPDATE,DELETE,TRUNCATE')"
    )
    if ledger_write_privileges:
        violations.append("runtime role can modify schema_migrations")
    if not await pool.fetchval(
        "SELECT has_table_privilege(current_user, "
        "'public.schema_migrations', 'SELECT')"
    ):
        violations.append("runtime role cannot read schema_migrations")

    policies = await pool.fetch(
        "SELECT policyname FROM pg_policies "
        "WHERE schemaname = 'public' AND tablename = 'memories'"
    )
    policy_names = {row["policyname"] for row in policies}
    required_policies = {
        "memories_select",
        "memories_insert",
        "memories_update",
        "memories_delete",
    }
    missing_policies = sorted(required_policies - policy_names)
    if missing_policies:
        violations.append(f"missing memories policies={missing_policies}")

    existing_tables = set(
        await pool.fetchval(
            "SELECT array_agg(table_name ORDER BY table_name) "
            "FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY($1::text[])",
            list(_REQUIRED_RUNTIME_TABLES),
        )
        or []
    )
    missing_tables = sorted(set(_REQUIRED_RUNTIME_TABLES) - existing_tables)
    if missing_tables:
        violations.append(f"missing required tables={missing_tables}")

    if violations:
        raise RuntimeError(
            "Runtime schema verification failed: " + "; ".join(violations)
        )


async def verify_migration_ledger(pool: asyncpg.Pool) -> None:
    """Fail unless the migration ledger exactly matches the code version set."""
    from weft.db.migrations import MIGRATIONS

    expected = {version for version, _, _ in MIGRATIONS}
    applied = await _get_applied_versions(pool)
    if not applied:
        raise RuntimeError(
            "Database migration verification failed: public.schema_migrations "
            "is missing or empty. Run migrations with the owner role before "
            "starting Weft in verify mode."
        )
    pending = sorted(expected - applied)
    unknown = sorted(applied - expected)
    if pending or unknown:
        details: list[str] = []
        if pending:
            details.append(f"pending versions={pending}")
        if unknown:
            details.append(f"unknown applied versions={unknown}")
        raise RuntimeError(
            "Database migration verification failed: "
            + "; ".join(details)
            + ". Run owner-managed migrations before restarting Weft."
        )


async def verify_migrations(pool: asyncpg.Pool) -> None:
    """Run every read-only restricted-runtime database gate."""
    await verify_migration_ledger(pool)
    await verify_runtime_invariants(pool)


async def run_migrations(pool: asyncpg.Pool) -> list[int]:
    """Run all pending migrations in order. Returns list of applied versions.

    Uses a Postgres advisory lock to serialize concurrent migration runs.
    """
    # Lazy import: this module is imported by the package __init__ that
    # builds MIGRATIONS, so a top-level import would re-enter the package
    # mid-discovery.
    from weft.db.migrations import MIGRATIONS

    applied: list[int] = []

    async with pool.acquire() as lock_conn:
        # Acquire session-level advisory lock (blocks until available)
        await lock_conn.execute(
            "SELECT pg_advisory_lock($1)",
            _MIGRATION_LOCK_ID,
            timeout=_MIGRATION_TIMEOUT,
        )
        try:
            existing = await _get_applied_versions(pool)

            for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
                if version in existing:
                    continue
                async with lock_conn.transaction():
                    await lock_conn.execute(sql, timeout=_MIGRATION_TIMEOUT)
                applied.append(version)
                logger.info("Applied migration %d: %s", version, description)

            # Record newly applied migrations (schema_migrations table
            # exists after migration 3 runs)
            if applied:
                for version, description, _ in MIGRATIONS:
                    if version not in existing:
                        await lock_conn.execute(
                            "INSERT INTO schema_migrations (version, description) "
                            "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                            version,
                            description,
                        )
        finally:
            await lock_conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID,
            )

    return applied
