"""RLS invariants — schema-level guards against silent tenant-isolation regressions.

Closes the gap between the existing RLS coverage (test_rls_e2e checks a
hardcoded list of 10 tables; test_rls_pentest checks enforcement against
that same list) and the actual scoped-table set after mig 36 / 38 / 40.

The motivating incidents:
  - weft-86a6d2b0: store_entity took create.user_id without a GUC fallback,
    silently wrote NULL for weeks.
  - weft-b985f651: backfill _TABLES list drifted to 7-of-16 coverage; caught
    by row-count anomaly in a dry run, not by a test.
  - weft-d4ba49d4: implicit-global anti-pattern (user_id IS NULL = global)
    let an unset GUC silently leak rows. Mig 36 killed the path with
    NOT NULL + named-string sentinel.

Invariants pinned here:
  1. Every table with a ``user_id`` column is classified explicitly — adding
     a new such column without updating ``EXPECTED_USER_ID_TABLES`` fails
     the test, forcing the author to choose scoped-CRUD or service-policy.
  2. Every scoped table has RLS enabled and the 4 CRUD policies.
  3. Scoped policies match the OR-sentinel pattern (not the legacy OR-NULL).
  4. Scoped ``user_id`` columns are NOT NULL with the session-GUC default.
  5. With ``app.user_id`` unset, INSERT into a scoped table fails loud
     (NotNullViolation), not silently with NULL — the regression that
     mig 36 fixed and that this test pins.

Requires a real PostgreSQL instance (testcontainers via conftest.py).
"""

from __future__ import annotations

import asyncpg
import pytest

from weft.schema import SYSTEM_GLOBAL_USER_ID


# Tables that follow the standard scoped-CRUD policy set
# ({table}_select / _insert / _update / _delete with sentinel-or-self).
# memories has a workspace branch in its SELECT policy; everything else
# is the simple sentinel-or-self pattern.
SCOPED_CRUD_TABLES: tuple[str, ...] = (
    "memories",
    "memory_relationships",
    "behaviors",
    "entities",
    "entity_mentions",
    "episodes",
    "episode_memories",
    "episode_turns",
    "modes",
    "alerts",
    "check_ins",
    "autonomy_policies",
    "policy_calibration_events",
    "cost_entries",
    "triggers",
    "calibration_records",
    "degradation_policies",
    "trackers",
    "autonomy_overrides",
    "cost_enforcement_state",
    "alert_state",
    "belief_claims",
    "weft_recall_queries",
    "replay_queue",
    # Topic-digest recall (v56/v57): both carry full GUC-scoped CRUD policies
    # plus the system-sentinel SELECT, same shape as belief_claims.
    "topic_resolution_aliases",
    "topic_digests",
    # Shuttle observer/synthesis blackboard (v58): same scoped-CRUD + sentinel
    # SELECT shape; landed via the shuttle_claims seam but never classified here.
    "shuttle_claims",
)

# Tables that have a ``user_id`` column but use a service policy by design.
# Auth lookup happens BEFORE app.user_id is set on the connection, so RLS
# scoping cannot apply to weft_tokens. Per-user listing filters at the
# application layer (see weft/credentials.py).
SERVICE_USER_ID_TABLES: tuple[str, ...] = (
    "weft_tokens",
)

EXPECTED_USER_ID_TABLES: frozenset[str] = frozenset(
    SCOPED_CRUD_TABLES + SERVICE_USER_ID_TABLES
)

# Substring (case-insensitive) of the GUC-resolution expression as it appears
# in pg_policies after Postgres normalization (NULLIF uppercased, ::text casts
# inserted). The legacy mig-36 cleanup also dropped a calibration_events_*
# alias, but the active policies all use the {table}_{op} naming.
_NULLIF_GUC_SUBSTRING = "nullif(current_setting('app.user_id'"

# The legacy implicit-global pattern that mig 36 killed. Search for the
# specific bad expression rather than just "IS NULL" — the SELECT policy on
# memories legitimately contains "workspace_id IS NOT NULL".
_LEGACY_NULL_PATTERN = "user_id is null"


def _has_guc(expr: str) -> bool:
    return _NULLIF_GUC_SUBSTRING in expr.lower()


def _has_legacy_null(expr: str) -> bool:
    return _LEGACY_NULL_PATTERN in expr.lower()


# ---------------------------------------------------------------------------
# 1. Drift detector — every user_id column is classified
# ---------------------------------------------------------------------------


class TestUserIdColumnInventory:
    """Every table with a user_id column matches the known classification.

    This is the primary drift guard. If a future migration adds a user_id
    column to a new table, this test fails until the author either:
      - adds the table to SCOPED_CRUD_TABLES (and wires up RLS in the same
        migration so the next test passes), or
      - adds it to SERVICE_USER_ID_TABLES (and documents WHY in this file).
    """

    async def test_user_id_columns_match_expected_set(self, pool):
        rows = await pool.fetch(
            """
            SELECT table_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name = 'user_id'
            ORDER BY table_name
            """
        )
        discovered = {r["table_name"] for r in rows}
        unexpected = discovered - EXPECTED_USER_ID_TABLES
        missing = EXPECTED_USER_ID_TABLES - discovered
        assert not unexpected, (
            f"New user_id column(s) discovered: {sorted(unexpected)}. "
            "Classify in SCOPED_CRUD_TABLES (and add RLS) or "
            "SERVICE_USER_ID_TABLES (with rationale)."
        )
        assert not missing, (
            f"Expected user_id tables missing from schema: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# 2. RLS enabled on every classified table
# ---------------------------------------------------------------------------


class TestRLSEnabled:
    """Every classified user_id table has rowsecurity=true.

    Both scoped and service tables enable RLS — service tables use a
    permissive service policy, but the relrowsecurity flag must still be
    on so Postgres enforces the policy set.
    """

    async def test_rls_enabled_on_all_classified_tables(self, pool):
        rows = await pool.fetch(
            "SELECT relname, relrowsecurity FROM pg_class "
            "WHERE relname = ANY($1::text[])",
            list(EXPECTED_USER_ID_TABLES),
        )
        enabled = {r["relname"]: r["relrowsecurity"] for r in rows}
        for table in EXPECTED_USER_ID_TABLES:
            assert enabled.get(table) is True, f"RLS not enabled on {table}"

    async def test_service_tables_have_at_least_one_policy(self, pool):
        """Service-exception tables must define at least one policy.

        With RLS enabled and no policies, Postgres denies all access to the
        table for non-superusers — which would break the application. The
        service pattern is RLS enabled + a permissive ``USING (true)`` policy,
        signalling intent (the table is service-scoped, not user-scoped)
        without locking the service role out.
        """
        rows = await pool.fetch(
            "SELECT tablename, count(*) AS n FROM pg_policies "
            "WHERE tablename = ANY($1::text[]) GROUP BY tablename",
            list(SERVICE_USER_ID_TABLES),
        )
        counts = {r["tablename"]: r["n"] for r in rows}
        for table in SERVICE_USER_ID_TABLES:
            assert counts.get(table, 0) >= 1, (
                f"Service-exception table {table} has RLS enabled but no "
                f"policies — the service role is fine (BYPASSRLS) but any "
                f"non-service role with table privileges would be locked out."
            )


# ---------------------------------------------------------------------------
# 3. CRUD policies + OR-sentinel pattern on scoped tables
# ---------------------------------------------------------------------------


class TestScopedTableCRUDPolicies:
    """Each scoped table has 4 named CRUD policies with the OR-sentinel pattern."""

    async def test_four_crud_policies_per_scoped_table(self, pool):
        rows = await pool.fetch(
            "SELECT tablename, policyname FROM pg_policies "
            "WHERE tablename = ANY($1::text[])",
            list(SCOPED_CRUD_TABLES),
        )
        by_table: dict[str, set[str]] = {}
        for r in rows:
            by_table.setdefault(r["tablename"], set()).add(r["policyname"])
        for table in SCOPED_CRUD_TABLES:
            policies = by_table.get(table, set())
            for op in ("select", "insert", "update", "delete"):
                expected = f"{table}_{op}"
                assert expected in policies, (
                    f"Missing policy {expected} on {table}. Found: {policies}"
                )

    async def test_insert_policy_has_with_check_against_session_guc(self, pool):
        """INSERT policy WITH CHECK pins user_id to the session GUC.

        Catches a regression where someone re-creates the policy with
        ``user_id IS NULL`` (the legacy implicit-global pattern that
        mig 36 killed) or drops the WITH CHECK entirely.
        """
        rows = await pool.fetch(
            "SELECT tablename, policyname, with_check FROM pg_policies "
            "WHERE tablename = ANY($1::text[]) AND cmd = 'INSERT'",
            list(SCOPED_CRUD_TABLES),
        )
        seen = {r["tablename"]: r for r in rows}
        for table in SCOPED_CRUD_TABLES:
            row = seen.get(table)
            assert row is not None, f"No INSERT policy found on {table}"
            with_check = row["with_check"] or ""
            assert _has_guc(with_check), (
                f"{table}_insert WITH CHECK does not pin user_id to "
                f"session GUC. Got: {with_check!r}"
            )
            assert not _has_legacy_null(with_check), (
                f"{table}_insert WITH CHECK contains user_id IS NULL — "
                f"the legacy implicit-global pattern that mig 36 killed. "
                f"Got: {with_check!r}"
            )

    async def test_update_delete_policies_use_session_guc(self, pool):
        """UPDATE / DELETE policy USING clauses pin to the session GUC."""
        rows = await pool.fetch(
            "SELECT tablename, policyname, qual, cmd FROM pg_policies "
            "WHERE tablename = ANY($1::text[]) AND cmd IN ('UPDATE', 'DELETE')",
            list(SCOPED_CRUD_TABLES),
        )
        for r in rows:
            qual = r["qual"] or ""
            assert _has_guc(qual), (
                f"{r['policyname']} on {r['tablename']} ({r['cmd']}) does "
                f"not pin user_id to session GUC. Got: {qual!r}"
            )
            assert not _has_legacy_null(qual), (
                f"{r['policyname']} on {r['tablename']} ({r['cmd']}) contains "
                f"user_id IS NULL — the legacy implicit-global pattern. "
                f"Got: {qual!r}"
            )

    async def test_select_policies_allow_sentinel_global(self, pool):
        """SELECT policy allows the SYSTEM_GLOBAL sentinel as well as the user.

        The sentinel branch is what makes ``__system_global_zathras__`` rows
        visible to every authenticated user (seeds, system facts). A SELECT
        policy without the sentinel branch would hide them.
        """
        rows = await pool.fetch(
            "SELECT tablename, policyname, qual FROM pg_policies "
            "WHERE tablename = ANY($1::text[]) AND cmd = 'SELECT'",
            list(SCOPED_CRUD_TABLES),
        )
        for r in rows:
            qual = r["qual"] or ""
            assert SYSTEM_GLOBAL_USER_ID in qual, (
                f"{r['policyname']} on {r['tablename']} (SELECT) is missing "
                f"the SYSTEM_GLOBAL sentinel branch. Got: {qual!r}"
            )
            assert _has_guc(qual), (
                f"{r['policyname']} on {r['tablename']} (SELECT) does not "
                f"pin to session GUC. Got: {qual!r}"
            )


# ---------------------------------------------------------------------------
# 4. NOT NULL + GUC default on every scoped user_id column
# ---------------------------------------------------------------------------


class TestUserIdColumnDefaults:
    """Every scoped table's user_id column is NOT NULL with the GUC default.

    This is the column-level half of mig 36's contract. Without it, an INSERT
    that omits user_id would silently write NULL — exactly the implicit-global
    regression weft-d4ba49d4 captures.
    """

    async def test_user_id_not_null_on_scoped_tables(self, pool):
        rows = await pool.fetch(
            """
            SELECT table_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name = 'user_id'
              AND table_name = ANY($1::text[])
            """,
            list(SCOPED_CRUD_TABLES),
        )
        seen = {r["table_name"]: r for r in rows}
        for table in SCOPED_CRUD_TABLES:
            row = seen.get(table)
            assert row is not None, f"{table}.user_id column not found"
            assert row["is_nullable"] == "NO", (
                f"{table}.user_id is nullable — would allow silent global "
                f"writes (the bug mig 36 fixed)."
            )
            default = row["column_default"] or ""
            # Postgres normalizes the default expression. Looser match is
            # fine — we're guarding against drift, not exact bytes.
            assert "current_setting" in default and "app.user_id" in default, (
                f"{table}.user_id has no session-GUC default. "
                f"Got: {default!r}"
            )


# ---------------------------------------------------------------------------
# 5. Unset GUC fails loud — runtime regression guard
# ---------------------------------------------------------------------------


class TestUnsetGUCFailsLoud:
    """With app.user_id unset, INSERT into a scoped table raises NotNullViolation.

    This is the runtime guarantee that the schema-level invariants combine to
    produce. Even if a developer forgets the COALESCE pattern at a call site,
    the column DEFAULT + NOT NULL turns the bug into a loud failure rather
    than a silent leak.
    """

    @pytest.mark.parametrize(
        "table,minimal_sql",
        [
            (
                "behaviors",
                # user_id intentionally omitted — relies on column DEFAULT.
                # When app.user_id is unset, DEFAULT evaluates to NULL and
                # NOT NULL kicks in.
                """
                INSERT INTO behaviors (
                    id, trigger_pattern, action, confidence, scope,
                    priority, enabled, access_count, token_count, status,
                    created_at, updated_at
                ) VALUES (
                    'weft-rls-inv-beh', 'pattern', 'action', 0.5, 'global',
                    0, true, 0, 1, 'active',
                    now(), now()
                )
                """,
            ),
            (
                "alerts",
                """
                INSERT INTO alerts (
                    id, alert_type, title, body, trigger_at, status,
                    channel, channel_target, payload
                ) VALUES (
                    'weft-rls-inv-alert', 'reminder', 't', 'b', now(),
                    'pending', 'log', 'stdout', '{}'::jsonb
                )
                """,
            ),
            (
                "trackers",
                """
                INSERT INTO trackers (
                    id, kind, title
                ) VALUES (
                    'weft-rls-inv-trk', 'task', 't'
                )
                """,
            ),
        ],
    )
    async def test_unset_guc_insert_raises_not_null(
        self, pool, table, minimal_sql
    ):
        # Acquire a raw connection so we can RESET app.user_id without the
        # conftest setup callback re-arming it. We open a transaction and
        # roll back at the end — the RESET is local to this transaction.
        async with pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                # Belt-and-braces: clear both session-level (set by conftest
                # setup callback) and any LOCAL value.
                await conn.execute("SET LOCAL app.user_id = ''")
                with pytest.raises(asyncpg.NotNullViolationError):
                    await conn.execute(minimal_sql)
            finally:
                await tr.rollback()
