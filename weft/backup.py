"""Full backup and restore for Weft memory data.

Exports the user's accumulated knowledge — memories, behaviors, entities,
episodes, modes, trackers, workspaces, and the relationship/membership
edges connecting them — as a self-contained JSON file that can be
restored to any Postgres instance.

What is *not* backed up: transient operational state (alerts, check_ins,
cost_entries, calibration_records, autonomy_policies, degradation_policies,
policy_calibration_events, memory_access_log, audit_backfill_user_id),
auth credentials (weft_tokens, oauth_*), and system metadata
(schema_migrations, weft_metadata). Restore re-bootstraps the schema via
``run_migrations`` and the operator re-issues credentials.

Backup and restore bypass RLS to operate on all rows across all user_ids.
This requires the database connection to use the table owner role (e.g.,
the Supabase service role key, not the anon key).

Backup format versions:
  1.0 — memories + memory_relationships only (legacy)
  1.1 — same shape as 1.0, gained workspace_id + author_identity columns
  1.2 — adds behaviors, entities, entity_mentions, episodes,
        episode_memories, modes, trackers, workspaces, workspace_members
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from weft.db.migrations import MIGRATIONS
from weft.db.migrations._runner import verify_migration_ledger

logger = logging.getLogger(__name__)

# Current backup format version — bump when the schema changes
BACKUP_VERSION = "1.2"
_LEGACY_VERSIONS = ("1.0", "1.1")

# Logical export is deliberately an allow-list.  In particular, never infer
# exportability from the set of tables visible to a database role: auth,
# migration, lease, telemetry, and infrastructure tables are not portable user
# data even when a role can read them.
_PORTABLE_TOP_LEVEL_KEYS = frozenset({
    "version", "schema_version", "exported_at", "checksum",
    "memory_count", "relationship_count", "memories", "relationships",
    "workspaces", "workspace_members", "behaviors", "entities",
    "entity_mentions", "episodes", "episode_memories", "modes", "trackers",
    "counts",
})
_NON_PORTABLE_SECTIONS = frozenset({
    "weft_tokens", "oauth_sessions", "oauth_clients", "oauth_codes",
    "schema_migrations", "weft_metadata", "leases", "worker_leases",
    "alerts", "check_ins", "cost_entries", "calibration_records",
    "autonomy_policies", "degradation_policies", "policy_calibration_events",
    "memory_access_log", "audit_backfill_user_id", "infrastructure",
})


# ---------------------------------------------------------------------------
# Table specs — declarative inventory of what gets backed up
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TableSpec:
    """A backed-up table: name, JSON section key, primary-key columns.

    ``pk`` columns drive the ``ON CONFLICT`` clause on restore so re-running
    a restore is idempotent.

    The actual column list is discovered from information_schema at
    backup/restore time so a new migration that adds a column doesn't
    require a code change here.
    """

    name: str
    section: str
    pk: tuple[str, ...]


# Order matters for restore: foreign-key dependencies first.
# memories.workspace_id → workspaces, so workspaces restores before memories.
# entity_mentions / episode_memories / memory_relationships / trackers all
# reference memories or entities and restore after them.
_TABLES: tuple[_TableSpec, ...] = (
    _TableSpec("workspaces", "workspaces", ("id",)),
    _TableSpec("workspace_members", "workspace_members", ("workspace_id", "member_identity")),
    _TableSpec("memories", "memories", ("id",)),
    _TableSpec("behaviors", "behaviors", ("id",)),
    _TableSpec("entities", "entities", ("id",)),
    _TableSpec("episodes", "episodes", ("id",)),
    _TableSpec("modes", "modes", ("id",)),
    _TableSpec("trackers", "trackers", ("id",)),
    _TableSpec("entity_mentions", "entity_mentions", ("entity_id", "memory_id")),
    _TableSpec("episode_memories", "episode_memories", ("episode_id", "memory_id")),
    _TableSpec("memory_relationships", "relationships", ("source_id", "target_id", "relation")),
)


# ---------------------------------------------------------------------------
# Column-type introspection + value (de)serialization
# ---------------------------------------------------------------------------


_KIND_DATETIME = "datetime"
_KIND_JSONB = "jsonb"
_KIND_INTERVAL = "interval"
_KIND_VECTOR = "vector"
_KIND_ARRAY = "array"
_KIND_PRIMITIVE = "primitive"


async def _introspect_columns(
    conn: asyncpg.Connection, table_name: str
) -> dict[str, str]:
    """Return ``{column_name: kind}`` for the table.

    ``kind`` is one of ``datetime``, ``jsonb``, ``interval``, ``vector``,
    ``array``, ``primitive``. We use this to drive type-aware (de)serialization
    rather than hand-rolling a column list per table.
    """
    rows = await conn.fetch(
        """
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = $1
        ORDER BY ordinal_position
        """,
        table_name,
    )
    out: dict[str, str] = {}
    for r in rows:
        name = r["column_name"]
        dt = r["data_type"]
        udt = r["udt_name"]
        if dt == "timestamp with time zone":
            out[name] = _KIND_DATETIME
        elif dt == "jsonb":
            out[name] = _KIND_JSONB
        elif dt == "interval":
            out[name] = _KIND_INTERVAL
        elif udt == "vector":
            out[name] = _KIND_VECTOR
        elif dt == "ARRAY":
            out[name] = _KIND_ARRAY
        else:
            out[name] = _KIND_PRIMITIVE
    return out


def _row_to_jsonable(row: asyncpg.Record, col_kinds: dict[str, str]) -> dict[str, Any]:
    """Convert an asyncpg Record into a JSON-friendly dict using column kinds."""
    out: dict[str, Any] = {}
    for col, kind in col_kinds.items():
        v = row[col]
        if v is None:
            out[col] = None
        elif kind == _KIND_DATETIME:
            out[col] = v.isoformat() if isinstance(v, datetime) else v
        elif kind == _KIND_INTERVAL:
            out[col] = v.total_seconds() if isinstance(v, timedelta) else v
        elif kind == _KIND_VECTOR:
            # pgvector hands back a list[float] when a vector codec is
            # registered on the connection (the app pool), but the bare
            # asyncpg pools the backup workflow / CLI open have none — there
            # the value arrives as its text literal "[0.1,0.2,...]". Parse
            # that form; do NOT fall through to list(v), which would explode
            # the ~10KB string into ~9500 single-character entries, bloating
            # the backup ~8x (1.5GB) and OOM-ing the runner mid-serialize.
            out[col] = json.loads(v) if isinstance(v, str) else list(v)
        elif kind == _KIND_ARRAY:
            out[col] = list(v) if not isinstance(v, list) else v
        elif kind == _KIND_JSONB:
            # asyncpg returns jsonb as a JSON-encoded string by default
            # (no codec registered). Parse to native types so the backup
            # holds nested objects/arrays, not escaped strings — and so
            # restore doesn't double-encode.
            out[col] = json.loads(v) if isinstance(v, str) else v
        else:
            out[col] = v
    return out


def _jsonable_to_args(
    row_dict: dict[str, Any], col_kinds: dict[str, str]
) -> tuple[list[str], list[Any]]:
    """Build ``(columns, values)`` for an INSERT, applying inverse converters.

    Only columns present in ``row_dict`` are included — old backups missing a
    column still restore cleanly (the new column picks up its default).
    """
    cols: list[str] = []
    args: list[Any] = []
    for col, kind in col_kinds.items():
        if col not in row_dict:
            continue
        v = row_dict[col]
        if v is None:
            cols.append(col)
            args.append(None)
            continue
        if kind == _KIND_DATETIME:
            args.append(datetime.fromisoformat(v) if isinstance(v, str) else v)
        elif kind == _KIND_INTERVAL:
            args.append(timedelta(seconds=v) if isinstance(v, (int, float)) else v)
        elif kind == _KIND_JSONB:
            # asyncpg expects a JSON string for jsonb columns
            args.append(json.dumps(v))
        elif kind == _KIND_VECTOR:
            # Symmetric to the backup side: restore connections are also
            # codec-less, so a Python list can't bind to the ${i}::vector
            # cast. Re-serialize to the pgvector text literal "[...]" (the
            # shape the codec's encoder produces); the ::vector cast parses
            # it. A value already a str (legacy/other backups) passes through.
            if isinstance(v, list):
                args.append("[" + ",".join(str(x) for x in v) + "]")
            else:
                args.append(v)
        else:
            args.append(v)
        cols.append(col)
    return cols, args


def _insert_sql(spec: _TableSpec, columns: list[str], col_kinds: dict[str, str]) -> str:
    """Build an idempotent INSERT statement for the spec.

    Vector columns get an explicit ``::vector`` cast. ON CONFLICT keys come
    from the spec's primary-key declaration.
    """
    placeholders = []
    for i, col in enumerate(columns, start=1):
        if col_kinds.get(col) == _KIND_VECTOR:
            # Cast through ::text first so asyncpg types the param as text and
            # binds the "[...]" literal directly, regardless of whether a
            # pgvector codec is registered on the connection. A bare
            # ${i}::vector routes the value through the codec's encoder when
            # one exists — which would re-encode our already-serialized string
            # character by character. ::text::vector lets Postgres parse it.
            placeholders.append(f"${i}::text::vector")
        else:
            placeholders.append(f"${i}")

    col_list = ", ".join(columns)
    val_list = ", ".join(placeholders)
    pk_list = ", ".join(spec.pk)
    # Quote table/column names defensively even though we control the inputs.
    return (
        f"INSERT INTO {spec.name} ({col_list}) VALUES ({val_list}) "
        f"ON CONFLICT ({pk_list}) DO NOTHING"
    )


# ---------------------------------------------------------------------------
# backup_all / restore_all
# ---------------------------------------------------------------------------


async def verify_backup_preconditions(pool: asyncpg.Pool) -> None:
    """Require an owner-capable connection and an exact, ready schema.

    Export/restore is an operator operation over every owner's rows.  It must
    not be reachable through the restricted application role, and it must not
    turn a restore into an implicit schema-upgrade operation.  This check is
    intentionally read-only and does not inspect or return credentials.
    """
    role = await pool.fetchrow(
        """
        SELECT current_user AS role, r.rolsuper, r.rolbypassrls,
               EXISTS (
                   SELECT 1 FROM pg_class c
                   JOIN pg_roles owner ON owner.oid = c.relowner
                   WHERE c.oid = 'public.memories'::regclass
                     AND owner.rolname = current_user
               ) AS owns_memories
        FROM pg_roles r
        WHERE r.rolname = current_user
        """
    )
    if role is None:
        raise PermissionError("backup/restore requires an owner-capable database role")
    role_name = role.get("role") or ""
    if role_name == "weft_app":
        raise PermissionError(
            "backup/restore requires an owner-capable database role; refusing restricted runtime role"
        )
    if not (
        role.get("rolsuper")
        or role.get("rolbypassrls")
        or role.get("owns_memories")
    ):
        raise PermissionError("backup/restore requires an owner-capable database role")

    # The ledger check is read-only and fails on missing/pending/unknown
    # migrations.  The CLI must direct operators to the owner migration command
    # rather than silently applying anything as part of restore.
    await verify_migration_ledger(pool)


async def backup_all(pool: asyncpg.Pool) -> dict:
    """Export user data as a self-contained JSON-friendly dict.

    Bypasses RLS to ensure ALL rows (across all user_ids) are exported.
    Requires the connection role to be the table owner and the schema ledger
    to be complete.  Preconditions are read-only and never expose secrets.
    """
    await verify_backup_preconditions(pool)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL row_security = off")

            sections: dict[str, list[dict[str, Any]]] = {}
            section_kinds: dict[str, dict[str, str]] = {}
            for spec in _TABLES:
                kinds = await _introspect_columns(conn, spec.name)
                section_kinds[spec.section] = kinds
                # Stable ordering for deterministic backups.
                order_col = (
                    "created_at"
                    if "created_at" in kinds
                    else (
                        "added_at"
                        if "added_at" in kinds
                        else "mentioned_at"
                        if "mentioned_at" in kinds
                        else next(iter(spec.pk))
                    )
                )
                rows = await conn.fetch(
                    f"SELECT * FROM {spec.name} ORDER BY {order_col} ASC"
                )
                sections[spec.section] = [_row_to_jsonable(r, kinds) for r in rows]

    schema_version = max(v for v, _, _ in MIGRATIONS)

    # Checksum is content-only and stable across format-version bumps —
    # spans memories ID+content (the historical 1.0 contract). Adding new
    # sections doesn't change what the checksum covers, so a backup taken
    # at v1.1 and re-loaded at v1.2 still verifies.
    memories = sections.get("memories", [])
    relationships = sections.get("relationships", [])
    content_hash = hashlib.sha256(
        json.dumps(
            [m["id"] + m["content"] for m in memories], sort_keys=True
        ).encode()
    ).hexdigest()

    counts = {f"{spec.section}_count": len(sections[spec.section]) for spec in _TABLES}

    return {
        "version": BACKUP_VERSION,
        "schema_version": schema_version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checksum": content_hash,
        # Legacy keys — kept for v1.0/v1.1 reader compatibility.
        "memory_count": len(memories),
        "relationship_count": len(relationships),
        "memories": memories,
        "relationships": relationships,
        # New sections (v1.2+).
        "workspaces": sections["workspaces"],
        "workspace_members": sections["workspace_members"],
        "behaviors": sections["behaviors"],
        "entities": sections["entities"],
        "entity_mentions": sections["entity_mentions"],
        "episodes": sections["episodes"],
        "episode_memories": sections["episode_memories"],
        "modes": sections["modes"],
        "trackers": sections["trackers"],
        "counts": counts,
    }


async def restore_all(
    pool: asyncpg.Pool,
    data: dict,
    *,
    dry_run: bool = False,
    skip_duplicates: bool = True,
) -> dict:
    """Restore user data from a backup dict.

    Bypasses RLS to restore rows with their original user_id values.

    Sections missing from the backup (legacy 1.0/1.1 backups don't have
    behaviors/entities/etc.) are simply skipped — no error.

    Returns a report dict with per-section restored/skipped counts plus
    the legacy ``memories_restored`` / ``relationships_restored`` keys for
    backward compat with CLI consumers.
    """
    if not isinstance(data, dict):
        raise ValueError("backup payload must be an object")
    # Dry-run is non-mutating, but still requires the same owner/schema gate so
    # it cannot be used to probe a restricted or partially migrated database.
    await verify_backup_preconditions(pool)

    version = data.get("version")
    if version not in (BACKUP_VERSION,) + _LEGACY_VERSIONS:
        logger.warning(
            "Backup version mismatch: expected %s, got %s",
            BACKUP_VERSION,
            version,
        )

    # Behaviors got added late; missing == [] is fine. Same for new sections.
    section_data: dict[str, list[dict[str, Any]]] = {
        spec.section: list(data.get(spec.section, []))
        for spec in _TABLES
    }

    report: dict[str, Any] = {
        "errors": [],
        "memories_restored": 0,
        "memories_skipped": 0,
        "relationships_restored": 0,
        "relationships_skipped": 0,
    }
    for spec in _TABLES:
        report[f"{spec.section}_restored"] = 0
        report[f"{spec.section}_skipped"] = 0

    if dry_run:
        # ``dry_run`` reports counts as if every row would insert. We can't
        # cheaply detect would-be-skipped rows when PKs include jsonb columns
        # (workspace_members.member_identity), so we approximate: rows whose
        # PK can be hashed get a real check; the rest count as inserts.
        async with pool.acquire() as conn:
            for spec in _TABLES:
                rows = section_data[spec.section]
                if not rows:
                    continue
                try:
                    pk_cols = ", ".join(spec.pk)
                    existing = await conn.fetch(
                        f"SELECT {pk_cols} FROM {spec.name}"
                    )
                    existing_keys = {tuple(r[c] for c in spec.pk) for r in existing}
                    hashable = True
                except TypeError:
                    existing_keys = set()
                    hashable = False
                for row_dict in rows:
                    if hashable:
                        try:
                            key = tuple(row_dict.get(c) for c in spec.pk)
                            if key in existing_keys:
                                report[f"{spec.section}_skipped"] += 1
                                continue
                        except TypeError:
                            pass
                    report[f"{spec.section}_restored"] += 1
        return report

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL row_security = off")

            for spec in _TABLES:
                rows = section_data[spec.section]
                if not rows:
                    continue
                kinds = await _introspect_columns(conn, spec.name)

                for row_dict in rows:
                    cols, args = _jsonable_to_args(row_dict, kinds)
                    sql = _insert_sql(spec, cols, kinds)
                    try:
                        result = await conn.execute(sql, *args)
                        # 'INSERT 0 1' = inserted; 'INSERT 0 0' = ON CONFLICT no-op.
                        if result.split()[-1] != "0":
                            report[f"{spec.section}_restored"] += 1
                        else:
                            if not skip_duplicates:
                                pk_repr = {c: row_dict.get(c) for c in spec.pk}
                                raise ValueError(
                                    f"restore collision in {spec.name}: {pk_repr}; "
                                    "rerun with duplicate skipping enabled"
                                )
                            report[f"{spec.section}_skipped"] += 1
                    except ValueError:
                        raise
                    except Exception as e:
                        pk_repr = {c: row_dict.get(c) for c in spec.pk}
                        report["errors"].append(f"{spec.name} {pk_repr}: {e}")

    if not skip_duplicates and any(
        report[f"{spec.section}_skipped"] for spec in _TABLES
    ):
        # ON CONFLICT DO NOTHING means we always silently skip duplicates.
        # The flag's only effect today is documentation; preserve the kwarg
        # for CLI compat but log when callers ask for the strict behavior.
        logger.debug(
            "restore_all called with skip_duplicates=False but duplicates "
            "were silently skipped via ON CONFLICT DO NOTHING."
        )

    return report


# ---------------------------------------------------------------------------
# verify_backup
# ---------------------------------------------------------------------------


def verify_backup(data: dict) -> dict:
    """Verify a backup file's integrity without connecting to a database."""
    issues: list[str] = []

    if not isinstance(data, dict):
        return {
            "valid": False,
            "issues": ["Backup payload must be an object"],
            "memory_count": 0,
            "relationship_count": 0,
            "memories_with_embeddings": 0,
            "exported_at": None,
            "schema_version": None,
        }

    unexpected = set(data) - _PORTABLE_TOP_LEVEL_KEYS
    for key in sorted(unexpected):
        if key in _NON_PORTABLE_SECTIONS or key.startswith(("oauth_", "auth_")):
            issues.append(f"Non-portable section rejected: {key}")
        else:
            issues.append(f"Unknown top-level section rejected: {key}")

    if "version" not in data:
        issues.append("Missing 'version' field")
    elif data["version"] not in (BACKUP_VERSION,) + _LEGACY_VERSIONS:
        issues.append(
            f"Version mismatch: expected {BACKUP_VERSION}, got {data['version']}"
        )

    if "memories" not in data:
        issues.append("Missing 'memories' field")
    elif not isinstance(data["memories"], list):
        issues.append("'memories' is not a list")

    if "relationships" in data and not isinstance(data["relationships"], list):
        issues.append("'relationships' is not a list")

    memories = data.get("memories", [])
    relationships = data.get("relationships", [])
    if not isinstance(memories, list):
        memories = []
    if not isinstance(relationships, list):
        relationships = []

    valid_memory_rows: list[dict[str, Any]] = []
    for i, memory in enumerate(memories):
        if not isinstance(memory, dict):
            issues.append(f"memories[{i}] must be an object")
            continue
        valid_memory_rows.append(memory)

    if data.get("checksum") and valid_memory_rows:
        checksum_parts: list[str] = []
        for i, memory in enumerate(valid_memory_rows):
            if not isinstance(memory.get("id"), str) or not isinstance(memory.get("content"), str):
                issues.append(f"memories[{i}] id and content must be strings")
                continue
            checksum_parts.append(memory["id"] + memory["content"])
        if len(checksum_parts) == len(valid_memory_rows):
            computed = hashlib.sha256(
                json.dumps(checksum_parts, sort_keys=True).encode()
            ).hexdigest()
            if computed != data["checksum"]:
                issues.append(
                    f"Checksum mismatch: expected {data['checksum']}, computed {computed}"
                )

    if data.get("memory_count") is not None and len(memories) != data["memory_count"]:
        issues.append(
            f"Memory count mismatch: header says {data['memory_count']}, "
            f"actual {len(memories)}"
        )
    if (
        data.get("relationship_count") is not None
        and len(relationships) != data["relationship_count"]
    ):
        issues.append(
            f"Relationship count mismatch: header says {data['relationship_count']}, "
            f"actual {len(relationships)}"
        )

    required_fields = {"id", "type", "content"}
    for i, m in enumerate(valid_memory_rows):
        missing = required_fields - set(m.keys())
        if missing:
            issues.append(f"Memory [{i}] missing fields: {missing}")

    memory_ids = {m.get("id") for m in valid_memory_rows if isinstance(m.get("id"), str)}
    for i, rel in enumerate(relationships):
        if not isinstance(rel, dict):
            issues.append(f"relationships[{i}] must be an object")
            continue
        if rel.get("source_id") not in memory_ids:
            issues.append(
                f"Relationship [{i}] references unknown source_id: {rel.get('source_id')}"
            )
        if rel.get("target_id") not in memory_ids:
            issues.append(
                f"Relationship [{i}] references unknown target_id: {rel.get('target_id')}"
            )

    # New v1.2 sections — verify only if present and only the FK edges that
    # legacy backups don't carry. ``entity_mentions``/``episode_memories``
    # reference memory_ids; ``workspace_members`` references workspaces.
    def _section_rows(name: str) -> list[dict[str, Any]]:
        rows = data.get(name, [])
        if not isinstance(rows, list):
            issues.append(f"'{name}' is not a list")
            return []
        valid: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                issues.append(f"{name}[{i}] must be an object")
                continue
            valid.append(row)
        return valid

    workspaces = _section_rows("workspaces")
    entities = _section_rows("entities")
    episodes = _section_rows("episodes")
    entity_mentions = _section_rows("entity_mentions")
    episode_memories = _section_rows("episode_memories")
    workspace_members = _section_rows("workspace_members")
    workspace_ids = {w.get("id") for w in workspaces}
    entity_ids = {e.get("id") for e in entities}
    episode_ids = {e.get("id") for e in episodes}

    for i, em in enumerate(entity_mentions):
        if em.get("entity_id") not in entity_ids:
            issues.append(
                f"entity_mentions[{i}] references unknown entity_id: {em.get('entity_id')}"
            )
        if em.get("memory_id") not in memory_ids:
            issues.append(
                f"entity_mentions[{i}] references unknown memory_id: {em.get('memory_id')}"
            )
    for i, em in enumerate(episode_memories):
        if em.get("episode_id") not in episode_ids:
            issues.append(
                f"episode_memories[{i}] references unknown episode_id: {em.get('episode_id')}"
            )
        if em.get("memory_id") not in memory_ids:
            issues.append(
                f"episode_memories[{i}] references unknown memory_id: {em.get('memory_id')}"
            )
    for i, wm in enumerate(workspace_members):
        if wm.get("workspace_id") not in workspace_ids:
            issues.append(
                f"workspace_members[{i}] references unknown workspace_id: "
                f"{wm.get('workspace_id')}"
            )

    with_embeddings = sum(1 for m in valid_memory_rows if m.get("embedding"))

    counts = {
        "memory_count": len(memories),
        "relationship_count": len(relationships),
        "memories_with_embeddings": with_embeddings,
    }
    for spec in _TABLES:
        if spec.section in ("memories", "relationships"):
            continue
        counts[f"{spec.section}_count"] = len(data.get(spec.section, []))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        **counts,
        "exported_at": data.get("exported_at"),
        "schema_version": data.get("schema_version"),
    }
