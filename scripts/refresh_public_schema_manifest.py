"""Refresh and validate the public database schema manifest.

This tool is deliberately local and side-effect limited: it reads migration
source and a separately authored policy fixture, then atomically writes the
requested documentation manifest.  It never connects to a database, edits a
migration, or writes hosted data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PACKAGE = ROOT / "weft" / "db" / "migrations"
_CREATE_TABLE = re.compile(
    r"(?is)\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?"
    r"(?:public\.)?([a-z_][a-z0-9_]*)\b"
)


def _migration_modules() -> dict[int, str]:
    """Return the source module for every discovered version."""
    modules: dict[int, str] = {}
    for _, name, is_package in pkgutil.iter_modules([str(MIGRATION_PACKAGE)]):
        if is_package or not re.fullmatch(r"v\d+_[a-z0-9_]+", name):
            continue
        module = importlib.import_module(f"weft.db.migrations.{name}")
        version = int(name[1:].split("_", 1)[0])
        modules[version] = name
        if getattr(module, "MIGRATION")[0] != version:
            raise ValueError(f"migration module/version mismatch: {name}")
    return modules


def _pending_files() -> list[dict[str, str]]:
    """Describe files deliberately excluded by the discovery convention."""
    return [
        {"path": path.relative_to(ROOT).as_posix(), "status": "not_discovered"}
        for path in sorted(MIGRATION_PACKAGE.glob("pending_*.py"))
    ]


def discover_source_state() -> dict[str, Any]:
    """Discover migration records and the created public table names."""
    from weft.db.migrations import MIGRATIONS

    modules = _migration_modules()
    migrations = [
        {
            "version": version,
            "description": description,
            "module": modules.get(version),
        }
        for version, description, _sql in MIGRATIONS
    ]
    if any(record["module"] is None for record in migrations):
        raise ValueError("a MIGRATION tuple has no discovered source module")

    tables: set[str] = set()
    for _version, _description, sql in MIGRATIONS:
        tables.update(match.group(1).lower() for match in _CREATE_TABLE.finditer(sql))
    return {
        "migration_head": max((record["version"] for record in migrations), default=None),
        "migrations": migrations,
        "source_tables": sorted(tables),
        "pending_files": _pending_files(),
    }


def _expected_pending(policy: dict[str, Any], discovered: list[dict[str, str]]) -> list[dict[str, str]]:
    """Combine discovered pending paths with literal policy dispositions."""
    policy_by_path = {item["path"]: item for item in policy.get("pending_files", [])}
    result: list[dict[str, str]] = []
    for item in discovered:
        decision = policy_by_path.get(item["path"])
        if decision is None:
            raise ValueError(f"pending-file disposition missing for {item['path']}")
        if decision.get("status") != "dormant":
            raise ValueError(f"pending-file disposition must be dormant: {item['path']}")
        result.append(
            {
                "path": item["path"],
                "status": "dormant",
                "disposition": decision["disposition"],
            }
        )
    extra = sorted(set(policy_by_path) - {item["path"] for item in discovered})
    if extra:
        raise ValueError(f"policy lists undiscovered pending files: {extra}")
    return result


def _expected_manifest(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    policy_tables = policy.get("public_tables")
    if not isinstance(policy_tables, list):
        raise ValueError("policy public_tables must be a list")
    by_name = {item.get("name"): item for item in policy_tables}
    if len(by_name) != len(policy_tables) or None in by_name:
        raise ValueError("policy contains duplicate or unnamed public tables")
    source_tables = set(state["source_tables"])
    policy_names = set(by_name)
    if source_tables != policy_names:
        raise ValueError(
            "policy/source public table inventory differs: "
            f"missing={sorted(source_tables - policy_names)} "
            f"extra={sorted(policy_names - source_tables)}"
        )
    return {
        "schema": "weft.public-schema-manifest.v1",
        "migration_head": state["migration_head"],
        "migration_discovery": policy["migration_policy"],
        "runtime_policy": policy["runtime_policy"],
        "migrations": state["migrations"],
        "tables": [by_name[name] for name in sorted(by_name)],
        "pending_files": _expected_pending(policy, state["pending_files"]),
    }


def validate_manifest(
    manifest: dict[str, Any], policy: dict[str, Any], *, state: dict[str, Any] | None = None
) -> None:
    """Raise ``ValueError`` when a manifest is stale or violates policy."""
    state = state or discover_source_state()
    migration_records = manifest.get("migrations", [])
    versions = [item.get("version") for item in migration_records]
    if len(versions) != len(set(versions)):
        raise ValueError("duplicate migration entries")
    if manifest.get("migration_head") != state["migration_head"]:
        raise ValueError("stale migration_head")
    if migration_records != state["migrations"]:
        raise ValueError("migration entries differ from source-discovered MIGRATIONS")

    declared_tables = manifest.get("tables", [])
    names = [item.get("name") for item in declared_tables]
    if len(names) != len(set(names)):
        raise ValueError("duplicate public table entries")
    policy_by_name = {item["name"]: item for item in policy.get("public_tables", [])}
    if set(names) != set(state["source_tables"]) or set(names) != set(policy_by_name):
        raise ValueError("public table inventory differs from source or policy")
    for item in declared_tables:
        if item != policy_by_name[item["name"]]:
            raise ValueError(f"public table policy differs for {item['name']}")

    expected_pending = _expected_pending(policy, state["pending_files"])
    if manifest.get("pending_files") != expected_pending:
        raise ValueError("pending-file disposition differs from source/policy")
    if manifest.get("schema") != "weft.public-schema-manifest.v1":
        raise ValueError("unsupported manifest schema")


def refresh(manifest_path: Path, policy_path: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    state = discover_source_state()
    manifest = _expected_manifest(policy, state)
    validate_manifest(manifest, policy, state=state)
    payload = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        f"refreshed {manifest_path}: head=v{state['migration_head']} "
        f"migrations={len(state['migrations'])} tables={len(state['source_tables'])} "
        f"pending={len(state['pending_files'])}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    refresh(args.manifest, args.policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
