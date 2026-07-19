"""Database migrations — sequential, append-only, one file per version.

Each migration lives in ``vNN_<slug>.py`` and exports a ``MIGRATION``
tuple ``(version, description, sql)``. The list is assembled here by
importing every ``vNN_*.py`` module and sorting by version.

Adding a migration: drop a new ``v{N+1}_<slug>.py`` file. No edits to
this file or to the runner are required.

Never modify existing migration files; always add a new numbered one.
"""

from __future__ import annotations

import importlib
import pkgutil

from weft.db.migrations._runner import (
    _MIGRATION_LOCK_ID,
    _get_applied_versions,
    run_migrations,
    verify_migration_ledger,
    verify_migrations,
    verify_runtime_invariants,
)


def _discover() -> list[tuple[int, str, str]]:
    """Import every ``vNN_*.py`` sibling and collect MIGRATION tuples."""
    tuples: list[tuple[int, str, str]] = []
    for _, name, ispkg in pkgutil.iter_modules(__path__):
        if ispkg or not name.startswith("v"):
            continue
        # Numeric prefix sanity: ``v01_foo`` has digits at [1:3].
        prefix = name[1:].split("_", 1)[0]
        if not prefix.isdigit():
            continue
        mod = importlib.import_module(f"{__name__}.{name}")
        tuples.append(mod.MIGRATION)

    tuples.sort(key=lambda m: m[0])
    versions = [m[0] for m in tuples]
    if len(set(versions)) != len(versions):
        duplicates = [v for v in versions if versions.count(v) > 1]
        raise RuntimeError(
            f"Duplicate migration versions detected: {sorted(set(duplicates))}"
        )
    return tuples


MIGRATIONS: list[tuple[int, str, str]] = _discover()

__all__ = [
    "MIGRATIONS",
    "run_migrations",
    "verify_migration_ledger",
    "verify_migrations",
    "verify_runtime_invariants",
    "_MIGRATION_LOCK_ID",
    "_get_applied_versions",
]
