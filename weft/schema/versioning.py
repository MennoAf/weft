"""Schema upgrade chain for memory rows.

Pattern lifted from Wick INTERCHANGE:
- ``schema_version`` on every row is the dispatch field
- ``UPGRADE_FNS[v]`` returns a row at version ``v + 1``
- Readers call ``upgrade_to_current`` before consuming
- Writers always write at ``CURRENT_VERSION``

When a future version ships, drop a new entry into ``UPGRADE_FNS`` and
bump ``CURRENT_VERSION``. Older versions stay in ``SUPPORTED_VERSIONS``
until the batch rewrite pass has cleared them from disk; then they
can be removed and the upgrade fn for them deleted.
"""

from __future__ import annotations

from typing import Callable

CURRENT_VERSION: int = 1
SUPPORTED_VERSIONS: tuple[int, ...] = (1,)

# Sentinel user_id for system-owned rows (seeds, shared modes, anything that
# was historically ``user_id IS NULL`` by convention). Migration 36 backfills
# every NULL to this value and adds a NOT NULL constraint so agents cannot
# accidentally write a global row by forgetting to set ``app.user_id``.
#
# Named-string format (rather than a UUID) so that an agent without context
# cannot guess it — "just give yourself a UUID" cannot land on this value.
SYSTEM_GLOBAL_USER_ID: str = "__system_global_zathras__"


def _upgrade_0_to_1(row: dict) -> dict:
    """Pre-schema rows lack the v1 fields. Backfill defaults that match
    migration 34's UPDATE clause so an in-memory row is indistinguishable
    from a freshly migrated DB row."""
    user_id = row.get("user_id")
    upgraded = dict(row)
    upgraded.setdefault("visibility", "global" if user_id is None else "private")
    if "author_identity" not in upgraded:
        if user_id is None:
            upgraded["author_identity"] = {"kind": "system", "component": "seed"}
        else:
            upgraded["author_identity"] = {"kind": "local_user", "user_id": user_id}
    upgraded.setdefault("provenance", {"source": "self"})
    upgraded.setdefault("sharing_metadata", {})
    upgraded.setdefault("workspace_id", None)
    upgraded["schema_version"] = 1
    return upgraded


UPGRADE_FNS: dict[int, Callable[[dict], dict]] = {
    0: _upgrade_0_to_1,
}


def upgrade_to_current(row: dict) -> dict:
    """Upgrade a row to ``CURRENT_VERSION``. No-op if already current."""
    version = row.get("schema_version") or 0
    while version < CURRENT_VERSION:
        upgrade = UPGRADE_FNS.get(version)
        if upgrade is None:
            raise ValueError(
                f"No upgrade path from schema_version={version}; "
                f"missing UPGRADE_FNS[{version}]"
            )
        row = upgrade(row)
        version = row.get("schema_version") or version + 1
    return row
