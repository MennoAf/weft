"""Hash-based change detection for Obsidian vault files."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path.home() / ".weft" / "obsidian_hashes.json"


class HashStore:
    """Tracks content hashes to avoid re-ingesting unchanged files."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            self._data = {"vault_path": None, "files": {}}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            self._data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Hash store corrupted, starting fresh: %s", exc)
            self._data = {"vault_path": None, "files": {}}

    def save(self):
        """Persist hash store to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2), encoding="utf-8"
        )

    def get_hash(self, rel_path: str) -> str | None:
        entry = self._data.get("files", {}).get(rel_path)
        return entry["hash"] if entry else None

    def get_memory_ids(self, rel_path: str) -> list[str]:
        entry = self._data.get("files", {}).get(rel_path)
        return entry.get("memory_ids", []) if entry else []

    def update(
        self, rel_path: str, content_hash: str, memory_ids: list[str]
    ):
        """Record a file's hash and the memory IDs it produced."""
        files = self._data.setdefault("files", {})
        files[rel_path] = {
            "hash": content_hash,
            "memory_ids": memory_ids,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    def remove(self, rel_path: str):
        """Remove a file from the hash store."""
        self._data.get("files", {}).pop(rel_path, None)

    def set_vault_path(self, vault_path: str):
        """Set or validate the vault path. Clears store if path changed."""
        stored = self._data.get("vault_path")
        if stored and stored != vault_path:
            logger.warning(
                "Vault path changed from %s to %s — clearing hash store",
                stored,
                vault_path,
            )
            self._data = {"vault_path": vault_path, "files": {}}
        else:
            self._data["vault_path"] = vault_path

    @property
    def file_count(self) -> int:
        return len(self._data.get("files", {}))

    @property
    def all_paths(self) -> set[str]:
        return set(self._data.get("files", {}).keys())
