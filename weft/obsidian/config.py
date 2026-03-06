"""Folder-to-memory-type mapping and vault configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from weft.models import MemoryType


@dataclass
class FolderMapping:
    """Maps a vault folder to a Weft memory type and topics."""

    memory_type: MemoryType
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.7


DEFAULT_FOLDER_MAP: dict[str, FolderMapping] = {
    "inbox": FolderMapping(MemoryType.fact, ["inbox"], confidence=0.5),
    "notes": FolderMapping(MemoryType.fact, ["reminders"]),
    "journal/daily": FolderMapping(MemoryType.fact, ["journal"]),
    "journal": FolderMapping(MemoryType.fact, ["journal"]),
    "people": FolderMapping(MemoryType.user_model, ["contacts"]),
    "wktw": FolderMapping(MemoryType.fact, ["wktw"]),
    "wktw/clients": FolderMapping(MemoryType.fact, ["wktw", "clients"]),
    "wktw/meetings": FolderMapping(MemoryType.fact, ["wktw", "meetings"]),
    "wktw/ideas": FolderMapping(MemoryType.fact, ["wktw", "ideas"]),
    "wktw/finances": FolderMapping(MemoryType.fact, ["wktw", "finances"]),
    "wktw/finances/income": FolderMapping(MemoryType.fact, ["wktw", "finances", "income"]),
    "wktw/finances/expenses": FolderMapping(MemoryType.fact, ["wktw", "finances", "expenses"]),
    "wktw/operations": FolderMapping(MemoryType.fact, ["wktw", "operations"]),
    "tools": FolderMapping(MemoryType.fact, ["tools"]),
    "recipes": FolderMapping(MemoryType.fact, ["recipes"]),
    "media": FolderMapping(MemoryType.fact, ["media"]),
    "writing/ideas": FolderMapping(MemoryType.fact, ["writing", "creative"]),
    "writing/blog": FolderMapping(MemoryType.fact, ["writing", "blog"]),
    "writing": FolderMapping(MemoryType.fact, ["writing"]),
}

DEFAULT_EXCLUDED_DIRS: set[str] = {
    "assets",
    ".obsidian",
    ".trash",
    ".git",
    "templates",
}

DEFAULT_SPLIT_THRESHOLD = 8192  # bytes
DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def resolve_folder_mapping(
    rel_path: str,
    folder_map: dict[str, FolderMapping] | None = None,
) -> FolderMapping:
    """Find the best folder mapping for a relative file path.

    Matches longest prefix first, falls back to a generic default.
    """
    if folder_map is None:
        folder_map = DEFAULT_FOLDER_MAP

    parts = Path(rel_path).parent.parts

    # Try longest prefix first
    for i in range(len(parts), 0, -1):
        prefix = "/".join(parts[:i])
        if prefix in folder_map:
            return folder_map[prefix]

    return FolderMapping(MemoryType.fact, ["notes"])
