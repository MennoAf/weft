"""Data models for the Weft Capability Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CapabilityEntry:
    """Structured capability record ready for Weft memory storage."""

    repo_slug: str
    file_path: str
    symbol_name: Optional[str] = None
    symbol_kind: Optional[str] = None
    docstring: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    reuse_notes: Optional[str] = None
    file_hash: str = ""
    capability_slugs: list[str] = field(default_factory=list)

    @property
    def topics(self) -> list[str]:
        """Return deterministic Weft topic tags for this capability."""
        topics = [
            f"repo:{self.repo_slug}",
            f"file:{self.file_path}",
        ]
        if self.symbol_name is not None:
            topics.append(f"symbol:{self.symbol_name}")
        topics.extend(f"capability:{slug}" for slug in self.capability_slugs)
        return topics

    @property
    def content(self) -> str:
        """Return the structured text block stored as Weft memory content."""
        capability = (
            ", ".join(self.capability_slugs)
            if self.capability_slugs
            else "(unclassified)"
        )
        lines = [
            f"CAPABILITY: {capability}",
            f"REPO: {self.repo_slug}",
            f"FILE: {self.file_path}",
        ]

        if self.symbol_name is not None:
            lines.append(f"SYMBOL: {self.symbol_name} ({self.symbol_kind})")
        if self.docstring is not None:
            lines.append(f"DOCSTRING: {self.docstring[:500]}")
        if self.imports:
            lines.append(f"IMPORTS: {', '.join(self.imports[:20])}")
        if self.reuse_notes is not None:
            lines.append(f"REUSE_NOTES: {self.reuse_notes}")

        lines.append(f"FILE_HASH: {self.file_hash}")
        return "\n".join(lines)
