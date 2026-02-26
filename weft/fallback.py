"""Weft fallback reader — graceful degradation when DB is unavailable.

When Postgres/Redis are down, Weft can still serve memories from a previously
exported markdown file (produced by ``weft export``).  This module provides:

* ``read_fallback(path)`` — return the raw markdown contents (or "" if missing).
* ``search_fallback(query, path, limit)`` — keyword search over exported sections.
* ``_parse_sections(text)`` — split exported markdown into structured dicts.
"""

from __future__ import annotations

import re
from pathlib import Path

FALLBACK_PATH = Path.home() / ".weft" / "fallback.md"

# Regex for the per-memory heading produced by exporter._format_memory_md
# Example: ### [fact] Redis is used for caching (confidence: 0.9, accessed: 3 times)
_HEADING_RE = re.compile(
    r"^### \[(?P<type>[a-z_]+)\]\s+"
    r"(?P<title>.+?)\s+"
    r"\(confidence:\s*(?P<confidence>[0-9.]+),\s*accessed:\s*(?P<access_count>\d+)\s*times?\)",
    re.MULTILINE,
)

# Topic heading: ## Topic: infrastructure
_TOPIC_RE = re.compile(r"^## Topic:\s*(.+)$", re.MULTILINE)


def read_fallback(path: str | Path | None = None) -> str:
    """Return the full contents of an exported markdown file, or '' if missing."""
    p = Path(path) if path is not None else FALLBACK_PATH
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def search_fallback(
    query: str,
    path: str | Path | None = None,
    limit: int = 10,
) -> list[dict]:
    """Keyword search over an exported markdown file.

    Each result dict has keys: type, title, confidence, access_count, topic,
    content, hits (number of keyword matches — used for ranking).

    Results are sorted by *hits* descending (best matches first).
    """
    text = read_fallback(path or FALLBACK_PATH)
    if not text:
        return []

    sections = _parse_sections(text)
    if not sections:
        return []

    # Normalise query into individual keywords for matching
    keywords = query.lower().split()
    if not keywords:
        return []

    scored: list[tuple[int, dict]] = []
    for section in sections:
        searchable = (section.get("content", "") + " " + section.get("title", "")).lower()
        hits = sum(searchable.count(kw) for kw in keywords)
        if hits > 0:
            section["hits"] = hits
            scored.append((hits, section))

    # Sort by hits descending
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [s for _, s in scored[:limit]]


def _parse_sections(text: str) -> list[dict]:
    """Parse exported markdown into a list of section dicts.

    Each dict contains: type, title, confidence (float), access_count (int),
    topic (str or ''), content (str).
    """
    sections: list[dict] = []

    # Determine current topic context by walking through the text
    # Strategy: find all heading positions, then extract content between them.
    heading_matches = list(_HEADING_RE.finditer(text))
    if not heading_matches:
        return sections

    # Build a map of line-offset → topic by scanning topic headings
    topic_positions: list[tuple[int, str]] = []
    for m in _TOPIC_RE.finditer(text):
        topic_positions.append((m.start(), m.group(1).strip()))
    # Also detect Uncategorized
    uncat_match = re.search(r"^## Uncategorized", text, re.MULTILINE)
    if uncat_match:
        topic_positions.append((uncat_match.start(), ""))
    topic_positions.sort(key=lambda t: t[0])

    def _topic_at(pos: int) -> str:
        """Return the topic that is active at a given text offset."""
        current = ""
        for tp, name in topic_positions:
            if tp > pos:
                break
            current = name
        return current

    for i, match in enumerate(heading_matches):
        # Content runs from end of heading line to start of next heading (or EOF)
        content_start = match.end()
        content_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)

        # Also stop at the next topic heading if it comes before the next memory heading
        for tp, _ in topic_positions:
            if tp > content_start and tp < content_end:
                content_end = tp
                break

        raw_content = text[content_start:content_end].strip()

        sections.append({
            "type": match.group("type"),
            "title": match.group("title").strip(),
            "confidence": float(match.group("confidence")),
            "access_count": int(match.group("access_count")),
            "topic": _topic_at(match.start()),
            "content": raw_content,
        })

    return sections
