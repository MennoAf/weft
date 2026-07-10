"""Topic-based capability lookup helpers for the Weft Capability Registry."""

from __future__ import annotations

import inspect
import re
from typing import Any, Awaitable, Callable

from capability_registry.slug_classifier import infer_slugs


MemoryRecord = dict[str, Any]
ListMemoriesFn = Callable[..., list[MemoryRecord] | Awaitable[list[MemoryRecord]]]

_EXPLICIT_TOPIC_PREFIXES = ("capability:", "repo:", "file:", "symbol:")

_CONTENT_FIELD_TO_KEY = {
    "CAPABILITY": "capability",
    "REPO": "repo",
    "FILE": "file",
    "SYMBOL": "symbol",
    "DOCSTRING": "docstring",
    "IMPORTS": "imports",
    "REUSE_NOTES": "reuse_notes",
    "FILE_HASH": "file_hash",
}
_CONTENT_LINE_RE = re.compile(r"^([A-Z_]+):\s*(.*)$")


def query_to_topics(query: str) -> list[str]:
    """Translate a free-text or explicit-tag query into Weft topic filters."""
    normalized = query.strip().lower()
    if not normalized:
        return []
    if normalized.startswith(_EXPLICIT_TOPIC_PREFIXES):
        return [normalized]

    slugs = infer_slugs(normalized, None, [])
    if slugs:
        return [f"capability:{slug}" for slug in slugs]
    return [f"capability:{normalized.replace(' ', '-')}"]


def parse_capability_entry_from_content(content: str) -> dict[str, str | None]:
    """Parse a capability memory content block into a field dict.

    First occurrence wins so stray field-shaped lines inside a docstring
    cannot overwrite real header values. Unknown or malformed lines are
    skipped; missing fields stay None.
    """
    parsed: dict[str, str | None] = {
        key: None for key in _CONTENT_FIELD_TO_KEY.values()
    }
    for line in content.splitlines():
        match = _CONTENT_LINE_RE.match(line.strip())
        if match is None:
            continue
        key = _CONTENT_FIELD_TO_KEY.get(match.group(1))
        if key is not None and parsed[key] is None:
            parsed[key] = match.group(2).strip()
    return parsed


async def _default_list_memories(
    pool: Any,
    topic: str,
    project_id: str | None,
    limit: int,
) -> list[MemoryRecord]:
    from weft.models import MemoryStatus
    from weft.store import list_memories

    memories = await list_memories(
        pool,
        status=MemoryStatus.active,
        topic=topic,
        project_id=project_id,
        limit=limit,
    )
    return [memory.to_dict() for memory in memories]


async def lookup_capabilities(
    query: str,
    pool: Any,
    project_id: str | None,
    limit: int = 10,
    *,
    list_memories_fn: ListMemoriesFn = _default_list_memories,
) -> list[dict[str, Any]]:
    """Resolve a query to capability topics and return parsed matches.

    Queries one topic at a time (list_memories takes a single topic filter),
    deduplicates by memory id preserving first occurrence, and truncates to
    ``limit`` results.
    """
    topics = query_to_topics(query)
    if not topics:
        return []

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for topic in topics:
        records = list_memories_fn(pool, topic, project_id, limit)
        if inspect.isawaitable(records):
            records = await records
        for record in records:
            memory_id = str(record.get("id", ""))
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            results.append(
                {
                    "memory_id": memory_id,
                    "topics": list(record.get("topic") or []),
                    "parsed": parse_capability_entry_from_content(
                        str(record.get("content", ""))
                    ),
                }
            )
            if len(results) >= limit:
                return results
    return results


def format_lookup_results(results: list[dict[str, Any]]) -> str:
    """Format lookup results as a reuse-oriented text report."""
    if not results:
        return "No capability entries found."

    blocks = []
    for result in results:
        parsed = result.get("parsed", {})
        symbol = parsed.get("symbol") or "(module)"
        docstring = parsed.get("docstring") or ""
        notes = parsed.get("reuse_notes") or docstring[:200] or "(no notes)"
        blocks.append(
            f"[repo:{parsed.get('repo')}] {parsed.get('file')} :: {symbol}\n"
            f"Capabilities: {parsed.get('capability')}\n"
            f"{notes}\n"
        )
    return "\n".join(blocks)
