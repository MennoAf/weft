"""Weft memory exporter — export memories as markdown or JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import asyncpg

from weft.models import MemoryStatus, MemoryType
from weft.store import list_memories


async def export_memories(
    pool: asyncpg.Pool,
    format: str = "md",
    memory_type: str | None = None,
    topic: str | None = None,
    status: str = "active",
    project_id: str | None = None,
) -> str:
    """Export memories as markdown or JSON string.

    Args:
        pool: asyncpg connection pool.
        format: Output format — "md" for markdown, "json" for JSON.
        memory_type: Filter by memory type (e.g. "fact", "preference").
        topic: Filter by topic tag.
        status: Filter by status (default "active").
        project_id: Scope export to a specific project (includes global memories).

    Returns:
        Formatted string of exported memories.
    """
    # Resolve enum filters
    status_enum = MemoryStatus(status) if status else None
    type_enum = MemoryType(memory_type) if memory_type else None

    # Fetch all matching memories (paginate to get everything)
    all_memories = []
    offset = 0
    batch_size = 100
    while True:
        batch = await list_memories(
            pool,
            status=status_enum,
            memory_type=type_enum,
            topic=topic,
            project_id=project_id,
            limit=batch_size,
            offset=offset,
        )
        all_memories.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    if format == "json":
        return _export_json(all_memories)
    else:
        return _export_markdown(all_memories)


def _export_json(memories: list) -> str:
    """Export memories as a JSON string."""
    return json.dumps(
        {
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )


def _export_markdown(memories: list) -> str:
    """Export memories as grouped markdown."""
    lines: list[str] = ["# Weft Memory Export", ""]

    if not memories:
        lines.append("_No memories found._")
        return "\n".join(lines)

    # Group memories by topic.  A memory can have multiple topics — it appears
    # under the first one found.  Memories with no topics go to "Uncategorized".
    by_topic: dict[str, list] = defaultdict(list)
    uncategorized: list = []

    for m in memories:
        if m.topic:
            # Place under the first topic (canonical grouping)
            by_topic[m.topic[0]].append(m)
        else:
            uncategorized.append(m)

    # Render topic groups in alphabetical order
    for topic_name in sorted(by_topic):
        lines.append(f"## Topic: {topic_name}")
        lines.append("")
        for m in by_topic[topic_name]:
            lines.append(_format_memory_md(m))
        lines.append("")

    # Render uncategorized section
    if uncategorized:
        lines.append("## Uncategorized")
        lines.append("")
        for m in uncategorized:
            lines.append(_format_memory_md(m))
        lines.append("")

    return "\n".join(lines)


def _format_memory_md(m) -> str:
    """Format a single memory as a markdown heading + body."""
    # Truncate content for the heading line, use full content in body
    title = m.content.split("\n")[0][:120]
    conf = round(m.confidence, 2)
    header = (
        f"### [{m.type.value}] {title} "
        f"(confidence: {conf}, accessed: {m.access_count} times)"
    )
    body = m.content
    return f"{header}\n{body}\n"
