"""Weft skills — higher-level query functions for structured insights."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

import asyncpg

from weft.store import search_by_vector


async def weekly_recap(
    pool: asyncpg.Pool,
    *,
    days: int = 7,
    project_id: str | None = None,
    limit: int = 100,
) -> dict:
    """Query memories from the last N days, grouped by topic and type."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    conditions = ["status = 'active'", f"created_at >= $1"]
    params: list = [cutoff]
    idx = 2

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = " AND ".join(conditions)
    query = f"""
        SELECT * FROM memories
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(query, *params)

    # Group by type
    by_type: dict[str, list[dict]] = {}
    # Group by source
    by_source: dict[str, int] = {}
    # Collect topics
    all_topics: dict[str, int] = {}

    for row in rows:
        mem_type = row["type"]
        source = row["source"]
        topics = row["topic"] or []

        entry = {
            "id": row["id"],
            "type": mem_type,
            "content": row["content"][:200],
            "topic": topics,
            "source": source,
            "created_at": row["created_at"].isoformat(),
            "confidence": float(row["confidence"]),
        }
        by_type.setdefault(mem_type, []).append(entry)
        by_source[source] = by_source.get(source, 0) + 1
        for t in topics:
            if not t.startswith("file:"):
                all_topics[t] = all_topics.get(t, 0) + 1

    # Sort topics by frequency
    top_topics = sorted(all_topics.items(), key=lambda x: -x[1])[:15]

    # Wellness section from check-in patterns
    wellness: dict | None = None
    try:
        from weft.check_in_patterns import analyze_all
        from weft.check_ins import list_check_ins

        check_ins = await list_check_ins(pool, limit=200)
        if check_ins:
            report = analyze_all(check_ins, trend_days=days, rolling_days=days)
            wellness = {
                "total_check_ins": report["total_check_ins"],
                "trends": report["trends"],
                "streaks": report["streaks"],
                "sleep_energy_correlation": report["sleep_energy_correlation"],
                "sleep_mood_correlation": report["sleep_mood_correlation"],
                "day_of_week": {
                    "best_day": report["day_of_week"]["best_day"],
                    "worst_day": report["day_of_week"]["worst_day"],
                },
            }
    except Exception as e:
        logger.debug("weekly_recap wellness_snapshot failed: %s", e, exc_info=True)

    result = {
        "period_days": days,
        "total_memories": len(rows),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "by_source": by_source,
        "top_topics": top_topics,
        "decisions": by_type.get("decision", []),
        "issues": by_type.get("issue", []),
        "milestones": by_type.get("milestone", []),
        "recent": [
            {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"][:200],
                "topic": row["topic"] or [],
                "source": row["source"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows[:20]
        ],
    }
    if wellness:
        result["wellness"] = wellness
    return result


async def search_all(
    pool: asyncpg.Pool,
    embedding_provider,
    *,
    query: str | None = None,
    topic: str | None = None,
    memory_type: str | None = None,
    days: int | None = None,
    limit: int = 20,
    retrieval_mode: str = "face",
) -> dict:
    """Cross-project search combining semantic + filter queries."""
    from weft.models import MemoryType
    from weft.retrieval_modes import (
        include_agent_provenance,
        sources_for_mode,
        wrap_untrusted_for_face,
    )

    if not any([query, topic, memory_type, days]):
        return {"error": "At least one filter required (query, topic, memory_type, or days)"}

    sources = sources_for_mode(retrieval_mode)
    agent_provenance_ok = include_agent_provenance(retrieval_mode)

    # If we have a text query, use vector search
    if query and embedding_provider:
        embedding = await embedding_provider.embed(query)
        mt = MemoryType(memory_type) if memory_type else None
        results = await search_by_vector(
            pool, embedding, limit=limit, topic=topic,
            memory_type=mt, project_id=None,  # brain-wide
            sources=sources,
            include_agent_provenance=agent_provenance_ok,
        )

        # Post-filter by days if specified
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            results = [r for r in results if r.memory.created_at >= cutoff]

        projected = []
        for r in results:
            d = r.to_dict()
            if retrieval_mode == "face":
                d["content"] = wrap_untrusted_for_face(
                    d["content"], r.memory.write_provenance,
                )
            projected.append(d)

        return {
            "query": query,
            "filters": {"topic": topic, "type": memory_type, "days": days},
            "count": len(results),
            "results": projected,
        }

    # Filter-only search (no semantic query)
    conditions = ["status = 'active'"]
    params: list = []
    idx = 1

    # Phase 2 Layer 2/3 filters mirror the search_by_vector branch above.
    if not agent_provenance_ok:
        conditions.append("write_provenance != 'agent'")
    conditions.append("review_status = 'active'")

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type)
        idx += 1

    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        conditions.append(f"created_at >= ${idx}")
        params.append(cutoff)
        idx += 1

    if sources:
        conditions.append(f"source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    where = " AND ".join(conditions)
    sql = f"""
        SELECT * FROM memories
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(sql, *params)
    items = []
    for row in rows:
        content = row["content"][:300]
        if retrieval_mode == "face":
            wp = row["write_provenance"] if "write_provenance" in row.keys() else "supervisor"
            content = wrap_untrusted_for_face(content, wp)
        items.append({
            "id": row["id"],
            "type": row["type"],
            "content": content,
            "topic": row["topic"] or [],
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "created_at": row["created_at"].isoformat(),
            "project_id": row["project_id"],
        })
    return {
        "query": query,
        "filters": {"topic": topic, "type": memory_type, "days": days},
        "count": len(rows),
        "results": items,
    }


async def project_status(
    pool: asyncpg.Pool,
    *,
    project_id: str,
    days: int = 30,
    limit: int = 50,
) -> dict:
    """Project-scoped memories weighted by importance."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await pool.fetch(
        """
        SELECT * FROM memories
        WHERE status = 'active'
          AND project_id = $1
          AND created_at >= $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        project_id, cutoff, limit * 2,  # fetch extra for weighting
    )

    # Weight by type
    type_weights = {
        "decision": 4, "issue": 4, "milestone": 3,
        "architecture": 3, "solution": 2, "pattern": 2,
        "fact": 1, "preference": 1, "user_model": 1,
        "handoff": 1,
    }

    weighted = sorted(
        rows,
        key=lambda r: (-type_weights.get(r["type"], 0), r["created_at"]),
    )[:limit]

    by_type: dict[str, list[dict]] = {}
    for row in weighted:
        entry = {
            "id": row["id"],
            "content": row["content"][:300],
            "topic": row["topic"] or [],
            "created_at": row["created_at"].isoformat(),
            "confidence": float(row["confidence"]),
        }
        by_type.setdefault(row["type"], []).append(entry)

    # Recent activity (last 10 regardless of type)
    recent = [
        {
            "id": row["id"],
            "type": row["type"],
            "content": row["content"][:200],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows[:10]
    ]

    return {
        "project_id": project_id,
        "period_days": days,
        "total_memories": len(rows),
        "decisions": by_type.get("decision", []),
        "issues": by_type.get("issue", []),
        "milestones": by_type.get("milestone", []),
        "architecture": by_type.get("architecture", []),
        "solutions": by_type.get("solution", []),
        "recent_activity": recent,
    }


_HANDOFF_SUMMARY_RE = re.compile(
    # `[ \t]*` (not `\s*`) so an empty Summary block doesn't slurp newlines
    # and bleed into the next **Section:** below it. `(\S.*?)` requires the
    # capture to start with non-whitespace, so an all-whitespace body fails
    # to match instead of returning the next section's body.
    r"\*\*Summary:\*\*[ \t]*(\S.*?)(?=\n\s*\*\*[A-Z][\w ]*:\*\*|\Z)",
    re.DOTALL,
)


def _extract_handoff_summary(content: str | None, max_chars: int = 200) -> str | None:
    # weft_handoff writes "## Session Handoff\n\n**Summary:** <text>\n\n**Next:** ..."
    # — pull the Summary block, stop at the next bold-section header or EOS.
    if not content:
        return None
    m = _HANDOFF_SUMMARY_RE.search(content)
    if not m:
        return None
    summary = m.group(1).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary or None


async def list_projects_with_handoffs(pool: asyncpg.Pool) -> list[dict]:
    """List known projects with last-handoff metadata for cross-project lookup.

    Returns one entry per project_id with at least one active memory:

        {
          "project_id": str,
          "last_handoff_at": ISO8601 str | None,
          "last_handoff_summary": str | None,   # truncated to 200 chars
          "last_activity_at": ISO8601 str | None,
          "memory_count": int,
        }

    Sorted by last_handoff_at DESC NULLS LAST, then project_id ASC. Designed
    so an agent in one project can discover the canonical project_id strings
    for other projects and form a `weft_prime(project_id=...)` call to read
    that project's most recent handoff.
    """
    rows = await pool.fetch(
        """
        SELECT
            project_id,
            MAX(created_at) FILTER (WHERE type = 'handoff' AND status = 'active')
                AS last_handoff_at,
            MAX(created_at) FILTER (WHERE status = 'active')
                AS last_activity_at,
            COUNT(*) FILTER (WHERE status = 'active') AS memory_count
        FROM memories
        WHERE project_id IS NOT NULL
        GROUP BY project_id
        HAVING COUNT(*) FILTER (WHERE status = 'active') > 0
        ORDER BY last_handoff_at DESC NULLS LAST, project_id ASC
        """,
    )

    summary_rows = await pool.fetch(
        """
        SELECT DISTINCT ON (project_id) project_id, content
        FROM memories
        WHERE type = 'handoff' AND status = 'active' AND project_id IS NOT NULL
        ORDER BY project_id, created_at DESC
        """,
    )
    summary_by_project = {
        r["project_id"]: _extract_handoff_summary(r["content"])
        for r in summary_rows
    }

    return [
        {
            "project_id": r["project_id"],
            "last_handoff_at": (
                r["last_handoff_at"].isoformat() if r["last_handoff_at"] else None
            ),
            "last_handoff_summary": summary_by_project.get(r["project_id"]),
            "last_activity_at": (
                r["last_activity_at"].isoformat() if r["last_activity_at"] else None
            ),
            "memory_count": int(r["memory_count"]),
        }
        for r in rows
    ]


async def meal_plan(
    pool: asyncpg.Pool,
    *,
    lissy_approved: bool | None = None,
    cuisine: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> dict:
    """Query recipe memories with optional filters."""
    conditions = ["status = 'active'", "'recipes' = ANY(topic)"]
    params: list = []
    idx = 1

    if tag:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(tag)
        idx += 1

    where = " AND ".join(conditions)
    sql = f"""
        SELECT * FROM memories
        WHERE {where}
        ORDER BY usefulness_score DESC, created_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(sql, *params)

    recipes = []
    for row in rows:
        content = row["content"]

        # Post-filter by content fields
        if lissy_approved is not None:
            marker = "Lissy approved: yes" if lissy_approved else "Lissy approved: no"
            if marker not in content:
                continue

        if cuisine:
            if f"Cuisine: {cuisine}" not in content.lower().replace(
                f"cuisine: {cuisine.lower()}", f"Cuisine: {cuisine}"
            ):
                # Case-insensitive cuisine check
                if cuisine.lower() not in content.lower():
                    continue

        recipes.append({
            "id": row["id"],
            "content": row["content"],
            "topic": row["topic"] or [],
            "created_at": row["created_at"].isoformat(),
        })

    return {
        "filters": {
            "lissy_approved": lissy_approved,
            "cuisine": cuisine,
            "tag": tag,
        },
        "count": len(recipes),
        "recipes": recipes,
    }


# Date pattern for extracting due dates from task topics
_DUE_DATE_RE = re.compile(r"^due:(\d{4}-\d{2}-\d{2})$")


@dataclass
class TaskEntry:
    """A single task-memory row, selected and parsed.

    Shared core consumed by both ``up_next()`` (below) and
    ``weft.board.task_adapter()`` — the weft_board subsumption contract
    requires task selection/parsing to live in exactly one place so the two
    callers can't drift apart. Do not reimplement the 'tasks' topic query or
    the due-date/priority topic parsing anywhere else; extend this instead.
    """

    id: str
    content: str
    topic: list[str]
    due_date: str | None  # "YYYY-MM-DD" if present in topic tags, else None
    priority: str | None
    created_at: datetime


async def fetch_task_entries(pool: asyncpg.Pool, limit: int) -> list[TaskEntry]:
    """Fetch open task-memories and parse due-date/priority out of topic tags.

    This is the shared up_next core (PRD weft-board §Source-Semantics /
    Critical Implementation Notes): the row selection (``status = 'active'
    AND 'tasks' = ANY(topic)``) and the ``due:YYYY-MM-DD`` / ``priority:``
    topic-tag parsing are the single source of truth for both ``up_next()``
    and ``weft.board.task_adapter()``.
    """
    rows = await pool.fetch(
        """
        SELECT * FROM memories
        WHERE status = 'active'
          AND 'tasks' = ANY(topic)
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )

    entries: list[TaskEntry] = []
    for row in rows:
        topics = row["topic"] or []
        due_date = None
        priority = None

        for t in topics:
            m = _DUE_DATE_RE.match(t)
            if m:
                due_date = m.group(1)
            if t.startswith("priority:"):
                priority = t.split(":", 1)[1]

        entries.append(
            TaskEntry(
                id=row["id"],
                content=row["content"],
                topic=topics,
                due_date=due_date,
                priority=priority,
                created_at=row["created_at"],
            )
        )

    return entries


async def up_next(
    pool: asyncpg.Pool,
    *,
    days: int = 7,
    include_no_date: bool = False,
    limit: int = 50,
) -> dict:
    """Find open tasks due in the next N days."""
    # Fetch + parse via the shared core (also consumed by weft.board.task_adapter)
    entries = await fetch_task_entries(pool, limit * 3)  # fetch extra, we'll filter

    now = datetime.now(timezone.utc).date()
    cutoff = now + timedelta(days=days)

    tasks_due: list[dict] = []
    tasks_overdue: list[dict] = []
    tasks_no_date: list[dict] = []

    for entry in entries:
        due_date = entry.due_date

        result_entry = {
            "id": entry.id,
            "content": entry.content,
            "topic": entry.topic,
            "due": due_date,
            "priority": entry.priority,
            "created_at": entry.created_at.isoformat(),
        }

        if due_date:
            try:
                due = datetime.strptime(due_date, "%Y-%m-%d").date()
            except ValueError:
                tasks_no_date.append(result_entry)
                continue

            if due < now:
                tasks_overdue.append(result_entry)
            elif due <= cutoff:
                tasks_due.append(result_entry)
        elif include_no_date:
            tasks_no_date.append(result_entry)

    # Sort by due date, then priority
    priority_order = {"highest": 0, "high": 1, "medium": 2, "low": 3, "lowest": 4}

    def sort_key(t):
        d = t.get("due") or "9999-12-31"
        p = priority_order.get(t.get("priority") or "", 5)
        return (d, p)

    tasks_overdue.sort(key=sort_key)
    tasks_due.sort(key=sort_key)

    result = {
        "period_days": days,
        "overdue": tasks_overdue,
        "due_soon": tasks_due,
        "overdue_count": len(tasks_overdue),
        "due_soon_count": len(tasks_due),
    }
    if include_no_date:
        result["no_date"] = tasks_no_date[:limit]
        result["no_date_count"] = len(tasks_no_date)

    return result
