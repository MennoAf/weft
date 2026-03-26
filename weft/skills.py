"""Weft skills — higher-level query functions for structured insights."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

import asyncpg

from weft.models import Memory, MemoryRecall
from weft.store import list_memories, search_by_vector


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
) -> dict:
    """Cross-project search combining semantic + filter queries."""
    from weft.models import MemoryType

    if not any([query, topic, memory_type, days]):
        return {"error": "At least one filter required (query, topic, memory_type, or days)"}

    # If we have a text query, use vector search
    if query and embedding_provider:
        embedding = await embedding_provider.embed(query)
        mt = MemoryType(memory_type) if memory_type else None
        results = await search_by_vector(
            pool, embedding, limit=limit, topic=topic,
            memory_type=mt, project_id=None,  # brain-wide
        )

        # Post-filter by days if specified
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            results = [r for r in results if r.memory.created_at >= cutoff]

        return {
            "query": query,
            "filters": {"topic": topic, "type": memory_type, "days": days},
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

    # Filter-only search (no semantic query)
    conditions = ["status = 'active'"]
    params: list = []
    idx = 1

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

    where = " AND ".join(conditions)
    sql = f"""
        SELECT * FROM memories
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await pool.fetch(sql, *params)
    return {
        "query": query,
        "filters": {"topic": topic, "type": memory_type, "days": days},
        "count": len(rows),
        "results": [
            {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"][:300],
                "topic": row["topic"] or [],
                "source": row["source"],
                "confidence": float(row["confidence"]),
                "created_at": row["created_at"].isoformat(),
                "project_id": row["project_id"],
            }
            for row in rows
        ],
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


async def up_next(
    pool: asyncpg.Pool,
    *,
    days: int = 7,
    include_no_date: bool = False,
    limit: int = 50,
) -> dict:
    """Find open tasks due in the next N days."""
    # Fetch all task memories
    rows = await pool.fetch(
        """
        SELECT * FROM memories
        WHERE status = 'active'
          AND 'tasks' = ANY(topic)
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit * 3,  # fetch extra, we'll filter
    )

    now = datetime.now(timezone.utc).date()
    cutoff = now + timedelta(days=days)

    tasks_due: list[dict] = []
    tasks_overdue: list[dict] = []
    tasks_no_date: list[dict] = []

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

        entry = {
            "id": row["id"],
            "content": row["content"],
            "topic": topics,
            "due": due_date,
            "priority": priority,
            "created_at": row["created_at"].isoformat(),
        }

        if due_date:
            try:
                due = datetime.strptime(due_date, "%Y-%m-%d").date()
            except ValueError:
                tasks_no_date.append(entry)
                continue

            if due < now:
                tasks_overdue.append(entry)
            elif due <= cutoff:
                tasks_due.append(entry)
        elif include_no_date:
            tasks_no_date.append(entry)

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
