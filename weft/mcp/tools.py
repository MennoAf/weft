"""Weft MCP tools — thin coordinators, ≤15 lines each."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import asyncpg
from fastmcp import Context

from weft.correlation import set_correlation_id
from weft.mcp.server import AppContext, mcp
from weft.behaviors import (
    delete_behavior,
    list_behaviors as list_behaviors_store,
    match_behaviors,
    store_behavior,
    touch_behavior,
)
from weft.entities import (
    get_entity,
    get_entity_memories,
    link_mention,
    search_entities,
    store_entity,
)
from weft.episodes import (
    add_memory_to_episode,
    close_episode,
    create_episode,
    get_episode,
    get_episode_memories,
    list_episodes,
    timeline_query,
)
from weft.models import (
    BehaviorCreate,
    BehaviorScope,
    EntityCreate,
    EntityType,
    EpisodeCreate,
    EpisodeStatus,
    EpisodeWithMemories,
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    MemoryTypeLiteral,
    RelationType,
)
from weft.tokens import estimate_tokens
from weft.session_tracking import boost_session_memories, log_memory_access
from weft.store import (
    add_relationship,
    count_by_vector,
    delete_memory,
    get_recent_writes,
    get_relationships,
    get_stats,
    list_memories,
    record_feedback,
    remove_relationship,
    search_by_vector,
    search_cross_project,
    store_memory,
    touch_memory,
    update_memory,
)

logger = logging.getLogger(__name__)

# Exceptions that indicate the database is unreachable (not application logic errors).
_DB_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionRefusedError)

# Exceptions from invalid input (bad enum values, wrong types, etc.)
_INPUT_ERRORS = (ValueError, TypeError)


def _coerce_list(value: str | list | None) -> list | None:
    """Coerce a JSON-string list to a native list.

    Some MCP clients serialize list params as JSON strings instead of arrays.
    E.g. topic arrives as '["a","b"]' instead of ["a","b"]. This helper
    transparently handles that so Pydantic validation succeeds.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return value  # let Pydantic raise if still wrong


_RELATIVE_DATE_RE = re.compile(r"^(\d+)\s*(d|days?|w|weeks?|m|months?)$", re.IGNORECASE)

_UNIT_DAYS = {"d": 1, "day": 1, "days": 1, "w": 7, "week": 7, "weeks": 7, "m": 30, "month": 30, "months": 30}


def _parse_review_after(value: str | None) -> datetime | None:
    """Parse a review_after value: ISO timestamp or relative like '30d', '2w', '3m'."""
    if value is None:
        return None
    match = _RELATIVE_DATE_RE.match(value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        days = amount * _UNIT_DAYS[unit]
        return datetime.now(timezone.utc) + timedelta(days=days)
    # Try ISO format
    return datetime.fromisoformat(value)


async def _detect_project_id(ctx: Context) -> str | None:
    """Auto-detect project_id from MCP client roots.

    Uses the directory name of the first root URI as the project identifier.
    E.g. file:///Users/jason/Projects/Weft → "weft"
    Returns None if roots are unavailable or empty.
    """
    try:
        roots = await ctx.list_roots()
        if roots:
            uri = str(roots[0].uri)
            # file:///path/to/ProjectName → "projectname"
            path = uri.replace("file://", "").rstrip("/")
            name = path.rsplit("/", 1)[-1] if "/" in path else path
            return name.lower() or None
    except Exception:
        pass
    return None


async def _resolve_project_id(ctx: Context, explicit: str | None) -> str | None:
    """Return the explicit project_id if provided, otherwise auto-detect."""
    if explicit is not None:
        return explicit
    return await _detect_project_id(ctx)


def _input_error_response(tool_name: str, error: Exception) -> dict:
    """Standard error response for invalid input parameters."""
    logger.info("Invalid input in %s: %s", tool_name, error)
    return {"error": "Invalid input", "detail": str(error), "tool": tool_name}


def _db_error_response(tool_name: str, error: Exception) -> dict:
    """Standard error response when database is unavailable."""
    detail = type(error).__name__
    if isinstance(error, asyncpg.InterfaceError):
        detail = "stale connection pool (will auto-recover)"
    elif isinstance(error, ConnectionRefusedError):
        detail = "database unreachable"
    logger.warning("Database unavailable in %s: %s — %s", tool_name, detail, error)
    return {"error": "Database unavailable", "detail": detail, "degraded": True, "tool": tool_name}


def _extract_primer_memory_ids(result: dict) -> list[str]:
    """Extract memory IDs from all sections of a primer result."""
    ids: list[str] = []
    # Sections that contain memory dicts with "id" keys
    for section in ("rules", "handoff", "recent_work", "decisions"):
        for item in result.get(section, []):
            if isinstance(item, dict) and "id" in item:
                ids.append(item["id"])
    # Issues have a nested structure
    issues = result.get("issues", {})
    for item in issues.get("items", []):
        if isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
    return ids


@mcp.tool()
async def weft_remember(
    ctx: Context,
    content: str,
    type: MemoryTypeLiteral = "fact",
    topic: list[str] | None = None,
    source: str = "conversation",
    confidence: float = 0.7,
    project_id: str | None = None,
    agent_id: str | None = None,
    check_contradictions: bool = True,
    pinned: bool = False,
    review_after: str | None = None,
) -> dict:
    """Store a new memory with type, topics, content, confidence, and source.
    If project_id is omitted, auto-detects from the client's working directory.

    review_after: optional lifecycle date. Accepts ISO timestamp or relative
    durations like '30d', '2w', '3m'. Memories past their review_after date
    are flagged in the primer so the agent can confirm, revise, or archive them."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_remember start [%s]", cid)
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        create = MemoryCreate(
            type=MemoryType(type),
            content=content,
            topic=_coerce_list(topic) or [],
            source=MemorySource(source),
            confidence=confidence,
            project_id=resolved_project,
            agent_id=agent_id,
            pinned=pinned,
            review_after=_parse_review_after(review_after),
        )
        embedding = await app.embedding.embed(content)
        memory = await store_memory(app.pool, create, embedding=embedding)
        await app.cache.set_memory(memory)
        await app.cache.invalidate_stats()
        result = memory.to_dict()
        if check_contradictions and embedding:
            from weft.consolidation import check_contradictions_on_store
            warnings = await check_contradictions_on_store(
                app.pool, memory.id, embedding,
                memory_type=create.type,
                project_id=resolved_project,
            )
            if warnings:
                result["contradiction_warnings"] = warnings
                result["contradiction_warnings_text"] = [
                    f'Warning: This may contradict memory {w["memory_id"]}: "{w["content_preview"]}"'
                    for w in warnings
                ]
        return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_remember", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_remember", e)


@mcp.tool()
async def weft_recall(
    ctx: Context,
    query: str,
    topic: str | None = None,
    type: MemoryTypeLiteral | None = None,
    status: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> dict:
    """Retrieve memories by semantic query, topic filter, type filter, status filter, or combination."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_recall start [%s] query=%r", cid, query[:50])
        app: AppContext = ctx.request_context.lifespan_context
        memory_type = MemoryType(type) if type else None
        memory_status = MemoryStatus(status) if status else MemoryStatus.active
        embedding = await app.embedding.embed(query)
        results = await search_by_vector(
            app.pool,
            embedding,
            limit=limit,
            threshold=threshold,
            status=memory_status,
            memory_type=memory_type,
            topic=topic,
            project_id=project_id,
            agent_id=agent_id,
        )
        # Touch accessed memories and enrich with entities
        enriched = []
        for r in results:
            await touch_memory(app.pool, r.memory.id)
            d = r.to_dict()
            try:
                from weft.entities import get_memory_entities
                ents = await get_memory_entities(app.pool, r.memory.id)
                if ents:
                    d["entities"] = [{"name": e.name, "type": e.entity_type.value} for e in ents]
            except Exception:
                pass
            enriched.append(d)
        # Fire-and-forget: log session access for implicit usefulness signals
        import asyncio
        if results:
            asyncio.create_task(
                log_memory_access(
                    app.pool,
                    [r.memory.id for r in results],
                    "recall",
                ),
                name="weft-session-log-recall",
            )

        # Count total matches above threshold (without LIMIT) so the agent
        # knows whether there are more results to explore.
        total_matches = await count_by_vector(
            app.pool,
            embedding,
            threshold=threshold,
            status=memory_status,
            memory_type=memory_type,
            topic=topic,
            project_id=project_id,
            agent_id=agent_id,
        )
        response: dict = {"query": query, "count": len(results), "results": enriched}
        if total_matches > len(results):
            response["total_matches"] = total_matches
            response["showing"] = f"Showing {len(results)} of {total_matches} matches"

        # Cross-project search: surface relevant memories from other projects
        resolved_project = await _resolve_project_id(ctx, project_id)
        if resolved_project is not None:
            try:
                from weft.config import load_config
                cfg = load_config()
                if cfg.retrieval.cross_project_search:
                    main_ids = {r.memory.id for r in results}
                    cross_results = await search_cross_project(
                        app.pool, embedding,
                        exclude_project_id=resolved_project,
                        limit=cfg.retrieval.cross_project_limit,
                        threshold=threshold,
                        status=memory_status,
                        memory_type=memory_type,
                        exclude_ids=list(main_ids),
                    )
                    if cross_results:
                        response["cross_project"] = [
                            {
                                **r.to_dict(),
                                "source_project": r.memory.project_id,
                            }
                            for r in cross_results
                        ]
            except Exception as exc:
                logger.warning("Cross-project search failed: %s", exc)

        return response
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_recall", e)
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_recall: %s", e)
        from weft.fallback import search_fallback
        results = search_fallback(query, limit=limit)
        return {"query": query, "count": len(results), "results": results, "degraded": True}


@mcp.tool()
async def weft_forget(
    ctx: Context,
    memory_id: str,
    hard: bool = False,
) -> dict:
    """Archive a memory (soft-delete) or hard-delete it."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        deleted = await delete_memory(app.pool, memory_id, hard=hard)
        await app.cache.invalidate_memory(memory_id)
        await app.cache.invalidate_stats()
        return {"memory_id": memory_id, "deleted": deleted, "hard": hard}
    except _DB_ERRORS as e:
        return _db_error_response("weft_forget", e)


@mcp.tool()
async def weft_context(
    ctx: Context,
    query: str,
    budget_tokens: int = 4000,
    topic: str | None = None,
    type: MemoryTypeLiteral | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    max_per_topic: int = 3,
) -> dict:
    """Budget-aware context loading: best memories for a situation within N tokens."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_context start [%s] budget=%d", cid, budget_tokens)
        from weft.context import build_context

        app: AppContext = ctx.request_context.lifespan_context
        memory_type = MemoryType(type) if type else None
        embedding = await app.embedding.embed(query)
        result = await build_context(
            app.pool, embedding,
            budget_tokens=budget_tokens, max_per_topic=max_per_topic,
            memory_type=memory_type, topic=topic, project_id=project_id,
            agent_id=agent_id,
        )
        # Touch the memories that made it into context
        for mem_dict in result["memories"]:
            await touch_memory(app.pool, mem_dict["id"])

        # Fire-and-forget: log session access for implicit usefulness signals
        import asyncio
        mem_ids = [m["id"] for m in result["memories"]]
        if mem_ids:
            asyncio.create_task(
                log_memory_access(app.pool, mem_ids, "context"),
                name="weft-session-log-context",
            )

        return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_context", e)
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_context: %s", e)
        from weft.fallback import search_fallback
        results = search_fallback(query)
        return {
            "memories": results,
            "count": len(results),
            "tokens_used": 0,
            "budget_tokens": budget_tokens,
            "degraded": True,
        }


@mcp.tool()
async def weft_revise(
    ctx: Context,
    memory_id: str,
    new_content: str,
    new_confidence: float | None = None,
    new_topic: list[str] | None = None,
    new_type: MemoryTypeLiteral | None = None,
    review_after: str | None = None,
) -> dict:
    """Update a memory's content, creating a new version that supersedes the old one.

    review_after: optional lifecycle date for the new version. Accepts ISO
    timestamp or relative durations like '30d', '2w', '3m'."""
    try:
        from weft.models import MemoryType as _MT
        from weft.revise import revise_memory

        resolved_type = _MT(new_type) if new_type else None
        app: AppContext = ctx.request_context.lifespan_context
        embedding = await app.embedding.embed(new_content)
        new, old = await revise_memory(
            app.pool, memory_id, new_content,
            embedding=embedding, new_confidence=new_confidence,
            new_topic=_coerce_list(new_topic), new_type=resolved_type,
            review_after=_parse_review_after(review_after),
        )
        await app.cache.set_memory(new)
        await app.cache.invalidate_memory(old.id)
        await app.cache.invalidate_stats()
        return {"new": new.to_dict(), "superseded": old.to_dict()}
    except _DB_ERRORS as e:
        return _db_error_response("weft_revise", e)


@mcp.tool()
async def weft_relate(
    ctx: Context,
    action: str,
    memory_id: str,
    target_id: str | None = None,
    relation: str | None = None,
) -> dict:
    """Manage relationships between memories: add, get, or remove."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        if action == "add":
            rel = await add_relationship(app.pool, source_id=memory_id, target_id=target_id, relation=RelationType(relation))
            return {"source_id": rel.source_id, "target_id": rel.target_id, "relation": rel.relation.value, "created_at": rel.created_at.isoformat()}
        elif action == "get":
            rels = await get_relationships(app.pool, memory_id, relation=RelationType(relation) if relation else None)
            return {"memory_id": memory_id, "count": len(rels), "relationships": [
                {"source_id": r.source_id, "target_id": r.target_id, "relation": r.relation.value, "created_at": r.created_at.isoformat()} for r in rels
            ]}
        elif action == "remove":
            removed = await remove_relationship(app.pool, source_id=memory_id, target_id=target_id, relation=RelationType(relation))
            return {"memory_id": memory_id, "target_id": target_id, "relation": relation, "removed": removed}
        else:
            return {"error": f"Unknown action: {action}. Use 'add', 'get', or 'remove'."}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_relate", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_relate", e)


@mcp.tool()
async def weft_consolidate(ctx: Context, dry_run: bool = False) -> dict:
    """Run consolidation: decay stale memories, merge duplicates, flag contradictions."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_consolidate start [%s] dry_run=%s", cid, dry_run)
        from weft.consolidation import consolidate
        app: AppContext = ctx.request_context.lifespan_context
        report = await consolidate(app.pool, dry_run=dry_run)
        await app.cache.invalidate_stats()
        return report.to_dict()
    except _DB_ERRORS as e:
        return _db_error_response("weft_consolidate", e)


@mcp.tool()
async def weft_feedback(
    ctx: Context,
    memory_id: str,
    helpful: bool,
) -> dict:
    """Record whether a memory was helpful. Adjusts usefulness score for future ranking."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        result = await record_feedback(app.pool, memory_id, helpful)
        await app.cache.invalidate_memory(memory_id)
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_feedback", e)


@mcp.tool()
async def weft_pin(
    ctx: Context,
    memory_id: str,
    pinned: bool = True,
) -> dict:
    """Pin or unpin a memory. Pinned memories are always included in prime and context calls."""
    try:

        app: AppContext = ctx.request_context.lifespan_context
        updated = await update_memory(app.pool, memory_id, pinned=pinned)
        if not updated:
            return {"error": f"Memory {memory_id} not found"}
        await app.cache.invalidate_memory(memory_id)
        return {"memory_id": memory_id, "pinned": updated.pinned}
    except _DB_ERRORS as e:
        return _db_error_response("weft_pin", e)


@mcp.tool()
async def weft_prime(
    ctx: Context,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 2400,
    query: str | None = None,
) -> dict:
    """Session primer: assemble structured context for session startup.
    If project_id is omitted, auto-detects from the client's working directory.

    query: optional intent/topic string to bias which behaviors, decisions,
    issues, and recent work are surfaced. When provided, those sections use
    semantic similarity to rank more relevant items higher."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_prime start [%s] project=%s", cid, project_id)
        from weft.primer import build_primer

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        # Compute query embedding if provided (best-effort).
        query_vec: list[float] | None = None
        if query and query.strip():
            try:
                query_vec = await app.embedding.embed(query.strip())
            except Exception as exc:
                logger.warning("Failed to embed primer query (non-fatal): %s", exc)

        result = await build_primer(
            app.pool,
            project_id=resolved_project,
            agent_id=agent_id,
            budget_tokens=budget_tokens,
            query_vec=query_vec,
        )

        # Fire-and-forget: trigger auto-consolidation if due (>24h since last run).
        # Never blocks the primer response; all exceptions are caught internally.
        try:
            import asyncio
            from weft.consolidation import consolidate_if_due
            asyncio.create_task(
                consolidate_if_due(app.pool),
                name="weft-auto-consolidation",
            )
        except Exception as exc:
            logger.debug("Auto-consolidation scheduling skipped: %s", exc)

        # Fire-and-forget: log session access for all memories in primer
        try:
            primer_mem_ids = _extract_primer_memory_ids(result)
            if primer_mem_ids:
                asyncio.create_task(
                    log_memory_access(app.pool, primer_mem_ids, "prime"),
                    name="weft-session-log-prime",
                )
        except Exception as exc:
            logger.debug("Prime session logging skipped: %s", exc)

        return result
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_prime: %s", e)
        from weft.fallback import read_fallback
        content = read_fallback()
        from weft.primer import _ONBOARDING_TEXT, _SECTION_HINTS
        from weft.tokens import estimate_tokens, truncate_to_token_budget

        # Truncate fallback content to budget — raw exports can be huge.
        handoff_section: list[dict] = []
        if content:
            cost = estimate_tokens(content)
            if cost > budget_tokens:
                content, cost = truncate_to_token_budget(content, budget_tokens)
            handoff_section = [{"content": content, "type": "fallback"}]

        return {
            "grounding": None,
            "rules": [],
            "behaviors": [],
            "handoff": handoff_section,
            "recent_work": [],
            "issues": {"count": 0, "items": []},
            "decisions": [],
            "entities": [],
            "total_tokens": 0,
            "budget_tokens": budget_tokens,
            "budget_remaining": budget_tokens,
            "excluded": 0,
            "degraded": True,
            "section_tokens": {},
            "hints": dict(_SECTION_HINTS),
            "onboarding": _ONBOARDING_TEXT,
        }


@mcp.tool()
async def weft_focus(
    ctx: Context,
    intent: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 1200,
) -> dict:
    """Post-intent re-prime: surface memories the generic primer missed.

    Call after weft_prime when you know what you're working on. Focus builds
    a compound query from your intent + last session context, excludes memories
    already surfaced by prime, and returns a tight-budget supplement.

    Unlike prime (broad session startup), focus is narrow and intent-driven.
    Can be called multiple times as intent shifts mid-session.

    intent: what you're focusing on (e.g., "implement session tracking")
    budget_tokens: max tokens in result (default 1200, supplemental to prime)"""
    try:
        cid = set_correlation_id()
        logger.debug("weft_focus start [%s] intent=%r", cid, intent[:50])
        from weft.focus import build_focus

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        result = await build_focus(
            app.pool,
            intent=intent,
            embedding_fn=app.embedding.embed,
            project_id=resolved_project,
            agent_id=agent_id,
            budget_tokens=budget_tokens,
        )

        # Fire-and-forget: log focused memories for session tracking
        import asyncio
        focused_ids = [m["id"] for m in result.focused_memories]
        if focused_ids:
            asyncio.create_task(
                log_memory_access(app.pool, focused_ids, "focus"),
                name="weft-session-log-focus",
            )

        return result.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_focus", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_focus", e)


@mcp.tool()
async def weft_status(ctx: Context) -> dict:
    """Memory statistics: total, by topic, by type, by confidence, recently accessed."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        cached = await app.cache.get_stats()
        if cached:
            return cached
        stats = await get_stats(app.pool)
        # Add recent writes with provenance (not cached — always fresh)
        stats["recent_writes"] = await get_recent_writes(app.pool, limit=10)
        await app.cache.set_stats(stats)
        return stats
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_status: %s", e)
        return {"degraded": True, "error": "Database unavailable", "total": 0}


@mcp.tool()
async def weft_extract(
    ctx: Context,
    text: str,
    min_confidence: float = 0.5,
) -> dict:
    """Extract memory candidates from text. Returns proposals for review — does NOT auto-store."""
    from weft.extract import extract_candidates
    candidates = extract_candidates(text, min_confidence=min_confidence)
    return {"count": len(candidates), "candidates": candidates}


@mcp.tool()
async def weft_learn(
    ctx: Context,
    content: str,
    task_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    min_confidence: float = 0.7,
) -> dict:
    """Store lessons learned from completed work. Extracts memories from free-text
    notes about what was learned (gotchas, patterns, fixes) and auto-stores
    candidates above the confidence threshold. Designed for post-task capture —
    call after loom_done with what the agent (or you) learned during the task.
    If project_id is omitted, auto-detects from the client's working directory."""
    try:
        from weft.extract import extract_candidates

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        candidates = extract_candidates(content, min_confidence=min_confidence)

        # Also store the raw content as a solution memory if no patterns matched
        # but the content is substantial enough to be useful
        if not candidates and len(content.split()) >= 10:
            candidates = [{
                "content": content.strip(),
                "type": "solution",
                "confidence": min_confidence,
                "topic": [],
            }]

        stored: list[dict] = []
        for c in candidates:
            create = MemoryCreate(
                type=MemoryType(c["type"]),
                content=c["content"],
                topic=c.get("topic", []) + ([f"task:{task_id}"] if task_id else []),
                source=MemorySource.conversation,
                confidence=c["confidence"],
                project_id=resolved_project,
                agent_id=agent_id,
            )
            embedding = await app.embedding.embed(c["content"])
            memory = await store_memory(app.pool, create, embedding=embedding)
            await app.cache.set_memory(memory)
            stored.append(memory.to_dict())

        # Auto-create a milestone when task_id is provided so primer's
        # recent_work section shows completed task breadcrumbs.
        milestone_dict: dict | None = None
        if task_id:
            # Build a concise milestone from the first ~100 words of content
            words = content.split()
            summary = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
            milestone_create = MemoryCreate(
                type=MemoryType.milestone,
                content=summary,
                topic=[f"task:{task_id}"],
                source=MemorySource.conversation,
                confidence=0.9,
                project_id=resolved_project,
                agent_id=agent_id,
            )
            ms_embedding = await app.embedding.embed(summary)
            milestone = await store_memory(
                app.pool, milestone_create, embedding=ms_embedding,
            )
            await app.cache.set_memory(milestone)
            milestone_dict = milestone.to_dict()

        await app.cache.invalidate_stats()

        # Boost usefulness for memories accessed this session
        session_boost: dict = {}
        try:
            session_boost = await boost_session_memories(app.pool)
        except Exception as exc:
            logger.warning("Session boost failed during learn: %s", exc)

        return {
            "candidates_found": len(candidates),
            "stored": len(stored),
            "memories": stored,
            "task_id": task_id,
            "milestone": milestone_dict,
            "session_boost": session_boost,
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_learn", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_learn", e)


@mcp.tool()
async def weft_feedback_general(
    ctx: Context,
    feedback: str,
    category: str = "suggestion",
    agent_id: str | None = None,
) -> dict:
    """Submit general feedback about Weft itself — friction points, feature requests,
    or praise. Unlike weft_feedback (per-memory ratings), this captures product-level
    observations from agents using Weft in the field."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        valid_categories = ("suggestion", "friction", "praise", "bug")
        if category not in valid_categories:
            return _input_error_response(
                "weft_feedback_general",
                ValueError(f"category must be one of {valid_categories}, got '{category}'"),
            )
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"[{category.upper()}] {feedback}",
            topic=["weft-feedback", category],
            source=MemorySource.conversation,
            confidence=0.8,
            project_id="weft",
            agent_id=agent_id,
        )
        embedding = await app.embedding.embed(feedback)
        memory = await store_memory(app.pool, create, embedding=embedding)
        await app.cache.set_memory(memory)
        await app.cache.invalidate_stats()
        return {"id": memory.id, "category": category, "stored": True}
    except _DB_ERRORS as e:
        return _db_error_response("weft_feedback_general", e)


@mcp.tool()
async def weft_handoff(
    ctx: Context,
    summary: str,
    in_progress: str | None = None,
    next_steps: str | None = None,
    open_questions: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Session handoff: capture context for the next session before clearing.

    Call this before ending a session to preserve continuity. The next session's
    weft_prime will surface the most recent handoff prominently so the incoming
    agent can pick up where you left off.

    Structure your handoff like a shift note:
    - summary: what was accomplished this session
    - in_progress: what's partially done or needs follow-up
    - next_steps: what you'd recommend doing next and why
    - open_questions: unresolved decisions or things to investigate"""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        # Build structured content
        parts = [f"## Session Handoff\n\n**Summary:** {summary}"]
        if in_progress:
            parts.append(f"\n**In Progress:** {in_progress}")
        if next_steps:
            parts.append(f"\n**Next Steps:** {next_steps}")
        if open_questions:
            parts.append(f"\n**Open Questions:** {open_questions}")
        content = "\n".join(parts)

        create = MemoryCreate(
            type=MemoryType.handoff,
            content=content,
            topic=["session-handoff"],
            source=MemorySource.conversation,
            confidence=1.0,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        embedding = await app.embedding.embed(summary)
        memory = await store_memory(app.pool, create, embedding=embedding)
        await app.cache.set_memory(memory)
        await app.cache.invalidate_stats()

        # Auto-prune: archive previous handoffs for this project so they
        # don't accumulate.  Only the most recent handoff matters.
        pruned_count = 0
        try:
            prev = await list_memories(
                app.pool,
                memory_type=MemoryType.handoff,
                status=MemoryStatus.active,
                project_id=resolved_project,
                limit=100,
            )
            for old in prev:
                if old.id != memory.id:
                    await update_memory(
                        app.pool, old.id, status=MemoryStatus.archived,
                    )
                    pruned_count += 1
        except Exception as exc:
            logger.warning("Handoff auto-prune failed: %s", exc)

        # Auto-episode: close open episodes and create a new one
        closed_ids = []
        new_episode_id = None
        try:
            open_eps = await list_episodes(
                app.pool, project_id=resolved_project,
                status=EpisodeStatus.open, limit=10,
            )
            for ep in open_eps:
                closed = await close_episode(app.pool, ep.id, summary=summary)
                if closed:
                    await add_memory_to_episode(app.pool, ep.id, memory.id)
                    closed_ids.append(ep.id)

            new_ep = await create_episode(app.pool, EpisodeCreate(
                title=f"Session after: {summary[:80]}",
                project_id=resolved_project,
                agent_id=agent_id,
            ))
            await add_memory_to_episode(app.pool, new_ep.id, memory.id)
            new_episode_id = new_ep.id
        except Exception as exc:
            logger.warning("Auto-episode failed during handoff: %s", exc)

        # Boost usefulness for memories accessed this session
        session_boost: dict = {}
        try:
            session_boost = await boost_session_memories(app.pool)
        except Exception as exc:
            logger.warning("Session boost failed during handoff: %s", exc)

        return {
            "id": memory.id,
            "project_id": resolved_project,
            "stored": True,
            "previous_handoffs_archived": pruned_count,
            "episodes_closed": closed_ids,
            "episode_opened": new_episode_id,
            "session_boost": session_boost,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_handoff", e)


# ── Skills ──────────────────────────────────────────────────────────


@mcp.tool()
async def weft_weekly_recap(
    ctx: Context,
    days: int = 7,
    project_id: str | None = None,
) -> dict:
    """Weekly recap: memories from the last N days grouped by type and topic.

    Returns decisions, issues, milestones, top topics, and recent activity.
    Use this to generate status updates, standup notes, or session summaries."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        from weft.skills import weekly_recap

        return await weekly_recap(app.pool, days=days, project_id=resolved_project)
    except _DB_ERRORS as e:
        return _db_error_response("weft_weekly_recap", e)


@mcp.tool()
async def weft_search_all(
    ctx: Context,
    query: str | None = None,
    topic: str | None = None,
    memory_type: MemoryTypeLiteral | None = None,
    days: int | None = None,
    limit: int = 20,
) -> dict:
    """Cross-project brain-wide search combining semantic and filter queries.

    At least one filter is required. Searches across ALL projects (not scoped).
    Use for finding information that spans projects or when you don't know
    which project something belongs to."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import search_all

        return await search_all(
            app.pool, app.embedding,
            query=query, topic=topic, memory_type=memory_type,
            days=days, limit=limit,
        )
    except _DB_ERRORS as e:
        return _db_error_response("weft_search_all", e)


@mcp.tool()
async def weft_project_status(
    ctx: Context,
    project_id: str | None = None,
    days: int = 30,
) -> dict:
    """Project status: memories for a project weighted by importance.

    Prioritizes decisions, issues, and milestones. Includes recent activity.
    Use to get a quick overview of where a project stands."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        if not resolved_project:
            return {"error": "project_id required — pass explicitly or run from a project directory"}
        from weft.skills import project_status

        return await project_status(app.pool, project_id=resolved_project, days=days)
    except _DB_ERRORS as e:
        return _db_error_response("weft_project_status", e)


@mcp.tool()
async def weft_meal_plan(
    ctx: Context,
    lissy_approved: bool | None = None,
    cuisine: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> dict:
    """Query recipe memories for meal planning.

    Filter by Lissy-approved, cuisine type, or tags. Returns full recipe
    content for planning meals."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import meal_plan

        return await meal_plan(
            app.pool, lissy_approved=lissy_approved,
            cuisine=cuisine, tag=tag, limit=limit,
        )
    except _DB_ERRORS as e:
        return _db_error_response("weft_meal_plan", e)


@mcp.tool()
async def weft_up_next(
    ctx: Context,
    days: int = 7,
    include_no_date: bool = False,
) -> dict:
    """Upcoming tasks: open tasks due in the next N days from Obsidian notes.

    Returns overdue tasks and tasks due soon, sorted by date and priority.
    Tasks come from Obsidian checkbox items with Tasks plugin emoji dates."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import up_next

        return await up_next(app.pool, days=days, include_no_date=include_no_date)
    except _DB_ERRORS as e:
        return _db_error_response("weft_up_next", e)


@mcp.tool()
async def weft_slack_sync(
    ctx: Context,
    channel_ids: list[str] | None = None,
    channel_names: list[str] | None = None,
    limit_per_channel: int = 50,
    bot_token: str | None = None,
) -> dict:
    """Sync Slack channel history into Weft memories.

    Two modes:
    1. With bot_token: uses slack_sdk directly (for CLI/cron).
    2. Without bot_token: caller must pre-fetch messages using Slack MCP
       tools and pass via channel_ids. This tool fetches what it can
       and stores the results.

    If bot_token is provided, syncs all non-excluded channels automatically.
    Otherwise, provide channel_ids or channel_names to target specific channels.
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context

        if bot_token:
            # SDK mode — fully automated
            from weft.slack.sync import sync_slack_sdk

            result = await sync_slack_sdk(
                app.pool,
                bot_token,
                app.embedding,
                limit_per_channel=limit_per_channel,
            )
            return {
                "mode": "sdk",
                "channels_synced": result.channels_synced,
                "messages_found": result.messages_found,
                "messages_synced": result.messages_synced,
                "messages_skipped": result.messages_skipped,
                "messages_updated": result.messages_updated,
                "memories_created": result.memories_created,
                "memories_archived": result.memories_archived,
            }
        else:
            # Return instructions for MCP-based sync workflow
            channel_ids = _coerce_list(channel_ids)
            channel_names = _coerce_list(channel_names)
            return {
                "mode": "mcp_assisted",
                "instructions": (
                    "To sync Slack channels without a bot token, use the Slack MCP "
                    "tools to read channel history, then call weft_slack_ingest with "
                    "the fetched messages. Steps:\n"
                    "1. slack_search_channels to find channel IDs\n"
                    "2. slack_read_channel for each channel\n"
                    "3. slack_read_thread for threaded messages\n"
                    "4. weft_slack_ingest with the collected data"
                ),
                "channel_ids": channel_ids,
                "channel_names": channel_names,
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_slack_sync", e)


@mcp.tool()
async def weft_slack_ingest(
    ctx: Context,
    channel_id: str,
    channel_name: str,
    messages: list[dict],
    threads: dict | None = None,
    user_names: dict | None = None,
) -> dict:
    """Ingest pre-fetched Slack messages into Weft memories.

    Use this after reading channel history via Slack MCP tools.
    Pass raw message objects from slack_read_channel and thread
    replies from slack_read_thread.

    Args:
        channel_id: The Slack channel ID (e.g. C0A8VD15M5X)
        channel_name: The channel name (e.g. 'general')
        messages: List of raw Slack message dicts from channel history
        threads: Optional {thread_ts: [reply_dicts]} for threaded messages
        user_names: Optional {user_id: display_name} mapping
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.slack.sync import ChannelInfo, sync_slack_messages

        channel = ChannelInfo(id=channel_id, name=channel_name)
        messages = _coerce_list(messages) or []
        if isinstance(threads, str):
            threads = json.loads(threads)

        result = await sync_slack_messages(
            app.pool,
            channels=[channel],
            messages_by_channel={channel_id: messages},
            embedding_provider=app.embedding,
            threads_by_channel={channel_id: threads} if threads else None,
            user_names=user_names,
        )
        return {
            "channel": channel_name,
            "messages_found": result.messages_found,
            "messages_synced": result.messages_synced,
            "messages_skipped": result.messages_skipped,
            "memories_created": result.memories_created,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_slack_ingest", e)
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_slack_ingest", e)


# --- Behavior tools ---


@mcp.tool()
async def weft_behavior_add(
    ctx: Context,
    trigger_pattern: str,
    action: str,
    confidence: float = 0.7,
    scope: str = "global",
    project_id: str | None = None,
    agent_id: str | None = None,
    priority: int = 0,
) -> dict:
    """Store a persistent behavioral rule for agents.
    If project_id is omitted, auto-detects from the client's working directory.

    trigger_pattern: describes WHEN this behavior should activate (embedded for semantic matching).
    action: describes WHAT the agent should do when the trigger matches.
    scope: 'global' (all projects), 'project' (specific project), or 'agent' (specific agent).
    priority: higher values override lower-priority behaviors (default 0)."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        create = BehaviorCreate(
            trigger_pattern=trigger_pattern,
            action=action,
            confidence=confidence,
            scope=BehaviorScope(scope),
            project_id=resolved_project,
            agent_id=agent_id,
            priority=priority,
        )
        embedding = await app.embedding.embed(trigger_pattern)
        behavior = await store_behavior(app.pool, create, embedding=embedding)
        return behavior.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_add", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_add", e)


@mcp.tool()
async def weft_behavior_match(
    ctx: Context,
    situation: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 5,
    threshold: float = 0.3,
) -> dict:
    """Find behavioral rules that match a described situation.
    If project_id is omitted, auto-detects from the client's working directory.

    situation: free-text description of the current context or task.
    Returns behaviors ranked by relevance (similarity * confidence * priority)."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        embedding = await app.embedding.embed(situation)
        results = await match_behaviors(
            app.pool,
            embedding,
            limit=limit,
            threshold=threshold,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        # Touch matched behaviors to track usage
        for r in results:
            await touch_behavior(app.pool, r.behavior.id)
        return {
            "situation": situation,
            "count": len(results),
            "behaviors": [r.to_dict() for r in results],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_match", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_match", e)


@mcp.tool()
async def weft_behavior_list(
    ctx: Context,
    scope: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    enabled: bool | None = True,
    limit: int = 50,
) -> dict:
    """List stored behavioral rules with optional filters.
    If project_id is omitted, auto-detects from the client's working directory.

    Returns behaviors ordered by priority (highest first)."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        behavior_scope = BehaviorScope(scope) if scope else None
        results = await list_behaviors_store(
            app.pool,
            scope=behavior_scope,
            project_id=resolved_project,
            agent_id=agent_id,
            enabled=enabled,
            limit=limit,
        )
        return {
            "count": len(results),
            "behaviors": [b.to_dict() for b in results],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_list", e)


# --- Episode tools ---


@mcp.tool()
async def weft_episode_create(
    ctx: Context,
    title: str,
    summary: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a new open episode — a time-bounded grouping of memories.

    Episodes track causal sequences of decisions, actions, and learnings.
    If project_id is omitted, auto-detects from the client's working directory."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        ep = await create_episode(app.pool, EpisodeCreate(
            title=title,
            summary=summary,
            project_id=resolved_project,
            agent_id=agent_id,
        ))
        return ep.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_episode_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_create", e)


@mcp.tool()
async def weft_episode_add(
    ctx: Context,
    episode_id: str,
    memory_id: str,
    position: int | None = None,
) -> dict:
    """Link a memory to an episode. Idempotent — linking the same memory twice is a no-op.

    If position is omitted, auto-assigns the next sequential position."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        created = await add_memory_to_episode(
            app.pool, episode_id, memory_id, position=position,
        )
        return {
            "episode_id": episode_id,
            "memory_id": memory_id,
            "created": created,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_add", e)


@mcp.tool()
async def weft_episode_timeline(
    ctx: Context,
    start: str | None = None,
    end: str | None = None,
    hours: int = 24,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Find episodes overlapping a time range.

    If start/end are omitted, defaults to the last N hours (default 24).
    start/end accept ISO timestamps or relative values like '2d', '1w'.
    If project_id is omitted, auto-detects from the client's working directory.
    Open episodes (no end time) match any range after their start."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        now = datetime.now(timezone.utc)
        if end is not None:
            parsed_end = _parse_review_after(end) or now
        else:
            parsed_end = now
        if start is not None:
            parsed_start = _parse_review_after(start) or (now - timedelta(hours=hours))
        else:
            parsed_start = now - timedelta(hours=hours)

        results = await timeline_query(
            app.pool,
            start=parsed_start,
            end=parsed_end,
            project_id=resolved_project,
            agent_id=agent_id,
            limit=limit,
        )
        return {
            "start": parsed_start.isoformat(),
            "end": parsed_end.isoformat(),
            "count": len(results),
            "episodes": [ep.to_dict() for ep in results],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_episode_timeline", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_timeline", e)


@mcp.tool()
async def weft_episode_context(
    ctx: Context,
    episode_id: str,
    budget_tokens: int = 2000,
    include_episode: bool = True,
) -> dict:
    """Load an episode and its memories within a token budget.

    Returns the episode metadata and as many linked memories as fit
    within budget_tokens. Memories are returned in position order."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        ep = await get_episode(app.pool, episode_id)
        if ep is None:
            return {"error": f"Episode {episode_id} not found"}

        memories = await get_episode_memories(app.pool, episode_id)

        # Pack memories within token budget
        budget = budget_tokens
        if include_episode:
            ep_text = f"{ep.title}: {ep.summary or ''}"
            budget -= estimate_tokens(ep_text)

        packed = []
        total_tokens = 0
        for mem in memories:
            cost = estimate_tokens(mem.content)
            if total_tokens + cost > budget:
                break
            packed.append(mem)
            total_tokens += cost

        ewm = EpisodeWithMemories(episode=ep, memories=packed)
        result = ewm.to_dict()
        result["tokens_used"] = total_tokens
        result["tokens_budget"] = budget_tokens
        result["memories_truncated"] = len(memories) - len(packed)
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_context", e)


# --- Entity tools ---


@mcp.tool()
async def weft_entity_create(
    ctx: Context,
    name: str,
    entity_type: str = "concept",
    aliases: list[str] | None = None,
    description: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a first-class entity (person, project, company, tool, concept).

    Entities group memories about a subject. Link memories via weft_entity_link.
    If project_id is omitted, auto-detects from the client's working directory.

    entity_type: 'person', 'project', 'company', 'tool', or 'concept'."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        aliases = _coerce_list(aliases) or []
        embed_text = name + (f": {description}" if description else "")
        embedding = await app.embedding.embed(embed_text)
        ent = await store_entity(app.pool, EntityCreate(
            name=name,
            entity_type=EntityType(entity_type),
            aliases=aliases,
            description=description,
            project_id=resolved_project,
            agent_id=agent_id,
        ), embedding=embedding)
        return ent.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_entity_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_create", e)


@mcp.tool()
async def weft_entity_link(
    ctx: Context,
    entity_id: str,
    memory_id: str,
) -> dict:
    """Link a memory to an entity. Idempotent — linking twice is a no-op.

    Increments the entity's mention count on first link."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        created = await link_mention(app.pool, entity_id, memory_id)
        return {
            "entity_id": entity_id,
            "memory_id": memory_id,
            "created": created,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_link", e)


@mcp.tool()
async def weft_entity_search(
    ctx: Context,
    query: str,
    entity_type: str | None = None,
    project_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> dict:
    """Search for entities by semantic similarity.

    If project_id is omitted, auto-detects from the client's working directory.
    Returns entities ranked by relevance to the query."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        embedding = await app.embedding.embed(query)
        etype = EntityType(entity_type) if entity_type else None
        results = await search_entities(
            app.pool, embedding,
            entity_type=etype,
            project_id=resolved_project,
            limit=limit,
            threshold=threshold,
        )
        return {
            "query": query,
            "count": len(results),
            "entities": [
                {**ent.to_dict(), "similarity": round(sim, 4)}
                for ent, sim in results
            ],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_entity_search", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_search", e)


@mcp.tool()
async def weft_entity_context(
    ctx: Context,
    entity_id: str,
    budget_tokens: int = 2000,
) -> dict:
    """Load an entity and its linked memories within a token budget.

    Returns the entity metadata and as many linked memories as fit
    within budget_tokens. Memories ordered by mention time (newest first)."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        ent = await get_entity(app.pool, entity_id)
        if ent is None:
            return {"error": f"Entity {entity_id} not found"}

        memories = await get_entity_memories(app.pool, entity_id)

        # Reserve tokens for entity metadata
        ent_text = f"{ent.name}: {ent.description or ''}"
        budget = budget_tokens - estimate_tokens(ent_text)

        packed = []
        total_tokens = 0
        for mem in memories:
            cost = estimate_tokens(mem.content)
            if total_tokens + cost > budget:
                break
            packed.append(mem)
            total_tokens += cost

        result = ent.to_dict()
        result["memories"] = [m.to_dict() for m in packed]
        result["memory_count"] = len(packed)
        result["tokens_used"] = total_tokens
        result["tokens_budget"] = budget_tokens
        result["memories_truncated"] = len(memories) - len(packed)
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_context", e)
