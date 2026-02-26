"""Weft MCP tools — thin coordinators, ≤15 lines each."""

from __future__ import annotations

import logging

import asyncpg
from fastmcp import Context

from weft.mcp.server import AppContext, mcp
from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType, RelationType
from weft.store import (
    add_relationship,
    delete_memory,
    get_relationships,
    get_stats,
    list_memories,
    record_feedback,
    remove_relationship,
    search_by_vector,
    store_memory,
    touch_memory,
)

logger = logging.getLogger(__name__)

# Broad tuple of exceptions that can surface when the database is unreachable.
_DB_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionRefusedError, Exception)


def _db_error_response(tool_name: str, error: Exception) -> dict:
    """Standard error response when database is unavailable."""
    logger.warning("Database unavailable in %s: %s", tool_name, error)
    return {"error": "Database unavailable", "degraded": True, "tool": tool_name}


@mcp.tool()
async def weft_remember(
    ctx: Context,
    content: str,
    type: str = "fact",
    topic: list[str] | None = None,
    source: str = "conversation",
    confidence: float = 0.7,
    project_id: str | None = None,
    agent_id: str | None = None,
    check_contradictions: bool = True,
) -> dict:
    """Store a new memory with type, topics, content, confidence, and source."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        create = MemoryCreate(
            type=MemoryType(type),
            content=content,
            topic=topic or [],
            source=MemorySource(source),
            confidence=confidence,
            project_id=project_id,
            agent_id=agent_id,
        )
        embedding = await app.embedding.embed(content)
        memory = await store_memory(app.pool, create, embedding=embedding)
        await app.cache.set_memory(memory)
        await app.cache.invalidate_stats()
        result = memory.to_dict()
        if check_contradictions and embedding:
            from weft.consolidation import check_contradictions_on_store
            warnings = await check_contradictions_on_store(app.pool, memory.id, embedding)
            if warnings:
                result["contradiction_warnings"] = warnings
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_remember", e)


@mcp.tool()
async def weft_recall(
    ctx: Context,
    query: str,
    topic: str | None = None,
    type: str | None = None,
    status: str | None = None,
    project_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> dict:
    """Retrieve memories by semantic query, topic filter, type filter, status filter, or combination."""
    try:
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
        )
        # Touch accessed memories
        for r in results:
            await touch_memory(app.pool, r.memory.id)
        return {"query": query, "count": len(results), "results": [r.to_dict() for r in results]}
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
    type: str | None = None,
    project_id: str | None = None,
    max_per_topic: int = 3,
) -> dict:
    """Budget-aware context loading: best memories for a situation within N tokens."""
    try:
        from weft.context import build_context

        app: AppContext = ctx.request_context.lifespan_context
        memory_type = MemoryType(type) if type else None
        embedding = await app.embedding.embed(query)
        result = await build_context(
            app.pool, embedding,
            budget_tokens=budget_tokens, max_per_topic=max_per_topic,
            memory_type=memory_type, topic=topic, project_id=project_id,
        )
        # Touch the memories that made it into context
        for mem_dict in result["memories"]:
            await touch_memory(app.pool, mem_dict["id"])
        return result
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
) -> dict:
    """Update a memory's content, creating a new version that supersedes the old one."""
    try:
        from weft.revise import revise_memory

        app: AppContext = ctx.request_context.lifespan_context
        embedding = await app.embedding.embed(new_content)
        new, old = await revise_memory(
            app.pool, memory_id, new_content,
            embedding=embedding, new_confidence=new_confidence, new_topic=new_topic,
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
    except _DB_ERRORS as e:
        return _db_error_response("weft_relate", e)


@mcp.tool()
async def weft_consolidate(ctx: Context, dry_run: bool = False) -> dict:
    """Run consolidation: decay stale memories, merge duplicates, flag contradictions."""
    try:
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
async def weft_prime(
    ctx: Context,
    project_id: str | None = None,
    budget_tokens: int = 4000,
    recent_days: int = 7,
) -> dict:
    """Session primer: assemble structured context with preferences, recent work, and relevant memories."""
    try:
        from weft.primer import build_primer

        app: AppContext = ctx.request_context.lifespan_context
        return await build_primer(
            app.pool,
            project_id=project_id,
            budget_tokens=budget_tokens,
            recent_days=recent_days,
        )
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_prime: %s", e)
        from weft.fallback import read_fallback
        content = read_fallback()
        return {
            "preferences": [],
            "recent_work": [],
            "relevant": [{"content": content, "type": "fallback"}] if content else [],
            "total_tokens": 0,
            "budget_tokens": budget_tokens,
            "budget_remaining": budget_tokens,
            "degraded": True,
        }


@mcp.tool()
async def weft_status(ctx: Context) -> dict:
    """Memory statistics: total, by topic, by type, by confidence, recently accessed."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        cached = await app.cache.get_stats()
        if cached:
            return cached
        stats = await get_stats(app.pool)
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
