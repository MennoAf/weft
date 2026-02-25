"""Weft MCP tools — thin coordinators, ≤15 lines each."""

from __future__ import annotations

from fastmcp import Context

from weft.mcp.server import AppContext, mcp
from weft.models import MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.store import (
    delete_memory,
    get_stats,
    list_memories,
    search_by_vector,
    store_memory,
    touch_memory,
)


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
) -> dict:
    """Store a new memory with type, topics, content, confidence, and source."""
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
    return memory.to_dict()


@mcp.tool()
async def weft_recall(
    ctx: Context,
    query: str,
    topic: str | None = None,
    type: str | None = None,
    project_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> dict:
    """Retrieve memories by semantic query, topic filter, type filter, or combination."""
    app: AppContext = ctx.request_context.lifespan_context
    embedding = await app.embedding.embed(query)
    results = await search_by_vector(
        app.pool,
        embedding,
        limit=limit,
        threshold=threshold,
        topic=topic,
        project_id=project_id,
    )
    # Touch accessed memories
    for r in results:
        await touch_memory(app.pool, r.memory.id)
    return {"query": query, "count": len(results), "results": [r.to_dict() for r in results]}


@mcp.tool()
async def weft_forget(
    ctx: Context,
    memory_id: str,
    hard: bool = False,
) -> dict:
    """Archive a memory (soft-delete) or hard-delete it."""
    app: AppContext = ctx.request_context.lifespan_context
    deleted = await delete_memory(app.pool, memory_id, hard=hard)
    await app.cache.invalidate_memory(memory_id)
    await app.cache.invalidate_stats()
    return {"memory_id": memory_id, "deleted": deleted, "hard": hard}


@mcp.tool()
async def weft_status(ctx: Context) -> dict:
    """Memory statistics: total, by topic, by type, by confidence, recently accessed."""
    app: AppContext = ctx.request_context.lifespan_context
    cached = await app.cache.get_stats()
    if cached:
        return cached
    stats = await get_stats(app.pool)
    await app.cache.set_stats(stats)
    return stats
