"""Version-aware memory updates — create new versions, archive old ones.

Revision workflow:
1. Get the existing memory
2. Create a new memory with updated content + new embedding
3. Add a 'supersedes' relationship (new → old)
4. Archive the old memory (status → archived)
"""

from __future__ import annotations

import asyncpg

from weft.models import (
    Memory,
    MemoryCreate,
    MemoryStatus,
    RelationType,
)
from weft.store import (
    add_relationship,
    get_memory,
    store_memory,
    update_memory,
)


async def revise_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    new_content: str,
    *,
    embedding: list[float] | None = None,
    new_confidence: float | None = None,
    new_topic: list[str] | None = None,
) -> tuple[Memory, Memory]:
    """Create a new version of a memory, superseding the old one.

    Returns (new_memory, old_memory) where old_memory has been archived.
    Raises ValueError if the original memory is not found.
    """
    old = await get_memory(pool, memory_id)
    if old is None:
        raise ValueError(f"Memory {memory_id} not found")

    # Create the new version, inheriting metadata from the old one
    create = MemoryCreate(
        type=old.type,
        content=new_content,
        topic=new_topic if new_topic is not None else old.topic,
        source=old.source,
        confidence=new_confidence if new_confidence is not None else old.confidence,
        project_id=old.project_id,
        agent_id=old.agent_id,
    )
    new = await store_memory(pool, create, embedding=embedding)

    # Link: new supersedes old
    await add_relationship(pool, new.id, old.id, RelationType.supersedes)

    # Archive the old memory
    archived_old = await update_memory(pool, old.id, status=MemoryStatus.archived)

    return new, archived_old or old
