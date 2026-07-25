"""Version-aware memory updates — create new versions, archive old ones.

Revision workflow:
1. Get the existing memory
2. Create a new memory with updated content + new embedding
3. Add a 'supersedes' relationship (new → old)
4. Archive the old memory (status → archived)
"""

from __future__ import annotations

from datetime import datetime, timezone

import asyncpg

from weft.db.connection import acquire
from weft.models import (
    Memory,
    MemoryStatus,
    PreferenceMetadata,
    MemoryType,
    RelationType,
    _weft_id,
)
from weft.store import get_memory
from weft.tokens import estimate_tokens

_UNSET = object()


async def revise_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    new_content: str,
    *,
    embedding: list[float] | None = None,
    new_confidence: float | None = None,
    new_topic: list[str] | None = None,
    new_type: MemoryType | None = None,
    new_project_id: str | None = _UNSET,
    new_pinned: bool | None = _UNSET,
    review_after: datetime | None = _UNSET,
    preference_metadata: object = _UNSET,
) -> tuple[Memory, Memory]:
    """Create a new version of a memory, superseding the old one.

    Returns (new_memory, old_memory) where old_memory has been archived.
    Raises ValueError if the original memory is not found.
    """
    old = await get_memory(pool, memory_id)
    if old is None:
        raise ValueError(f"Memory {memory_id} not found")

    # Prepare all data before opening the transaction
    resolved_review = old.review_after if review_after is _UNSET else review_after
    resolved_type = new_type if new_type is not None else old.type
    resolved_topic = new_topic if new_topic is not None else old.topic
    resolved_confidence = new_confidence if new_confidence is not None else old.confidence
    resolved_project_id = old.project_id if new_project_id is _UNSET else new_project_id
    resolved_pinned = old.pinned if new_pinned is _UNSET else bool(new_pinned)
    if preference_metadata is _UNSET:
        resolved_preference_metadata = old.preference_metadata
    elif preference_metadata is None:
        resolved_preference_metadata = None
    else:
        resolved_preference_metadata = PreferenceMetadata.model_validate(preference_metadata)
    if resolved_type is not MemoryType.preference and resolved_preference_metadata is not None:
        raise ValueError(
            "retyping away from preference requires explicit preference_metadata=null"
        )
    new_id = _weft_id()
    now = datetime.now(timezone.utc)
    token_count = estimate_tokens(new_content)
    async with acquire(pool) as conn:
        async with conn.transaction():
            # 1. Insert new memory
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, project_id, agent_id, embedding, status, pinned,
                    review_after, user_id, preference_metadata, workspace_id,
                    project_facets, write_provenance, review_status, embed_composition_version
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $8, $8,
                    0, $9, $10, $11::vector, 'active', $13,
                    $12, nullif(current_setting('app.user_id', true), ''), $14::jsonb,
                    $15, $16, $17, $18, $19
                )
                """,
                new_id,
                resolved_type.value,
                resolved_topic,
                new_content,
                old.source.value,
                resolved_confidence,
                token_count,
                now,
                resolved_project_id,
                old.agent_id,
                embedding,
                resolved_review,
                resolved_pinned,
                resolved_preference_metadata.model_dump_json()
                if resolved_preference_metadata is not None else None,
                old.workspace_id,
                old.project_facets,
                old.write_provenance,
                old.review_status,
                1,
            )

            # 2. Link: new supersedes old
            await conn.execute(
                """
                INSERT INTO memory_relationships (source_id, target_id, relation, created_at, user_id)
                VALUES ($1, $2, $3, $4, nullif(current_setting('app.user_id', true), ''))
                ON CONFLICT (source_id, target_id, relation) DO NOTHING
                """,
                new_id,
                old.id,
                RelationType.supersedes.value,
                now,
            )

            # 3. Archive the old memory
            await conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = now() WHERE id = $1",
                old.id,
            )

    new = Memory(
        id=new_id,
        type=resolved_type,
        topic=resolved_topic,
        content=new_content,
        source=old.source,
        confidence=resolved_confidence,
        token_count=token_count,
        created_at=now,
        updated_at=now,
        accessed_at=now,
        access_count=0,
        project_id=resolved_project_id,
        agent_id=old.agent_id,
        status=MemoryStatus.active,
        pinned=resolved_pinned,
        review_after=resolved_review,
        preference_metadata=resolved_preference_metadata,
        workspace_id=old.workspace_id,
        project_facets=old.project_facets,
        write_provenance=old.write_provenance,
        review_status=old.review_status,
    )

    return new, Memory(
        **{**old.model_dump(), "status": MemoryStatus.archived, "updated_at": now},
    )
