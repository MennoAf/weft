"""Acceptance test — "What do I need to know before meeting Jim Boblaw?"

This is the canonical personal-agent query shape: given a person, surface
(a) the entity record itself, (b) memories mentioning them, and (c) any
open trackers tied to them. A working personal agent should be able to
hit all three from a single name.

Jim Boblaw is the synthetic test persona — never substitute a real name
in committed fixtures.
"""

from __future__ import annotations

import pytest

from weft.entities import search_entities, store_entity
from weft.models import (
    EntityCreate,
    EntityType,
    MemoryCreate,
    MemorySource,
    MemoryType,
    NudgeMode,
    TrackerCreate,
    TrackerKind,
)
from weft.store import search_hybrid, store_memory
from weft.trackers import create_tracker, list_trackers

from tests.acceptance.conftest import (
    SYNTHETIC_PERSON,
    cleanup_project,
    sandbox_project_id,
)


CASE_ID = "meeting_prep_jim_boblaw"


@pytest.mark.asyncio
async def test_meeting_prep_jim_boblaw(pool, embedder) -> None:
    project_id = sandbox_project_id(CASE_ID)
    try:
        # --- Seed -------------------------------------------------------
        # An entity with role + description so "what do I know about X"
        # has structured ground to hit.
        person_embedding = await embedder.embed(SYNTHETIC_PERSON)
        jim = await store_entity(
            pool,
            EntityCreate(
                name=SYNTHETIC_PERSON,
                entity_type=EntityType.person,
                description="VP Engineering at Acme; leading the data platform migration.",
                project_id=project_id,
            ),
            embedding=person_embedding,
        )

        # Three memories, each a distinct fact a personal agent should
        # surface separately. Embeddings on each so hybrid recall ranks them.
        seed_memories = [
            "Met with Jim Boblaw to scope the data platform migration. He's pushing for a Q3 cutover and wants Snowflake → BigQuery.",
            "Jim Boblaw mentioned his team is short-staffed on data engineering — he's hiring two senior ICs in May.",
            "Jim Boblaw prefers async-first communication. Slack DMs > email > meetings.",
        ]
        for content in seed_memories:
            mem_emb = await embedder.embed(content)
            await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content=content,
                    source=MemorySource.conversation,
                    project_id=project_id,
                ),
                embedding=mem_emb,
            )

        # An open follow-up tracker tied to Jim. Personal agents should
        # surface this on a "meeting prep" prompt without being asked.
        tracker = await create_tracker(
            pool,
            TrackerCreate(
                kind=TrackerKind.outreach,
                title=f"Follow up with {SYNTHETIC_PERSON} on migration timeline",
                project_id=project_id,
                entity_id=jim.id,
                nudge_mode=NudgeMode.once,
            ),
        )

        # --- Query ------------------------------------------------------
        # Path 1: entity lookup — the agent resolves the name first.
        query_emb = await embedder.embed(SYNTHETIC_PERSON)
        entity_hits = await search_entities(
            pool, query_emb, project_id=project_id, limit=5,
        )
        # Path 2: memory recall — the agent pulls relevant context.
        question = f"What do I need to know before meeting {SYNTHETIC_PERSON}?"
        question_emb = await embedder.embed(question)
        recall_hits = await search_hybrid(
            pool, question, question_emb,
            limit=10, project_id=project_id,
        )
        # Path 3: open trackers tied to that entity — the "don't forget
        # the open thread" signal.
        entity_trackers = await list_trackers(
            pool, project_id=project_id, entity_id=jim.id, open_only=True,
        )

        # --- Assertions -------------------------------------------------
        # Entity is found and linked to the right project.
        assert any(e.id == jim.id for e, _ in entity_hits), (
            "entity search did not surface Jim Boblaw's record"
        )

        # All three memories surface in recall. We don't lock to top-N
        # ranking here — we just require they're retrievable, since the
        # agent is the one synthesizing the response.
        recalled_contents = [r.memory.content for r in recall_hits]
        assert any("data platform migration" in c for c in recalled_contents), (
            "recall missed the migration context memory"
        )
        assert any("hiring two senior" in c for c in recalled_contents), (
            "recall missed the staffing context memory"
        )
        assert any("async-first" in c for c in recalled_contents), (
            "recall missed the communication-preference memory"
        )

        # Tracker surfaces — meeting prep without the open follow-up is
        # the failure mode personal agents must not hit.
        assert any(t.id == tracker.id for t in entity_trackers), (
            "open tracker tied to entity was not surfaced"
        )

    finally:
        await cleanup_project(pool, project_id)
