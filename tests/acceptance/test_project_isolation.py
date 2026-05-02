"""Acceptance test — project_id isolation guardrail.

The personal-agent acceptance suite relies on per-case sandbox project_ids
to keep tests hermetic. If retrieval primitives ever leak across project
boundaries, every other case in this directory becomes unreliable. This
test pins the boundary directly:

  * Identical-content memories in two distinct projects.
  * Identical-name entities in two distinct projects.
  * A query scoped to project A must NOT return rows from project B.

If this fails, the rest of the acceptance suite's hermeticity guarantees
are also broken — fix this first.
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
)
from weft.store import search_hybrid, store_memory

from tests.acceptance.conftest import cleanup_project, sandbox_project_id


CASE_ID_A = "project_isolation_a"
CASE_ID_B = "project_isolation_b"
SHARED_NAME = "Sam"  # synthetic; identical across both projects on purpose
SHARED_FACT = "Had coffee with Sam to talk about the API redesign."


@pytest.mark.asyncio
async def test_project_isolation(pool, embedder) -> None:
    project_a = sandbox_project_id(CASE_ID_A)
    project_b = sandbox_project_id(CASE_ID_B)
    try:
        # --- Seed identical content in both projects --------------------
        name_emb = await embedder.embed(SHARED_NAME)
        fact_emb = await embedder.embed(SHARED_FACT)

        sam_a = await store_entity(
            pool,
            EntityCreate(
                name=SHARED_NAME,
                entity_type=EntityType.person,
                description="Sam from project A — works on the API redesign.",
                project_id=project_a,
            ),
            embedding=name_emb,
        )
        sam_b = await store_entity(
            pool,
            EntityCreate(
                name=SHARED_NAME,
                entity_type=EntityType.person,
                description="Sam from project B — totally different Sam.",
                project_id=project_b,
            ),
            embedding=name_emb,
        )

        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=f"[A] {SHARED_FACT}",
                source=MemorySource.conversation,
                project_id=project_a,
            ),
            embedding=fact_emb,
        )
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=f"[B] {SHARED_FACT}",
                source=MemorySource.conversation,
                project_id=project_b,
            ),
            embedding=fact_emb,
        )

        # --- Query scoped to project A only -----------------------------
        entity_hits_a = await search_entities(
            pool, name_emb, project_id=project_a, limit=10,
        )
        recall_hits_a = await search_hybrid(
            pool, SHARED_FACT, fact_emb,
            limit=10, project_id=project_a,
        )

        # --- Assertions -------------------------------------------------
        # Project A's entity is found, project B's entity is not.
        a_ids = {e.id for e, _ in entity_hits_a}
        # NB: search_entities currently includes globally-scoped (NULL
        # project_id) entities as a deliberate feature. We only assert
        # that B's entity is excluded — a stricter "exactly A only"
        # assertion would couple this test to the sandbox seed never
        # creating global entities, which is fragile.
        assert sam_a.id in a_ids, "project A's entity not surfaced under its own project_id"
        assert sam_b.id not in a_ids, (
            "LEAK: project B's entity surfaced when querying scoped to project A"
        )

        recalled_contents = [r.memory.content for r in recall_hits_a]
        assert any(c.startswith("[A] ") for c in recalled_contents), (
            "project A's memory not surfaced under its own project_id"
        )
        assert not any(c.startswith("[B] ") for c in recalled_contents), (
            "LEAK: project B's memory surfaced when querying scoped to project A"
        )

    finally:
        await cleanup_project(pool, project_a)
        await cleanup_project(pool, project_b)
