#!/usr/bin/env python3
"""
fixtures.py — Jim Boblaw synthetic persona fixtures with known-membership seeds.

Defines several seeded collections (plants, medications, etc.) with known
cardinality and topic tags, so the oracle gather_topic_memories can enumerate
them fully, and the stochastic weft_recall candidate path can be measured
against a ground truth.

House rule: never use real names — Jim Boblaw is a synthetic sentinel persona
from the Weft test suite. All fixture data is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from weft.models import Memory, MemorySource, MemoryStatus, MemoryType, MemoryCreate


class FixtureCategory(str, Enum):
    """Fixture taxonomy — each category is a topic tag + known membership set."""
    PLANTS = "plants"
    MEDICATIONS = "medications"
    BOOKS = "books"


# ─────────────────────────────────────────────────────────────────────────
# Jim Boblaw Synthetic Persona — Global Sentinel
# ─────────────────────────────────────────────────────────────────────────

JIM_BOBLAW_USER_ID = "jim-boblaw-synthetic-test"
JIM_BOBLAW_PROJECT_ID = "jim-boblaw-enumeration-eval"


@dataclass
class FixtureCollection:
    """Known-membership set: topic tag + list of memory contents."""
    category: FixtureCategory
    topic_tag: str
    members: list[str]

    def __post_init__(self):
        """Validate cardinality for test traceability."""
        if not self.members:
            raise ValueError(f"FixtureCollection {self.category} has zero members")


# Fixtures with known cardinality — each is a seeded collection Jim Boblaw
# "remembers" about a topic, tagged for topic gather.

def get_fixtures() -> list[FixtureCollection]:
    """Return all known-membership fixture collections for Jim Boblaw."""
    return [
        FixtureCollection(
            category=FixtureCategory.PLANTS,
            topic_tag="plants",
            members=[
                "Jim grows tomatoes in his backyard.",
                "Jim has a pothos plant in his bedroom.",
                "Jim tends to basil in the kitchen garden.",
                "Jim's favorite flower is the sunflower.",
                "Jim bought orchids last spring.",
                "Jim propagates succulents on his windowsill.",
                "Jim planted roses along the fence.",
                "Jim composted leaves from his maple tree.",
                "Jim grows carrots in his vegetable patch.",
                "Jim waters his ferns every morning.",
                "Jim harvested lavender for dried flowers.",
                "Jim keeps a cactus collection in the office.",
            ],
        ),
        FixtureCollection(
            category=FixtureCategory.MEDICATIONS,
            topic_tag="medications",
            members=[
                "Jim takes aspirin for headaches.",
                "Jim uses metformin for blood sugar management.",
                "Jim applies antibiotic cream to minor cuts.",
                "Jim takes vitamin D supplements.",
                "Jim uses ibuprofen for joint pain.",
                "Jim takes omeprazole for acid reflux.",
                "Jim uses nasal decongestant spray.",
                "Jim takes allergy medication in spring.",
            ],
        ),
        FixtureCollection(
            category=FixtureCategory.BOOKS,
            topic_tag="books",
            members=[
                "Jim read 'The Great Gatsby' in college.",
                "Jim enjoys science fiction novels.",
                "Jim has a signed copy of 'Dune'.",
                "Jim re-read 'To Kill a Mockingbird' last year.",
                "Jim listens to audiobooks during his commute.",
            ],
        ),
    ]


async def seed_fixtures(
    pool,
    user_id: str = JIM_BOBLAW_USER_ID,
    project_id: str = JIM_BOBLAW_PROJECT_ID,
    embedder=None,
) -> dict[str, FixtureCollection]:
    """Seed all Jim Boblaw fixtures into the DB and return collection map.

    Each memory is stored with a REAL embedding computed via the same provider
    the production write path uses (``app.embedding.embed(content)``; see
    ``weft_remember`` in weft/mcp/tools.py). Without a vector, the stochastic
    vector/hybrid CANDIDATE recall path can never match a seeded row, which
    would make the min/median/max spread — the entire point of the candidate
    arm (PRD §V4 "the spread is the never-miss signal") — structurally
    meaningless. So embeddings are NOT optional for a valid eval.

    Args:
        pool: asyncpg pool for writing memories.
        user_id: Override for user identity (default: Jim Boblaw).
        project_id: Override for project sandbox (default: Jim Boblaw enumeration).
        embedder: Embedding provider with an async ``embed(content)`` method.
            Defaults to the FastEmbed 768-dim provider the harness/app uses.

    Returns:
        Dictionary mapping category.value -> FixtureCollection.
    """
    from weft.store import store_memory
    from weft.auth import current_user_id
    from weft.db.connection import acquire

    if embedder is None:
        from weft.embeddings import get_provider
        embedder = get_provider("fastembed", dimensions=768)

    collections = get_fixtures()

    # Set user_id context and use acquire() to ensure RLS-scoped connection
    token = current_user_id.set(user_id)
    try:
        async with acquire(pool) as conn:
            for collection in collections:
                for content in collection.members:
                    # Compute a real embedding the same way the production
                    # write path does — no faked vectors.
                    embedding = await embedder.embed(content)
                    await store_memory(
                        pool,
                        create=MemoryCreate(
                            type=MemoryType.fact,
                            content=content,
                            topic=[collection.topic_tag],
                            source=MemorySource.conversation,
                            confidence=0.9,
                            project_id=project_id,
                        ),
                        embedding=embedding,
                    )
    finally:
        current_user_id.reset(token)

    return {c.category.value: c for c in collections}
