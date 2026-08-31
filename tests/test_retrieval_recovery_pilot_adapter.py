from __future__ import annotations

import pytest

from benchmarks.retrieval_recovery_pilot.adapter import LivePilotAdapter, SnapshotNamespace
from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.models import RelationType
from weft.store import get_relationships


@pytest.mark.asyncio
async def test_seed_creates_persisted_conflict_edge(pool) -> None:
    adapter = LivePilotAdapter(
        pool,
        namespace=SnapshotNamespace("pilot-user", "pilot-project", "run"),
    )
    await adapter.seed()
    first = adapter.snapshot.stable_id("memory:pilot-conflict-a").split(":", 1)[1]
    second = adapter.snapshot.stable_id("memory:pilot-conflict-b").split(":", 1)[1]
    token = current_user_id.set("pilot-user")
    try:
        async with acquire(pool):
            relationships = await get_relationships(pool, first, relation=RelationType.contradicts)
    finally:
        current_user_id.reset(token)
    assert any(
        {relationship.source_id, relationship.target_id} == {first, second}
        for relationship in relationships
    )
