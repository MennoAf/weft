"""Acceptance-test fixtures — sandbox project_ids + a real embedder.

These tests verify Weft behaves correctly for *personal-agent* query shapes
("what do I know about <person>", "what do I have this week", project
isolation). The model is:

  * Each test runs against a unique ``project_id`` derived from the case
    name. Seed data lives in that scope, query runs against that scope,
    teardown deletes everything in that scope. No cross-test pollution.
  * Embeddings come from FastEmbed — same provider used in
    benchmarks/longmemeval/tests/test_smoke.py, so retrieval behaviour
    matches what the rest of the test surface sees.

The committed test fixtures use the synthetic persona "Jim Boblaw" — never
substitute a real name. Real-data cases live under ``cases_local/``
(gitignored) and may be seeded from snapshots of the user's actual Weft
projects via the (forthcoming) snapshot CLI.
"""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.embeddings.base import EmbeddingProvider


SYNTHETIC_PERSON = "Jim Boblaw"


@pytest.fixture(scope="session")
def embedder() -> EmbeddingProvider:
    """A session-scoped FastEmbed provider — model load is the slow part,
    so reusing it across all acceptance tests cuts setup time meaningfully."""
    return get_provider("fastembed", dimensions=768)


def sandbox_project_id(case_id: str) -> str:
    """Stable per-case project_id for sandbox isolation.

    The ``personal_test_`` prefix is the convention every acceptance case
    uses; nothing in production code ever writes under this prefix, so the
    teardown's blanket DELETE is safe.
    """
    return f"personal_test_{case_id}"


async def cleanup_project(pool, project_id: str) -> None:
    """Hard-delete every row tied to ``project_id``.

    Order respects FK dependencies: mention links and tracker rows
    referencing memories/entities go first, then the entities/memories
    themselves. Any table without a project_id column (e.g. user-scoped
    metadata) is intentionally left alone.
    """
    await pool.execute(
        "DELETE FROM entity_mentions WHERE memory_id IN "
        "(SELECT id FROM memories WHERE project_id = $1)",
        project_id,
    )
    await pool.execute("DELETE FROM trackers WHERE project_id = $1", project_id)
    await pool.execute("DELETE FROM memories WHERE project_id = $1", project_id)
    await pool.execute("DELETE FROM entities WHERE project_id = $1", project_id)
