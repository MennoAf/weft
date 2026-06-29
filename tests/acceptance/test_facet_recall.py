"""Acceptance test — L5 facet-recall: OAuth belief cross-project merge + recall.

Exercises the real weft_remember write path (store_memory + init_project_facets
+ check_dedup_on_store) and the real weft_recall path (search_hybrid with
facet_boost_project_id) end-to-end against a testcontainer Postgres DB.
No production data is touched — the pool fixture is a fresh ephemeral DB per test.

Spec: loom-739761de  (parent epic: loom-1d6c8e5c)
Depends on: loom-a7664166 (L2, write path) + loom-cbef102a (L3, facet-boost recall)

Done-when assertions:
  1. Store OAuth belief under project=weft, then semantically-identical belief
     under project=loom → exactly ONE active memories row,
     project_facets == {weft-sandbox, loom-sandbox}.
  2. weft_recall (search_hybrid + facet_boost_project_id) under project=weft
     returns the belief.
  3. weft_recall under project=loom also returns the same belief.
  4. The recalled result dict carries a 'project_facets' key (L3 mapping verified).
  5. A third semantically-distinct belief stored under project=loom is NOT merged
     (stays its own row — total 2 active rows after both stores).

Synthetic persona: Jim Boblaw (never real names from conversation).
"""

from __future__ import annotations

import pytest

from weft.consolidation import check_dedup_on_store, init_project_facets
from weft.models import MemoryCreate, MemoryType
from weft.store import search_hybrid, store_memory

from tests.acceptance.conftest import (
    SYNTHETIC_PERSON,
    cleanup_project,
    sandbox_project_id,
)


CASE_ID = "facet_recall_acceptance"

# Jim Boblaw's OAuth preference — the belief that will be cross-project merged.
OAUTH_CONTENT = (
    f"{SYNTHETIC_PERSON} requires OAuth 2.0 with PKCE flow for all new authentication "
    "integrations. Client secrets must never be stored in browser-accessible storage; "
    "access tokens are short-lived (15 min) and refresh tokens rotate on every use."
)

# Semantically distinct belief — must NOT merge with the OAuth belief above.
DISTINCT_CONTENT = (
    f"{SYNTHETIC_PERSON} versions all infrastructure-as-code using Terraform, keeping "
    "modules in a dedicated Git monorepo separate from application code."
)

# Query used to simulate weft_recall against the stored OAuth belief.
OAUTH_QUERY = f"What is {SYNTHETIC_PERSON}'s authentication strategy for new integrations?"


@pytest.mark.asyncio
async def test_facet_recall_oauth_cross_project(pool, embedder) -> None:
    """End-to-end proof of the facet-recall epic (loom-739761de).

    Stores an OAuth belief under project=weft, cross-project deduplicates it
    under project=loom (simulating a second agent's write), then asserts that
    weft_recall surfaces the merged belief from BOTH project contexts and that
    the recalled result carries project_facets in its payload.

    Uses the testcontainer Postgres DB (same fixture the unit tests use) so the
    real dedup + embedding + search code paths run without touching production data.
    """
    project_weft = sandbox_project_id(f"{CASE_ID}_weft")
    project_loom = sandbox_project_id(f"{CASE_ID}_loom")

    try:
        # ── 1. Embed once; reuse across all store + recall operations ────────
        oauth_emb = await embedder.embed(OAUTH_CONTENT)
        query_emb = await embedder.embed(OAUTH_QUERY)

        # ── 2. Store OAuth belief under project=weft (real weft_remember path) ──
        #       store_memory creates the row; init_project_facets seeds facets.
        mem_weft = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.preference,
                content=OAUTH_CONTENT,
                confidence=0.85,
                project_id=project_weft,
            ),
            embedding=oauth_emb,
        )
        await init_project_facets(pool, mem_weft.id, project_weft)

        # ── 3. Cross-project dedup: same belief arrives from project=loom ────
        #       This mirrors what weft_remember does when a second agent in the
        #       loom project tries to store the same belief.
        dedup = await check_dedup_on_store(
            pool,
            OAUTH_CONTENT,
            oauth_emb,
            new_confidence=0.85,
            memory_type=MemoryType.preference,
            project_id=project_loom,
        )

        # Must be a clean auto-merge — no new row inserted.
        assert dedup.is_duplicate is True, (
            f"Expected is_duplicate=True for semantically-identical cross-project "
            f"belief; got action={dedup.action!r}"
        )
        assert dedup.action == "facet_appended", (
            f"Expected action='facet_appended'; got {dedup.action!r}"
        )
        assert dedup.existing_memory is not None
        assert dedup.existing_memory.id == mem_weft.id, (
            "facet_appended should reference the original weft memory"
        )

        # ── 4. DB: exactly ONE active row, facets cover BOTH projects ─────────
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        )
        assert count == 1, (
            f"Cross-project dedup must not create a second row; found {count}"
        )

        row = await pool.fetchrow(
            "SELECT project_facets FROM memories WHERE id = $1", mem_weft.id
        )
        assert row is not None, "Original memory row disappeared after dedup"
        facets = set(row["project_facets"])
        assert facets == {project_weft, project_loom}, (
            f"Expected project_facets={{{project_weft!r}, {project_loom!r}}}; "
            f"got {facets!r}"
        )

        # ── 5. weft_recall from project=weft surfaces the merged belief ───────
        #       facet_boost_project_id mirrors what weft_recall(retrieval_mode='face')
        #       does: drop the hard project wall and rank by facet overlap.
        results_weft = await search_hybrid(
            pool,
            OAUTH_QUERY,
            query_emb,
            limit=5,
            threshold=0.0,
            facet_boost_project_id=project_weft,
        )
        ids_weft = [r.memory.id for r in results_weft]
        assert mem_weft.id in ids_weft, (
            "weft_recall (facet_boost=weft) did not surface the merged OAuth belief"
        )

        # ── 6. weft_recall from project=loom also surfaces the same belief ────
        results_loom = await search_hybrid(
            pool,
            OAUTH_QUERY,
            query_emb,
            limit=5,
            threshold=0.0,
            facet_boost_project_id=project_loom,
        )
        ids_loom = [r.memory.id for r in results_loom]
        assert mem_weft.id in ids_loom, (
            "weft_recall (facet_boost=loom) did not surface the merged OAuth belief"
        )

        # ── 7. Recalled result carries project_facets in its payload (L3 check) ──
        #       MemoryRecall.to_dict() calls memory.to_dict() which calls
        #       model_dump(mode='json') — project_facets is a native list field.
        match_weft = next(r for r in results_weft if r.memory.id == mem_weft.id)
        result_dict = match_weft.to_dict()
        assert "project_facets" in result_dict, (
            "weft_recall result dict must contain 'project_facets' key (L3 mapping)"
        )
        assert set(result_dict["project_facets"]) == {project_weft, project_loom}, (
            f"project_facets in recalled result must equal "
            f"{{{project_weft!r}, {project_loom!r}}}; "
            f"got {result_dict['project_facets']!r}"
        )
        # Also verify directly via the Memory object (proves _row_to_memory maps it).
        assert set(match_weft.memory.project_facets) == {project_weft, project_loom}, (
            "Memory.project_facets must be populated (not empty default)"
        )

        # ── 8. Semantically-distinct belief under loom stays its own row ──────
        distinct_emb = await embedder.embed(DISTINCT_CONTENT)
        distinct_dedup = await check_dedup_on_store(
            pool,
            DISTINCT_CONTENT,
            distinct_emb,
            new_confidence=0.85,
            memory_type=MemoryType.preference,
            project_id=project_loom,
        )

        # Terraform / infra-as-code content must NOT auto-merge with OAuth belief.
        assert distinct_dedup.is_duplicate is False, (
            f"Semantically-distinct belief must not merge with the OAuth row; "
            f"got action={distinct_dedup.action!r}"
        )

        # Store the distinct belief as a new row (what weft_remember would do after
        # check_dedup_on_store returns is_duplicate=False).
        mem_distinct = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.preference,
                content=DISTINCT_CONTENT,
                confidence=0.85,
                project_id=project_loom,
            ),
            embedding=distinct_emb,
        )
        await init_project_facets(pool, mem_distinct.id, project_loom)

        # Two active rows now: the merged OAuth belief + the distinct Terraform one.
        count_final = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        )
        assert count_final == 2, (
            f"Expected 2 active rows after distinct store; found {count_final}"
        )

        # Verify the distinct row carries only the loom facet (no cross-project merge).
        row_distinct = await pool.fetchrow(
            "SELECT project_facets FROM memories WHERE id = $1", mem_distinct.id
        )
        assert set(row_distinct["project_facets"]) == {project_loom}, (
            f"Distinct belief must only carry the {project_loom!r} facet; "
            f"got {set(row_distinct['project_facets'])!r}"
        )

    finally:
        await cleanup_project(pool, project_weft)
        await cleanup_project(pool, project_loom)
