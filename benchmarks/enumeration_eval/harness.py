#!/usr/bin/env python3
"""
harness.py — Enumeration eval harness for recall@membership measurement.

Two paths:
1. ORACLE: gather_topic_memories (deterministic) — must return ALL M members
   in a single call. Verifies recall@membership == 1.0 as the ground truth.
2. CANDIDATE: weft_recall (stochastic NL) — run k>=5 times, report spread.

Metric: recall@membership = |returned ∩ members(C)| / |members(C)|

A fixture is a FixtureCollection with known membership set. For each collection
and each of k runs, we measure whether the recall path retrieves each member.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import asyncpg

from weft.topic_gather import gather_topic_memories
from weft.auth import current_user_id
from weft.db.connection import acquire

logger = logging.getLogger(__name__)


@dataclass
class RecallStats:
    """Recall@membership statistics for a single fixture collection."""
    collection_name: str
    topic_tag: str
    expected_members: int

    # Oracle path (single run, deterministic)
    oracle_returned: int
    oracle_recall_at_membership: float  # Should always be 1.0

    # Candidate path (k runs, stochastic)
    candidate_runs: int
    candidate_recalls_per_run: list[float]

    @property
    def candidate_min_recall(self) -> float:
        """Minimum recall@membership across k runs."""
        return min(self.candidate_recalls_per_run) if self.candidate_recalls_per_run else 0.0

    @property
    def candidate_median_recall(self) -> float:
        """Median recall@membership across k runs."""
        return median(self.candidate_recalls_per_run) if self.candidate_recalls_per_run else 0.0

    @property
    def candidate_max_recall(self) -> float:
        """Maximum recall@membership across k runs."""
        return max(self.candidate_recalls_per_run) if self.candidate_recalls_per_run else 0.0

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "collection": self.collection_name,
            "topic_tag": self.topic_tag,
            "expected_members": self.expected_members,
            "oracle": {
                "returned": self.oracle_returned,
                "recall_at_membership": self.oracle_recall_at_membership,
            },
            "candidate": {
                "runs": self.candidate_runs,
                "min_recall": self.candidate_min_recall,
                "median_recall": self.candidate_median_recall,
                "max_recall": self.candidate_max_recall,
                "spread": self.candidate_max_recall - self.candidate_min_recall,
            },
        }


async def enum_gather_oracle(
    pool: asyncpg.Pool,
    topic_tag: str,
    user_id: str,
    expected_members: list[str],
) -> tuple[list[str], float]:
    """Oracle path: gather_topic_memories returns ALL M members (deterministic).

    Args:
        pool: asyncpg pool.
        topic_tag: Topic tag to gather on.
        user_id: User identity for RLS.
        expected_members: Known membership set (for recall calculation).

    Returns:
        (returned_contents, recall_at_membership).
        Recall is |returned ∩ expected| / |expected|.
    """
    token = current_user_id.set(user_id)
    try:
        result = await gather_topic_memories(pool, [topic_tag], user_id)
        returned_memories = result["memories"]
        returned_contents = [m.content for m in returned_memories]

        # Calculate recall@membership: intersection / expected cardinality
        intersection = set(returned_contents) & set(expected_members)
        recall_at_membership = len(intersection) / len(expected_members)

        logger.info(
            "enum_gather_oracle: topic=%s returned=%d expected=%d "
            "intersection=%d recall=%.2f",
            topic_tag,
            len(returned_contents),
            len(expected_members),
            len(intersection),
            recall_at_membership,
        )

        return returned_contents, recall_at_membership
    finally:
        current_user_id.reset(token)


async def enum_recall_candidate(
    pool: asyncpg.Pool,
    topic_tag: str,
    user_id: str,
    expected_members: list[str],
    k_runs: int = 5,
) -> list[float]:
    """Candidate path: weft_recall stochastic, run k times, measure spread.

    Models the natural-language recall path an agent actually uses: each of the
    k runs phrases the enumeration request DIFFERENTLY (the way a human/agent
    would ask "list Jim's plants" vs "what plants does Jim have"), embeds that
    phrasing, and runs hybrid (semantic + keyword) recall.

    Two design choices make the spread a real "never-miss signal" (PRD §V4)
    rather than a degenerate constant:

    1. **Per-run phrasing variation.** A single fixed query embeds to one
       vector and ``search_hybrid`` is deterministic, so identical queries
       would give 0 spread by construction — an uninformative meter. Distinct
       phrasings produce distinct rankings, so the run-to-run variance reflects
       how sensitive recall is to how the question is asked.
    2. **Realistic retrieval pressure.** ``limit`` is set to the membership
       cardinality M (the natural enumeration target — "give me all M of my
       plants"). To score 1.0 the path must rank ALL M members into its top-M
       against the full multi-collection corpus. A limit far above the corpus
       size would trivially return everything every run (recall always 1.0,
       spread always 0) — the same class of degenerate meter as zero vectors.

    Args:
        pool: asyncpg pool.
        topic_tag: Topic tag to query (as natural language or direct semantic).
        user_id: User identity for RLS.
        expected_members: Known membership set (for recall calculation).
        k_runs: Number of stochastic runs (default 5).

    Returns:
        List of recall@membership scores, one per run.
    """
    from weft.store import search_hybrid
    from weft.embeddings import get_provider

    # Natural-language phrasing bank — the way an agent might ask to enumerate
    # a collection. Cycled across the k runs so each run is a distinct query.
    phrasing_templates = [
        "Jim Boblaw's {t}",
        "What {t} does Jim Boblaw have?",
        "List all of Jim Boblaw's {t}",
        "Tell me everything about Jim Boblaw's {t}",
        "Enumerate every one of Jim Boblaw's {t}",
        "Which {t} has Jim Boblaw mentioned?",
        "Give me the complete set of Jim Boblaw's {t}",
    ]

    # Retrieval pressure: ask for exactly the membership cardinality. To reach
    # recall 1.0 the NL path must rank all M members into its top-M.
    candidate_limit = len(expected_members)

    token = current_user_id.set(user_id)
    try:
        # Get embedder for semantic search
        embedder = get_provider("fastembed", dimensions=768)

        recalls_per_run = []
        for run_idx in range(k_runs):
            # Distinct phrasing per run — cycle the bank if k exceeds its size.
            query = phrasing_templates[run_idx % len(phrasing_templates)].format(
                t=topic_tag
            )

            # Embed the query for semantic search
            embedding = await embedder.embed(query)

            # Hybrid search: combines semantic + keyword matching
            try:
                results = await search_hybrid(
                    pool,
                    query=query,
                    embedding=embedding,
                    limit=candidate_limit,  # = M, realistic enumeration target
                    project_id=None,  # Search across all projects
                    threshold=0.0,  # Accept even low-confidence results
                )

                # Extract returned contents from results
                returned_contents = [r.memory.content for r in results]

                # Calculate recall@membership: intersection / expected cardinality
                intersection = set(returned_contents) & set(expected_members)
                recall_at_membership = len(intersection) / len(expected_members)
                recalls_per_run.append(recall_at_membership)

                logger.info(
                    "enum_recall_candidate run %d/%d: query=%r limit=%d returned=%d "
                    "expected=%d intersection=%d recall=%.2f",
                    run_idx + 1,
                    k_runs,
                    query,
                    candidate_limit,
                    len(returned_contents),
                    len(expected_members),
                    len(intersection),
                    recall_at_membership,
                )
            except Exception as exc:
                logger.warning(
                    "enum_recall_candidate run %d: query=%r error=%s",
                    run_idx + 1,
                    query,
                    exc,
                )
                recalls_per_run.append(0.0)

        return recalls_per_run
    finally:
        current_user_id.reset(token)


async def run_enumeration_eval(
    pool: asyncpg.Pool,
    fixtures: dict[str, "FixtureCollection"],
    user_id: str,
) -> list[RecallStats]:
    """Run enumeration eval harness over all fixtures.

    Args:
        pool: asyncpg pool.
        fixtures: Dict of category.value -> FixtureCollection.
        user_id: User identity for all operations.

    Returns:
        List of RecallStats, one per fixture collection.
    """
    results = []

    for category_key, collection in fixtures.items():
        logger.info(
            "run_enumeration_eval: collection=%s topic=%s members=%d",
            category_key,
            collection.topic_tag,
            len(collection.members),
        )

        # Oracle path: single deterministic run
        oracle_returned, oracle_recall = await enum_gather_oracle(
            pool,
            topic_tag=collection.topic_tag,
            user_id=user_id,
            expected_members=collection.members,
        )

        # Candidate path: k stochastic runs
        candidate_recalls = await enum_recall_candidate(
            pool,
            topic_tag=collection.topic_tag,
            user_id=user_id,
            expected_members=collection.members,
            k_runs=5,
        )

        stats = RecallStats(
            collection_name=category_key,
            topic_tag=collection.topic_tag,
            expected_members=len(collection.members),
            oracle_returned=len(oracle_returned),
            oracle_recall_at_membership=oracle_recall,
            candidate_runs=len(candidate_recalls),
            candidate_recalls_per_run=candidate_recalls,
        )

        results.append(stats)
        logger.info("RecallStats: %s", stats.to_dict())

    return results


async def run_minted_case_eval(
    pool: asyncpg.Pool,
    user_id: str,
    *,
    path: "Path | str | None" = None,
    top_k: int = 10,
    embedder: Any = None,
) -> list[dict]:
    """Exercise minted known-answer eval cases from the JSONL store.

    Reads every case from the minted_cases JSONL and runs ``search_by_vector``
    for each query, checking whether ``satisfying_memory_id`` appears in the
    top-K results.  Returns one result dict per case.

    This is the "Feedback" leg of the CL1 compounding loop (loom-add4d5c8):
    Signal(miss) → Store(eval case) → **Feedback(harness exercises it)** → Proof.

    Args:
        pool: asyncpg pool.
        user_id: User identity for RLS scoping of ``search_by_vector``.
        path: Path to the JSONL store.  Defaults to
            ``benchmarks.enumeration_eval.mint.DEFAULT_MINTED_CASES_PATH``.
        top_k: Number of results to retrieve per case; the satisfying memory
            must appear within this many results to count as a hit.
        embedder: Embedding provider (``async embed(text) -> list[float]``).
            Defaults to ``FastEmbedProvider`` (768-dim).

    Returns:
        List of dicts, one per minted case::

            {
                "query": str,
                "satisfying_memory_id": str,
                "source": str,          # "canary" | "reask"
                "hit": bool,            # True if satisfying_memory_id in top_k
            }

        Empty list if the JSONL store has no cases.
    """
    from benchmarks.enumeration_eval.mint import load_minted_cases
    from weft.store import search_by_vector
    from weft.embeddings import get_provider

    cases = load_minted_cases(path)
    if not cases:
        logger.debug("run_minted_case_eval: 0 minted cases found (path=%s)", path)
        return []

    if embedder is None:
        embedder = get_provider("fastembed", dimensions=768)

    results: list[dict] = []
    for case in cases:
        query: str = case["query"]
        satisfying_memory_id: str = case["satisfying_memory_id"]

        try:
            embedding = await embedder.embed(query)
            search_results = await search_by_vector(
                pool,
                embedding,
                limit=top_k,
                threshold=0.0,
                user_id=user_id,
            )
            result_ids = {r.memory.id for r in search_results}
            hit = satisfying_memory_id in result_ids
        except Exception as exc:
            logger.warning(
                "run_minted_case_eval: error for query=%r memory_id=%s: %s",
                query,
                satisfying_memory_id,
                exc,
            )
            hit = False

        logger.info(
            "run_minted_case_eval: query=%r satisfying_memory_id=%s hit=%s",
            query[:60],
            satisfying_memory_id,
            hit,
        )
        results.append(
            {
                "query": query,
                "satisfying_memory_id": satisfying_memory_id,
                "source": case.get("source", "unknown"),
                "hit": hit,
            }
        )

    return results
