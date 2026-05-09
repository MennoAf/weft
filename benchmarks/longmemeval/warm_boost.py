"""Pre-warm the turn-tier boost loop for benchmark runs (P1.A5).

The cold-DB LongMemEval harness ingests each question's haystack into a
fresh project sandbox and queries against it once. Turns enter at the
default ``usefulness_score = 0.7`` and ``last_boosted_at = None`` —
which makes the P1.A3 rerank's usefulness factor a constant ~0.85
across all turns. Constant factors cancel in ranking, so the rerank's
boost-loop signal is unexercised.

This module simulates N rounds of recall + access-log + boost so the
rerank has divergent usefulness scores to work with at query time.

Recall-driven, not gold-keyed: warmup queries come from the corpus
itself (random sampled turn content), not from question-answer evidence.
That keeps the warmup unbiased w.r.t. which turns are "right" — the
boost is pumped wherever the recall layer keeps surfacing the same
turns under random queries, which is the production loop's behavior.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import asyncpg

from weft.episode_turns import recall_turns
from weft.session_tracking import boost_session_turns, log_turn_access

if TYPE_CHECKING:
    from weft.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


_DEFAULT_QUERIES_PER_ROUND = 10
_QUERY_PREFIX_CHARS = 120


async def warm_boost_turns(
    pool: asyncpg.Pool,
    embedder: "EmbeddingProvider",
    *,
    project_id: str,
    rounds: int,
    queries_per_round: int = _DEFAULT_QUERIES_PER_ROUND,
    top_k: int = 10,
    rng_seed: int = 0,
) -> dict:
    """Run ``rounds`` warmup passes against the project's turns.

    Each round:
      1. Sample ``queries_per_round`` random turns from the project.
      2. For each sampled turn, embed its content prefix and call
         :func:`recall_turns` with that query.
      3. Log every recalled turn id via :func:`log_turn_access` under a
         per-round session id.
      4. Apply :func:`boost_session_turns` for that session id, which
         dedupes by turn and bumps ``usefulness_score`` +
         ``last_boosted_at`` for every accessed turn.

    Returns a telemetry dict::

        {"rounds": int, "queries": int, "accessed_turns": int,
         "boosted_turns": int, "n_population": int}

    The number of boosts is upper-bounded by the number of distinct
    turns the recall layer ever returns across all rounds. Turns that
    never surface stay at the ingest default — which is the desired
    asymmetry: the rerank gets divergent signal across the population.
    """
    if rounds <= 0:
        return {
            "rounds": 0, "queries": 0, "accessed_turns": 0,
            "boosted_turns": 0, "n_population": 0,
        }

    sql = """
        SELECT t.id, t.content
          FROM episode_turns t
          JOIN episodes e ON e.id = t.episode_id
         WHERE e.project_id = $1
           AND t.content IS NOT NULL
           AND length(t.content) > 0
    """
    rows = await pool.fetch(sql, project_id)
    if not rows:
        logger.debug("warm_boost: no turns for project_id=%s", project_id)
        return {
            "rounds": rounds, "queries": 0, "accessed_turns": 0,
            "boosted_turns": 0, "n_population": 0,
        }

    population = [(r["id"], r["content"]) for r in rows]
    rng = random.Random(rng_seed)

    total_queries = 0
    total_accessed = 0
    total_boosted = 0

    for round_idx in range(rounds):
        sample = rng.sample(population, min(queries_per_round, len(population)))
        session_id = f"warmboost-{project_id}-r{round_idx}"

        for _, content in sample:
            query = content[:_QUERY_PREFIX_CHARS].strip()
            if not query:
                continue
            try:
                embedding = await embedder.embed(query)
            except Exception as e:
                logger.warning("warm_boost embed failed: %s", e)
                continue

            try:
                recalled = await recall_turns(
                    pool, query,
                    project_id=project_id,
                    top_k=top_k,
                    embedding=embedding,
                )
            except Exception as e:
                # PG FTS can throw on pathological inputs (e.g. "tsquery
                # stack too small" when a content-prefix sample parses
                # into a deeply-nested tsquery). One bad warmup query
                # must not take down the whole benchmark question — the
                # rest of the round can still produce signal.
                logger.warning("warm_boost recall_turns failed: %s", e)
                continue
            total_queries += 1
            if recalled:
                try:
                    await log_turn_access(
                        pool,
                        [t.id for t in recalled],
                        tool_name="warm_boost",
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.warning("warm_boost log_turn_access failed: %s", e)
                    continue
                total_accessed += len(recalled)

        boost_result = await boost_session_turns(pool, session_id=session_id)
        total_boosted += int(boost_result.get("boosted", 0))

    return {
        "rounds": rounds,
        "queries": total_queries,
        "accessed_turns": total_accessed,
        "boosted_turns": total_boosted,
        "n_population": len(population),
    }
