#!/usr/bin/env python3
"""
__main__.py — Runnable entrypoint for enumeration_eval harness.

Usage:
    uv run python -m benchmarks.enumeration_eval

Requires Docker running for testcontainers (Postgres pgvector + Redis).
Reuses the test suite's conftest.py for pool/redis fixtures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the enumeration_eval harness end-to-end."""
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer

    from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
    from weft.db.migrations import run_migrations
    from benchmarks.enumeration_eval.fixtures import (
        seed_fixtures,
        JIM_BOBLAW_USER_ID,
        JIM_BOBLAW_PROJECT_ID,
    )
    from benchmarks.enumeration_eval.harness import run_enumeration_eval

    logger.info("="*70)
    logger.info("Enumeration Eval Harness")
    logger.info("="*70)

    # Start testcontainers
    logger.info("Starting testcontainers (Postgres pgvector + Redis)...")
    pg_container = PostgresContainer("pgvector/pgvector:pg16")
    redis_container = RedisContainer("redis:7-alpine")
    pg_container.start()
    redis_container.start()

    try:
        # Create pool
        dsn = pg_container.get_connection_url().replace("+psycopg2", "")

        async def _test_init(conn):
            """pgvector codec init."""
            from weft.db.connection import _pgvector_codec_init
            await _pgvector_codec_init(conn)

        async def _test_setup(conn):
            """Set app.user_id for RLS."""
            await conn.execute(f"SET app.user_id = '{JIM_BOBLAW_USER_ID}'")

        pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=5, init=_test_init, setup=_test_setup,
        )

        try:
            # Run migrations
            logger.info("Running migrations...")
            await run_migrations(pool)
            await register_pgvector_codec(pool)

            # Clean slate
            logger.info("Truncating tables...")
            await pool.execute(
                "TRUNCATE memory_access_log, turn_access_log, entity_mentions, "
                "belief_claims, shuttle_claims, episode_turns, episode_memories, "
                "memory_relationships, entities, episodes, memories, behaviors, "
                "weft_metadata, modes, alerts, alert_state, check_ins, "
                "autonomy_policies, policy_calibration_events, autonomy_overrides, "
                "cost_entries, cost_enforcement_state, triggers, calibration_records, "
                "degradation_policies, audit_backfill_user_id, workspace_members, "
                "workspaces, trackers, weft_tokens, weft_recall_queries, "
                "replay_queue, weft_counters, topic_digests, "
                "topic_resolution_aliases CASCADE"
            )

            # Seed Jim Boblaw fixtures
            logger.info("Seeding Jim Boblaw fixtures...")
            fixtures = await seed_fixtures(
                pool,
                user_id=JIM_BOBLAW_USER_ID,
                project_id=JIM_BOBLAW_PROJECT_ID,
            )
            logger.info("Seeded %d fixture collections", len(fixtures))

            # Run enumeration eval
            logger.info("Running enumeration evaluation...")
            results = await run_enumeration_eval(pool, fixtures, JIM_BOBLAW_USER_ID)

            # Print report
            logger.info("="*70)
            logger.info("ENUMERATION EVAL RESULTS")
            logger.info("="*70)

            for result in results:
                stats_dict = result.to_dict()
                logger.info("")
                logger.info("Collection: %s", result.collection_name)
                logger.info("  Topic: %s", result.topic_tag)
                logger.info("  Expected members: %d", result.expected_members)
                logger.info("")
                logger.info("  Oracle Path (Deterministic):")
                logger.info("    Returned: %d", result.oracle_returned)
                logger.info("    Recall@membership: %.4f", result.oracle_recall_at_membership)
                logger.info("")
                logger.info("  Candidate Path (Stochastic, k=%d runs):", result.candidate_runs)
                logger.info("    Min recall@membership:    %.4f", result.candidate_min_recall)
                logger.info("    Median recall@membership: %.4f", result.candidate_median_recall)
                logger.info("    Max recall@membership:    %.4f", result.candidate_max_recall)
                logger.info("    Spread (max - min):       %.4f", result.candidate_max_recall - result.candidate_min_recall)

            # Validation: oracle MUST have recall == 1.0
            logger.info("")
            logger.info("="*70)
            logger.info("VALIDATION")
            logger.info("="*70)

            oracle_failures = [r for r in results if r.oracle_recall_at_membership < 1.0]
            if oracle_failures:
                logger.error("FAILED: Oracle path did not return all members for:")
                for r in oracle_failures:
                    logger.error(
                        "  %s (recall=%.4f, expected=%d, returned=%d)",
                        r.collection_name,
                        r.oracle_recall_at_membership,
                        r.expected_members,
                        r.oracle_returned,
                    )
                return 1
            else:
                logger.info("PASS: All oracle paths returned 100% recall@membership")

            # Summary report
            logger.info("")
            logger.info("="*70)
            logger.info("SUMMARY")
            logger.info("="*70)
            logger.info("Fixtures evaluated: %d", len(results))
            logger.info("Oracle consistency: 100%% (all collections recall=1.0)")
            avg_candidate_min = sum(r.candidate_min_recall for r in results) / len(results)
            avg_candidate_median = sum(r.candidate_median_recall for r in results) / len(results)
            avg_candidate_max = sum(r.candidate_max_recall for r in results) / len(results)
            logger.info("Candidate path average min recall:    %.4f", avg_candidate_min)
            logger.info("Candidate path average median recall: %.4f", avg_candidate_median)
            logger.info("Candidate path average max recall:    %.4f", avg_candidate_max)

            # JSON output for CI/CD consumption
            output_path = Path(__file__).parent / "results.json"
            output_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "oracle_consistency": 1.0,
                "candidate_avg_min_recall": avg_candidate_min,
                "candidate_avg_median_recall": avg_candidate_median,
                "candidate_avg_max_recall": avg_candidate_max,
                "collections": [r.to_dict() for r in results],
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info("Results saved to: %s", output_path)

            return 0

        finally:
            await pool.close()

    finally:
        pg_container.stop()
        redis_container.stop()
        logger.info("Testcontainers stopped.")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
