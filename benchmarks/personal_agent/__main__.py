#!/usr/bin/env python3
"""
__main__.py — Runnable entrypoint for the PAAH enumeration harness.

Usage:
    uv run python -m benchmarks.personal_agent

Requires Docker (testcontainers: Postgres pgvector + Redis). Seeds the
known-cardinality manifest through the real ``weft_remember`` write path, drives
``weft_recall`` with enumeration queries, and writes ``results.json`` — the
repeatable scoreboard number the project has lacked since the 2026-05-03
LongMemEval baseline.

Exit code:
    0 — seed integrity held AND every collection's enumeration answer was
        correct: the explicit count fields (response["count"] and
        enumeration.count) matched the manifest and enumeration.members covered
        the full membership on every run.
    1 — an explicit count field was wrong, the members list was incomplete, or
        the seed cardinality was corrupted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Same clean-slate table list the enumeration_eval harness truncates.
_TRUNCATE = (
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


async def main() -> int:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer

    from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
    from weft.db.migrations import run_migrations
    from benchmarks.personal_agent.manifest import PAAH_USER_ID
    from benchmarks.personal_agent.seed import (
        seed_corpus, seed_turns, seed_entity_brief, seed_agenda,
    )
    from benchmarks.personal_agent.harness import run_enumeration_paah
    from benchmarks.personal_agent.temporal_harness import run_temporal_paah
    from benchmarks.personal_agent.entity_harness import run_entity_brief_paah
    from benchmarks.personal_agent.agenda_harness import run_agenda_paah

    logger.info("=" * 70)
    logger.info(
        "PAAH — Personal-Agent Acceptance Harness "
        "(enumeration + temporal + entity + agenda)"
    )
    logger.info("=" * 70)

    logger.info("Starting testcontainers (Postgres pgvector + Redis)...")
    pg_container = PostgresContainer("pgvector/pgvector:pg16")
    redis_container = RedisContainer("redis:7-alpine")
    pg_container.start()
    redis_container.start()

    try:
        dsn = pg_container.get_connection_url().replace("+psycopg2", "")

        async def _test_init(conn):
            await _pgvector_codec_init(conn)

        async def _test_setup(conn):
            await conn.execute(f"SET app.user_id = '{PAAH_USER_ID}'")

        pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=5, init=_test_init, setup=_test_setup,
        )
        try:
            logger.info("Running migrations...")
            await run_migrations(pool)
            await register_pgvector_codec(pool)

            logger.info("Truncating tables (clean slate)...")
            await pool.execute(_TRUNCATE)

            logger.info("Seeding PAAH manifest through the real write path...")
            seed_results = await seed_corpus(pool)

            # Ground-truth integrity gate: if dedup collapsed a collection, the
            # manifest count is no longer trustworthy — say so loudly.
            seed_ok = all(sr.clean for sr in seed_results)
            for sr in seed_results:
                logger.info(
                    "  seed %-12s intended=%2d stored=%2d clean=%s%s",
                    sr.collection.name, sr.intended, sr.stored, sr.clean,
                    "" if sr.clean else f"  COLLISIONS={len(sr.dedup_collisions)}",
                )

            logger.info("Running enumeration recall over seeded corpus...")
            stats = await run_enumeration_paah(pool, seed_results, limit=10)

            # ---- Headline scoreboard ------------------------------------
            # The misread-proof number: across every (collection, run), how
            # often does each answer signal equal the truth?
            #   obvious_count = response["count"]                (the fix)
            #   enum_count    = response["enumeration"]["count"] (explicit)
            #   naive_results = len(results[])                   (legacy slice)
            total_runs = sum(len(s.runs) for s in stats)
            obvious_correct = sum(
                1 for s in stats for r in s.runs if r.obvious_count == s.manifest_count
            )
            enum_correct = sum(
                1 for s in stats for r in s.runs if r.enum_count == s.manifest_count
            )
            naive_correct = sum(
                1 for s in stats for r in s.runs if r.naive_results == s.manifest_count
            )
            # Render altitude: the answer as the agent READS it, not the dict.
            render_legacy_wrong = sum(
                1 for s in stats for r in s.runs
                if r.render_legacy_rows != s.manifest_count
            )
            render_enum_complete = sum(
                1 for s in stats for r in s.runs
                if r.render_enum_rows == s.manifest_count and r.render_count_shown
            )

            # ---- Report --------------------------------------------------
            logger.info("=" * 70)
            logger.info("PAAH ENUMERATION RESULTS")
            logger.info("=" * 70)
            router_all_correct = True
            for s in stats:
                logger.info("")
                logger.info("Collection: %s (tag=%s)", s.name, s.topic_tag)
                logger.info("  Manifest count (ground truth): %d", s.manifest_count)
                logger.info("  Recall limit:                  %d", s.limit)
                logger.info(
                    "  enumeration block present:     %.0f%%",
                    s.enum_present_rate * 100,
                )
                logger.info(
                    "  correct-count rate — obvious response['count']: %.0f%% | "
                    "explicit enumeration.count: %.0f%% | naive len(results): %.0f%%",
                    s.obvious_count_correct_rate * 100,
                    s.enum_count_correct_rate * 100,
                    s.naive_correct_rate * 100,
                )
                logger.info(
                    "  recall@membership (enumeration.members): min=%.3f median=%.3f max=%.3f",
                    s.recall_min, s.recall_median, s.recall_max,
                )
                logger.info(
                    "  RENDER (agent-facing) — legacy results-only render wrong: "
                    "%.0f%% (finding) | enum-aware render complete: %.0f%% | "
                    "count readable in header: %.0f%% | summary carries count: %.0f%%",
                    s.render_legacy_wrong_rate * 100,
                    s.render_close_rate * 100,
                    s.render_count_shown_rate * 100,
                    s.summary_shows_count_rate * 100,
                )
                if not (
                    s.enum_count_correct_rate == 1.0
                    and s.obvious_count_correct_rate == 1.0
                    and s.recall_min == 1.0
                    # Loop-closer: the answer must survive rendering, not just
                    # be correct in the response dict.
                    and s.render_close_rate == 1.0
                    and s.render_count_shown_rate == 1.0
                    and s.render_legacy_wrong_rate == 1.0
                ):
                    router_all_correct = False

            # ---- Verdict -------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("VERDICT")
            logger.info("=" * 70)
            logger.info("Seed integrity (all collections clean): %s", seed_ok)
            logger.info("Enumeration answer correct everywhere:  %s", router_all_correct)
            logger.info(
                "SCOREBOARD — correct count via obvious response['count']: %d/%d | "
                "explicit enumeration.count: %d/%d | naive len(results): %d/%d",
                obvious_correct, total_runs, enum_correct, total_runs,
                naive_correct, total_runs,
            )
            contract_closed = (
                obvious_correct == total_runs and enum_correct == total_runs
            )
            if contract_closed:
                logger.info(
                    "CONSUMPTION CONTRACT: CLOSED — the answer is now an explicit "
                    "field the agent KNOWS (response['count'] and "
                    "enumeration.count both correct on every run); the complete "
                    "list is in enumeration.members. The legacy len(results) "
                    "slice stays wrong (%d/%d), which is exactly why the explicit "
                    "fields matter.", naive_correct, total_runs,
                )
            else:
                logger.info(
                    "CONSUMPTION CONTRACT: STILL OPEN — an explicit count field "
                    "was wrong on some run; investigate."
                )

            # ---- Render-altitude verdict (the loop-closer) --------------
            render_contract_closed = (
                render_enum_complete == total_runs
                and render_legacy_wrong == total_runs
            )
            logger.info(
                "RENDER SCOREBOARD — enum-aware render complete+readable: %d/%d | "
                "legacy results-only render wrong: %d/%d (the gap it closes)",
                render_enum_complete, total_runs, render_legacy_wrong, total_runs,
            )
            if render_contract_closed:
                logger.info(
                    "RENDER CONTRACT: CLOSED — the answer survives into the text "
                    "an agent actually reads (format_recall_context surfaces the "
                    "corrected count + complete membership on every run); the "
                    "legacy results-only render stays wrong (%d/%d), which is "
                    "exactly the gap the enum-aware render closes.",
                    render_legacy_wrong, total_runs,
                )
            else:
                logger.info(
                    "RENDER CONTRACT: STILL OPEN — the enum-aware render dropped "
                    "the answer on some run; investigate format_recall_context."
                )

            # ============ SHAPE 2: temporal / dialogue (turn-tier) ============
            logger.info("")
            logger.info("Seeding temporal dialogue trace (real weft_turn_append)...")
            seeded_turns = await seed_turns(pool)
            logger.info(
                "  turns intended=%d stored=%d clean=%s",
                seeded_turns.intended, seeded_turns.stored, seeded_turns.clean,
            )
            logger.info("Running temporal/dialogue probes over the trace...")
            turn_stats = await run_temporal_paah(pool, seeded_turns)

            t_total = sum(len(s.runs) for s in turn_stats)
            t_routed_ok = sum(
                1 for s in turn_stats for r in s.runs if r.routed_correct
            )
            t_anchor_hit = sum(
                1 for s in turn_stats for r in s.runs if r.anchor_present
            )

            logger.info("=" * 70)
            logger.info("PAAH TEMPORAL/DIALOGUE RESULTS (Branch-A turn-tier probe)")
            logger.info("=" * 70)
            temporal_all_correct = seeded_turns.clean
            for s in turn_stats:
                logger.info("")
                logger.info("Probe: %s (want tier=%s, anchor=%s)",
                            s.key, s.expected_tier, s.anchor_key)
                logger.info("  routed to expected tier: %.0f%%",
                            s.routed_correct_rate * 100)
                logger.info(
                    "  anchor-turn present: rate=%.0f%% min=%.0f (never-miss floor)",
                    s.anchor_present_rate * 100, s.anchor_present_min,
                )
                if not (s.routed_correct_rate == 1.0 and s.anchor_present_min == 1.0):
                    temporal_all_correct = False
            logger.info("")
            logger.info(
                "TEMPORAL SCOREBOARD — routed correctly: %d/%d | anchor surfaced: %d/%d",
                t_routed_ok, t_total, t_anchor_hit, t_total,
            )
            logger.info(
                "TURN-TIER (Branch-A): %s",
                "ANSWERS — anchor surfaced on every probe/phrasing"
                if t_anchor_hit == t_total
                else f"GAP — anchor missed on {t_total - t_anchor_hit}/{t_total} runs",
            )

            # ============ SHAPE 3: entity brief (beliefs + graph) ============
            logger.info("")
            logger.info("Seeding entity brief (weft_entity_create + link)...")
            seeded_entity = await seed_entity_brief(pool)
            entity_stats = await run_entity_brief_paah(pool, seeded_entity)

            logger.info("=" * 70)
            logger.info("PAAH ENTITY-BRIEF RESULTS (beliefs + graph)")
            logger.info("=" * 70)
            logger.info("Entity: %s (%d linked facts)",
                        entity_stats.name, entity_stats.fact_count)
            logger.info(
                "  ORACLE weft_entity_context recall@links: %.3f (complete=%s)",
                entity_stats.oracle_recall, entity_stats.oracle_complete,
            )
            logger.info(
                "  CANDIDATE weft_recall recall@links: min=%.3f median=%.3f max=%.3f",
                entity_stats.candidate_min, entity_stats.candidate_median,
                entity_stats.candidate_max,
            )
            for c in entity_stats.candidate:
                logger.info(
                    "    routed=%-6s fell_back=%-5s recall@links=%.3f  %s",
                    c.routed_tier or "belief", str(c.fell_back),
                    c.recall_at_links, c.query,
                )
            if entity_stats.fallbacks:
                logger.info(
                    "  NEVER-MISS: %d/%d brief phrasings routed to the turns tier "
                    "(temporal word) and RECOVERED via the empty-tier belief "
                    "fallback — no phrasing returned empty (min recall@links=%.3f).",
                    len(entity_stats.fallbacks), len(entity_stats.candidate),
                    entity_stats.candidate_min,
                )
            entity_ok = (
                seeded_entity.brief.clean
                and entity_stats.oracle_complete
                and entity_stats.never_empty
            )

            # ============ SHAPE 4: agenda (trackers / open loops) ============
            logger.info("")
            logger.info("Seeding agenda trackers (real tracker lifecycle path)...")
            seeded_agenda = await seed_agenda(pool)
            logger.info(
                "  agenda seeded=%d due=%d excluded=%d clean=%s",
                seeded_agenda.intended, len(seeded_agenda.due_ids),
                len(seeded_agenda.excluded_ids), seeded_agenda.clean,
            )
            agenda_stats = await run_agenda_paah(pool, seeded_agenda)

            logger.info("=" * 70)
            logger.info("PAAH AGENDA RESULTS (trackers / open loops)")
            logger.info("=" * 70)
            logger.info(
                "  ORACLE weft_tracker_due: recall@due=%.3f (complete=%s) "
                "precise=%s (leaks=%d) keep-pushing-first=%s",
                agenda_stats.oracle_recall, agenda_stats.oracle_complete,
                agenda_stats.oracle_precise, len(agenda_stats.oracle_leaks),
                agenda_stats.keep_pushing_first,
            )
            logger.info(
                "  AGENT-FACING weft_daily_brief: open-loop coverage=%d/%d (%.0f%%) "
                "surfaces_open_loops=%s",
                agenda_stats.brief_due_covered, agenda_stats.due_expected,
                agenda_stats.brief_coverage * 100,
                agenda_stats.brief_surfaces_open_loops,
            )
            if not agenda_stats.brief_surfaces_open_loops:
                logger.info(
                    "  FINDING: the due-loop query answers correctly, but the "
                    "agent-facing daily brief surfaces %d/%d open loops — "
                    "weft_tracker_due's docstring points at a daily-brief "
                    "'open loops' section that assemble_daily_brief does not build "
                    "(sections present: %s). Consumption-contract gap, same family "
                    "as the enumeration results[] slice.",
                    agenda_stats.brief_due_covered, agenda_stats.due_expected,
                    agenda_stats.brief_sections or "[]",
                )
            agenda_ok = (
                seeded_agenda.clean
                and agenda_stats.oracle_complete
                and agenda_stats.oracle_precise
                and agenda_stats.keep_pushing_first
            )
            logger.info(
                "AGENDA SHAPE: %s",
                "ANSWERS — due query complete, precise, keep-pushing surfaces first"
                if agenda_ok
                else "GAP — oracle due query is incomplete/imprecise/misordered",
            )

            output_path = Path(__file__).parent / "results.json"
            output_data = {
                "shapes": ["enumeration", "temporal"],
                "enumeration": {
                    "limit": 10,
                    "seed_integrity_clean": seed_ok,
                    "answer_correct_everywhere": router_all_correct,
                    "consumption_contract_closed": contract_closed,
                    "render_contract_closed": render_contract_closed,
                    "scoreboard": {
                        "total_runs": total_runs,
                        "correct_via_obvious_response_count": obvious_correct,
                        "correct_via_explicit_enumeration_count": enum_correct,
                        "correct_via_naive_len_results": naive_correct,
                        "render_enum_complete_and_readable": render_enum_complete,
                        "render_legacy_wrong": render_legacy_wrong,
                    },
                    "collections": [s.to_dict() for s in stats],
                    "seed": [
                        {
                            "collection": sr.collection.name,
                            "intended": sr.intended,
                            "stored": sr.stored,
                            "clean": sr.clean,
                            "dedup_collisions": sr.dedup_collisions,
                        }
                        for sr in seed_results
                    ],
                },
                "temporal": {
                    "limit": turn_stats[0].limit if turn_stats else None,
                    "seed_turns": {
                        "intended": seeded_turns.intended,
                        "stored": seeded_turns.stored,
                        "clean": seeded_turns.clean,
                    },
                    "turn_tier_answers": temporal_all_correct,
                    "scoreboard": {
                        "total_runs": t_total,
                        "routed_correct": t_routed_ok,
                        "anchor_surfaced": t_anchor_hit,
                    },
                    "probes": [s.to_dict() for s in turn_stats],
                },
                "entity_brief": {
                    "seed_brief_clean": seeded_entity.brief.clean,
                    "oracle_complete": entity_stats.oracle_complete,
                    "belief_routed_all_complete": all(
                        c.recall_at_links == 1.0
                        for c in entity_stats.candidate if not c.fell_back
                    ),
                    "recovered_via_fallback": len(entity_stats.fallbacks),
                    "finding": (
                        "temporal-worded briefs ('before our meeting') route off "
                        "the belief tier to the turns tier and recover via the "
                        "empty-tier belief fallback (never-miss); none return empty"
                        if entity_stats.fallbacks else None
                    ),
                    "stats": entity_stats.to_dict(),
                },
                "agenda": {
                    "seed_clean": seeded_agenda.clean,
                    "due_expected": agenda_stats.due_expected,
                    "oracle_complete": agenda_stats.oracle_complete,
                    "oracle_precise": agenda_stats.oracle_precise,
                    "keep_pushing_first": agenda_stats.keep_pushing_first,
                    "agenda_answers": agenda_ok,
                    "agent_facing_surfaces_open_loops": (
                        agenda_stats.brief_surfaces_open_loops
                    ),
                    "finding": (
                        "weft_tracker_due answers the plate correctly, but the "
                        "agent-facing daily brief surfaces "
                        f"{agenda_stats.brief_due_covered}/{agenda_stats.due_expected} "
                        "open loops — assemble_daily_brief has no trackers section "
                        "despite weft_tracker_due's docstring pointing at one"
                        if not agenda_stats.brief_surfaces_open_loops else None
                    ),
                    "stats": agenda_stats.to_dict(),
                },
            }
            output_data["shapes"] = [
                "enumeration", "temporal", "entity_brief", "agenda",
            ]
            output_path.write_text(json.dumps(output_data, indent=2))
            logger.info("Results saved to: %s", output_path)

            all_clean = (
                seed_ok
                and router_all_correct
                and temporal_all_correct
                and entity_ok
                and agenda_ok
            )
            return 0 if all_clean else 1
        finally:
            await pool.close()
    finally:
        pg_container.stop()
        redis_container.stop()
        logger.info("Testcontainers stopped.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
