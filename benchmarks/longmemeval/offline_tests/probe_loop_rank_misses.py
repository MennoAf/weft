"""Standalone read-only rank probe for two confirmed LongMemEval-S misses.

Replays the exact answer-time ``weft_recall(tier=turns)`` queries for:

* ``0a995998`` (multi-session, gold 3) — both recall tool calls
* ``0ddfec37`` (knowledge-update, gold 15) — the single recall tool call

through the real retrieval path (``weft.turn_recall.temporal_anchor`` →
``weft.episode_turns.recall_turns`` → RRF → ``weft.relevance.rank_turns``),
capturing for every probe:

(a) raw vector candidates with cosine distances and pre-fusion ranks,
(b) raw keyword (BM25/FTS) candidates with ts_rank scores and pre-fusion ranks,
(c) fused RRF ranks (recomputed with the production ``_rrf_fuse_turn_rows``
    over the captured rows — same function, same arguments),
(d) the final window that reached the answerer (post-rerank), verified
    against the ids recorded in the run checkpoint.

For each gold-supporting turn it reports presence / rank / score in each
candidate list, its project-global rank (untruncated), and its margin to the
``top_k`` boundary, so the miss can be classified as:

(i)   never a candidate (absent from both raw halves),
(ii)  candidate but cut by top_k truncation,
(iii) candidate but ranked low by fusion / rerank.

Read-only contract: the probe runs SELECT statements only. It never writes,
never migrates, never embeds corpus rows (query embeddings only, computed
locally by FastEmbed). The database is the run's own disposable benchmark DB
pointed at by ``LONGMEMEVAL_DATABASE_URL``.

Usage (from the repo root):

    export LONGMEMEVAL_DATABASE_URL="postgresql://weft:weft_local@127.0.0.1:5433/lme_bench"
    uv run python benchmarks/longmemeval/offline_tests/probe_loop_rank_misses.py \
        [--out artifacts/loop-rank-probe-20260929]

The expected answer-time tool calls are pinned from
``artifacts/longmemeval-gpt6-full-s-turns/faithful/session-checkpoint.json``
(``evidence.<qid>.tool_results``); the probe fails loudly if the replayed
final window diverges from the checkpointed one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OWNER_ID = "faithful-gpt6-fulls-turns-20260929"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 768

# ---------------------------------------------------------------------------
# Pinned case definitions (from session-checkpoint.json evidence.tool_results)
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    {
        "case_id": "0a995998:tool_call_0",
        "question_id": "0a995998",
        "project_id": "longmemeval-0a995998",
        "tool_call_index": 0,
        "limit": 5,
        "query": "items of clothing pick up or return from a store clothing count",
        "checkpoint_returned_ids": [
            "et-37ebe585ea9a47659e06aef1cb36f369",
            "et-2524a9ef85de421288e38691c7530729",
        ],
    },
    {
        "case_id": "0a995998:tool_call_1",
        "question_id": "0a995998",
        "project_id": "longmemeval-0a995998",
        "tool_call_index": 1,
        "limit": 10,
        "query": "need to pick up from store return clothes clothing shopping item(s)",
        "checkpoint_returned_ids": [
            "et-37ebe585ea9a47659e06aef1cb36f369",
            "et-0e563b06af0b4807baf026367ad74849",
            "et-43434b8d452c40218d36524df2697819",
            "et-0185d42c8a82412db2997cecbd8189f4",
            "et-05877ac948684f41ba069362367ecf84",
        ],
    },
    {
        "case_id": "0ddfec37:tool_call_0",
        "question_id": "0ddfec37",
        "project_id": "longmemeval-0ddfec37",
        "tool_call_index": 0,
        "limit": 5,
        "query": "autographed baseballs collection first three months how many added",
        "checkpoint_returned_ids": [
            "et-3485c9c747874d1b9e5495835fc464fa",
            "et-11f48ab37ed944989a88338060468773",
        ],
    },
]

# Gold-supporting turns, pinned by DB id (resolved from the dataset answer
# sessions via source_session_id + content match on 2026-09-29).
GOLD_TURNS: dict[str, list[dict[str, str]]] = {
    "0a995998": [
        {
            "id": "et-37ebe585ea9a47659e06aef1cb36f369",
            "label": "boots_pickup (answer_afa9873b_1 t4, 16:19)",
            "note": "gold item 1: pick up new pair of Zara boots (exchange on 2/5)",
        },
        {
            "id": "et-95df0c34a1054029b174b0bd7d495c68",
            "label": "boots_return (answer_afa9873b_3 t6, 11:13)",
            "note": "gold item 1 (other half): return boots to Zara, exchanged for larger size",
        },
        {
            "id": "et-b12256b35af44ed28de8737983ef7a7f",
            "label": "blazer_pickup (answer_afa9873b_2 t10, 06:30)",
            "note": "gold item 2: pick up dry cleaning for navy blue blazer",
        },
        {
            "id": "et-b4b6b947173e4524b19d754c55ba63a1",
            "label": "green_sweater (answer_afa9873b_3 t2, 11:13)",
            "note": "gold item 3: green sweater lent to sister, awaiting return",
        },
    ],
    "0ddfec37": [
        {
            "id": "et-0a429ba0b44f4f5890a871d842740c25",
            "label": "gold_15 (answer_a22b654d_1 t0, 2023-07-11)",
            "note": "GOLD: 'that's 15 autographed baseballs since I started' — the time-scoped early count",
        },
        {
            "id": "et-6a08a2e1b85949fb88ebabaf0b19bede",
            "label": "later_20_user (answer_a22b654d_2 t0, 2023-12-30)",
            "note": "competing later fact: 'I have added 20 autographed baseballs to my collection'",
        },
        {
            "id": "et-11f48ab37ed944989a88338060468773",
            "label": "later_20_assistant (answer_a22b654d_2 t1, 2023-12-30)",
            "note": "competing later fact: assistant echoes 'adding 20 autographed baseballs in just a few months'",
        },
    ],
}


# ---------------------------------------------------------------------------
# Probe implementation
# ---------------------------------------------------------------------------


def _record_to_meta(row: Any, rank: int, half: str) -> dict[str, Any]:
    """Narrow an asyncpg candidate row to probe-relevant fields (no vectors)."""
    meta: dict[str, Any] = {
        "half": half,
        "pre_fusion_rank": rank,
        "id": str(row["id"]),
        "occurred_at": row["occurred_at"].isoformat(),
        "role": str(row["role"]),
        "source_session_id": row.get("source_session_id"),
        "content_head": " ".join(str(row["content"]).split())[:160],
    }
    try:
        meta["cosine_distance"] = float(row["_distance"])
    except (KeyError, IndexError):
        pass
    return meta


async def _tsrank_scores(
    conn: Any, query: str, ids: list[str]
) -> dict[str, float]:
    """Read-only ts_rank values for specific turn ids (production expression)."""
    if not ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT id,
               ts_rank(to_tsvector('english', content),
                       websearch_to_tsquery('english', $1)) AS score
          FROM episode_turns
         WHERE id = ANY($2::text[])
        """,
        query,
        ids,
    )
    return {str(r["id"]): float(r["score"]) for r in rows}


async def _keyword_match_count(conn: Any, project_id: str, query: str) -> int:
    """How many project turns match the FTS query at all (read-only)."""
    val = await conn.fetchval(
        """
        SELECT count(*)
          FROM episode_turns t
          JOIN episodes e ON t.episode_id = e.id
         WHERE e.project_id = $1
           AND to_tsvector('english', t.content)
               @@ websearch_to_tsquery('english', $2)
        """,
        project_id,
        query,
    )
    return int(val)


async def _global_vector_ranks(
    conn: Any, project_id: str, embedding: list[float], top_n: int,
    watch_ids: list[str],
) -> dict[str, Any]:
    """Untruncated vector ranking of the whole project (read-only).

    Returns the top ``top_n`` ids plus the exact rank/distance of every
    watched (gold) id — what the vector half WOULD have returned without
    the SQL LIMIT.
    """
    rows = await conn.fetch(
        """
        WITH d AS (
            SELECT t.id,
                   t.embedding <=> $1::vector AS dist,
                   row_number() OVER (
                       ORDER BY t.embedding <=> $1::vector, t.id
                   ) AS rnk
              FROM episode_turns t
              JOIN episodes e ON t.episode_id = e.id
             WHERE e.project_id = $2 AND t.embedding IS NOT NULL
        )
        SELECT id, dist, rnk FROM d
         WHERE rnk <= $3 OR id = ANY($4::text[])
         ORDER BY rnk
        """,
        embedding,
        project_id,
        top_n,
        watch_ids,
    )
    top = [
        {"rank": int(r["rnk"]), "id": str(r["id"]), "cosine_distance": float(r["dist"])}
        for r in rows
        if int(r["rnk"]) <= top_n
    ]
    watched = {
        str(r["id"]): {"rank": int(r["rnk"]), "cosine_distance": float(r["dist"])}
        for r in rows
        if str(r["id"]) in set(watch_ids)
    }
    return {"top": top, "watched": watched, "project_size_with_embedding": None}


async def _global_keyword_ranks(
    conn: Any, project_id: str, query: str, top_n: int, watch_ids: list[str]
) -> dict[str, Any]:
    """Untruncated FTS ranking of the whole project (read-only)."""
    rows = await conn.fetch(
        """
        WITH s AS (
            SELECT t.id,
                   ts_rank(to_tsvector('english', t.content),
                           websearch_to_tsquery('english', $1)) AS score,
                   row_number() OVER (
                       ORDER BY ts_rank(to_tsvector('english', t.content),
                                        websearch_to_tsquery('english', $1)) DESC,
                                t.id
                   ) AS rnk
              FROM episode_turns t
              JOIN episodes e ON t.episode_id = e.id
             WHERE e.project_id = $2
        )
        SELECT id, score, rnk FROM s
         WHERE rnk <= $3 OR id = ANY($4::text[])
         ORDER BY rnk
        """,
        query,
        project_id,
        top_n,
        watch_ids,
    )
    total = await conn.fetchval(
        "SELECT count(*) FROM episode_turns t JOIN episodes e ON t.episode_id = e.id WHERE e.project_id = $1",
        project_id,
    )
    top = [
        {"rank": int(r["rnk"]), "id": str(r["id"]), "ts_rank": float(r["score"])}
        for r in rows
        if int(r["rnk"]) <= top_n
    ]
    watched = {
        str(r["id"]): {"rank": int(r["rnk"]), "ts_rank": float(r["score"])}
        for r in rows
        if str(r["id"]) in set(watch_ids)
    }
    return {"top": top, "watched": watched, "project_turn_count": int(total)}


def _analyse_case(
    case: dict[str, Any],
    probes: list[dict[str, Any]],
    final_ids: list[str],
    gold_meta: dict[str, dict[str, Any]],
    vector_global: dict[str, Any],
    keyword_global: dict[str, Any],
    keyword_match_total: int,
) -> dict[str, Any]:
    """Classify each gold turn: absent-from-candidates vs top_k-cut vs low fusion."""
    top_k_per_anchor = max(1, case["limit"] // 2)
    candidate_limit = top_k_per_anchor * 3
    fusion_top_k = top_k_per_anchor
    absent_rank = candidate_limit + 1

    per_gold: dict[str, dict[str, Any]] = {}
    for gold_id, info in gold_meta.items():
        vec = info.get("vector")  # {"rank":..,"cosine_distance":..} or None
        kw = info.get("keyword")  # {"rank":..,"ts_rank":..} or None
        fused = info.get("fused")  # {"rank":..,"rrf_score":..} or None
        final_rank = info.get("final_rank")
        in_window = final_rank is not None

        if vec is None and kw is None:
            verdict = "absent_from_candidates"
        elif fused is None or fused["rank"] > fusion_top_k:
            verdict = "candidate_cut_by_top_k"
        else:
            verdict = "returned"

        entry = dict(info)  # preserve margin_to_boundary / ts_rank_if_matched
        entry.update(
            {
                "vector_global_untruncated": vector_global["watched"].get(gold_id),
                "keyword_global_untruncated": keyword_global["watched"].get(gold_id),
                "final_window_rank": final_rank,
                "in_final_window": in_window,
                "verdict": verdict,
                "absent_rank_penalty": absent_rank,
            }
        )
        per_gold[gold_id] = entry

    return {
        "case_id": case["case_id"],
        "question_id": case["question_id"],
        "limit": case["limit"],
        "derived": {
            "top_k_per_anchor": top_k_per_anchor,
            "candidate_sql_limit": candidate_limit,
            "fusion_top_k": fusion_top_k,
            "rrf_absent_rank": absent_rank,
        },
        "keyword_matched_turns_in_project": keyword_match_total,
        "probes": probes,
        "final_window_ids": final_ids,
        "checkpoint_returned_ids": case["checkpoint_returned_ids"],
        "matches_checkpoint": final_ids == case["checkpoint_returned_ids"],
        "gold_turns": per_gold,
        "vector_global_top": vector_global["top"],
        "keyword_global_top": keyword_global["top"],
    }


async def run_case(
    case: dict[str, Any],
    pool: Any,
    embedder: Any,
    owner_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Replay one checkpointed recall call through the real production path."""
    from weft.auth import current_user_id
    from weft.db.connection import acquire
    from weft.turn_recall import temporal_anchor

    gold_ids = [g["id"] for g in GOLD_TURNS[case["question_id"]]]
    gold_labels = {g["id"]: g for g in GOLD_TURNS[case["question_id"]]}
    top_k_per_anchor = max(1, case["limit"] // 2)
    candidate_limit = top_k_per_anchor * 3

    probes: list[dict[str, Any]] = []
    final_ids: list[str] = []

    def _diag(anchor: str, vector_rows: list, keyword_rows: list, turns: list) -> None:
        final_ids.extend(str(t.id) for t in turns)

    def _probe(payload: dict[str, Any]) -> None:
        probes.append(dict(payload))

    token = current_user_id.set(owner_id)
    try:
        async with acquire(pool) as conn:
            anchored = await temporal_anchor(
                pool,
                case["query"],
                project_id=case["project_id"],
                top_k_per_anchor=top_k_per_anchor,
                embedder=embedder,
                diag_callback=_diag,
                probe_diag_callback=_probe,
            )
            for turns in anchored.values():
                for t in turns:
                    if str(t.id) not in final_ids:
                        final_ids.append(str(t.id))

            # --- Raw halves per probe were captured only as id lists by
            # probe_diag_callback; the full rows (with distances) come from
            # the diag_callback path. Re-run capture explicitly per variant
            # below is unnecessary: temporal_anchor already invoked
            # diag_callback per probe with the raw rows, but we need them
            # per-variant. Capture them via a second instrumented pass over
            # the SAME variants using recall_turns directly is a fork — so
            # instead re-derive raw halves with read-only SQL that mirrors
            # the production queries exactly.
            embedding = await embedder.embed(case["query"])
            vector_rows = await conn.fetch(
                """
                SELECT candidates.*
                  FROM (
                    SELECT t.*, t.embedding <=> $1::vector AS _distance
                      FROM episode_turns t
                      JOIN episodes e ON t.episode_id = e.id
                     WHERE t.embedding IS NOT NULL
                       AND e.project_id = $2
                     ORDER BY t.embedding <=> $1::vector
                     LIMIT $3
                  ) AS candidates
                 ORDER BY candidates._distance, candidates.id
                """,
                embedding,
                case["project_id"],
                candidate_limit,
            )
            keyword_rows = await conn.fetch(
                """
                SELECT t.*
                  FROM episode_turns t
                  JOIN episodes e ON t.episode_id = e.id
                 WHERE to_tsvector('english', t.content)
                       @@ websearch_to_tsquery('english', $1)
                   AND e.project_id = $2
                 ORDER BY ts_rank(
                     to_tsvector('english', t.content),
                     websearch_to_tsquery('english', $1)
                 ) DESC, t.id
                 LIMIT $3
                """,
                case["query"],
                case["project_id"],
                candidate_limit,
            )

            # Exact ts_rank floats for captured keyword candidates + gold ids.
            all_watch = [str(r["id"]) for r in keyword_rows] + gold_ids
            tsrank = await _tsrank_scores(conn, case["query"], all_watch)

            # Untruncated global ranks (what LIMIT would have had to cut).
            vector_global = await _global_vector_ranks(
                conn, case["project_id"], embedding, candidate_limit, gold_ids
            )
            keyword_global = await _global_keyword_ranks(
                conn, case["project_id"], case["query"], candidate_limit, gold_ids
            )
            keyword_match_total = await _keyword_match_count(
                conn, case["project_id"], case["query"]
            )

            # Fused RRF via the production fusion function on the captured rows.
            from weft.episode_turns import _rrf_fuse_turn_rows
            from weft.relevance import rank_turns, score_turn

            fused_pairs = _rrf_fuse_turn_rows(
                vector_rows,
                keyword_rows,
                candidate_limit=candidate_limit,
                top_k=top_k_per_anchor,
                vector_weight=0.5,
                keyword_weight=0.5,
            )
            fused_scores = {
                str(t.id): float(s) for t, s in fused_pairs
            }
            fused_rank = {
                tid: i + 1 for i, tid in enumerate(fused_scores)
            }
            ranked = rank_turns(fused_pairs)  # production rerank (now = wall clock)
            rerank_report = []
            for st in ranked:
                rerank_report.append(
                    {
                        "id": str(st.turn.id),
                        "rrf_base_score": fused_scores[str(st.turn.id)],
                        "recency_factor": st.recency_factor,
                        "usefulness_factor": st.usefulness_factor,
                        "final_score": st.score,
                        "occurred_at": st.turn.occurred_at.isoformat(),
                    }
                )

            vector_list = [
                _record_to_meta(r, i + 1, "vector") for i, r in enumerate(vector_rows)
            ]
            keyword_list = []
            for i, r in enumerate(keyword_rows):
                meta = _record_to_meta(r, i + 1, "keyword")
                meta["ts_rank"] = round(tsrank.get(str(r["id"]), 0.0), 8)
                keyword_list.append(meta)

            gold_meta: dict[str, dict[str, Any]] = {}
            for gid in gold_ids:
                vec_entry = next(
                    (m for m in vector_list if m["id"] == gid), None
                )
                kw_entry = next(
                    (m for m in keyword_list if m["id"] == gid), None
                )
                gold_meta[gid] = {
                    "label": gold_labels[gid]["label"],
                    "note": gold_labels[gid]["note"],
                    "vector": (
                        {
                            "rank": vec_entry["pre_fusion_rank"],
                            "cosine_distance": vec_entry["cosine_distance"],
                            "cosine_similarity": round(
                                1.0 - vec_entry["cosine_distance"], 6
                            ),
                        }
                        if vec_entry
                        else None
                    ),
                    "keyword": (
                        {"rank": kw_entry["pre_fusion_rank"], "ts_rank": kw_entry["ts_rank"]}
                        if kw_entry
                        else None
                    ),
                    "ts_rank_if_matched": tsrank.get(gid),
                    "fused": (
                        {
                            "rank": fused_rank[gid],
                            "rrf_score": round(fused_scores[gid], 6),
                        }
                        if gid in fused_scores
                        else None
                    ),
                    "final_rank": (
                        final_ids.index(gid) + 1 if gid in final_ids else None
                    ),
                }

            # Boundary margin: gold fused score vs the score of the LAST
            # turn selected by fusion (rank == fusion_top_k).
            boundary_scores = [
                (r, s)
                for r, s in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
            boundary_entry = None
            if len(boundary_scores) >= top_k_per_anchor:
                bid, bscore = boundary_scores[top_k_per_anchor - 1]
                boundary_entry = {
                    "boundary_turn_id": bid,
                    "boundary_fused_rank": top_k_per_anchor,
                    "boundary_rrf_score": round(bscore, 6),
                }
                for gid in gold_ids:
                    if gid in fused_scores:
                        gold_meta[gid]["margin_to_boundary"] = round(
                            fused_scores[gid] - bscore, 6
                        )
                    else:
                        gold_meta[gid]["margin_to_boundary"] = None

            probe_payload = {
                "query": case["query"],
                "temporal_anchor_probe": probes[0] if probes else None,
                "anchor_candidate_ids_match": (
                    bool(probes)
                    and sorted(probes[0].get("candidate_ids", []))
                    == sorted(
                        {str(r["id"]) for r in vector_rows}
                        | {str(r["id"]) for r in keyword_rows}
                    )
                ),
                "anchor_vector_count_match": (
                    bool(probes)
                    and probes[0].get("vector_candidate_count") == len(vector_rows)
                ),
                "anchor_keyword_count_match": (
                    bool(probes)
                    and probes[0].get("keyword_candidate_count") == len(keyword_rows)
                ),
                "anchor_retrieved_ids_match": (
                    bool(probes)
                    and probes[0].get("retrieved_turn_ids") == final_ids[: len(probes[0].get("retrieved_turn_ids", []))]
                ),
                "vector_candidates": vector_list,
                "keyword_candidates": keyword_list,
                "fused_rank_order": [
                    {"rank": i + 1, "id": tid, "rrf_score": round(s, 6)}
                    for i, (tid, s) in enumerate(
                        sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
                    )
                ],
                "fusion_boundary": boundary_entry,
                "rerank_report": rerank_report,
                "candidate_ids_union_size": len(
                    {str(r["id"]) for r in vector_rows}
                    | {str(r["id"]) for r in keyword_rows}
                ),
            }

            report = _analyse_case(
                case,
                [probe_payload],
                final_ids,
                gold_meta,
                vector_global,
                keyword_global,
                keyword_match_total,
            )
            report["rrf_k"] = 60
            report["rerank_final_order"] = rerank_report
            return report, gold_meta
    finally:
        current_user_id.reset(token)


async def main_async(out_dir: Path) -> int:
    started = datetime.now(timezone.utc)
    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL")
    if not dsn:
        print("LONGMEMEVAL_DATABASE_URL is required", file=sys.stderr)
        return 2

    from urllib.parse import urlsplit

    u = urlsplit(dsn)
    env_note = {
        "dsn": {
            "host": u.hostname,
            "port": u.port,
            "database": u.path.lstrip("/"),
            "user": u.username,
        },
        "WEFT_HIERARCHICAL": os.environ.get("WEFT_HIERARCHICAL"),
        "WEFT_TURN_RERANK_DISABLE": os.environ.get("WEFT_TURN_RERANK_DISABLE"),
        "read_only": True,
        "live_db": True,
    }
    if os.environ.get("WEFT_HIERARCHICAL") == "1":
        print(
            "WARNING: WEFT_HIERARCHICAL=1 — production run did NOT use the "
            "hierarchical path; results would not mirror answer time.",
            file=sys.stderr,
        )
        return 2

    from weft.config import WeftConfig
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider

    config = WeftConfig()
    config.database.url = dsn
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    config.retrieval.recovery_mode = "off"
    assert config.embedding.model == EMBED_MODEL, config.embedding.model
    assert config.embedding.dimensions == EMBED_DIMS, config.embedding.dimensions
    embedder = get_provider(
        config.embedding.provider,
        model_name=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )
    pool = await create_pool(config)

    results: list[dict[str, Any]] = []
    try:
        for case in CASES:
            print(f"probing {case['case_id']} ...", flush=True)
            report, _ = await run_case(case, pool, embedder, OWNER_ID)
            if not report["matches_checkpoint"]:
                print(
                    f"ERROR: replay window for {case['case_id']} diverges from checkpoint:\n"
                    f"  replay:     {report['final_window_ids']}\n"
                    f"  checkpoint: {report['checkpoint_returned_ids']}",
                    file=sys.stderr,
                )
            results.append(report)
            await asyncio.sleep(0)
    finally:
        await pool.close()

    payload = {
        "probe": "loop-rank-probe",
        "generated_at": started.isoformat(),
        "owner_id": OWNER_ID,
        "embedding": {"provider": config.embedding.provider, "model": EMBED_MODEL, "dimensions": EMBED_DIMS},
        "retrieval_path": (
            "weft.turn_recall.temporal_anchor -> weft.episode_turns.recall_turns "
            "(vector+FTS LIMIT top_k*3) -> _rrf_fuse_turn_rows (RRF k=60, absent=limit+1) "
            "-> weft.relevance.rank_turns (rerank now=wall clock)"
        ),
        "environment": env_note,
        "cases": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {out_path}")

    ok = all(r["matches_checkpoint"] for r in results)
    print("checkpoint window match:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/loop-rank-probe-20260929"),
        help="output directory (default: artifacts/loop-rank-probe-20260929)",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
