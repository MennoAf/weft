#!/usr/bin/env python3
"""Read-only local rank probe for LongMemEval-S preference misses.

This reuses the exact exp8 worktree product retrieval path, but calls its
lower-level retrieval function rather than ``weft_recall`` so it does not write
turn-access telemetry. Query embeddings use local FastEmbed; no judge, writer,
network provider, ingestion, or benchmark run is used.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORKTREE = ROOT / ".recall-lift-worktree"
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

EVIDENCE = ROOT / "evidence/single-session-preference"
OUT = Path(__file__).resolve().parent / "probe-results.json"
CLASSIFICATION = json.loads((EVIDENCE / "classification.json").read_text())
DATASET_PATH = ROOT / "benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json"
MANIFEST_PATH = WORKTREE / "benchmarks/longmemeval/manifests/longmemeval_s_full_turns_manifest.json"
ARTIFACT_ROOT = WORKTREE / "artifacts/longmemeval-gpt6-full-s-turns"
RUNS = ["faithful", "faithful-run1-20260930", "faithful-run2-20260930"]
PRIMARY = ["32260d93", "75832dbd", "d6233ab6", "0a34ad58", "1c0ddc50"]
OWNER_ID = "faithful-gpt6-fulls-exp8-20260930"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 768
DEFAULT_DSN = "postgresql://weft:weft_local@127.0.0.1:5433/lme_bench"


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def rank_of(rows: list[Any], turn_id: str) -> int | None:
    for index, row in enumerate(rows, 1):
        row_id = row.get("id") if isinstance(row, dict) else row["id"]
        if str(row_id) == turn_id:
            return index
    return None


async def global_vector_rank(conn: Any, project_id: str, embedding: list[float], turn_id: str) -> int | None:
    return await conn.fetchval(
        """WITH ranked AS (
               SELECT t.id, row_number() OVER (ORDER BY t.embedding <=> $1::vector, t.id) AS rank
                 FROM episode_turns t JOIN episodes e ON e.id = t.episode_id
                WHERE e.project_id = $2 AND t.embedding IS NOT NULL
           ) SELECT rank FROM ranked WHERE id = $3""",
        embedding, project_id, turn_id,
    )


async def global_keyword_rank(conn: Any, project_id: str, query: str, turn_id: str) -> int | None:
    return await conn.fetchval(
        """WITH ranked AS (
               SELECT t.id,
                      row_number() OVER (
                          ORDER BY ts_rank(to_tsvector('english', t.content),
                                           websearch_to_tsquery('english', $1)) DESC,
                                   t.id
                      ) AS rank
                 FROM episode_turns t JOIN episodes e ON e.id = t.episode_id
                WHERE e.project_id = $2
                  AND to_tsvector('english', t.content) @@ websearch_to_tsquery('english', $1)
           ) SELECT rank FROM ranked WHERE id = $3""",
        query, project_id, turn_id,
    )


async def analyse_query(
    *, pool: Any, embedder: Any, question_id: str, query: str, limit: int,
    gold_turn_id: str, source: str,
) -> dict[str, Any]:
    from weft.db.connection import acquire
    from weft.episode_turns import _rrf_fuse_turn_rows
    from weft.relevance import rank_turns
    from weft.turn_recall import temporal_anchor

    project_id = f"longmemeval-{question_id}"
    top_k_per_anchor = min(limit, 10)
    candidate_limit = top_k_per_anchor * 5  # MCP _weft_recall_turns exp8 path
    embedding = await embedder.embed(query)
    diag: list[dict[str, Any]] = []

    def on_probe(anchor: str, vectors: list[Any], keywords: list[Any], returned: list[Any]) -> None:
        diag.append({"anchor": anchor, "vectors": list(vectors), "keywords": list(keywords), "returned": list(returned)})

    async with acquire(pool) as conn:
        anchored = await temporal_anchor(
            pool,
            query,
            project_id=project_id,
            top_k_per_anchor=top_k_per_anchor,
            candidate_sql_limit=candidate_limit,
            anchor_result_limit=limit,
            expansion_slots=8,
            embedder=embedder,
            diag_callback=on_probe,
        )

        probe_rows: list[dict[str, Any]] = []
        for probe_index, raw in enumerate(diag):
            vec_rank = rank_of(raw["vectors"], gold_turn_id)
            kw_rank = rank_of(raw["keywords"], gold_turn_id)
            candidates = {str(row["id"]) for row in [*raw["vectors"], *raw["keywords"]]}
            fused = _rrf_fuse_turn_rows(
                raw["vectors"], raw["keywords"],
                candidate_limit=candidate_limit,
                top_k=max(1, len(candidates)),
                vector_weight=1.0,
                keyword_weight=0.3,
            )
            fused_rank = next((i for i, (turn, _) in enumerate(fused, 1) if str(turn.id) == gold_turn_id), None)
            reranked = rank_turns(fused)
            rerank_rank = next((i for i, score in enumerate(reranked, 1) if str(score.turn.id) == gold_turn_id), None)
            probe_rows.append({
                "probe_index": probe_index,
                "anchor": raw["anchor"],
                "vector_candidate_count": len(raw["vectors"]),
                "keyword_candidate_count": len(raw["keywords"]),
                "gold_vector_rank_in_candidate_window": vec_rank,
                "gold_vector_project_rank_untruncated": await global_vector_rank(conn, project_id, embedding, gold_turn_id),
                "gold_keyword_rank_in_candidate_window": kw_rank,
                "gold_keyword_project_rank_if_match": await global_keyword_rank(conn, project_id, query, gold_turn_id),
                "gold_rrf_rank_in_candidate_union": fused_rank,
                "gold_recency_usefulness_rank_within_rrf_pool": rerank_rank,
                "production_candidate_limit": candidate_limit,
                "production_rrf_output_width": limit + 8,
            })

        returned = []
        seen: set[str] = set()
        for group in anchored.values():
            for turn in group:
                if str(turn.id) in seen:
                    continue
                seen.add(str(turn.id))
                returned.append(turn)
                if len(returned) >= limit + 8:
                    break
            if len(returned) >= limit + 8:
                break
        final_rank = next((i for i, turn in enumerate(returned, 1) if str(turn.id) == gold_turn_id), None)
        content_rows = await conn.fetch(
            """SELECT t.id, t.role, t.source_session_id, t.turn_index, t.content
                 FROM episode_turns t JOIN episodes e ON t.episode_id=e.id
                WHERE e.project_id=$1 AND t.id=$2""",
            project_id, gold_turn_id,
        )
        gold_row = dict(content_rows[0]) if content_rows else None

    return {
        "source": source,
        "query": query,
        "limit": limit,
        "project_id": project_id,
        "gold_vector_rank_project_untruncated": probe_rows[0]["gold_vector_project_rank_untruncated"] if probe_rows else None,
        "gold_keyword_rank_project_if_match": probe_rows[0]["gold_keyword_project_rank_if_match"] if probe_rows else None,
        "gold_vector_rank_in_recall_candidate_window": probe_rows[0]["gold_vector_rank_in_candidate_window"] if probe_rows else None,
        "gold_keyword_rank_in_recall_candidate_window": probe_rows[0]["gold_keyword_rank_in_candidate_window"] if probe_rows else None,
        "gold_rrf_rank_candidate_union": probe_rows[0]["gold_rrf_rank_in_candidate_union"] if probe_rows else None,
        "gold_rank_after_production_rerank": probe_rows[0]["gold_recency_usefulness_rank_within_rrf_pool"] if probe_rows else None,
        "gold_rank_in_production_return_limit_plus_8": final_rank,
        "gold_in_protected_limit_prefix": bool(final_rank is not None and final_rank <= limit),
        "returned_turn_ids": [str(turn.id) for turn in returned],
        "probe_details": probe_rows,
        "gold_turn": gold_row,
    }


async def main() -> None:
    import asyncpg
    from weft.auth import current_user_id
    from weft.config import WeftConfig
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider

    dataset_rows = json.loads(DATASET_PATH.read_text())
    dataset = {row["question_id"]: row for row in dataset_rows}
    classified = {row["question_id"]: row for row in CLASSIFICATION["stable_failure_items"]}
    manifest = json.loads(MANIFEST_PATH.read_text())
    retrieval = manifest["retrieval"]
    checkpoints = {
        run: json.loads((ARTIFACT_ROOT / run / "session-checkpoint.json").read_text())["evidence"]
        for run in RUNS
    }

    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL", DEFAULT_DSN)
    cfg = WeftConfig()
    cfg.database.url = dsn
    cfg.database.pool_min_size = 1
    cfg.database.pool_max_size = 2
    cfg.retrieval.recovery_mode = "off"
    pool = await create_pool(cfg)
    embedder = get_provider("fastembed", model_name=EMBED_MODEL, dimensions=EMBED_DIMS)
    token = current_user_id.set(OWNER_ID)
    try:
        cases: list[dict[str, Any]] = []
        all_queries: list[tuple[str, str, int, str]] = []
        gold_ids: dict[str, str] = {}
        gold_session_ids: dict[str, str] = {}
        gold_sources: dict[str, str] = {}

        # Resolve the source-user turn in the local DB by source_session_id +
        # verbatim dataset turn content. This is a post-ingest readback.
        for qid in PRIMARY:
            item = classified[qid]
            source = item["supporting_statements"][0]
            sid = source["session_id"]
            target_content = source["source_turn_content"]
            project_id = f"longmemeval-{qid}"
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT t.id, t.role, t.source_session_id, t.turn_index, t.content
                         FROM episode_turns t JOIN episodes e ON t.episode_id=e.id
                        WHERE e.project_id=$1 AND t.source_session_id=$2 AND t.role='user'
                        ORDER BY t.turn_index, t.id""",
                    project_id, sid,
                )
            exact = [row for row in rows if norm(row["content"]) == norm(target_content)]
            if not exact:
                exact = [row for row in rows if norm(target_content) in norm(row["content"]) or norm(row["content"]) in norm(target_content)]
            if not exact:
                gold_ids[qid] = ""
                gold_session_ids[qid] = sid
                gold_sources[qid] = target_content
            else:
                gold_ids[qid] = str(exact[0]["id"])
                gold_session_ids[qid] = sid
                gold_sources[qid] = target_content

            qrow = dataset[qid]
            all_queries.append((qid, qrow["question"], retrieval["top_k"], "original_dataset_question"))

            # Replay every recorded weft_recall query from every run. The
            # generated retrieval phrasing varies by run; retaining each query
            # lets the report distinguish stable misses from query drift.
            for actual_run in RUNS:
                calls = [
                    x for x in checkpoints[actual_run][qid].get("tool_results", [])
                    if x.get("name") == "weft_recall"
                ]
                for i, call in enumerate(calls):
                    args = call.get("arguments", {})
                    all_queries.append((
                        qid, args["query"],
                        int(args.get("limit", retrieval["top_k"])),
                        f"checkpoint:{actual_run}:call{i}",
                    ))

        query_results: dict[str, list[dict[str, Any]]] = {qid: [] for qid in PRIMARY}
        # Repeated queries across runs are deduplicated so every embedding is
        # local, bounded, and computed only once.
        seen_queries: set[tuple[str, str, int]] = set()
        for qid, query, limit, source in all_queries:
            key = (qid, query, limit)
            if key in seen_queries:
                continue
            seen_queries.add(key)
            if not gold_ids.get(qid):
                query_results[qid].append({"source": source, "query": query, "error": "gold source turn did not resolve in local DB"})
                continue
            result = await analyse_query(
                pool=pool, embedder=embedder, question_id=qid, query=query,
                limit=limit, gold_turn_id=gold_ids[qid], source=source,
            )
            query_results[qid].append(result)

        # Exact context parity check on a stable item using its recorded run2 query.
        parity_qid = "d6233ab6"
        parity_record = checkpoints["faithful-run2-20260930"][parity_qid]
        parity_call = next(x for x in parity_record["tool_results"] if x.get("name") == "weft_recall")
        parity_query = parity_call["arguments"]["query"]
        expected_turns = parity_call["result"].get("turns", [])
        local_match = next((r for r in query_results[parity_qid] if r["query"] == parity_query), None)
        local_ids = local_match["returned_turn_ids"] if local_match else []
        expected_ids = [str(t.get("id")) for t in expected_turns]
        id_match = local_ids == expected_ids
        expected_texts = [norm(t.get("content", "")) for t in expected_turns]
        local_texts: list[str] = []
        if local_match:
            async with pool.acquire() as conn:
                ids = local_match["returned_turn_ids"]
                by_id = await conn.fetch(
                    "SELECT id, content FROM episode_turns WHERE id=ANY($1::text[])", ids,
                )
            content_by_id = {str(r["id"]): norm(r["content"]) for r in by_id}
            local_texts = [content_by_id.get(tid, "") for tid in ids]
        text_match = local_texts == expected_texts

        # Recompute main faithful-run exact reservation totals from the actual
        # budget ledger, independently of the generated classification report.
        ledger = json.loads((ARTIFACT_ROOT / "faithful" / "budget-ledger.json").read_text())
        reservations = {row["reservation_id"]: row for row in ledger["reservations"]}
        costs: dict[str, dict[str, Any]] = {}
        for qid in CLASSIFICATION["validation_plan"]["question_ids"]:
            record = checkpoints["faithful"].get(qid, {})
            answer_res = record.get("answer", {}).get("reservations", [])
            judge_res = record.get("judge", {}).get("reservation_id")
            ids = [*answer_res, *([judge_res] if judge_res else [])]
            rows = [reservations[rid] for rid in ids if rid in reservations]
            costs[qid] = {
                "reservation_count": len(ids),
                "ledger_rows_found": len(rows),
                "actual_usd": round(sum(float(r.get("actual_usd", 0) or 0) for r in rows), 8),
                "reserved_estimate_usd": round(sum(float(r.get("estimated_usd", 0) or 0) for r in rows), 8),
            }

        results = {
            "probe": "single-session-preference-turn-ranks",
            "database": "local lme_bench (LONGMEMEVAL_DATABASE_URL override accepted; no DSN persisted)",
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "profile": manifest["profile"],
            "retrieval_settings": retrieval,
            "production_recall_path": "weft.mcp.tools._weft_recall_turns -> weft.turn_recall.temporal_anchor -> weft.episode_turns.recall_turns -> _rrf_fuse_turn_rows -> rank_turns -> protected limit prefix + same-session expansion",
            "product_parameters": {
                "weights": {"vector": 1.0, "keyword": 0.3},
                "candidate_sql_limit": "5 * min(limit, 10)",
                "expansion_slots": retrieval["turn_tier_expansion_slots"],
                "stored_search_tsv": False,
                "embedding_provider": "fastembed (local)",
                "embedding_model": EMBED_MODEL,
                "embedding_dimensions": EMBED_DIMS,
                "paid_provider_calls": 0,
                "paid_embedding_calls": 0,
                "query_embedding_count_after_dedup": len(seen_queries),
                "public_weft_recall_called": False,
                "read_only_database_queries": True,
            },
            "parity": {
                "question_id": parity_qid,
                "checkpoint_run": "faithful-run2-20260930",
                "query": parity_query,
                "checkpoint_turn_count": len(expected_ids),
                "local_turn_count": len(local_ids),
                "ordered_turn_ids_match": id_match,
                "ordered_turn_content_match": text_match,
                "checkpoint_ids": expected_ids,
                "local_ids": local_ids,
            },
            "primary_cases": [
                {
                    "question_id": qid,
                    "question": dataset[qid]["question"],
                    "gold_statement": gold_sources[qid],
                    "gold_source_session_id": gold_session_ids[qid],
                    "gold_turn_id": gold_ids[qid] or None,
                    "checkpoint_context_presence_by_run": classified[qid]["context_presence_by_run"],
                    "checkpoint_recall_queries_by_run": {
                        run: [
                            {
                                "query": call.get("arguments", {}).get("query"),
                                "limit": call.get("arguments", {}).get("limit"),
                                "returned_turn_count": len(call.get("result", {}).get("turns", [])),
                            }
                            for call in checkpoints[run][qid].get("tool_results", [])
                            if call.get("name") == "weft_recall"
                        ]
                        for run in RUNS
                    },
                    "local_probe_results": query_results[qid],
                }
                for qid in PRIMARY
            ],
            "secondary_reader_item_context_absence": {
                qid: classified[qid]["context_presence_by_run"]
                for qid in ["09d032c9", "0edc2aef", "35a27287", "afdc33df"]
            },
            "budget_ledger_validation": {
                "source": str((ARTIFACT_ROOT / "faithful" / "budget-ledger.json").relative_to(ROOT)),
                "costs_by_question": costs,
                "focused_18_item_actual_usd": round(sum(v["actual_usd"] for v in costs.values()), 8),
                "focused_18_item_reserved_usd": round(sum(v["reserved_estimate_usd"] for v in costs.values()), 8),
                "classification_report_actual_usd": CLASSIFICATION["validation_plan"]["18_item_actual_cost_estimate_usd"],
                "classification_report_reserved_usd": CLASSIFICATION["validation_plan"]["18_item_reserved_estimate_usd"],
            },
        }
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({
            "output": str(OUT.relative_to(ROOT)),
            "profile": results["profile"],
            "parity": {k: results["parity"][k] for k in ("checkpoint_turn_count", "local_turn_count", "ordered_turn_ids_match", "ordered_turn_content_match")},
            "focused_18_item_actual_usd": results["budget_ledger_validation"]["focused_18_item_actual_usd"],
            "focused_18_item_reserved_usd": results["budget_ledger_validation"]["focused_18_item_reserved_usd"],
            "primary_items": len(results["primary_cases"]),
            "local_query_embeddings": len(seen_queries),
        }, indent=2))
    finally:
        current_user_id.reset(token)
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
