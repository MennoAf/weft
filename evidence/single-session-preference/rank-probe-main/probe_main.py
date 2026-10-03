#!/usr/bin/env python3
"""Read-only turn-recall rank probe against the single-session main worktree."""
from __future__ import annotations
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WORKTREE = ROOT / ".single-session-worktree"
INPUT_WORKTREE = ROOT / ".recall-lift-worktree"
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
OUT = Path(__file__).resolve().parent / "probe-results.json"
CLASSIFICATION_PATH = ROOT / "evidence/single-session-preference/classification.json"
DATASET_PATH = ROOT / "benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json"
MANIFEST_PATH = INPUT_WORKTREE / "benchmarks/longmemeval/manifests/longmemeval_s_full_turns_manifest.json"
ARTIFACT_ROOT = INPUT_WORKTREE / "artifacts/longmemeval-gpt6-full-s-turns"
RUNS = ["faithful", "faithful-run1-20260930", "faithful-run2-20260930"]
PRIMARY = ["32260d93", "75832dbd", "d6233ab6", "0a34ad58", "1c0ddc50"]
READER_STABLE = ["09d032c9", "0edc2aef", "35a27287", "afdc33df"]
STABLE_IDS = PRIMARY + READER_STABLE
ALWAYS_CHECK = ["06f04340", "07b6f563", "1a1907b4", "1d4e3b97", "1da05512"]
OWNER_ID = "faithful-gpt6-fulls-exp8-20260930"
MODEL = "BAAI/bge-small-en-v1.5"
DIMS = 768
DSN = "postgresql://weft:weft_local@127.0.0.1:5433/lme_bench"


def norm(s: str) -> str:
    return " ".join(s.casefold().split())


def row_rank(rows: list[Any], tid: str) -> int | None:
    for i, r in enumerate(rows, 1):
        if str(r["id"]) == tid:
            return i
    return None


async def full_vector_rank(conn: Any, project: str, emb: list[float], tid: str) -> int | None:
    return await conn.fetchval("""WITH r AS (
        SELECT t.id,row_number() OVER(ORDER BY t.embedding <=> $1::vector,t.id) n
        FROM episode_turns t JOIN episodes e ON e.id=t.episode_id
        WHERE e.project_id=$2 AND t.embedding IS NOT NULL)
        SELECT n FROM r WHERE id=$3""", emb, project, tid)


async def full_or_keyword_rank(conn: Any, project: str, tsq: str | None, tid: str) -> int | None:
    if not tsq:
        return None
    return await conn.fetchval("""WITH r AS (
        SELECT t.id,row_number() OVER(ORDER BY ts_rank(to_tsvector('english',t.content),to_tsquery('english',$1)) DESC,t.id) n
        FROM episode_turns t JOIN episodes e ON e.id=t.episode_id
        WHERE e.project_id=$2 AND to_tsvector('english',t.content) @@ to_tsquery('english',$1))
        SELECT n FROM r WHERE id=$3""", tsq, project, tid)


async def probe_query(*, pool: Any, embedder: Any, qid: str, query: str, limit: int,
                      gold: dict[str, Any], source: str, or_query: Any) -> dict[str, Any]:
    from weft.db.connection import acquire
    from weft.episode_turns import _rrf_fuse_turn_rows
    from weft.relevance import rank_turns
    from weft.turn_recall import temporal_anchor

    project = f"longmemeval-{qid}"
    per_anchor = min(limit, 10)
    sql_width = per_anchor * 5
    tsq = or_query(query)
    diag: list[dict[str, Any]] = []
    def callback(anchor: str, vectors: list[Any], keywords: list[Any], returned: list[Any]) -> None:
        diag.append({"anchor": anchor, "vectors": list(vectors), "keywords": list(keywords), "returned": list(returned)})

    async with acquire(pool) as conn:
        # This is the code-under-test. It executes vector + OR keyword, RRF,
        # production rerank, and same-session expansion with the receipt's cap.
        actual = await temporal_anchor(
            pool, query, project_id=project, top_k_per_anchor=per_anchor,
            candidate_sql_limit=sql_width, anchor_result_limit=limit,
            expansion_slots=8, embedder=embedder, diag_callback=callback,
        )
        emb = await embedder.embed(query)
        detail_rows = []
        for n, d in enumerate(diag):
            vrank = row_rank(d["vectors"], gold["id"])
            krank = row_rank(d["keywords"], gold["id"])
            union_ids = {str(r["id"]) for r in (*d["vectors"], *d["keywords"])}
            # Full-union rank is diagnostic only. Production prefix is capped
            # at the request limit before reranking; compute that actual lane too.
            fused_full = _rrf_fuse_turn_rows(d["vectors"], d["keywords"], candidate_limit=sql_width,
                                               top_k=max(1, len(union_ids)), vector_weight=1.0, keyword_weight=0.3)
            full_rrf_rank = next((i for i,(t,_) in enumerate(fused_full,1) if str(t.id)==gold["id"]),None)
            fused_prefix = _rrf_fuse_turn_rows(d["vectors"], d["keywords"], candidate_limit=sql_width,
                                                 top_k=limit, vector_weight=1.0, keyword_weight=0.3)
            prefix_rrf_rank = next((i for i,(t,_) in enumerate(fused_prefix,1) if str(t.id)==gold["id"]),None)
            reranked_prefix = rank_turns(fused_prefix)
            prefix_rerank_rank = next((i for i,x in enumerate(reranked_prefix,1) if str(x.turn.id)==gold["id"]),None)
            full_rerank = rank_turns(fused_full)
            full_rerank_rank = next((i for i,x in enumerate(full_rerank,1) if str(x.turn.id)==gold["id"]),None)
            detail_rows.append({
                "anchor": d["anchor"],
                "candidate_window": sql_width,
                "vector_candidates": len(d["vectors"]),
                "keyword_candidates": len(d["keywords"]),
                "gold_vector_candidate_rank": vrank,
                "gold_keyword_candidate_rank_or": krank,
                "gold_rrf_rank_full_union_diagnostic": full_rrf_rank,
                "gold_rrf_rank_production_prefix": prefix_rrf_rank,
                "gold_rerank_rank_production_prefix": prefix_rerank_rank,
                "gold_rerank_rank_full_union_diagnostic": full_rerank_rank,
                "gold_vector_project_rank": await full_vector_rank(conn, project, emb, gold["id"]),
                "gold_keyword_project_rank_or": await full_or_keyword_rank(conn, project, tsq, gold["id"]),
            })
        returned: list[Any] = []
        seen: set[str] = set()
        for group in actual.values():
            for t in group:
                if str(t.id) not in seen:
                    seen.add(str(t.id)); returned.append(t)
                if len(returned) >= limit + 8:
                    break
            if len(returned) >= limit + 8:
                break
        position = next((i for i,t in enumerate(returned,1) if str(t.id)==gold["id"]),None)
        prefix_position = next((i for i,t in enumerate(returned[:limit],1) if str(t.id)==gold["id"]),None)
    return {
        "source": source, "query": query, "limit": limit, "effective_window": limit+8,
        "gold_turn_id": gold["id"], "gold_source_session_id": gold.get("source_session_id"),
        "gold_turn_index": gold.get("turn_index"),
        "gold_vector_rank": detail_rows[0]["gold_vector_candidate_rank"] if detail_rows else None,
        "gold_keyword_candidate_window_rank_or": detail_rows[0]["gold_keyword_candidate_rank_or"] if detail_rows else None,
        "gold_rrf_rank_full_union_diagnostic": detail_rows[0]["gold_rrf_rank_full_union_diagnostic"] if detail_rows else None,
        "gold_rerank_rank_production_prefix": detail_rows[0]["gold_rerank_rank_production_prefix"] if detail_rows else None,
        "gold_post_rerank_rank_full_union_diagnostic": detail_rows[0]["gold_rerank_rank_full_union_diagnostic"] if detail_rows else None,
        "gold_position_protected_prefix": prefix_position,
        "gold_position_effective_window": position,
        "gold_in_effective_window": position is not None,
        "returned_turn_ids": [str(t.id) for t in returned], "probe_details": detail_rows,
    }


def extract_dataset_statement_turns(dataset_row: dict[str, Any], session_id: str) -> list[str]:
    # Answers can map to multiple user statements; locate exact dataset turn text
    # for receipt evidence, rather than treating a generated answer as the gold.
    sessions = dataset_row.get("haystack_sessions", [])
    ids = dataset_row.get("haystack_session_ids", [])
    out = []
    for sid, turns in zip(ids, sessions):
        if sid == session_id:
            out.extend(t for t in turns if isinstance(t,str))
    return out


async def main() -> None:
    from weft.auth import current_user_id
    from weft.config import WeftConfig
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider
    from weft.store import build_or_tsquery
    cls = json.loads(CLASSIFICATION_PATH.read_text())
    stable = {x["question_id"]: x for x in cls["stable_failure_items"]}
    ds = {x["question_id"]: x for x in json.loads(DATASET_PATH.read_text())}
    manifest = json.loads(MANIFEST_PATH.read_text())
    checkpoints = {run: json.loads((ARTIFACT_ROOT/run/"session-checkpoint.json").read_text())["evidence"] for run in RUNS}
    cfg = WeftConfig(); cfg.database.url = os.environ.get("LONGMEMEVAL_DATABASE_URL", DSN)
    cfg.database.pool_min_size=1; cfg.database.pool_max_size=2; cfg.retrieval.recovery_mode="off"
    pool = await create_pool(cfg)
    embedder = get_provider("fastembed", model_name=MODEL, dimensions=DIMS)
    token = current_user_id.set(OWNER_ID)
    try:
        gold_rows: dict[str,dict[str,Any]] = {}
        query_list: list[tuple[str,str,int,str]] = []
        async with pool.acquire() as conn:
            for qid in STABLE_IDS:
                item=stable[qid]; statement=item["supporting_statements"][0]
                sid=statement["session_id"]; content=statement["source_turn_content"]
                rows=await conn.fetch("""SELECT t.id,t.role,t.source_session_id,t.turn_index,t.content
                    FROM episode_turns t JOIN episodes e ON e.id=t.episode_id
                    WHERE e.project_id=$1 AND t.source_session_id=$2 AND t.role='user'
                    ORDER BY t.turn_index,t.id""",f"longmemeval-{qid}",sid)
                exact=[dict(r) for r in rows if norm(r["content"])==norm(content)]
                if not exact: exact=[dict(r) for r in rows if norm(content) in norm(r["content"]) or norm(r["content"]) in norm(content)]
                if not exact: raise RuntimeError(f"gold source unresolved {qid} {sid}")
                gold_rows[qid]=exact[0]
                if qid in PRIMARY:
                    query_list.append((qid,ds[qid]["question"],manifest["retrieval"]["top_k"],"original_dataset_question"))
                    for run in RUNS:
                        for i,call in enumerate(x for x in checkpoints[run][qid].get("tool_results",[]) if x.get("name")=="weft_recall"):
                            a=call.get("arguments",{}); query_list.append((qid,a["query"],int(a.get("limit",10)),f"checkpoint:{run}:call{i}"))
        results={qid:[] for qid in PRIMARY}; seen=set()
        for qid,q,limit,source in query_list:
            k=(qid,q,limit)
            if k in seen: continue
            seen.add(k)
            results[qid].append(await probe_query(pool=pool,embedder=embedder,qid=qid,query=q,limit=limit,gold=gold_rows[qid],source=source,or_query=build_or_tsquery))
        # Checkpoint parity: same run-2 call used by the prior probe; compare text/order because local ingest IDs are reminted.
        pqid="d6233ab6"; call=next(x for x in checkpoints["faithful-run2-20260930"][pqid]["tool_results"] if x.get("name")=="weft_recall")
        pr=next(x for x in results[pqid] if x["query"]==call["arguments"]["query"])
        expected=call["result"].get("turns",[])
        async with pool.acquire() as conn:
            byid=await conn.fetch("SELECT id,content FROM episode_turns WHERE id=ANY($1::text[])",pr["returned_turn_ids"])
        contents={str(x["id"]):norm(x["content"]) for x in byid}
        parity={"checkpoint_turn_count":len(expected),"local_turn_count":len(pr["returned_turn_ids"]),
                "ordered_turn_content_match":[contents.get(tid,"") for tid in pr["returned_turn_ids"]]==[norm(x.get("content","")) for x in expected],
                "ordered_turn_ids_match":pr["returned_turn_ids"]==[str(x.get("id")) for x in expected],"query":pr["query"]}
        # Five always-correct checks: choose answer-bearing user turns from one saved
        # checkpoint context where possible, matching by answer-session membership.
        # Persist the exact target turn text and rank it under the user's original question.
        always=[]
        for qid in ALWAYS_CHECK:
            row=ds[qid]; answer_sids=set(row.get("answer_session_ids",[])); checkpoint_turns=[]
            for run in RUNS:
                rec=checkpoints[run].get(qid,{})
                for tr in rec.get("tool_results",[]):
                    if tr.get("name")!="weft_recall": continue
                    for turn in tr.get("result",{}).get("turns",[]):
                        if turn.get("role")=="user" and turn.get("source_session_id") in answer_sids:
                            checkpoint_turns.append(turn)
                if checkpoint_turns: break
            if not checkpoint_turns: raise RuntimeError(f"always-correct successful receipt has no answer-session user turn: {qid}")
            target=checkpoint_turns[0]
            async with pool.acquire() as conn:
                source_rows=await conn.fetch("""SELECT t.id,t.role,t.source_session_id,t.turn_index,t.content
                    FROM episode_turns t JOIN episodes e ON e.id=t.episode_id
                    WHERE e.project_id=$1 AND t.source_session_id=$2 AND t.role='user'
                    ORDER BY t.turn_index,t.id""",f"longmemeval-{qid}",target["source_session_id"])
            matches=[dict(t) for t in source_rows if norm(t["content"])==norm(target["content"])]
            if not matches: matches=[dict(t) for t in source_rows if norm(target["content"]) in norm(t["content"]) or norm(t["content"]) in norm(target["content"])]
            if not matches: raise RuntimeError(f"always-correct answer-bearing turn unresolved locally: {qid} {target['source_session_id']}")
            chosen=matches[0]
            probe=await probe_query(pool=pool,embedder=embedder,qid=qid,query=row["question"],limit=10,gold=chosen,source="always_correct_original_question",or_query=build_or_tsquery)
            always.append({"question_id":qid,"question":row["question"],"gold_turn":{k:chosen.get(k) for k in ["id","role","source_session_id","turn_index","content"]},"probe":probe})
        output={"probe":"single-session-preference-rank-probe-main","code_worktree":".single-session-worktree","code_head":"c204ab5ad83f488d6dff6058c735068f7a8ddb68","origin_main_base":"b1e6b88ec5d450ce5f0b1460f1c4d438bf61b384","keyword_semantics":"build_or_tsquery + parameterized to_tsquery; empty query skips keyword SQL",
            "input_artifact_worktree":".recall-lift-worktree","profile":manifest["profile"],"retrieval_settings":manifest["retrieval"],
            "product_parameters":{"vector_weight":1.0,"keyword_weight":0.3,"candidate_sql_width":"5 * min(limit,10)","expansion_slots":8,"local_embedding_model":MODEL,"dimensions":DIMS,"deduplicated_query_embeddings":len(seen),"paid_provider_calls":0,"benchmark_or_judge_calls":0,"database_access":"local lme_bench; SELECT only"},
            "parity":parity,"primary_cases":[{"question_id":qid,"question":ds[qid]["question"],"gold_turn":gold_rows[qid],"checkpoint_context_presence_by_run":stable[qid]["context_presence_by_run"],"queries":results[qid]} for qid in PRIMARY],
            "always_correct_spot_checks":always}
        OUT.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"output":str(OUT.relative_to(ROOT)),"primary_query_count":sum(map(len,results.values())),"deduplicated_query_embeddings":len(seen),"parity":parity,"always_checks":len(always),"always_correct_in_window":sum(x['probe']['gold_in_effective_window'] for x in always)},indent=2))
    finally:
        current_user_id.reset(token); await pool.close()

if __name__=="__main__": asyncio.run(main())
