"""Round-2 measurement census for the turn-tier recall lift (SELECT-only).

Replays EVERY judged-false multi-session / knowledge-update question's
turn-tier ``weft_recall`` calls from the saved checkpoint through both
funnels, plus a judged-TRUE retention-guard sample:

* AFTER (post-fix, current production code): capture through
  ``temporal_anchor`` (OR keyword half, per-anchor window ``min(limit,10)``,
  SQL width 5x, weighted RRF 1.0/0.3) and record the authentic returned
  window.
* BEFORE (old funnel at cb76a689, mirrored via direct SELECTs): vector half
  ``ORDER BY embedding <=>`` with width ``3 * (limit // 2)``, keyword half
  ``websearch_to_tsquery`` full-AND with ``ts_rank`` ordering, unweighted
  RRF re-fusion (0.5/0.5) via the production ``_rrf_fuse_turn_rows``, and
  the production recency rerank; window = fused top ``limit // 2``.

Gold evidence for FALSE cases is every turn in the checkpoint's ``answer_*``
sessions (session-level diagnostic labels, same convention as the cycle
probe evidence — not judged relevance). TRUE-guard retention is the
checkpointed answer-time window being a subset of the post-fix window.

Read-only contract: SELECT statements only; embeddings are computed locally
by FastEmbed. Partial results are rewritten every ``--partial-every`` calls
so a crash preserves progress; completed calls are skipped on re-run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT = (
    ROOT.parent / "artifacts/longmemeval-gpt6-full-s-turns/faithful/session-checkpoint.json"
)
DATASET = (
    ROOT.parent
    / "benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json"
)
OUT = ROOT / "artifacts/recall-lift-20260930"
OWNER_ID = "faithful-gpt6-fulls-turns-20260929"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 768
QUESTION_TYPES = {"multi-session", "knowledge-update"}
TRUE_SAMPLE_SIZE = 28
PARTIAL_EVERY = 25

VECTOR_WEIGHT_AFTER = 1.0
KEYWORD_WEIGHT_AFTER = 0.3
WEIGHTS_BEFORE = (0.5, 0.5)

_VECTOR_SQL = """
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
"""

_KEYWORD_SQL_BEFORE = """
    SELECT t.*
      FROM episode_turns t
      JOIN episodes e ON t.episode_id = e.id
     WHERE e.project_id = $1
       AND to_tsvector('english', t.content)
             @@ websearch_to_tsquery('english', $2)
     ORDER BY ts_rank(
         to_tsvector('english', t.content),
         websearch_to_tsquery('english', $2)
     ) DESC, t.id
     LIMIT $3
"""

_GOLD_SQL = """
    SELECT t.id, t.source_session_id, t.turn_index
      FROM episode_turns t
      JOIN episodes e ON t.episode_id = e.id
     WHERE e.project_id = $1
       AND t.source_session_id = ANY($2::text[])
     ORDER BY t.source_session_id, t.turn_index, t.id
"""

_KEYWORD_COUNT_AFTER_SQL = """
    SELECT count(*)
      FROM episode_turns t
      JOIN episodes e ON t.episode_id = e.id
     WHERE e.project_id = $1
       AND to_tsvector('english', t.content) @@ to_tsquery('english', $2)
"""

_KEYWORD_COUNT_BEFORE_SQL = """
    SELECT count(*)
      FROM episode_turns t
      JOIN episodes e ON t.episode_id = e.id
     WHERE e.project_id = $1
       AND to_tsvector('english', t.content) @@ websearch_to_tsquery('english', $2)
"""


def _row_id(row: Any) -> str:
    return str(row["id"])


def _dedupe(rows: list[Any]) -> list[Any]:
    return list({_row_id(r): r for r in rows}.values())


def load_cohort() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic FALSE cohort (every judged-false target-type question,
    every turn-tier call) plus the TRUE retention-guard sample."""
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    types = {
        r["question_id"]: r["question_type"]
        for r in json.loads(DATASET.read_text(encoding="utf-8"))
    }
    evidence = checkpoint["evidence"]
    false_cases: list[dict[str, Any]] = []
    true_cases: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def _calls(row: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        return [
            (i, t)
            for i, t in enumerate(row.get("tool_results") or [])
            if t.get("name") == "weft_recall"
            and (t.get("arguments") or {}).get("tier") == "turns"
        ]

    def _answer_sessions(row: dict[str, Any]) -> list[str]:
        return [
            str(s["session_id"])
            for s in row.get("sessions") or []
            if isinstance(s, dict)
            and str(s.get("session_id", "")).startswith("answer_")
        ]

    false_ids = sorted(
        q for q, r in evidence.items()
        if (r.get("judge") or {}).get("label") is False
        and types.get(q) in QUESTION_TYPES
    )
    true_ids = sorted(
        q for q, r in evidence.items()
        if (r.get("judge") or {}).get("label") is True
        and types.get(q) in QUESTION_TYPES
    )

    for qid in false_ids:
        row = evidence[qid]
        calls = _calls(row)
        if not calls:
            skipped.append({
                "question_id": qid, "segment": "false",
                "reason": "no turn-tier recall calls",
            })
            continue
        for call_index, tool_call in calls:
            args = tool_call["arguments"]
            false_cases.append({
                "case_id": f"{qid}:tool_call_{call_index}",
                "question_id": qid,
                "question_type": types[qid],
                "segment": "false",
                "project_id": args["project_id"],
                "query": args["query"],
                "limit": int(args["limit"]),
                "checkpoint_returned_ids": [
                    str(t["id"]) for t in (tool_call.get("result") or {}).get("turns") or []
                ],
                "answer_session_ids": _answer_sessions(row),
            })

    for qid in true_ids[:TRUE_SAMPLE_SIZE]:
        row = evidence[qid]
        calls = _calls(row)
        if not calls:
            skipped.append({
                "question_id": qid, "segment": "true-sample",
                "reason": "no turn-tier recall calls",
            })
            continue
        for call_index, tool_call in calls:
            args = tool_call["arguments"]
            true_cases.append({
                "case_id": f"{qid}:tool_call_{call_index}",
                "question_id": qid,
                "question_type": types[qid],
                "segment": "true-sample",
                "project_id": args["project_id"],
                "query": args["query"],
                "limit": int(args["limit"]),
                "checkpoint_returned_ids": [
                    str(t["id"]) for t in (tool_call.get("result") or {}).get("turns") or []
                ],
                "answer_session_ids": _answer_sessions(row),
            })

    return false_cases, true_cases, skipped


def _rank_maps(vector_rows: list[Any], keyword_rows: list[Any]):
    vector_rank = {_row_id(r): i + 1 for i, r in enumerate(vector_rows)}
    keyword_rank = {_row_id(r): i + 1 for i, r in enumerate(keyword_rows)}
    return vector_rank, keyword_rank


def analyze_funnel(
    vector_rows: list[Any],
    keyword_rows: list[Any],
    *,
    candidate_limit: int,
    boundary_index: int,
    output_k: int,
    vec_w: float,
    kw_w: float,
    returned: list[str],
    gold_ids: list[str],
) -> dict[str, Any]:
    """Re-fuse captured halves with the production function and classify
    every gold id, mirroring the cycle-probe verdict semantics."""
    from weft.episode_turns import _rrf_fuse_turn_rows
    from weft.relevance import rank_turns

    vector_rank, keyword_rank = _rank_maps(vector_rows, keyword_rows)
    union = len(set(vector_rank) | set(keyword_rank))
    fused = _rrf_fuse_turn_rows(
        vector_rows, keyword_rows,
        candidate_limit=candidate_limit, top_k=union,
        vector_weight=vec_w, keyword_weight=kw_w,
    )
    fused_rank = {str(t.id): i + 1 for i, (t, _) in enumerate(fused)}
    fused_score = {str(t.id): float(s) for t, s in fused}
    fused_cap = fused[: max(boundary_index, output_k)]
    ranked = [s.turn for s in rank_turns(fused_cap, now=None)]
    window = [str(t.id) for t in ranked[:output_k]]
    boundary = (
        fused[boundary_index - 1][1]
        if len(fused) >= boundary_index
        else (fused[-1][1] if fused else None)
    )
    gold: dict[str, Any] = {}
    for gid in gold_ids:
        if gid not in vector_rank and gid not in keyword_rank:
            verdict = "absent_from_candidates"
        elif gid in returned:
            verdict = "returned"
        else:
            verdict = "candidate_cut_by_top_k"
        score = fused_score.get(gid)
        margin = (
            score - float(boundary)
            if score is not None and boundary is not None
            else None
        )
        gold[gid] = {
            "verdict": verdict,
            "vector_rank": vector_rank.get(gid),
            "keyword_rank": keyword_rank.get(gid),
            "fused_rank": fused_rank.get(gid),
            "margin": margin,
            "margin_percent": (
                100.0 * margin / float(boundary)
                if margin is not None and boundary
                else None
            ),
            "in_answer_window": gid in set(returned),
        }
    return {
        "window": window,
        "gold": gold,
        "vector_candidates": len(vector_rows),
        "keyword_candidates": len(keyword_rows),
        "union_candidates": union,
    }


async def capture_after(
    case: dict[str, Any], pool: Any, embedder: Any, conn: Any,
    expansion_slots: int = 0,
):
    from weft.auth import current_user_id
    from weft.turn_recall import temporal_anchor

    limit = int(case["limit"])
    top_k = min(limit, 10)
    sql_width = top_k * 5
    vector_rows: list = []
    keyword_rows: list = []
    returned: list[str] = []

    def on_diag(anchor, vec, kw, turns):
        vector_rows.extend(vec)
        keyword_rows.extend(kw)
        returned.extend(str(t.id) for t in turns)

    token = current_user_id.set(OWNER_ID)
    try:
        await temporal_anchor(
            pool, case["query"],
            project_id=case["project_id"],
            top_k_per_anchor=top_k,
            candidate_sql_limit=sql_width,
            anchor_result_limit=limit,
            embedder=embedder,
            diag_callback=on_diag,
            expansion_slots=expansion_slots,
        )
    finally:
        current_user_id.reset(token)
    return _dedupe(vector_rows), _dedupe(keyword_rows), returned, top_k, sql_width


async def capture_before(case: dict[str, Any], conn: Any, embedding: list[float]):
    limit = int(case["limit"])
    top_k = max(1, limit // 2)
    sql_width = top_k * 3
    vector_rows = await conn.fetch(
        _VECTOR_SQL, embedding, case["project_id"], sql_width,
    )
    keyword_rows = await conn.fetch(
        _KEYWORD_SQL_BEFORE, case["project_id"], case["query"], sql_width,
    )
    return vector_rows, keyword_rows, top_k, sql_width


async def run(args: argparse.Namespace) -> int:
    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL")
    if not dsn:
        print("LONGMEMEVAL_DATABASE_URL is required", file=sys.stderr)
        return 2
    parsed = urlsplit(dsn)
    if (parsed.hostname, parsed.port, parsed.path.lstrip("/")) != (
        "127.0.0.1", 5433, "lme_bench"
    ):
        print("Refusing non-local/non-benchmark DSN", file=sys.stderr)
        return 2

    from weft.config import WeftConfig
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider

    false_cases, true_cases, skipped = load_cohort()
    if args.max_calls:
        false_cases = false_cases[: args.max_calls]
        true_cases = true_cases[: max(0, args.max_calls // 4)]
    cohort = false_cases + true_cases
    print(
        f"cohort: {len(false_cases)} false calls / {len(set(c['question_id'] for c in false_cases))}"
        f" false questions; {len(true_cases)} true-sample calls; {len(skipped)} skipped",
        flush=True,
    )

    partial_path = OUT / f"{args.out_name}_partial.json"
    final_path = OUT / f"{args.out_name}_results.json"
    baseline_map: dict[str, list[str]] = {}
    if args.baseline and Path(args.baseline).exists():
        try:
            _baseline_doc = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            baseline_map = {
                c["case_id"]: c["after"]["returned"] for c in _baseline_doc["records"]
            }
            print(f"prefix baseline loaded: {args.baseline}", flush=True)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"baseline load failed ({exc}); prefix assertion disabled", flush=True)
    records: list[dict[str, Any]] = []
    if not args.fresh and partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8")).get("records", [])
        print(f"resuming with {len(records)} completed calls", flush=True)
    done = {r["case_id"] for r in records}

    config = WeftConfig()
    config.database.url = dsn
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    config.retrieval.recovery_mode = "off"
    assert config.embedding.model == EMBED_MODEL
    assert config.embedding.dimensions == EMBED_DIMS
    embedder = get_provider(
        config.embedding.provider,
        model_name=EMBED_MODEL,
        dimensions=EMBED_DIMS,
    )
    pool = await create_pool(config)

    embeddings: dict[str, list[float]] = {}
    gold_cache: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}
    processed_since_partial = 0

    def write_partial() -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps({
                "schema": "weft.longmemeval.round2-census.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "read_only": True,
                "dsn": {"host": parsed.hostname, "port": parsed.port,
                        "database": parsed.path.lstrip("/"), "user": parsed.username},
                "fusion_after": {"vector": VECTOR_WEIGHT_AFTER, "keyword": KEYWORD_WEIGHT_AFTER},
                "fusion_before": {"vector": WEIGHTS_BEFORE[0], "keyword": WEIGHTS_BEFORE[1]},
                "expansion_slots": args.expansion_slots,
                "records": records,
                "skipped": skipped,
            }, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    try:
        from weft.auth import current_user_id
        from weft.db.connection import acquire
        from weft.store import build_or_tsquery

        for case in cohort:
            if case["case_id"] in done:
                continue
            started = time.perf_counter()
            record: dict[str, Any] = {
                "case_id": case["case_id"],
                "question_id": case["question_id"],
                "question_type": case["question_type"],
                "segment": case["segment"],
                "limit": case["limit"],
                "checkpoint_returned_ids": case["checkpoint_returned_ids"],
            }
            token = current_user_id.set(OWNER_ID)
            try:
                async with acquire(pool) as conn:
                    gold_turns: dict[str, str] = {}
                    if case["segment"] == "false":
                        gkey = (case["project_id"], tuple(case["answer_session_ids"]))
                        if gkey not in gold_cache:
                            rows = await conn.fetch(
                                _GOLD_SQL, case["project_id"],
                                list(case["answer_session_ids"]),
                            )
                            gold_cache[gkey] = {
                                str(r["id"]): f"{r['source_session_id']} t{r['turn_index']}"
                                for r in rows
                            }
                        gold_turns = gold_cache[gkey]

                    embedding = embeddings.get(case["query"])
                    if embedding is None:
                        embedding = await embedder.embed(case["query"])
                        embeddings[case["query"]] = embedding

                    # AFTER: authentic production path — captured TWICE in the
                    # same run (same wall clock) so the prefix assertion is
                    # drift-free: base = expansion 0, expanded = production.
                    vec_b0, kw_b0, returned_base, top_k_a, width_a = await capture_after(
                        case, pool, embedder, conn, expansion_slots=0,
                    )
                    vec_a, kw_a, returned_a, top_k_a, width_a = await capture_after(
                        case, pool, embedder, conn,
                        expansion_slots=args.expansion_slots,
                    )
                    record["after_base"] = {
                        "returned": returned_base,
                        "expansion_slots": 0,
                    }
                    record["after"] = {
                        "returned": returned_a,
                        "top_k_per_anchor": top_k_a,
                        "candidate_sql_limit": width_a,
                        "expansion_slots": args.expansion_slots,
                    }
                    baseline_window = baseline_map.get(case["case_id"])
                    if baseline_window is not None:
                        record["after"]["prefix_matches_baseline"] = (
                            returned_a[: case["limit"]] == baseline_window[: case["limit"]]
                        )
                    # Drift-free prefix assertion: same-run, same wall clock.
                    record["after"]["prefix_preserved_same_run"] = (
                        returned_a[: case["limit"]] == returned_base[: case["limit"]]
                    )
                    if case["segment"] == "false":
                        fts = build_or_tsquery(case["query"])
                        record["after"]["keyword_match_total"] = int(
                            await conn.fetchval(
                                _KEYWORD_COUNT_AFTER_SQL,
                                case["project_id"], fts or "",
                            )
                        ) if fts else 0
                        analysis = analyze_funnel(
                            vec_a, kw_a,
                            candidate_limit=width_a,
                            boundary_index=case["limit"],
                            output_k=case["limit"],
                            vec_w=VECTOR_WEIGHT_AFTER,
                            kw_w=KEYWORD_WEIGHT_AFTER,
                            returned=returned_a,
                            gold_ids=list(gold_turns),
                        )
                        record["after"]["analysis"] = analysis

                    # BEFORE: old funnel mirrored via direct SELECTs.
                    vec_b, kw_b, top_k_b, width_b = await capture_before(
                        case, conn, embedding,
                    )
                    record["before"] = {
                        "top_k_per_anchor": top_k_b,
                        "candidate_sql_limit": width_b,
                    }
                    if case["segment"] == "false":
                        record["before"]["keyword_match_total"] = int(
                            await conn.fetchval(
                                _KEYWORD_COUNT_BEFORE_SQL,
                                case["project_id"], case["query"],
                            )
                        )
                        before_returned_analysis = analyze_funnel(
                            vec_b, kw_b,
                            candidate_limit=width_b,
                            boundary_index=top_k_b,
                            output_k=top_k_b,
                            vec_w=WEIGHTS_BEFORE[0],
                            kw_w=WEIGHTS_BEFORE[1],
                            returned=[],  # filled from the computed window below
                            gold_ids=[],
                        )
                        before_window = before_returned_analysis["window"]
                        record["before"]["returned"] = before_window
                        analysis_b = analyze_funnel(
                            vec_b, kw_b,
                            candidate_limit=width_b,
                            boundary_index=top_k_b,
                            output_k=top_k_b,
                            vec_w=WEIGHTS_BEFORE[0],
                            kw_w=WEIGHTS_BEFORE[1],
                            returned=before_window,
                            gold_ids=list(gold_turns),
                        )
                        analysis_b.pop("window", None)
                        record["before"]["analysis"] = analysis_b

                    if case["segment"] == "true-sample":
                        checkpoint_ids = set(case["checkpoint_returned_ids"])
                        retained = checkpoint_ids.issubset(set(returned_a))
                        record["retention"] = {
                            "checkpoint_ids": len(checkpoint_ids),
                            "retained": retained,
                            "lost_ids": sorted(checkpoint_ids - set(returned_a)),
                        }
                        if not retained:
                            vector_rank, keyword_rank = _rank_maps(vec_a, kw_a)
                            from weft.episode_turns import _rrf_fuse_turn_rows
                            fused = _rrf_fuse_turn_rows(
                                vec_a, kw_a,
                                candidate_limit=width_a,
                                top_k=len(set(vector_rank) | set(keyword_rank)),
                                vector_weight=VECTOR_WEIGHT_AFTER,
                                keyword_weight=KEYWORD_WEIGHT_AFTER,
                            )
                            fused_rank = {str(t.id): i + 1 for i, (t, _) in enumerate(fused)}
                            record["retention"]["lost_detail"] = [
                                {
                                    "id": lost,
                                    "vector_rank": vector_rank.get(lost),
                                    "keyword_rank": keyword_rank.get(lost),
                                    "fused_rank": fused_rank.get(lost),
                                }
                                for lost in record["retention"]["lost_ids"]
                            ]
            finally:
                current_user_id.reset(token)

            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            records.append(record)
            processed_since_partial += 1
            print(
                f"[{len(records)}/{len(cohort)}] {case['case_id']} "
                f"({case['question_type']}, limit={case['limit']}) "
                f"{record['elapsed_ms']}ms",
                flush=True,
            )
            if processed_since_partial >= PARTIAL_EVERY:
                write_partial()
                processed_since_partial = 0
    finally:
        await pool.close()

    write_partial()
    final_path.write_text(
        json.dumps({
            "schema": "weft.longmemeval.round2-census.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "fusion_after": {"vector": VECTOR_WEIGHT_AFTER, "keyword": KEYWORD_WEIGHT_AFTER},
            "fusion_before": {"vector": WEIGHTS_BEFORE[0], "keyword": WEIGHTS_BEFORE[1]},
            "expansion_slots": args.expansion_slots,
            "records": records,
            "skipped": skipped,
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {final_path} ({len(records)} calls, {len(skipped)} skipped)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=0,
                        help="smoke-test bound on cohort size (0 = full census)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing partial results")
    parser.add_argument("--expansion-slots", type=int, default=5,
                        help="additive session-expansion slots on the AFTER path "
                             "(production turn-tier default)")
    parser.add_argument("--out-name", default="round2_census",
                        help="output basename: <out-name>_results.json / _partial.json")
    parser.add_argument("--baseline", default=str(OUT / "round2_census_results.json"),
                        help="census results whose windows define the protected "
                             "prefix for the per-call prefix assertion")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
