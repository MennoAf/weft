"""Checkpoint-driven, SELECT-only before/after probe for turn retrieval.

Run through probe_loop_rank_misses.py. Extra false-judgment questions are
selected from the saved checkpoint; for those cases the reported per-turn
gold set is every turn in the checkpoint's answer_* sessions. That broad
session-level set is diagnostic, not a replacement for semantic gold labels.
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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT.parent / "artifacts/longmemeval-gpt6-full-s-turns/faithful/session-checkpoint.json"
OWNER_ID = "faithful-gpt6-fulls-turns-20260929"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 768
# Must mirror the turn-tier production fusion weights
# (weft.turn_recall.temporal_anchor → weft.episode_turns._rrf_fuse_turn_rows).
FUSION_VECTOR_WEIGHT = 1.0
FUSION_KEYWORD_WEIGHT = 0.3

FIXED_GOLD: dict[str, dict[str, dict[str, str]]] = {
    "0a995998": {
        "et-37ebe585ea9a47659e06aef1cb36f369": {
            "label": "boots_pickup", "note": "Zara boots pickup"
        },
        "et-95df0c34a1054029b174b0bd7d495c68": {
            "label": "boots_return", "note": "Zara boots return/exchange"
        },
        "et-b12256b35af44ed28de8737983ef7a7f": {
            "label": "blazer_pickup", "note": "navy-blue blazer dry-cleaning pickup"
        },
        "et-b4b6b947173e4524b19d754c55ba63a1": {
            "label": "green_sweater", "note": "sweater lent to sister"
        },
    },
    "0ddfec37": {
        "et-0a429ba0b44f4f5890a871d842740c25": {
            "label": "gold_15", "note": "July 15-baseball count"
        },
        "et-6a08a2e1b85949fb88ebabaf0b19bede": {
            "label": "later_20_user", "note": "later 20-baseball user statement"
        },
        "et-11f48ab37ed944989a88338060468773": {
            "label": "later_20_assistant", "note": "later 20-baseball assistant echo"
        },
    },
}


def _recall_calls(row: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (i, item)
        for i, item in enumerate(row.get("tool_results", []))
        if item.get("name") == "weft_recall"
        and (item.get("arguments") or {}).get("tier") == "turns"
    ]


def load_cases(max_extra_questions: int = 5) -> list[dict[str, Any]]:
    """Load the original saved misses and five deterministic extra misses."""
    from weft.turn_recall import extract_anchors, temporal_query_variants

    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    evidence = checkpoint["evidence"]
    cases: list[dict[str, Any]] = []
    fixed_ids = ("0a995998", "0ddfec37")
    for qid in fixed_ids:
        row = evidence[qid]
        for call_index, tool_call in _recall_calls(row):
            args = tool_call["arguments"]
            cases.append({
                "case_id": f"{qid}:tool_call_{len([c for c in cases if c['question_id'] == qid])}",
                "question_id": qid,
                "project_id": args["project_id"],
                "query": args["query"],
                "limit": int(args["limit"]),
                "checkpoint_returned_ids": [
                    str(turn["id"]) for turn in (tool_call.get("result") or {}).get("turns", [])
                ],
                "gold_turns": FIXED_GOLD[qid],
                "gold_session_ids": [],
                "gold_source": "brief-pinned gold turn IDs",
                "tool_call_index": call_index,
            })

    extra_questions = 0
    for qid, row in evidence.items():
        judge = row.get("judge") or {}
        if (
            qid in fixed_ids
            or qid == "09d032c9"
            or row.get("status") != "completed"
            or not isinstance(judge, dict)
            or judge.get("label") is not False
        ):
            continue
        answer_sessions = [
            str(session["session_id"])
            for session in row.get("sessions", [])
            if isinstance(session, dict)
            and str(session.get("session_id", "")).startswith("answer_")
        ]
        if not answer_sessions:
            continue
        selected = None
        for call_index, tool_call in _recall_calls(row):
            args = tool_call.get("arguments") or {}
            query = args.get("query")
            if (
                not isinstance(query, str)
                or not args.get("project_id")
                or not isinstance(args.get("limit"), int)
                or extract_anchors(query)
                or temporal_query_variants(query) != [query]
            ):
                continue
            selected = (call_index, tool_call, args)
            break
        if selected is None:
            continue
        call_index, tool_call, args = selected
        cases.append({
            "case_id": f"{qid}:tool_call_{call_index}",
            "question_id": qid,
            "project_id": args["project_id"],
            "query": args["query"],
            "limit": int(args["limit"]),
            "checkpoint_returned_ids": [
                str(turn["id"]) for turn in (tool_call.get("result") or {}).get("turns", [])
            ],
            "gold_turns": {},
            "gold_session_ids": answer_sessions,
            "gold_source": "all turns in checkpoint answer_* sessions",
            "tool_call_index": call_index,
            "judge_label": False,
        })
        extra_questions += 1
        if extra_questions >= max_extra_questions:
            break
    if extra_questions < max_extra_questions:
        raise RuntimeError(
            f"checkpoint yielded only {extra_questions} eligible extra false-judgment cases"
        )
    return cases


def _row_id(row: Any) -> str:
    return str(row["id"])


async def run_case(case: dict[str, Any], pool: Any, embedder: Any, phase: str,
                   candidate_multiplier: int) -> dict[str, Any]:
    from weft.auth import current_user_id
    from weft.db.connection import acquire
    from weft.turn_recall import temporal_anchor

    limit = case["limit"]
    top_k = max(1, limit // 2) if phase == "before" else min(limit, 10)
    fusion_window = top_k if phase == "before" else limit
    sql_width = top_k * (3 if phase == "before" else candidate_multiplier)
    vector_rows: list[Any] = []
    keyword_rows: list[Any] = []
    returned: list[str] = []
    anchor_probes: list[dict[str, Any]] = []
    user_token = current_user_id.set(OWNER_ID)
    try:
        async with acquire(pool) as conn:
            if case["gold_session_ids"]:
                gold_rows = await conn.fetch(
                    """
                    SELECT t.id, t.source_session_id, t.turn_index
                      FROM episode_turns t
                      JOIN episodes e ON t.episode_id = e.id
                     WHERE e.project_id = $1
                       AND t.source_session_id = ANY($2::text[])
                     ORDER BY t.source_session_id, t.turn_index, t.id
                    """,
                    case["project_id"], case["gold_session_ids"],
                )
                gold_turns = {
                    str(row["id"]): {
                        "label": f"{row['source_session_id']} t{row['turn_index']}",
                        "note": "checkpoint answer-session turn",
                    }
                    for row in gold_rows
                }
            else:
                gold_turns = dict(case["gold_turns"])
            gold_ids = list(gold_turns)

            def on_diag(anchor: str, vectors: list, keywords: list, turns: list) -> None:
                vector_rows.extend(vectors)
                keyword_rows.extend(keywords)
                returned.extend(str(turn.id) for turn in turns)

            def on_probe(payload: dict[str, Any]) -> None:
                anchor_probes.append(dict(payload))

            await temporal_anchor(
                pool,
                case["query"],
                project_id=case["project_id"],
                top_k_per_anchor=top_k,
                candidate_sql_limit=sql_width,
                anchor_result_limit=limit if phase == "after" else None,
                embedder=embedder,
                diag_callback=on_diag,
                probe_diag_callback=on_probe,
            )

            # Match the live keyword half for this phase, with only fixed
            # expressions interpolated; query text remains a bind parameter.
            if phase == "after":
                from weft.store import build_or_tsquery
                fts_query = build_or_tsquery(case["query"])
                if fts_query:
                    keyword_total = int(await conn.fetchval(
                        """SELECT count(*) FROM episode_turns t
                             JOIN episodes e ON t.episode_id = e.id
                            WHERE e.project_id = $1
                              AND to_tsvector('english', t.content)
                                  @@ to_tsquery('english', $2)""",
                        case["project_id"], fts_query,
                    ))
                else:
                    keyword_total = 0
            else:
                keyword_total = int(await conn.fetchval(
                    """SELECT count(*) FROM episode_turns t
                         JOIN episodes e ON t.episode_id = e.id
                        WHERE e.project_id = $1
                          AND to_tsvector('english', t.content)
                              @@ websearch_to_tsquery('english', $2)""",
                    case["project_id"], case["query"],
                ))
    finally:
        current_user_id.reset(user_token)

    # Collapse duplicate rows from instrumentation while preserving each
    # half's SQL rank order.
    vector_by_id = {_row_id(row): row for row in vector_rows}
    keyword_by_id = {_row_id(row): row for row in keyword_rows}
    vector_rows = list(vector_by_id.values())
    keyword_rows = list(keyword_by_id.values())
    vector_rank = {_row_id(row): i + 1 for i, row in enumerate(vector_rows)}
    keyword_rank = {_row_id(row): i + 1 for i, row in enumerate(keyword_rows)}
    from weft.episode_turns import _rrf_fuse_turn_rows
    from weft.relevance import rank_turns

    union_size = len(set(vector_rank) | set(keyword_rank))
    fused_all = _rrf_fuse_turn_rows(
        vector_rows,
        keyword_rows,
        candidate_limit=sql_width,
        top_k=union_size,
        vector_weight=FUSION_VECTOR_WEIGHT,
        keyword_weight=FUSION_KEYWORD_WEIGHT,
    )
    fused_score = {str(turn.id): float(score) for turn, score in fused_all}
    fused_rank = {str(turn.id): i + 1 for i, (turn, _) in enumerate(fused_all)}
    boundary = fused_all[fusion_window - 1][1] if len(fused_all) >= fusion_window else None

    gold_verdicts = {}
    for gold_id, details in gold_turns.items():
        if gold_id not in vector_rank and gold_id not in keyword_rank:
            verdict = "absent_from_candidates"
        elif gold_id not in fused_rank or fused_rank[gold_id] > fusion_window:
            verdict = "candidate_cut_by_top_k"
        elif gold_id in returned:
            verdict = "returned"
        else:
            verdict = "candidate_cut_by_top_k"
        score = fused_score.get(gold_id)
        margin = score - float(boundary) if score is not None and boundary is not None else None
        gold_verdicts[gold_id] = {
            "label": details["label"],
            "note": details["note"],
            "vector_rank": vector_rank.get(gold_id),
            "keyword_rank": keyword_rank.get(gold_id),
            "fused_rank": fused_rank.get(gold_id),
            "fused_score": score,
            "final_rank": returned.index(gold_id) + 1 if gold_id in returned else None,
            "margin_to_boundary": margin,
            "margin_percent": (100.0 * margin / float(boundary))
                if margin is not None and boundary else None,
            "verdict": verdict,
        }

    return {
        "case_id": case["case_id"],
        "question_id": case["question_id"],
        "query": case["query"],
        "limit": limit,
        "phase": phase,
        "keyword_mode": "or" if phase == "after" else "websearch_and",
        "top_k_per_anchor": top_k,
        "candidate_sql_limit": sql_width,
        "fusion_window": fusion_window,
        "keyword_matched_turns_in_project": keyword_total,
        "vector_candidate_count": len(vector_rows),
        "keyword_candidate_count": len(keyword_rows),
        "returned_ids": returned,
        "checkpoint_returned_ids": case["checkpoint_returned_ids"],
        "matches_checkpoint": returned == case["checkpoint_returned_ids"],
        "no_new_truncation_below_old_coverage": set(case["checkpoint_returned_ids"]).issubset(returned),
        "gold_source": case["gold_source"],
        "gold_turns": gold_verdicts,
        "probe_diagnostics": anchor_probes,
    }


def render_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Question / call | Gold turn | Verdict | vec rank | keyword rank | fused rank | margin vs boundary | margin % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        for turn in case["gold_turns"].values():
            margin = turn["margin_to_boundary"]
            margin_text = "—" if margin is None else f"{margin:+.7f}"
            pct = turn["margin_percent"]
            pct_text = "—" if pct is None else f"{pct:+.2f}%"
            lines.append(
                f"| {case['case_id']} | {turn['label']} | {turn['verdict']} | "
                f"{turn['vector_rank'] or '—'} | {turn['keyword_rank'] or '—'} | "
                f"{turn['fused_rank'] or '—'} | {margin_text} | {pct_text} |"
            )
    lines.extend([
        "",
        "| Question / call | keyword matches | keyword candidates | return window | checkpoint match | old IDs retained |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for case in results:
        lines.append(
            f"| {case['case_id']} | {case['keyword_matched_turns_in_project']} | "
            f"{case['keyword_candidate_count']} | {len(case['returned_ids'])} | "
            f"{case['matches_checkpoint']} | {case['no_new_truncation_below_old_coverage']} |"
        )
    return "\n".join(lines) + "\n"


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

    config = WeftConfig()
    config.database.url = dsn
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    config.retrieval.recovery_mode = "off"
    assert config.embedding.model == EMBED_MODEL, config.embedding.model
    assert config.embedding.dimensions == EMBED_DIMS, config.embedding.dimensions
    embedder = get_provider(
        config.embedding.provider,
        model_name=EMBED_MODEL,
        dimensions=EMBED_DIMS,
    )
    pool = await create_pool(config)
    cases = load_cases(args.extra_questions)
    results = []
    try:
        for case in cases:
            print(f"probing {case['case_id']} [{args.phase}] ...", flush=True)
            result = await run_case(
                case, pool, embedder, args.phase, args.candidate_multiplier
            )
            if args.phase == "before" and not result["matches_checkpoint"]:
                print(f"WARNING: checkpoint divergence for {case['case_id']}", file=sys.stderr)
            results.append(result)
    finally:
        await pool.close()

    args.out.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.label}{args.phase}_"
    (args.out / f"{prefix}probe_results.json").write_text(
        json.dumps({
            "probe": "turn-recall-lift",
            "phase": args.phase,
            "label": args.label,
            "fusion_vector_weight": FUSION_VECTOR_WEIGHT,
            "fusion_keyword_weight": FUSION_KEYWORD_WEIGHT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "dsn": {"host": parsed.hostname, "port": parsed.port,
                    "database": parsed.path.lstrip("/"), "user": parsed.username},
            "extra_false_judgment_questions": args.extra_questions,
            "cases": results,
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    table = render_table(results)
    (args.out / f"{prefix}verdicts.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {args.out}/{prefix}probe_results.json and verdicts.md")
    if args.phase == "before":
        return 0 if all(r["matches_checkpoint"] for r in results) else 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--out", type=Path, default=Path("artifacts/recall-lift-20260930"))
    parser.add_argument("--extra-questions", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument(
        "--label", default="",
        help="filename prefix for outputs (e.g. cycle3_ keeps prior evidence intact)",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
