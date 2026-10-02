"""Round-8 measurement: price the session-expansion depth curve.

One capture pass over the round-2 census cohort; each call yields:

* the authentic production base window (expansion 0),
* the full union fused ranking (for query-ranked filler ordering),
* the UNBOUNDED ordered session-sibling list (R7 rule: matched sessions =
  top-3 by best window rank; candidates = matched-session turns not in the
  window; order = session priority, |turn_index − hit|, turn_index, id),
* token_count for every candidate turn (context pricing).

Depths {5, 8, 10} are then evaluated in-memory with production semantics:
window = base + siblings[:depth] + query-ranked filler to reach `limit+depth`.
For each depth: missed-evidence returned, strict/extended absence, TRUE-guard
retention, named golds, FALSE checkpoint drops, fill rate, added-turns and
summed token_count stats, and the per-slot yield curve (slots 1..10).

Read-only contract: SELECT statements only; local FastEmbed embeddings.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFLINE = ROOT / "benchmarks/longmemeval/offline_tests"
if str(OFFLINE) not in sys.path:
    sys.path.insert(0, str(OFFLINE))

import probe_round2_census as rc  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "artifacts/recall-lift-20260930"
OWNER_ID = "faithful-gpt6-fulls-turns-20260929"
K_RRF = 60
VECTOR_W, KEYWORD_W = 1.0, 0.3
DEPTHS = (5, 8, 10)
MAX_SESSIONS = 3
SLOT_CURVE_DEPTH = 10

NAMED_GOLD = {
    "et-0a429ba0b44f4f5890a871d842740c25": ("gold_15", "0ddfec37:tool_call_0"),
    "et-95df0c34a1054029b174b0bd7d495c68": ("boots_return", "0a995998:tool_call_1"),
    "et-b12256b35af44ed28de8737983ef7a7f": ("blazer_pickup", "0a995998:tool_call_1"),
}
NAMED_CALLS = {cid for _, cid in NAMED_GOLD.values()}

_WINDOW_SESSION_SQL = """
    SELECT id, source_session_id, turn_index
      FROM episode_turns
     WHERE id = ANY($1::text[])
"""

_SESSION_TURNS_SQL = """
    SELECT t.id, t.source_session_id, t.turn_index, t.token_count
      FROM episode_turns t
      JOIN episodes e ON t.episode_id = e.id
     WHERE e.project_id = $1
       AND t.source_session_id = ANY($2::text[])
     ORDER BY t.source_session_id, t.turn_index, t.id
"""


def _row_id(row):
    return str(row["id"])


def _dedupe(rows):
    return list({_row_id(r): r for r in rows}.values())


async def capture(case, pool, embedder):
    from weft.auth import current_user_id
    from weft.db.connection import acquire
    from weft.turn_recall import temporal_anchor

    limit = int(case["limit"])
    top_k = min(limit, 10)
    sql_width = top_k * 5
    vector_rows: list = []
    keyword_rows: list = []
    token = current_user_id.set(OWNER_ID)
    try:
        async with acquire(pool) as conn:
            def on_diag(anchor, vec, kw, turns):
                vector_rows.extend(vec)
                keyword_rows.extend(kw)

            await temporal_anchor(
                pool, case["query"],
                project_id=case["project_id"],
                top_k_per_anchor=top_k,
                candidate_sql_limit=sql_width,
                anchor_result_limit=limit,
                embedder=embedder,
                diag_callback=on_diag,
            )
    finally:
        current_user_id.reset(token)
    return _dedupe(vector_rows), _dedupe(keyword_rows), top_k, sql_width


async def main() -> int:
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

    from weft.auth import current_user_id
    from weft.config import WeftConfig
    from weft.db.connection import acquire, create_pool
    from weft.embeddings import get_provider
    from weft.episode_turns import _rrf_fuse_turn_rows, _row_to_turn
    from weft.relevance import rank_turns

    census = json.loads((OUT / "round2_census_results.json").read_text(encoding="utf-8"))
    records = census["records"]
    cohort = {c["case_id"]: c for c in rc.load_cohort()[0] + rc.load_cohort()[1]}

    config = WeftConfig()
    config.database.url = dsn
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    config.retrieval.recovery_mode = "off"
    embedder = get_provider(
        config.embedding.provider,
        model_name="BAAI/bge-small-en-v1.5",
        dimensions=768,
    )
    pool = await create_pool(config)

    acc = {
        d: {"missed_returned": 0, "true_retained": 0, "true_total": 0,
            "strict_absent": 0, "rescued_absent": 0, "drops": 0, "false_total": 0,
            "len_ok": True, "named": {}, "fill_full": 0, "fill_any": 0,
            "added_turns": [], "added_tokens": [], "by_type": defaultdict(lambda: [0, 0])}
        for d in DEPTHS
    }
    slot_yield = [0] * SLOT_CURVE_DEPTH
    sibling_sizes: list[int] = []
    per_call = []
    session_turn_cache: dict[tuple[str, tuple[str, ...]], dict[str, list[tuple[str, int]]]] = {}
    token_by_id: dict[str, int] = {}
    token = current_user_id.set(OWNER_ID)
    try:
        for i, call in enumerate(records, 1):
            case = cohort.get(call["case_id"])
            limit = int(call["limit"])
            ck = set(call["checkpoint_returned_ids"])
            gold = call["after"].get("analysis", {}).get("gold") or {}

            vec, kw, top_k, width = await capture(case, pool, embedder)
            fused = _rrf_fuse_turn_rows(
                vec, kw,
                candidate_limit=width,
                top_k=max(1, len({*{_row_id(r) for r in vec}, *{_row_id(r) for r in kw}})),
                vector_weight=VECTOR_W, keyword_weight=KEYWORD_W,
            )
            fused_turns = [t for t, _ in fused]
            fused_scores = [float(s) for _, s in fused]
            # Production semantics: the recency rerank applies ONLY within
            # the fused top-`limit` (the prefix pipeline); the filler order
            # comes from the wider fused[:limit+max_depth] slice ranked the
            # same way. Ranking the full union here would let wall-clock
            # recency pull far-below-boundary turns into the base window —
            # not what the production funnel does.
            ranked_capped = [
                s.turn for s in rank_turns(fused[:limit], now=None)
            ]
            base_window = [t.id for t in ranked_capped[:limit]]
            ranked_filler = [
                s.turn for s in rank_turns(
                    fused[: limit + SLOT_CURVE_DEPTH], now=None,
                )
            ]
            window_set = set(base_window)

            for r in (*vec, *kw):
                try:
                    token_by_id[_row_id(r)] = int(r["token_count"])
                except (KeyError, IndexError):
                    pass

            async with acquire(pool) as conn:
                wrows = await conn.fetch(_WINDOW_SESSION_SQL, base_window)
                session_of = {
                    str(r["id"]): (str(r["source_session_id"]) if r["source_session_id"] else None)
                    for r in wrows
                }
                index_of = {str(r["id"]): int(r["turn_index"]) for r in wrows}
                sessions = sorted({s for s in session_of.values() if s})
                sturns: dict[str, list[tuple[str, int]]] = {}
                if sessions:
                    skey = (case["project_id"], tuple(sessions))
                    if skey not in session_turn_cache:
                        srows = await conn.fetch(
                            _SESSION_TURNS_SQL, case["project_id"], sessions,
                        )
                        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
                        for r in srows:
                            grouped[str(r["source_session_id"])].append(
                                (str(r["id"]), int(r["turn_index"]))
                            )
                            token_by_id.setdefault(str(r["id"]), int(r["token_count"] or 0))
                        session_turn_cache[skey] = dict(grouped)
                    sturns = session_turn_cache[skey]

            priority: dict[str, int] = {}
            hit_index: dict[str, int] = {}
            for pos, tid in enumerate(base_window):
                sid = session_of.get(tid)
                if sid is None:
                    continue
                if sid not in priority:
                    priority[sid] = pos
                    hit_index[sid] = index_of.get(tid, 0)
            matched = sorted(priority, key=lambda s: (priority[s], s))[:MAX_SESSIONS]
            candidates = []
            for prio, sid in enumerate(matched):
                hit = hit_index.get(sid, 0)
                for tid, tidx in sturns.get(sid, []):
                    if tid in window_set:
                        continue
                    candidates.append((prio, abs(tidx - hit), tidx, tid))
            candidates.sort()
            sibling_list = [c[3] for c in candidates]
            sibling_sizes.append(len(sibling_list))

            call_slots = sibling_list + [
                t.id for t in ranked_filler
                if t.id not in window_set and t.id not in set(sibling_list)
            ]
            call_meta = {
                "case_id": call["case_id"],
                "segment": call["segment"],
                "question_type": call["question_type"],
                "limit": limit,
                "checkpoint_returned_ids": call["checkpoint_returned_ids"],
                "base_window": base_window,
                "sibling_pool": len(sibling_list),
                "windows": {},
                "appended_tokens": {},
            }

            for depth in DEPTHS:
                a = acc[depth]
                appended = call_slots[:depth]
                window = base_window + appended
                rec_window_len = limit + depth
                if len(window) < rec_window_len:
                    window = window + [
                        t.id for t in ranked_filler
                        if t.id not in set(window)
                    ][: rec_window_len - len(window)]
                wset = set(window)
                call_meta.setdefault("windows", {})[str(depth)] = window

                tokens = [token_by_id.get(tid, 0) for tid in appended]
                a["added_turns"].append(len(appended))
                a["added_tokens"].append(sum(tokens))
                if len(sibling_list) >= depth:
                    a["fill_full"] += 1
                if sibling_list:
                    a["fill_any"] += 1
                if call["segment"] == "true-sample":
                    a["true_total"] += 1
                    if ck.issubset(wset):
                        a["true_retained"] += 1
                if call["segment"] == "false":
                    a["false_total"] += 1
                    if not ck.issubset(wset):
                        a["drops"] += 1
                    for n, tid in enumerate(appended[:SLOT_CURVE_DEPTH]):
                        if tid in gold and tid not in ck:
                            slot_yield[n] += 1
                    for gid, g in gold.items():
                        t = call["question_type"]
                        if gid in wset and gid not in ck:
                            a["missed_returned"] += 1
                            a["by_type"][t][0] += 1
                        if g["verdict"] == "absent_from_candidates":
                            a["strict_absent"] += 1
                            if gid in wset:
                                a["rescued_absent"] += 1
                            else:
                                a["by_type"][t][1] += 1
                    if call["case_id"] in NAMED_CALLS:
                        w_named = set(window)
                        for gid, (lbl, cid) in NAMED_GOLD.items():
                            if cid == call["case_id"] and gid in gold:
                                prev = a["named"].get(lbl, True)
                                a["named"][lbl] = prev and (gid in w_named)
            per_call.append(call_meta)
            if i % 50 == 0:
                print(f"[{i}/{len(records)}]", flush=True)
    finally:
        current_user_id.reset(token)
        await pool.close()

    def p90(values):
        s = sorted(values)
        return s[min(len(s) - 1, int(0.9 * len(s)))]

    summary = {
        "schema": "weft.longmemeval.round8-depth-curve.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "depths": list(DEPTHS),
        "sibling_pool_sizes": {
            "min": min(sibling_sizes), "median": statistics.median(sibling_sizes),
            "max": max(sibling_sizes),
        },
        "per_slot_yield_slots_1_to_10": slot_yield,
        "per_slot_marginal": [
            round(slot_yield[n] / max(1, slot_yield[0]), 3) for n in range(len(slot_yield))
        ],
        "depths": {},
    }
    for depth in DEPTHS:
        a = acc[depth]
        summary["depths"][f"depth={depth}"] = {
            "missed_evidence_returned": a["missed_returned"],
            "true_retention": f"{a['true_retained']}/{a['true_total']}",
            "strict_absent": a["strict_absent"],
            "extended_absent": a["strict_absent"] - a["rescued_absent"],
            "absent_rescued_via_expansion": a["rescued_absent"],
            "false_checkpoint_drops": f"{a['drops']}/{a['false_total']}",
            "named_gold": dict(a["named"]),
            "fill_rate_full": f"{a['fill_full']}/{a['false_total'] + a['true_total']}",
            "fill_rate_any": f"{a['fill_any']}/{a['false_total'] + a['true_total']}",
            "added_turns_mean": round(statistics.mean(a["added_turns"]), 2),
            "added_turns_p90": p90(a["added_turns"]),
            "added_tokens_mean_per_call": round(statistics.mean(a["added_tokens"]), 1),
            "added_tokens_p90_per_call": p90(a["added_tokens"]),
            "added_tokens_total": sum(a["added_tokens"]),
            "by_type_missed_returned_and_still_strict_absent": {
                t: {"returned": v[0], "still_absent": v[1]} for t, v in a["by_type"].items()
            },
        }
    (OUT / "round8_depth_curve.json").write_text(
        json.dumps({**summary, "calls": per_call}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
