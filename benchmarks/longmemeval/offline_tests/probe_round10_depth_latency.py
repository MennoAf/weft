"""Round-10 measurement: real-path latency, expansion depth 5 vs 8.

Drives the REAL production recall (``temporal_anchor`` with the round-7
``expansion_slots`` parameter) over a timing-focused sample of the census
cohort — every 3rd call (74 of 220), covering both segments and question
types. Each sampled call runs BOTH depths in the same run; the invocation
order alternates by call index to cancel warm-up bias.

Per depth: elapsed_ms mean/median/p95, failures/timeouts (must be zero),
returned-window length, and answer-side context cost (added turns and
summed token_count over the appended window tail).

Read-only contract: SELECT statements only; local FastEmbed embeddings.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFLINE = ROOT / "benchmarks/longmemeval/offline_tests"
if str(OFFLINE) not in sys.path:
    sys.path.insert(0, str(OFFLINE))

import probe_round2_census as rc  # noqa: E402

OUT = ROOT / "artifacts/recall-lift-20260930"
OWNER_ID = "faithful-gpt6-fulls-turns-20260929"
DEPTHS = (5, 8)
SAMPLE_STRIDE = 3


def _stats(values: list[float]) -> dict:
    s = sorted(values)
    return {
        "n": len(s),
        "mean_ms": round(statistics.mean(s), 1),
        "median_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


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
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider
    from weft.turn_recall import temporal_anchor

    census = json.loads((OUT / "round2_census_results.json").read_text(encoding="utf-8"))
    records = census["records"]
    cohort = {c["case_id"]: c for c in rc.load_cohort()[0] + rc.load_cohort()[1]}
    sample = [r for i, r in enumerate(records) if i % SAMPLE_STRIDE == 0]
    print(f"sample: {len(sample)} of {len(records)} calls (every {SAMPLE_STRIDE}rd)", flush=True)

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

    per_call: list[dict] = []
    failures = {d: 0 for d in DEPTHS}
    token = current_user_id.set(OWNER_ID)
    try:
        for i, call in enumerate(sample):
            case = cohort.get(call["case_id"])
            if case is None:
                continue
            limit = int(call["limit"])
            top_k = min(limit, 10)
            sql_width = top_k * 5
            entry: dict = {
                "case_id": call["case_id"],
                "segment": call["segment"],
                "question_type": call["question_type"],
                "limit": limit,
                "order": "d5-first" if i % 2 == 0 else "d8-first",
            }
            token_ = current_user_id.set(OWNER_ID)
            try:
                for depth in DEPTHS:
                    t0 = time.perf_counter()
                    try:
                        anchored = await temporal_anchor(
                            pool, case["query"],
                            project_id=case["project_id"],
                            top_k_per_anchor=top_k,
                            candidate_sql_limit=sql_width,
                            anchor_result_limit=limit,
                            embedder=embedder,
                            expansion_slots=depth,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000
                    except Exception as exc:
                        failures[depth] += 1
                        entry[f"d{depth}"] = {
                            "error": f"{type(exc).__name__}: {exc}"[:200],
                        }
                        continue
                    turns = [t for turns in anchored.values() for t in turns]
                    appended = turns[limit:]
                    entry[f"d{depth}"] = {
                        "elapsed_ms": round(elapsed, 1),
                        "window_len": len(turns),
                        "added_turns": len(appended),
                        "added_tokens": sum(t.token_count or 0 for t in appended),
                    }
            finally:
                current_user_id.reset(token_)
            per_call.append(entry)
            print(
                f"[{len(per_call)}/{len(sample)}] {call['case_id']} "
                + " ".join(
                    f"d{d}={entry.get(f'd{d}', {}).get('elapsed_ms', 'ERR')}ms"
                    for d in DEPTHS
                ),
                flush=True,
            )
    finally:
        current_user_id.reset(token)
        await pool.close()

    ok_calls = [e for e in per_call if all(f"d{d}" in e and "error" not in e[f"d{d}"] for d in DEPTHS)]
    summary = {
        "schema": "weft.longmemeval.round10-depth-latency.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "sample": {
            "requested": len(sample),
            "measured": len(ok_calls),
            "stride": SAMPLE_STRIDE,
            "failures": failures,
            "order_balanced": {
                "d5-first": sum(1 for e in ok_calls if e["order"] == "d5-first"),
                "d8-first": sum(1 for e in ok_calls if e["order"] == "d8-first"),
            },
        },
        "latency": {
            f"depth={d}": _stats([e[f"d{d}"]["elapsed_ms"] for e in ok_calls])
            for d in DEPTHS
        },
        "windows": {
            f"depth={d}": {
                "window_len_set": sorted({e[f"d{d}"]["window_len"] for e in ok_calls}),
                "added_turns_mean": round(statistics.mean(
                    e[f"d{d}"]["added_turns"] for e in ok_calls), 2),
                "added_tokens_mean": round(statistics.mean(
                    e[f"d{d}"]["added_tokens"] for e in ok_calls), 1),
                "added_tokens_p90": sorted(
                    e[f"d{d}"]["added_tokens"] for e in ok_calls)[
                    min(len(ok_calls) - 1, int(0.9 * len(ok_calls)))],
                "added_tokens_total": sum(e[f"d{d}"]["added_tokens"] for e in ok_calls),
            }
            for d in DEPTHS
        },
        "paired_delta_ms": {
            "mean": round(statistics.mean(
                e["d8"]["elapsed_ms"] - e["d5"]["elapsed_ms"] for e in ok_calls), 1),
            "median": round(statistics.median(
                e["d8"]["elapsed_ms"] - e["d5"]["elapsed_ms"] for e in ok_calls), 1),
        },
        "calls": per_call,
    }
    (OUT / "round10_depth_latency.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
