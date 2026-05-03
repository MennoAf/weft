#!/usr/bin/env python3
"""
diagnose.py — Recall-precision diagnostic for failing LongMemEval questions.

Answers ONE question: for the questions Weft got wrong, was the gold
evidence in the top-K retrieved set? If YES, the Reader is the bottleneck.
If NO, retrieval is the bottleneck and bumping top_k won't help.

For each failing question (per the judge's eval-results JSONL):

    1. Sandbox + ingest the haystack (extracted mode by default — same as
       the run we are diagnosing).
    2. Retrieve top-K using the same router policy the adapter uses.
    3. Identify gold evidence sessions (those flagged ``has_answer`` in
       the dataset).
    4. For each gold session, compute the max embedding similarity
       against any retrieved memory. Bucket as strong / weak / miss.
    5. Cleanup the sandbox.

Aggregate report:
    - per-question table of gold-session recall
    - distribution of best-match similarities
    - the headline number: % of failing questions where AT LEAST ONE
      gold session has strong overlap with a top-K retrieved memory

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-05-02
Python:  >= 3.12

Dependencies:
    weft (this repo), asyncpg, click, numpy

Usage:
    See the bottom of this file for run commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
import click

from weft.embeddings.base import EmbeddingProvider

from benchmarks.longmemeval.adapter import _make_embedder, _make_pool
from benchmarks.longmemeval.dataset import Instance, load_split
from benchmarks.longmemeval.ingest import (
    IngestMode,
    cleanup_haystack,
    load_haystack,
    project_id_for,
)
from benchmarks.longmemeval.router import policy_for, retrieve

logger = logging.getLogger(__name__)


STRONG_THRESHOLD = 0.7  # cosine — gold session content clearly in top-K
WEAK_THRESHOLD = 0.4    # cosine — partial overlap, ambiguous signal


@dataclass(slots=True)
class GoldSessionResult:
    """One gold evidence session's match against the top-K retrieved set."""

    session_id: str
    best_similarity: float
    best_memory_excerpt: str  # first 120 chars of the best-matching memory


@dataclass(slots=True)
class QuestionDiagnostic:
    """Recall diagnostic for one failing question."""

    question_id: str
    question_type: str
    question: str
    n_sessions: int
    n_gold_sessions: int
    n_retrieved: int
    top_k: int
    gold_results: list[GoldSessionResult] = field(default_factory=list)

    @property
    def has_strong_match(self) -> bool:
        """True iff at least one gold session has a strong match in top-K."""
        return any(g.best_similarity >= STRONG_THRESHOLD for g in self.gold_results)

    @property
    def all_gold_strong(self) -> bool:
        """True iff every gold session has a strong match in top-K."""
        return bool(self.gold_results) and all(
            g.best_similarity >= STRONG_THRESHOLD for g in self.gold_results
        )

    @property
    def best_overall_similarity(self) -> float:
        return max((g.best_similarity for g in self.gold_results), default=0.0)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Both vectors must be the same dimensionality."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_failing_question_ids(eval_results_path: Path) -> set[str]:
    """Read judge's eval-results JSONL, return question_ids the judge marked false."""
    failing: set[str] = set()
    with eval_results_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = row["autoeval_label"]
            # autoeval_label was a bare bool in older runs; current judge
            # wraps it as {"model": ..., "label": bool}. Handle both.
            if isinstance(label, dict):
                label = label["label"]
            if not label:
                failing.add(row["question_id"])
    return failing


async def _diagnose_one(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    instance: Instance,
    *,
    mode: IngestMode,
) -> QuestionDiagnostic:
    """Re-ingest one question's haystack and measure gold-evidence recall."""
    project_id = project_id_for(instance.question_id)

    # 1. Ingest the haystack — costs LLM tokens in extracted mode.
    await load_haystack(pool, embedder, instance, mode)

    # 2. Retrieve using the same router policy the adapter uses.
    policy = policy_for(instance.question_type)
    memories = await retrieve(
        pool, embedder,
        question=instance.question,
        question_type=instance.question_type,
        project_id=project_id,
        policy=policy,
    )

    # 3. Embed gold-session texts and each retrieved memory; compute
    # per-gold-session best match. Reuses the same embedder so the
    # similarity space matches what retrieval saw.
    gold_sessions = [s for s in instance.sessions if s.has_answer]
    diag = QuestionDiagnostic(
        question_id=instance.question_id,
        question_type=instance.question_type,
        question=instance.question,
        n_sessions=len(instance.sessions),
        n_gold_sessions=len(gold_sessions),
        n_retrieved=len(memories),
        top_k=policy.top_k,
    )

    if not gold_sessions or not memories:
        return diag

    retrieved_embeddings = [
        await embedder.embed(m.memory.content) for m in memories
    ]
    for session in gold_sessions:
        gold_emb = await embedder.embed(session.to_text())
        sims = [_cosine(gold_emb, e) for e in retrieved_embeddings]
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        diag.gold_results.append(
            GoldSessionResult(
                session_id=session.session_id,
                best_similarity=sims[best_idx],
                best_memory_excerpt=memories[best_idx].memory.content[:120],
            )
        )
    return diag


def _summarize(diagnostics: list[QuestionDiagnostic]) -> dict:
    """Aggregate per-question diagnostics into a headline-number summary."""
    if not diagnostics:
        return {"n": 0}

    # Per-question signals.
    n = len(diagnostics)
    n_with_strong = sum(1 for d in diagnostics if d.has_strong_match)
    n_all_strong = sum(1 for d in diagnostics if d.all_gold_strong)
    n_zero_match = sum(
        1 for d in diagnostics if d.best_overall_similarity < WEAK_THRESHOLD
    )

    # Best-similarity distribution (one value per question — its single best).
    bests = sorted(d.best_overall_similarity for d in diagnostics)
    median_best = bests[n // 2]
    mean_best = sum(bests) / n

    # Per-gold-session signals (more granular than per-question).
    all_gold = [g for d in diagnostics for g in d.gold_results]
    g_strong = sum(1 for g in all_gold if g.best_similarity >= STRONG_THRESHOLD)
    g_weak = sum(
        1 for g in all_gold
        if WEAK_THRESHOLD <= g.best_similarity < STRONG_THRESHOLD
    )
    g_miss = sum(1 for g in all_gold if g.best_similarity < WEAK_THRESHOLD)

    return {
        "n_questions": n,
        "n_with_any_strong_match": n_with_strong,
        "pct_with_any_strong_match": round(n_with_strong / n, 3),
        "n_all_gold_strong": n_all_strong,
        "pct_all_gold_strong": round(n_all_strong / n, 3),
        "n_zero_match": n_zero_match,
        "pct_zero_match": round(n_zero_match / n, 3),
        "best_similarity": {
            "median": round(median_best, 3),
            "mean": round(mean_best, 3),
            "min": round(bests[0], 3),
            "max": round(bests[-1], 3),
        },
        "gold_sessions_total": len(all_gold),
        "gold_sessions_strong": g_strong,
        "gold_sessions_weak": g_weak,
        "gold_sessions_miss": g_miss,
        "interpretation": (
            "If pct_with_any_strong_match is HIGH, retrieval is finding the "
            "evidence — the Reader is the bottleneck (prompting / "
            "enumeration / chain-of-thought). If LOW, retrieval is the "
            "bottleneck — bumping top_k will not help; need query "
            "reformulation, multi-hop, or BM25 weighting changes."
        ),
    }


async def run_diagnosis(
    *,
    dataset_path: Path,
    eval_results_path: Path,
    output_path: Path,
    mode: IngestMode = "extracted",
    limit: int | None = None,
    pool: asyncpg.Pool | None = None,
    embedder: EmbeddingProvider | None = None,
) -> dict:
    """Run the diagnostic over every failing question and write a report."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not eval_results_path.exists():
        raise FileNotFoundError(f"eval results not found: {eval_results_path}")

    failing_qids = _load_failing_question_ids(eval_results_path)
    logger.info("found %d failing questions", len(failing_qids))

    instances = [i for i in load_split(dataset_path) if i.question_id in failing_qids]
    if limit is not None:
        instances = instances[:limit]
    logger.info("diagnosing %d instances", len(instances))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    owns_pool = pool is None
    if pool is None:
        pool = await _make_pool()
    if embedder is None:
        embedder = _make_embedder()

    started = time.monotonic()
    diagnostics: list[QuestionDiagnostic] = []
    try:
        for instance in instances:
            try:
                diag = await _diagnose_one(pool, embedder, instance, mode=mode)
            except Exception as exc:
                logger.exception(
                    "diagnose failed for %s: %s", instance.question_id, exc,
                )
                continue
            else:
                diagnostics.append(diag)
                logger.info(
                    "%s: best_sim=%.2f gold_sessions=%d/%d",
                    instance.question_id,
                    diag.best_overall_similarity,
                    sum(1 for g in diag.gold_results
                        if g.best_similarity >= STRONG_THRESHOLD),
                    diag.n_gold_sessions,
                )
            await cleanup_haystack(pool, instance)
    finally:
        if owns_pool:
            await pool.close()

    summary = _summarize(diagnostics)
    payload = {
        "summary": summary,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "mode": mode,
        "dataset": str(dataset_path),
        "eval_results": str(eval_results_path),
        "thresholds": {"strong": STRONG_THRESHOLD, "weak": WEAK_THRESHOLD},
        "per_question": [
            {
                "question_id": d.question_id,
                "question_type": d.question_type,
                "question": d.question,
                "n_sessions": d.n_sessions,
                "n_gold_sessions": d.n_gold_sessions,
                "n_retrieved": d.n_retrieved,
                "top_k": d.top_k,
                "best_overall_similarity": round(d.best_overall_similarity, 3),
                "has_strong_match": d.has_strong_match,
                "all_gold_strong": d.all_gold_strong,
                "gold_results": [
                    {
                        "session_id": g.session_id,
                        "best_similarity": round(g.best_similarity, 3),
                        "best_memory_excerpt": g.best_memory_excerpt,
                    }
                    for g in d.gold_results
                ],
            }
            for d in diagnostics
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


@click.command()
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="LongMemEval split JSON (the same one the run used).",
)
@click.option(
    "--eval-results",
    "eval_results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Judge eval-results JSONL (.eval-results-gpt-4o sidecar).",
)
@click.option(
    "--mode",
    type=click.Choice(["raw", "extracted"]),
    default="extracted",
    show_default=True,
    help="Ingest mode — must match the run being diagnosed for valid signal.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Diagnostic report JSON path. Defaults to <eval-results>.recall-diagnostic.json.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Diagnose only the first N failing questions (smoke check).",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def cli(
    dataset_path: Path,
    eval_results_path: Path,
    mode: IngestMode,
    output_path: Path | None,
    limit: int | None,
    log_level: str,
) -> None:
    """Was the gold evidence in top-K? Run a recall-precision diagnostic."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")
    if mode == "extracted" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException(
            "ANTHROPIC_API_KEY required for extracted-mode ingest. "
            "Set it in your shell or add it to ~/.weft/.env."
        )

    output = output_path or eval_results_path.with_suffix(
        eval_results_path.suffix + ".recall-diagnostic.json"
    )

    summary = asyncio.run(
        run_diagnosis(
            dataset_path=dataset_path,
            eval_results_path=eval_results_path,
            output_path=output,
            mode=mode,
            limit=limit,
        )
    )

    click.echo(f"\nWrote: {output}")
    click.echo(json.dumps(summary, indent=2))


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# Diagnose the multi-session failures from the 2026-05-02 run:
#
#    uv run python -m benchmarks.longmemeval.diagnose \
#        --dataset /Users/jasonbauman/Documents/code_projects/Personal/langchain/LongMemEval/data/longmemeval_oracle.json \
#        --eval-results benchmarks/longmemeval/results/longmemeval_oracle_extracted_multi-session_20260502T233202Z.jsonl.eval-results-gpt-4o
#
# Smoke check on the first 5 failing questions:
#    add --limit 5
#
# Cost: extracted-mode ingest only (no Reader, no judge). Roughly the
# same per-question as a full benchmark run, but only on the failing
# subset — typically ~$1–2 for a multi-session subset of ~50 questions.
#
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
