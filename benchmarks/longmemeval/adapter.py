#!/usr/bin/env python3
"""
adapter.py — LongMemEval × Weft benchmark harness.

Pipeline per question:

    1. Sandbox    — derive project_id "lme_<qid>" so haystacks never bleed.
    2. Ingest     — load all sessions via the chosen mode (raw | extracted).
    3. Recall     — hybrid search (vector + BM25 + RRF) for the question.
    4. Read       — Claude Reader produces a short hypothesis string.
    5. Emit       — append {question_id, hypothesis} as one JSONL line.
    6. Cleanup    — optional purge of the question's memories (keeps DB tidy).

Output JSONL is the exact contract LongMemEval's evaluator expects, so the
file feeds directly into ``src/evaluation/evaluate_qa.py`` from the upstream
repo without any post-processing.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-04-30
Python:  >= 3.12

Dependencies:
    weft (this repo), anthropic, asyncpg, click, tqdm (optional, for progress)

Usage:
    See README.md and the bottom of this file for run commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import click

from weft.config import load_config
from weft.db.connection import (
    _pgvector_codec_init,
    register_pgvector_codec,
)
from weft.embeddings import get_provider
from weft.embeddings.base import EmbeddingProvider

from benchmarks.longmemeval.dataset import Instance, load_split
from benchmarks.longmemeval.ingest import (
    IngestMode,
    cleanup_haystack,
    load_haystack,
    project_id_for,
)
from benchmarks.longmemeval.reader import Reader
from benchmarks.longmemeval.router import Tier, policy_for, retrieve

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 10
BENCHMARK_USER_ID = "longmemeval-bench"


@dataclass(slots=True)
class RunStats:
    """Aggregate token + timing stats across a benchmark run."""

    questions_total: int = 0
    questions_done: int = 0
    questions_failed: int = 0
    sessions_ingested: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    started_at: float = 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0


# ----------------------------------------------------------------------
# Pool setup — adapter-owned so this module can run standalone against any
# Weft Postgres instance without dragging in the test fixtures.
# ----------------------------------------------------------------------


async def _bench_setup(conn: asyncpg.Connection) -> None:
    """Pool ``setup`` callback — sets app.user_id for migration-34 NOT NULL.

    Mirrors the pattern in tests/conftest.py: every direct ``pool.execute``
    call needs a non-null user_id to satisfy the schema. We use a stable
    sentinel so every benchmark write goes under the same identity.
    """
    await conn.execute(f"SET app.user_id = '{BENCHMARK_USER_ID}'")


async def _make_pool() -> asyncpg.Pool:
    """Create an asyncpg pool from WeftConfig (env / ~/.weft/config.toml)."""
    config = load_config()
    dsn = config.database.url
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")
    pool = await asyncpg.create_pool(
        dsn,
        min_size=2,
        max_size=8,
        init=_pgvector_codec_init,
        setup=_bench_setup,
    )
    await register_pgvector_codec(pool)
    return pool


def _make_embedder() -> EmbeddingProvider:
    """Build the embedding provider from WeftConfig.

    Honors the user's configured provider (openai / fastembed / etc.) so
    benchmark embeddings live in the same vector space as the rest of the
    Weft instance — important if anyone later queries across project_ids.
    """
    config = load_config()
    return get_provider(
        config.embedding.provider,
        model_name=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )


# ----------------------------------------------------------------------
# Per-question pipeline
# ----------------------------------------------------------------------


async def _run_one(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    reader: Reader,
    instance: Instance,
    *,
    mode: IngestMode,
    top_k: int,
    tier: Tier = "belief",
) -> tuple[str, dict]:
    """Run the full pipeline for one question.

    Returns:
        (hypothesis, telemetry_dict). The hypothesis goes into the JSONL
        results file; telemetry is aggregated into RunStats.
    """
    project_id = project_id_for(instance.question_id)

    # 1+2. Ingest haystack into a per-question project sandbox.
    n_sessions = await load_haystack(pool, embedder, instance, mode)

    # 3. Recall — question-type-aware policy lives in router.policy_for().
    # The CLI top_k acts as a floor: if a caller bumps top_k above the
    # policy default (e.g. running ablations), honor that. Sandbox isolation
    # (over-fetch + post-filter to project_id) is handled inside retrieve().
    policy = policy_for(instance.question_type)
    if top_k > policy.top_k:
        from benchmarks.longmemeval.router import RetrievalPolicy
        policy = RetrievalPolicy(top_k=top_k, overfetch_multiplier=policy.overfetch_multiplier)
    memories = await retrieve(
        pool, embedder,
        question=instance.question,
        question_type=instance.question_type,
        project_id=project_id,
        policy=policy,
        tier=tier,
    )

    # 4. Read — Claude synthesizes the answer.
    response = await reader.read_answer(
        question=instance.question,
        question_date=instance.question_date,
        question_type=instance.question_type,
        memories=memories,
        top_k=policy.top_k,
    )

    telemetry = {
        "n_sessions": n_sessions,
        "n_recalled": len(memories),
        "input_tokens": response.input_tokens,
        "cached_tokens": response.cached_input_tokens,
        "output_tokens": response.output_tokens,
        "model": response.model,
    }
    return response.hypothesis, telemetry


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def _stratified_sample(
    instances: list[Instance],
    *,
    frac: float,
    seed: int = 0,
) -> list[Instance]:
    """Take a stratified sample preserving question-type proportions.

    Splits ``instances`` by ``question_type`` (abstention and non-abstention
    variants are treated as distinct strata, since the Reader's prompt
    differs and they're functionally separate categories), then takes
    ``ceil(frac * |stratum|)`` from each stratum using a deterministic
    seeded shuffle so re-runs sample the same questions.

    A fixed seed is the right default for benchmark sampling: identical
    samples across runs make subset numbers comparable. Pass a different
    seed only when you specifically want a fresh draw.
    """
    import math
    import random
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1]; got {frac}")
    by_type: dict[str, list[Instance]] = {}
    for inst in instances:
        by_type.setdefault(inst.question_type, []).append(inst)
    rng = random.Random(seed)
    sampled: list[Instance] = []
    for qtype, group in by_type.items():
        n_take = max(1, math.ceil(len(group) * frac))
        n_take = min(n_take, len(group))
        shuffled = group.copy()
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:n_take])
    # Preserve original order so JSONL output and judge inputs line up
    # with intuitions from the full split (helpful when scanning logs).
    sampled_ids = {inst.question_id for inst in sampled}
    return [inst for inst in instances if inst.question_id in sampled_ids]


async def run_benchmark(
    *,
    dataset_path: Path,
    output_path: Path,
    mode: IngestMode = "raw",
    top_k: int = DEFAULT_TOP_K,
    cleanup: bool = True,
    limit: int | None = None,
    question_types: frozenset[str] | None = None,
    stratified_frac: float | None = None,
    sample_seed: int = 0,
    tier: Tier = "belief",
    pool: asyncpg.Pool | None = None,
    embedder: EmbeddingProvider | None = None,
    reader: Reader | None = None,
) -> RunStats:
    """Run the LongMemEval adapter over a dataset split.

    Args:
        dataset_path: Path to a longmemeval_*.json split file.
        output_path: Where to write the {question_id, hypothesis} JSONL.
            Existing files are NOT overwritten — they are appended to, so
            partial runs can be resumed by deduping question_ids upstream.
        mode: "raw" (write sessions verbatim, full fidelity) or
            "extracted" (run Weft's LLM ingest pipeline).
        top_k: Memories to feed the Reader.
        cleanup: If True, hard-delete each question's memories after
            recording the hypothesis. Recommended for dev loops; disable
            if you want to inspect the DB after the run.
        limit: Stop after this many questions (None = full split). Useful
            for smoke tests against a small slice. Applied AFTER the
            question_types filter.
        question_types: If provided, only process instances whose
            ``question_type`` is in this set. Exact-match — abstention
            variants like ``"multi-session_abs"`` must be listed
            explicitly. Used for cheap subset A/B runs (e.g., iterate on
            the router for multi-session only).
        pool / embedder / reader: Inject dependencies for testing. If
            omitted, defaults are constructed from WeftConfig + env.

    Returns:
        RunStats summarizing the run. Written to ``<output_path>.stats.json``.

    Raises:
        FileNotFoundError: dataset_path does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    instances = load_split(dataset_path)
    if question_types:
        before = len(instances)
        instances = [i for i in instances if i.question_type in question_types]
        logger.info(
            "question_type filter %s: %d → %d instances",
            sorted(question_types), before, len(instances),
        )
    if stratified_frac is not None:
        before = len(instances)
        instances = _stratified_sample(
            instances, frac=stratified_frac, seed=sample_seed,
        )
        logger.info(
            "stratified sample (frac=%.3f, seed=%d): %d → %d instances",
            stratified_frac, sample_seed, before, len(instances),
        )
    if limit is not None:
        instances = instances[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    owns_pool = pool is None
    if pool is None:
        pool = await _make_pool()
    if embedder is None:
        embedder = _make_embedder()
    if reader is None:
        reader = Reader()

    stats = RunStats(questions_total=len(instances), started_at=time.monotonic())

    try:
        with output_path.open("a", encoding="utf-8") as out:
            for instance in instances:
                try:
                    hypothesis, telemetry = await _run_one(
                        pool, embedder, reader, instance,
                        mode=mode, top_k=top_k, tier=tier,
                    )
                except Exception as exc:
                    logger.exception(
                        "question %s failed: %s", instance.question_id, exc,
                    )
                    stats.questions_failed += 1
                    continue
                else:
                    stats.questions_done += 1
                    stats.sessions_ingested += telemetry["n_sessions"]
                    stats.input_tokens += telemetry["input_tokens"]
                    stats.cached_tokens += telemetry["cached_tokens"]
                    stats.output_tokens += telemetry["output_tokens"]

                    out.write(
                        json.dumps(
                            {
                                "question_id": instance.question_id,
                                "hypothesis": hypothesis,
                            }
                        )
                        + "\n"
                    )
                    out.flush()

                if cleanup:
                    await cleanup_haystack(pool, instance)
    finally:
        if owns_pool:
            await pool.close()

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "questions_total": stats.questions_total,
                "questions_done": stats.questions_done,
                "questions_failed": stats.questions_failed,
                "sessions_ingested": stats.sessions_ingested,
                "input_tokens": stats.input_tokens,
                "cached_tokens": stats.cached_tokens,
                "output_tokens": stats.output_tokens,
                "elapsed_seconds": stats.elapsed(),
                "mode": mode,
                "tier": tier,
                "top_k": top_k,
                "dataset": str(dataset_path),
                "question_types": sorted(question_types) if question_types else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


@click.command()
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to longmemeval_{oracle,s,m}.json split file.",
)
@click.option(
    "--mode",
    type=click.Choice(["raw", "extracted", "turns"]),
    default="raw",
    show_default=True,
    help=(
        "Ingest mode: 'raw' writes sessions verbatim; 'extracted' runs "
        "Weft's LLM pipeline; 'turns' writes each conversational turn into "
        "episode_turns for turn-tier hybrid recall (Branch A)."
    ),
)
@click.option(
    "--tier",
    type=click.Choice(["belief", "turns", "auto"]),
    default="belief",
    show_default=True,
    help=(
        "Retrieval tier. 'belief' (default) hits hybrid recall over memories. "
        "'turns' queries episode_turns directly (use with --mode turns). "
        "'auto' routes multi-session and temporal-reasoning to turns; "
        "everything else stays on belief."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("benchmarks/longmemeval/results"),
    show_default=True,
    help="Directory where the JSONL hypotheses file is written.",
)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Memories returned from hybrid recall and fed to the Reader.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Only run the first N questions (smoke testing).",
)
@click.option(
    "--question-type",
    "question_types",
    multiple=True,
    default=(),
    help=(
        "Only process instances of this question_type. Repeatable; exact-"
        "match (pass 'multi-session' AND 'multi-session_abs' for both). "
        "Cheap A/B subset runs — e.g. router tuning on multi-session only."
    ),
)
@click.option(
    "--stratified-frac",
    type=float,
    default=None,
    help=(
        "Take a stratified sample by question_type, keeping the original "
        "proportions (e.g. 0.1 = 10% of each type). Deterministic via "
        "--sample-seed so re-runs hit the same questions. Applied AFTER "
        "--question-type, BEFORE --limit."
    ),
)
@click.option(
    "--sample-seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for the stratified-frac sampler.",
)
@click.option(
    "--no-cleanup",
    is_flag=True,
    default=False,
    help="Do not delete a question's memories after answering. Default cleans up.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def cli(
    dataset_path: Path,
    mode: IngestMode,
    tier: Tier,
    output_dir: Path,
    top_k: int,
    limit: int | None,
    question_types: tuple[str, ...],
    stratified_frac: float | None,
    sample_seed: int,
    no_cleanup: bool,
    log_level: str,
) -> None:
    """Run Weft against the LongMemEval benchmark, write hypotheses JSONL."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Match Weft's config loader: load ~/.weft/.env before checking env.
    # Without this, keys placed in the standard Weft .env would be rejected
    # because the check runs before load_config() triggers load_dotenv().
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException(
            "ANTHROPIC_API_KEY is required for the Reader stage. "
            "Set it in your shell or add it to ~/.weft/.env."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    split_name = dataset_path.stem  # e.g. "longmemeval_oracle"
    qt_set = frozenset(question_types) if question_types else None
    if qt_set and len(qt_set) == 1:
        slug = f"_{next(iter(qt_set))}"
    elif qt_set:
        slug = f"_filtered{len(qt_set)}"
    else:
        slug = ""
    if stratified_frac is not None:
        slug = f"{slug}_strat{int(round(stratified_frac * 100))}s{sample_seed}"
    # Suffix the tier into the filename when it's not the default belief
    # path so A/B comparison runs don't fight over filenames.
    tier_slug = "" if tier == "belief" else f"_tier-{tier}"
    output_path = output_dir / f"{split_name}_{mode}{tier_slug}{slug}_{timestamp}.jsonl"

    stats = asyncio.run(
        run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            mode=mode,
            top_k=top_k,
            cleanup=not no_cleanup,
            limit=limit,
            question_types=qt_set,
            stratified_frac=stratified_frac,
            sample_seed=sample_seed,
            tier=tier,
        )
    )

    click.echo(f"\nWrote: {output_path}")
    click.echo(
        f"  done={stats.questions_done}/{stats.questions_total} "
        f"failed={stats.questions_failed} "
        f"elapsed={stats.elapsed():.1f}s"
    )
    click.echo(
        f"  tokens: in={stats.input_tokens} "
        f"cached={stats.cached_tokens} "
        f"out={stats.output_tokens}"
    )
    click.echo(
        f"\nNext: feed this file to LongMemEval's evaluator:\n"
        f"  python LongMemEval/src/evaluation/evaluate_qa.py "
        f"gpt-4o {output_path} LongMemEval/data/{split_name}.json"
    )


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 0. Prerequisites:
#    - ANTHROPIC_API_KEY exported in your shell (Reader stage).
#    - A running Weft Postgres (docker-compose.weft.yml or DATABASE_URL set).
#    - LongMemEval dataset downloaded:
#         git clone https://github.com/xiaowu0162/LongMemEval ../LongMemEval
#         (or pull JSON from huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
#
# 1. Smoke test against 5 questions (Oracle split is cheapest — evidence only):
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_oracle.json \
#        --mode raw --limit 5
#
# 2. Full Oracle run, raw ingest mode (~$1-2 in judge cost when scored):
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_oracle.json \
#        --mode raw
#
# 3. Compare to extracted mode on the same split:
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_oracle.json \
#        --mode extracted
#
# 3a. Subset run on a single question type — cheap A/B for router tuning
#     (~$2–4 instead of ~$15 for the full Oracle split). Pass each type
#     explicitly; abstention variants are not auto-included.
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_oracle.json \
#        --mode extracted \
#        --question-type multi-session \
#        --question-type multi-session_abs
#
# 4. Score the resulting JSONL with the LongMemEval judge wrapper. This
#    runs the upstream evaluator inside an ephemeral uv env (no Weft dep
#    pinning) and writes a structured per-question-type breakdown JSON
#    next to the upstream output:
#    uv run python -m benchmarks.longmemeval.judge \
#        --hyp benchmarks/longmemeval/results/<file>.jsonl
#    Requires OPENAI_API_KEY. Cost: ~$5–15 for a 500-question oracle run.
#
# 5. Output:
#    - results/<split>_<mode>_<timestamp>.jsonl   — hypotheses (eval contract)
#    - results/<split>_<mode>_<timestamp>.jsonl.stats.json   — token/timing stats
#
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
