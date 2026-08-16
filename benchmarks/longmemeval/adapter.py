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
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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
    expected_turn_count,
    load_haystack,
    project_id_for,
)
from benchmarks.longmemeval.materialize import (
    Detector,
    MaterializationAborted,
    materialize_question,
)
from benchmarks.longmemeval.reader import Reader
from benchmarks.longmemeval.replay_drive import ReplayExecutorKind, drive_replay
from benchmarks.longmemeval.router import RetrievalDiagnostics, Tier, policy_for, retrieve

logger = logging.getLogger(__name__)


def _answer_text_match(gold: object, content: str) -> bool:
    """Check whether a scalar gold answer is present in a turn's content.

    Heuristic (documented for transparency):
    - Both sides are lowercased and whitespace-stripped before comparison.
    - For gold answers longer than 3 characters: substring containment
      (``gold in content``). This means "Paris" matches "Parisian" as a
      side effect; short, common tokens are more likely to false-positive.
    - For gold answers 3 characters or shorter (e.g. "no", "yes", "UK"):
      word-boundary match via ``re.search(r"\\b<gold>\\b", content)`` to
      prevent "no" matching "north" or "not".

    Known false positives:
    - "Paris" will match a turn that mentions "Parisian" (substring, no
      boundary guard for long tokens by design — boundary guards on long
      tokens would miss plurals, possessives, etc.).
    - Single-word gold answers that appear inside compound words sharing
      that root (e.g. "art" inside "artifact") — accepted trade-off.

    Known false negatives:
    - Gold answers that appear only in paraphrased form (e.g. "NYC" when
      the turn says "New York City"). Text-match cannot resolve synonyms.
    - Hyphenated or punctuation-adjacent gold tokens may miss a boundary
      match for the short-token path (``re`` \\b is ASCII-boundary-aware).

    Updated behavior for ≤3-char gold answers:
    - Word-boundary matching (``\\b``) requires a word character (``[A-Za-z0-9_]``)
      immediately adjacent to the boundary. If the gold answer starts or ends
      with a non-word character (e.g. ``"(b)"``, ``"no."``), ``\\b`` can never
      anchor and the regex always fails. For such tokens, substring containment
      is used instead. Pure alphanumeric short tokens still use ``\\b`` as before.

    Args:
        gold: The gold answer string (lowercased + stripped before use).
        content: The turn content to search within.

    Returns:
        True if the gold answer is found in the content, False otherwise.
    """
    gold = str(gold).strip().lower()
    content = content.strip().lower()
    if not gold:
        return False
    if len(gold) <= 3:
        # Word-boundary regex requires word chars at both ends; otherwise \b
        # can't anchor and the match always fails. Fall back to substring.
        if gold[0].isalnum() and gold[-1].isalnum():
            return bool(re.search(rf"\b{re.escape(gold)}\b", content))
        return gold in content
    return gold in content


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
    # Recall@k instrumentation (turn-mode + turn-tier only). None when
    # recall capture didn't run; populated at the end of run_benchmark
    # so the CLI can print the headline number without re-reading files.
    recall_at_k: float | None = None
    recall_k: int = 10
    recall_n_hits: int = 0
    recall_n_questions: int = 0
    recall_jsonl_path: str | None = None
    recall_summary_path: str | None = None
    # Warm-boost aggregates (P1.A5). Zero when --warm-boost-rounds=0.
    warm_boost_rounds: int = 0
    warm_boost_queries: int = 0
    warm_boost_accessed_turns: int = 0
    warm_boost_boosted_turns: int = 0
    # Belief-view materialization aggregates (tier=belief-view only). Errors
    # here mean turns whose claims were silently skipped — a non-zero count
    # on a gate run means the score under-reads the belief-view.
    materialize_turns_total: int = 0
    materialize_claims_written: int = 0
    materialize_claims_superseded: int = 0
    materialize_abstentions: int = 0
    materialize_errors: int = 0
    # Replay-loop aggregates (tier=replay only). The recall-gap replay loop is
    # enqueued + drained per question on top of belief-view materialization;
    # these count the multi-turn aggregate claims it wrote. A run with
    # claims_written=0 across all questions means the loop ran but the detector
    # found no enumerations — a real (null) result, not an inert path.
    replay_rows_enqueued: int = 0
    replay_rows_processed: int = 0
    replay_rows_done: int = 0
    replay_rows_failed: int = 0
    replay_claims_written: int = 0
    replay_claims_superseded: int = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0


# ----------------------------------------------------------------------
# Pool setup — adapter-owned so this module can run standalone against any
# Weft Postgres instance without dragging in the test fixtures.
# ----------------------------------------------------------------------


async def _bench_setup(conn: asyncpg.Connection) -> None:
    """Pool ``setup`` callback — runs on every connection acquire.

    Sets app.user_id for migration-34 NOT NULL and search_path for pgvector
    type resolution. Both are needed because the default pool reset runs
    RESET ALL on connection release, which wipes both GUCs. The init callback
    (_pgvector_codec_init) sets search_path on connection creation, but that
    only runs once — this setup callback re-sets it on every acquire so it's
    always correct after a reset cycle.
    """
    await conn.execute(f"SET app.user_id = '{BENCHMARK_USER_ID}'")
    await conn.execute("SET search_path TO public, extensions")


DEFAULT_LOCAL_BENCHMARK_DSN = (
    "postgresql://postgres@127.0.0.1:55432/weftbench"
)


def _redacted_database_target(dsn: str) -> str:
    """Render a DSN target without logging its password or query options."""
    try:
        parsed = urlsplit(dsn)
        if not parsed.scheme or not parsed.hostname:
            return "<keyword-style-or-invalid-dsn>"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parsed.port or 5432
        database = parsed.path.lstrip("/") or "<default>"
        user = parsed.username or "<default>"
        return f"{parsed.scheme}://{user}@{host}:{port}/{database}"
    except ValueError:
        return "<invalid-dsn>"


async def _make_pool() -> asyncpg.Pool:
    """Create the benchmark pool from an explicit benchmark-only DSN.

    LongMemEval must never inherit ``DATABASE_URL``/``WEFT_DATABASE_URL``.
    Those normally point at hosted Weft/Supabase, where a benchmark can be
    slow, throttled, and—more importantly—silently run against the wrong
    materialization. The detached local runner supplies the canonical DSN;
    direct callers must export ``LONGMEMEVAL_DATABASE_URL`` themselves.
    """
    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "LONGMEMEVAL_DATABASE_URL is required for LongMemEval; refusing "
            "to fall back to DATABASE_URL/WEFT_DATABASE_URL. For the canonical "
            "local benchmark container, export "
            f"{DEFAULT_LOCAL_BENCHMARK_DSN!r}."
        )
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")

    target = _redacted_database_target(dsn)
    password = os.environ.get("LONGMEMEVAL_DATABASE_PASSWORD") or None
    logger.info(
        "longmemeval.database_target target=%s password_source=%s",
        target,
        "env" if password else "dsn_or_none",
    )
    pool_kwargs = {
        "min_size": 2,
        "max_size": 8,
        "init": _pgvector_codec_init,
        "setup": _bench_setup,
    }
    if password:
        pool_kwargs["password"] = password
    pool = await asyncpg.create_pool(dsn, **pool_kwargs)
    try:
        await register_pgvector_codec(pool)
        identity = await pool.fetchrow(
            "SELECT current_database() AS database_name, current_user AS role_name, "
            "inet_server_addr()::text AS server_addr, inet_server_port() AS server_port"
        )
        logger.info(
            "longmemeval.database_identity target=%s database=%s role=%s server=%s:%s",
            target,
            identity["database_name"],
            identity["role_name"],
            identity["server_addr"],
            identity["server_port"],
        )
    except Exception:
        await pool.close()
        raise
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


async def _snapshot_live_identity(
    pool: asyncpg.Pool,
    manifest: "Manifest",
) -> dict[str, object]:
    """Read and validate the live shape for a fixed-materialization run.

    This is deliberately read-only.  It is shared by the snapshot Reader
    runner and its tests so a Reader artifact cannot be produced against a
    partial restore or a different database shape.  The canonical baseline
    has a stronger, literal count gate than arbitrary manifests because a
    small/subset materialization must never masquerade as
    ``baseline_v1_local``.
    """
    row = await pool.fetchrow(
        "SELECT count(t.id) AS turns, count(DISTINCT e.id) AS episodes, "
        "count(t.id) FILTER (WHERE t.embedding IS NOT NULL) AS embedded, "
        "current_database() AS database_name, current_user AS role_name, "
        "inet_server_addr()::text AS server_addr, inet_server_port() AS server_port "
        "FROM episode_turns t JOIN episodes e ON t.episode_id = e.id "
        "WHERE e.project_id LIKE 'lme_%'"
    )
    if row is None:
        raise RuntimeError("fixed-materialization identity query returned no row")

    counts: dict[str, object] = {
        "turns": int(row["turns"]),
        "episodes": int(row["episodes"]),
        "embedded": int(row["embedded"]),
        "database": row["database_name"],
        "role": row["role_name"],
        "server_addr": row["server_addr"],
        "server_port": row["server_port"],
    }
    expected = {
        "turns": manifest.total_turns,
        "episodes": manifest.total_episodes,
        "embedded": manifest.total_turns,
    }
    if manifest.snapshot_name == "baseline_v1_local":
        expected = {"turns": 2_446_993, "episodes": 500, "embedded": 2_446_993}
    live_shape = {key: counts[key] for key in expected}
    if live_shape != expected:
        raise ValueError(
            "fixed-materialization row-count/embedding mismatch: "
            f"live={live_shape}, expected={expected}, "
            f"snapshot={manifest.snapshot_name!r}, database={counts['database']!r}"
        )
    return counts


def _manifest_identity(manifest: "Manifest") -> str:
    """Return the stable hash used to bind outputs to one manifest."""
    return hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest()


def _assert_local_benchmark_target() -> None:
    """Refuse the fixed-snapshot CLI unless its DSN is the dedicated local DB."""
    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL", "").strip()
    if not dsn:
        raise click.ClickException(
            "LONGMEMEVAL_DATABASE_URL is required for fixed-snapshot runs; "
            "refusing to inspect an implicit/hosted database."
        )
    parsed = urlsplit(dsn.replace("+psycopg2", ""))
    host = (parsed.hostname or "").lower()
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")
    if host not in {"127.0.0.1", "localhost", "::1"} or port != 55432 or database != "weftbench":
        raise click.ClickException(
            "fixed-snapshot runs require the dedicated local target "
            f"{DEFAULT_LOCAL_BENCHMARK_DSN!r}; got {_redacted_database_target(dsn)!r}"
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
    capture_recall: bool = False,
    recall_k: int = 10,
    warm_boost_rounds: int = 0,
    warm_boost_queries_per_round: int = 10,
    detector: Detector | None = None,
    replay_executor: ReplayExecutorKind = "inline",
) -> tuple[str, dict]:
    """Run the full pipeline for one question.

    Args:
        capture_recall: When True (turn-mode + turn-tier only), build an
            in-memory ``{turn_id: session_id}`` side-map at ingest time
            and record per-question recall@``recall_k`` against the gold
            ``answer_session_ids``. The metric is returned in the
            telemetry dict under the ``recall`` key. Other modes don't
            have a stable id-to-session mapping, so this flag is a no-op
            outside of ``mode='turns'`` + ``tier='turns'``.
        recall_k: Cutoff for the recall metric (default 10). Independent
            of the Reader's ``top_k`` so we can report recall@10 even
            when the policy widens to 30 for multi-session questions.

    Returns:
        (hypothesis, telemetry_dict). The hypothesis goes into the JSONL
        results file; telemetry is aggregated into RunStats. When
        capture_recall is enabled, telemetry["recall"] holds a dict with
        per-question recall fields ready to serialize to JSONL.
    """
    project_id = project_id_for(instance.question_id)

    # 1+2. Ingest haystack into a per-question project sandbox.
    # Allocate per-question side-maps only when recall@k capture is on
    # — keeps memory pressure flat for non-instrumented runs.
    turn_session_map: dict[str, str] | None = (
        {} if capture_recall and mode == "turns" and tier == "turns" else None
    )
    turn_content_map: dict[str, str] | None = (
        {} if capture_recall and mode == "turns" and tier == "turns" else None
    )
    n_sessions = await load_haystack(
        pool, embedder, instance, mode,
        turn_session_map=turn_session_map,
        turn_content_map=turn_content_map,
    )

    # 2.5. Warm-boost (P1.A5) — pre-warm the boost loop so usefulness
    # scores diverge before the actual recall query runs. No-op when
    # warm_boost_rounds=0, which is the cold-DB baseline path.
    warm_boost_telemetry: dict | None = None
    if warm_boost_rounds > 0 and tier == "turns" and mode == "turns":
        from benchmarks.longmemeval.warm_boost import warm_boost_turns
        warm_boost_telemetry = await warm_boost_turns(
            pool, embedder,
            project_id=project_id,
            rounds=warm_boost_rounds,
            queries_per_round=warm_boost_queries_per_round,
            rng_seed=hash(instance.question_id) & 0xFFFFFFFF,
        )

    # 2.6. Belief-view materialization (loom-1fe75d00) — run the detector +
    # supersession writer over this question's just-ingested turns so the
    # belief-view query has claims to read. Sandbox-scoped, cursor-free (see
    # materialize.py for why the production global cursor is wrong here). Only
    # meaningful with --mode turns; a belief-view run over raw/extracted mode
    # has no episode_turns to materialize, so the tier degrades to its
    # turn-recall fallback (which is also empty) — guard with a clear warning.
    materialize_telemetry: dict | None = None
    if tier in ("belief-view", "replay"):
        if mode != "turns":
            logger.warning(
                "tier=%r requires --mode turns (no episode_turns to "
                "materialize in mode=%r); claims will be empty for q=%s",
                tier, mode, instance.question_id,
            )
        mat_stats = await materialize_question(
            pool, project_id,
            detector=detector,
        )
        materialize_telemetry = mat_stats.to_dict()
        if mode == "turns" and mat_stats.turns_total == 0:
            # Turns mode ingested sessions but the sandbox SELECT saw nothing —
            # the signature of an identity/RLS mismatch (e.g. a caller-supplied
            # pool whose role neither owns the tables nor sets app.user_id),
            # not of an empty haystack. Loud because the run would otherwise
            # complete and bill the Reader against claim-less recalls.
            logger.warning(
                "tier=%r materialization fetched ZERO turns for q=%s "
                "(project_id=%s) despite mode=turns — check pool identity/RLS "
                "(see materialize._fetch_question_turns)",
                tier, instance.question_id, project_id,
            )

    # 2.7. Replay loop (tier=replay only) — enqueue + drain the recall-gap
    # replay substrate on top of the per-turn claims just materialized. This is
    # the load-bearing step that makes the replay path actually execute on the
    # benchmark: without it the substrate is inert and a before/after measures
    # nothing (see replay_drive.py). The aggregate detector writes its multi-turn
    # 'replay-' claims into belief_claims so the recall below can read them.
    replay_telemetry: dict | None = None
    if tier == "replay":
        replay_stats = await drive_replay(
            pool,
            question=instance.question,
            user_id=BENCHMARK_USER_ID,
            executor=replay_executor,
        )
        replay_telemetry = replay_stats.to_dict()

    # 3. Recall — question-type-aware policy lives in router.policy_for().
    # The CLI top_k acts as a floor: if a caller bumps top_k above the
    # policy default (e.g. running ablations), honor that. Sandbox isolation
    # (over-fetch + post-filter to project_id) is handled inside retrieve().
    policy = policy_for(instance.question_type)
    if top_k > policy.top_k:
        from benchmarks.longmemeval.router import RetrievalPolicy
        policy = RetrievalPolicy(top_k=top_k, overfetch_multiplier=policy.overfetch_multiplier)
    # Ground-truth turn count for diagnostics: if the side-map was
    # populated during ingest, its length is the exact number of turns
    # inserted for this question. Used by the 0-return diagnostic path.
    expected_turn_count = len(turn_session_map) if turn_session_map else 0
    # Gold turn IDs for per-half miss diagnostics: all turn IDs from
    # the side-map are gold candidates (any of them could surface the
    # gold answer). Used to log where gold turns ranked in vector vs.
    # keyword halves when recall@k misses.
    gold_turn_ids = list(turn_session_map.keys()) if turn_session_map else None

    memories = await retrieve(
        pool, embedder,
        question=instance.question,
        question_type=instance.question_type,
        project_id=project_id,
        policy=policy,
        tier=tier,
        user_id=BENCHMARK_USER_ID,
        expected_turn_count=expected_turn_count,
        gold_turn_ids=gold_turn_ids,
        turn_session_map=turn_session_map,
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
    if warm_boost_telemetry is not None:
        telemetry["warm_boost"] = warm_boost_telemetry
    if materialize_telemetry is not None:
        telemetry["materialize"] = materialize_telemetry
    if replay_telemetry is not None:
        telemetry["replay"] = replay_telemetry

    # Recall@k instrumentation — only meaningful when we have side-maps.
    # Session-level: compare source sessions of top-k retrieved turns
    # against the gold ``has_answer`` sessions (loaded from
    # ``answer_session_ids`` in dataset.py). A session-level "hit" means
    # at least one gold session appears among the retrieved sessions.
    #
    # Turn-level: additionally check whether the gold answer text appears
    # verbatim in any of the retrieved turns' content (substring match
    # for long gold answers, word-boundary match for short ones). See
    # ``_answer_text_match`` for the documented heuristic and its
    # known false-positive / false-negative cases.
    if turn_session_map is not None:
        topk = memories[:recall_k]
        retrieved_turn_ids = [m.memory.id for m in topk]
        retrieved_session_ids = [
            turn_session_map[tid] for tid in retrieved_turn_ids
            if tid in turn_session_map
        ]
        retrieved_session_set = set(retrieved_session_ids)
        gold_session_ids = {
            s.session_id for s in instance.sessions if s.has_answer
        }
        hit = bool(gold_session_ids & retrieved_session_set)

        # Turn-level recall: check whether the gold answer text appears in
        # any of the top-k retrieved turns. Requires the turn_content_map
        # side-map, which is always populated alongside turn_session_map
        # when capture_recall is True in turns mode.
        gold_answer_text = str(instance.answer).strip().lower()
        retrieved_turn_contents: list[str] = []
        if turn_content_map is not None:
            retrieved_turn_contents = [
                turn_content_map[tid]
                for tid in retrieved_turn_ids
                if tid in turn_content_map
            ]
        n_retrieved_turns_with_content = len(retrieved_turn_contents)
        turn_level_hit = any(
            _answer_text_match(gold_answer_text, content)
            for content in retrieved_turn_contents
        )

        telemetry["recall"] = {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "k": recall_k,
            "retrieved_turn_ids": retrieved_turn_ids,
            "retrieved_session_ids": sorted(retrieved_session_set),
            "gold_session_ids": sorted(gold_session_ids),
            "recall_at_k_hit": hit,
            "n_turns_indexed": len(turn_session_map),
            "turn_level_recall_at_k_hit": turn_level_hit,
            "n_retrieved_turns_with_content": n_retrieved_turns_with_content,
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
    resume: bool = False,
    question_types: frozenset[str] | None = None,
    stratified_frac: float | None = None,
    sample_seed: int = 0,
    tier: Tier = "belief",
    warm_boost_rounds: int = 0,
    warm_boost_queries_per_round: int = 10,
    replay_executor: ReplayExecutorKind = "inline",
    pool: asyncpg.Pool | None = None,
    embedder: EmbeddingProvider | None = None,
    reader: Reader | None = None,
    detector: Detector | None = None,
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

    if resume and output_path.exists():
        completed_ids: set[str] = set()
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question_id = row.get("question_id")
                if question_id:
                    completed_ids.add(question_id)
        instances = [i for i in instances if i.question_id not in completed_ids]
        logger.info("resume: skipping %d completed questions", len(completed_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    owns_pool = pool is None
    if pool is None:
        pool = await _make_pool()
    if embedder is None:
        embedder = _make_embedder()
    if reader is None:
        reader = Reader()

    # Refresh the connection pool every N questions to simulate realistic usage.
    # In production, a user asks one or a few questions per session — not 500
    # rapid-fire recalls on the same pool. Sustained pool churn causes
    # nondeterministic state degradation (stale snapshots, leaked transaction
    # state) that doesn't reflect real usage. Recreating the pool every few
    # questions gives a clean, fair benchmark that matches how the system is
    # actually used: a few recalls per session, then a fresh connection.
    _POOL_REFRESH_INTERVAL = 3

    stats = RunStats(questions_total=len(instances), started_at=time.monotonic())

    # Recall@10 instrumentation is only meaningful when we have a stable
    # turn-id → session-id mapping, which only exists in turn-mode + turn
    # tier. Other configurations skip the side-map allocation entirely.
    capture_recall = mode == "turns" and tier == "turns"
    recall_k = 10
    recall_records: list[dict] = []
    recall_jsonl_path = output_path.parent / f"{output_path.stem}_recall_at_{recall_k}.jsonl"
    recall_summary_path = output_path.parent / f"{output_path.stem}_recall_at_{recall_k}_summary.json"
    recall_out = recall_jsonl_path.open("a", encoding="utf-8") if capture_recall else None

    try:
        with output_path.open("a", encoding="utf-8") as out:
            for q_idx, instance in enumerate(instances):
                # Refresh pool every N questions for clean connection state.
                if owns_pool and q_idx > 0 and q_idx % _POOL_REFRESH_INTERVAL == 0:
                    await pool.close()
                    pool = await _make_pool()
                    logger.info("pool refreshed at question %d/%d", q_idx, len(instances))
                try:
                    hypothesis, telemetry = await _run_one(
                        pool, embedder, reader, instance,
                        mode=mode, top_k=top_k, tier=tier,
                        capture_recall=capture_recall,
                        recall_k=recall_k,
                        warm_boost_rounds=warm_boost_rounds,
                        warm_boost_queries_per_round=warm_boost_queries_per_round,
                        detector=detector,
                        replay_executor=replay_executor,
                    )
                except MaterializationAborted:
                    # Consecutive-failure burst — detector/DB is down. Every
                    # subsequent question would fail the same way while still
                    # billing the Reader, so fail the whole run loudly instead
                    # of recording one more questions_failed and moving on.
                    raise
                except Exception as exc:
                    logger.exception(
                        "question %s failed: %s", instance.question_id, exc,
                    )
                    stats.questions_failed += 1
                    # No continue: fall through to the per-question cleanup
                    # below. Skipping cleanup on failure would leak this
                    # question's belief_claims (which have no project_id
                    # scoping) into every later question's belief-view recall.
                else:
                    stats.questions_done += 1
                    stats.sessions_ingested += telemetry["n_sessions"]
                    stats.input_tokens += telemetry["input_tokens"]
                    stats.cached_tokens += telemetry["cached_tokens"]
                    stats.output_tokens += telemetry["output_tokens"]
                    wb = telemetry.get("warm_boost")
                    if wb is not None:
                        stats.warm_boost_rounds += int(wb.get("rounds", 0))
                        stats.warm_boost_queries += int(wb.get("queries", 0))
                        stats.warm_boost_accessed_turns += int(wb.get("accessed_turns", 0))
                        stats.warm_boost_boosted_turns += int(wb.get("boosted_turns", 0))
                    mat = telemetry.get("materialize")
                    if mat is not None:
                        stats.materialize_turns_total += int(mat.get("turns_total", 0))
                        stats.materialize_claims_written += int(mat.get("claims_written", 0))
                        stats.materialize_claims_superseded += int(mat.get("claims_superseded", 0))
                        stats.materialize_abstentions += int(mat.get("abstentions", 0))
                        stats.materialize_errors += int(mat.get("errors", 0))
                    rp = telemetry.get("replay")
                    if rp is not None:
                        stats.replay_rows_enqueued += int(rp.get("rows_enqueued", 0))
                        stats.replay_rows_processed += int(rp.get("rows_processed", 0))
                        stats.replay_rows_done += int(rp.get("rows_done", 0))
                        stats.replay_rows_failed += int(rp.get("rows_failed", 0))
                        stats.replay_claims_written += int(rp.get("claims_written", 0))
                        stats.replay_claims_superseded += int(rp.get("claims_superseded", 0))

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

                    recall_record = telemetry.get("recall")
                    if recall_record is not None and recall_out is not None:
                        recall_records.append(recall_record)
                        recall_out.write(json.dumps(recall_record) + "\n")
                        recall_out.flush()

                if cleanup:
                    await cleanup_haystack(pool, instance)
    finally:
        if recall_out is not None:
            recall_out.close()
        if owns_pool:
            await pool.close()

    # Roll up recall@k for turn-tier runs and emit a summary JSON next to
    # the per-question JSONL. Print the headline number to stdout so smoke
    # tests don't require opening a file to read it.
    if capture_recall and recall_records:
        n_total = len(recall_records)
        n_hits = sum(1 for r in recall_records if r["recall_at_k_hit"])
        per_type: dict[str, dict[str, int | float]] = {}
        for r in recall_records:
            qt = r["question_type"]
            slot = per_type.setdefault(qt, {"n": 0, "hits": 0})
            slot["n"] = int(slot["n"]) + 1
            slot["hits"] = int(slot["hits"]) + (1 if r["recall_at_k_hit"] else 0)
        for qt, slot in per_type.items():
            n = int(slot["n"])
            hits = int(slot["hits"])
            slot["recall_at_k"] = (hits / n) if n else 0.0
        summary = {
            "k": recall_k,
            "n_questions": n_total,
            "n_hits": n_hits,
            "recall_at_k": (n_hits / n_total) if n_total else 0.0,
            "per_question_type": per_type,
            "mode": mode,
            "tier": tier,
            "dataset": str(dataset_path),
        }
        recall_summary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        logger.info(
            "recall@%d = %.3f (%d/%d) — written to %s",
            recall_k, summary["recall_at_k"], n_hits, n_total, recall_summary_path,
        )
        stats.recall_at_k = float(summary["recall_at_k"])
        stats.recall_k = recall_k
        stats.recall_n_hits = n_hits
        stats.recall_n_questions = n_total
        stats.recall_jsonl_path = str(recall_jsonl_path)
        stats.recall_summary_path = str(recall_summary_path)

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
                "warm_boost": {
                    "config_rounds": warm_boost_rounds,
                    "config_queries_per_round": warm_boost_queries_per_round,
                    "rerank_disabled": os.environ.get(
                        "WEFT_TURN_RERANK_DISABLE"
                    ) == "1",
                    "total_rounds_ran": stats.warm_boost_rounds,
                    "total_queries": stats.warm_boost_queries,
                    "total_accessed_turns": stats.warm_boost_accessed_turns,
                    "total_boosted_turns": stats.warm_boost_boosted_turns,
                } if warm_boost_rounds > 0 else None,
                "materialize": {
                    "turns_total": stats.materialize_turns_total,
                    "claims_written": stats.materialize_claims_written,
                    "claims_superseded": stats.materialize_claims_superseded,
                    "abstentions": stats.materialize_abstentions,
                    "errors": stats.materialize_errors,
                } if tier in ("belief-view", "replay") else None,
                "replay": {
                    "executor": replay_executor,
                    "rows_enqueued": stats.replay_rows_enqueued,
                    "rows_processed": stats.replay_rows_processed,
                    "rows_done": stats.replay_rows_done,
                    "rows_failed": stats.replay_rows_failed,
                    "claims_written": stats.replay_claims_written,
                    "claims_superseded": stats.replay_claims_superseded,
                } if tier == "replay" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


# ----------------------------------------------------------------------
# Materialize-only and recall-only modes (Phase 2 A/B framework)
# ----------------------------------------------------------------------


async def _run_materialize_only(
    *,
    dataset_path: Path,
    snapshot_dir: Path,
    snapshot_name: str,
    limit: int | None = None,
    question_types: frozenset[str] | None = None,
    stratified_frac: float | None = None,
    sample_seed: int = 0,
) -> None:
    """Ingest all questions into the DB, persist manifest + snapshot, exit.

    No recall, no Reader, no cleanup. Creates the fixed-materialization
    state all A/B comparisons run against.
    """
    from benchmarks.longmemeval.snapshot import (
        Manifest,
        QuestionManifest,
        code_version,
        dataset_checksum,
        snapshot as snapshot_fn,
    )
    from benchmarks.longmemeval.ingest import load_haystack

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    instances = load_split(dataset_path)
    if question_types:
        instances = [i for i in instances if i.question_type in question_types]
    if stratified_frac is not None:
        instances = _stratified_sample(instances, frac=stratified_frac, seed=sample_seed)
    if limit is not None:
        instances = instances[:limit]

    pool = await _make_pool()
    embedder = _make_embedder()

    try:
        retrieval_as_of = datetime.now(timezone.utc).isoformat()
        questions: list[QuestionManifest] = []
        total_episodes = 0
        total_turns = 0

        for idx, instance in enumerate(instances):
            project_id = project_id_for(instance.question_id)

            # Resume only a complete question sandbox. A forced close can
            # leave an episode with some inserted turns; accepting any
            # positive count would silently publish an under-materialized
            # snapshot. Partial sandboxes are deleted and re-ingested.
            existing_turns = await pool.fetchval(
                "SELECT count(*) FROM episode_turns t "
                "JOIN episodes e ON t.episode_id = e.id "
                "WHERE e.project_id = $1",
                project_id,
            )
            existing_episodes = await pool.fetchval(
                "SELECT count(*) FROM episodes WHERE project_id = $1",
                project_id,
            )
            expected_turns = expected_turn_count(instance)
            if (
                int(existing_episodes or 0) == 1
                and int(existing_turns or 0) == expected_turns
            ):
                logger.info(
                    "materialize-only: %d/%d (%s) — SKIPPED (complete: %d turns)",
                    idx + 1, len(instances), instance.question_id,
                    expected_turns,
                )
                rows = await pool.fetch(
                    "SELECT t.id, e.id AS ep_id FROM episode_turns t "
                    "JOIN episodes e ON t.episode_id = e.id "
                    "WHERE e.project_id = $1 ORDER BY t.turn_index",
                    project_id,
                )
                turn_session_map = {r["id"]: r["ep_id"] for r in rows}
                ep_count = await pool.fetchval(
                    "SELECT count(*) FROM episodes WHERE project_id = $1",
                    project_id,
                )
                turn_count = int(existing_turns or 0)
                total_episodes += int(ep_count or 0)
                total_turns += turn_count
                questions.append(QuestionManifest(
                    question_id=instance.question_id,
                    question_type=instance.question_type,
                    project_id=project_id,
                    gold_session_ids=list(getattr(instance, "answer_session_ids", [])),
                    turn_ids=list(turn_session_map.keys()),
                    turn_session_map=dict(turn_session_map),
                    row_count=turn_count,
                ))
                continue
            if int(existing_episodes or 0) > 0:
                logger.warning(
                    "materialize-only: %s has partial sandbox (%d episodes, %d/%d turns); cleaning it before retry",
                    instance.question_id, int(existing_episodes),
                    int(existing_turns or 0), expected_turns,
                )
                await cleanup_haystack(pool, instance)

            turn_session_map: dict[str, str] = {}
            turn_content_map: dict[str, str] = {}
            await load_haystack(
                pool, embedder, instance, "turns",
                turn_session_map=turn_session_map,
                turn_content_map=turn_content_map,
            )

            # Count episodes and turns for this question
            ep_count = await pool.fetchval(
                "SELECT count(*) FROM episodes WHERE project_id = $1",
                project_id,
            )
            turn_count = await pool.fetchval(
                "SELECT count(*) FROM episode_turns t "
                "JOIN episodes e ON t.episode_id = e.id "
                "WHERE e.project_id = $1",
                project_id,
            )
            total_episodes += int(ep_count or 0)
            total_turns += int(turn_count or 0)

            questions.append(QuestionManifest(
                question_id=instance.question_id,
                question_type=instance.question_type,
                project_id=project_id,
                gold_session_ids=list(getattr(instance, "answer_session_ids", [])),
                turn_ids=list(turn_session_map.keys()),
                turn_session_map=dict(turn_session_map),
                turn_content_map=dict(turn_content_map),
                row_count=int(turn_count or 0),
            ))
            logger.info(
                "materialize-only: %d/%d (%s) — %d turns",
                idx + 1, len(instances), instance.question_id,
                turn_count or 0,
            )

        # Get embedding info
        config = load_config()
        manifest = Manifest(
            snapshot_name=snapshot_name,
            dataset_path=str(dataset_path),
            dataset_checksum=dataset_checksum(dataset_path),
            code_version=code_version(),
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model or "",
            embedding_dimensions=config.embedding.dimensions or 0,
            retrieval_as_of=retrieval_as_of,
            questions=questions,
            total_episodes=total_episodes,
            total_turns=total_turns,
        )

        await snapshot_fn(pool, name=snapshot_name, manifest=manifest, base_dir=snapshot_dir)
    finally:
        await pool.close()


async def _run_recall_only(
    *,
    dataset_path: Path,
    snapshot_dir: Path,
    snapshot_name: str,
    output_dir: Path,
    split_name: str,
    timestamp: str,
    limit: int | None = None,
    question_types: frozenset[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Load manifest, run recall-only, record recall@k. No ingest, no Reader.

    Assumes the snapshot is already loaded in the DB (via restore_snapshot
    or a prior --materialize-only run that left data in place).
    """
    from benchmarks.longmemeval.snapshot import Manifest, snapshot_dir as snap_dir_fn
    from benchmarks.longmemeval.router import (
        RetrievalDiagnostics,
        policy_for,
        retrieve,
    )

    sdir = snap_dir_fn(snapshot_dir, snapshot_name)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = Manifest.from_json(manifest_path.read_text(encoding="utf-8"))

    # Load question texts from the dataset (needed for embedding the query)
    instances = load_split(dataset_path)
    question_text_map = {i.question_id: i.question for i in instances}

    # Filter questions
    qs = manifest.questions
    if question_types:
        qs = [q for q in qs if q.question_type in question_types]
    if limit is not None:
        qs = qs[:limit]

    pool = await _make_pool()
    embedder = _make_embedder()
    from datetime import datetime as _dt
    as_of = _dt.fromisoformat(manifest.retrieval_as_of)

    recall_k = 10
    recall_records: list[dict] = []

    try:
        for idx, q in enumerate(qs):
            policy = policy_for(q.question_type)
            if top_k > policy.top_k:
                from benchmarks.longmemeval.router import RetrievalPolicy
                policy = RetrievalPolicy(
                    top_k=top_k, overfetch_multiplier=policy.overfetch_multiplier,
                )

            text = question_text_map.get(q.question_id, q.question_id)
            diagnostics = RetrievalDiagnostics()
            indexed_turn_ids = list(q.turn_ids)
            gold_session_ids = set(q.gold_session_ids)
            gold_session_turn_ids = [
                tid for tid in indexed_turn_ids
                if q.turn_session_map.get(tid) in gold_session_ids
            ]
            diagnostics.indexed_turn_ids = indexed_turn_ids
            diagnostics.gold_session_turn_ids = gold_session_turn_ids
            memories = await retrieve(
                pool, embedder,
                question=text,
                question_type=q.question_type,
                project_id=q.project_id,
                policy=policy,
                tier="turns",
                expected_turn_count=q.row_count,
                as_of=as_of,
                gold_turn_ids=gold_session_turn_ids,
                diagnostics=diagnostics,
            )

            topk = memories[:recall_k]
            retrieved_turn_ids = [m.memory.id for m in topk]
            retrieved_session_ids = set()
            for tid in retrieved_turn_ids:
                sid = q.turn_session_map.get(tid)
                if sid:
                    retrieved_session_ids.add(sid)
            gold_set = set(q.gold_session_ids)
            recall_hit = bool(retrieved_session_ids & gold_set)

            recall_records.append({
                "question_id": q.question_id,
                "question_type": q.question_type,
                "recall_at_k_hit": recall_hit,
                "retrieved_turn_ids": retrieved_turn_ids,
                "gold_session_ids": q.gold_session_ids,
                "retrieval_diagnostics": diagnostics.to_dict(
                    gold_turn_ids=q.turn_ids,
                    top_k=policy.top_k,
                    recall_hit=recall_hit,
                ),
            })

            if (idx + 1) % 50 == 0:
                logger.info("recall-only: %d/%d", idx + 1, len(qs))
    finally:
        await pool.close()

    # Build summary
    n_total = len(recall_records)
    n_hits = sum(1 for r in recall_records if r["recall_at_k_hit"])
    per_type: dict[str, dict[str, int | float]] = {}
    for r in recall_records:
        qt = r["question_type"]
        slot = per_type.setdefault(qt, {"n": 0, "hits": 0})
        slot["n"] += 1
        slot["hits"] += 1 if r["recall_at_k_hit"] else 0
    for qt, slot in per_type.items():
        slot["recall_at_k"] = slot["hits"] / slot["n"] if slot["n"] else 0.0

    diagnostic_counts: dict[str, int] = {}
    diagnostic_by_type: dict[str, dict[str, int]] = {}
    for record in recall_records:
        diag = record["retrieval_diagnostics"]
        cause = diag["cause"]
        diagnostic_counts[cause] = diagnostic_counts.get(cause, 0) + 1
        for key in (
            "initial_empty", "retry_attempted", "retry_rescued",
            "fallback_attempted", "fallback_rescued", "final_empty",
        ):
            if diag[key]:
                diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1
        qt_counts = diagnostic_by_type.setdefault(record["question_type"], {})
        for key in (cause, diag["path"]):
            qt_counts[key] = qt_counts.get(key, 0) + 1
        if record["recall_at_k_hit"]:
            qt_counts["recall_hits"] = qt_counts.get("recall_hits", 0) + 1
        else:
            qt_counts["recall_misses"] = qt_counts.get("recall_misses", 0) + 1

    summary = {
        "k": recall_k,
        "n_questions": n_total,
        "n_hits": n_hits,
        "recall_at_k": n_hits / n_total if n_total else 0.0,
        "per_question_type": per_type,
        "diagnostic_counts": diagnostic_counts,
        "diagnostic_by_question_type": diagnostic_by_type,
        "mode": "turns",
        "tier": "turns",
        "dataset": str(dataset_path),
        "snapshot_name": snapshot_name,
        "retrieval_as_of": manifest.retrieval_as_of,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{split_name}_recall_only_{snapshot_name}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    jsonl_path = output_dir / f"{split_name}_recall_only_{snapshot_name}_{timestamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in recall_records:
            f.write(json.dumps(r) + "\n")

    summary["summary_path"] = str(summary_path)
    return summary


async def _run_snapshot_reader_only(
    *,
    dataset_path: Path,
    snapshot_dir: Path,
    snapshot_name: str,
    output_path: Path,
    telemetry_path: Path | None = None,
    summary_path: Path | None = None,
    limit: int | None = None,
    question_types: frozenset[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    resume: bool = False,
    pool: asyncpg.Pool | None = None,
    embedder: EmbeddingProvider | None = None,
    reader: Reader | None = None,
) -> dict:
    """Read answers from an already-restored turn snapshot.

    This is intentionally a separate path from :func:`run_benchmark`.  It
    never calls ``load_haystack``, ``cleanup_haystack``, materialization,
    replay, snapshot restore, or any other benchmark write during the
    question loop.  The only per-question operations are query embedding,
    the existing turn-tier ``retrieve`` router, and ``Reader.read_answer``.

    ``output_path`` is evaluator-compatible JSONL.  ``telemetry_path`` is a
    separate resumable audit trail so provenance and token accounting never
    pollute the evaluator input.  A question is resumable only after a
    successful telemetry row and a hypothesis row both exist; failed rows are
    retried on the next invocation.
    """
    from benchmarks.longmemeval.snapshot import Manifest, snapshot_dir as snap_dir_fn

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    sdir = snap_dir_fn(snapshot_dir, snapshot_name)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not (sdir / ".complete").exists():
        raise RuntimeError(f"snapshot is not marked complete: {sdir}")

    manifest = Manifest.from_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.snapshot_name != snapshot_name:
        raise ValueError(
            f"snapshot name mismatch: requested={snapshot_name!r}, "
            f"manifest={manifest.snapshot_name!r}"
        )
    actual_dataset_checksum = _dataset_checksum_for_reader(dataset_path)
    if actual_dataset_checksum != manifest.dataset_checksum:
        raise ValueError(
            "dataset checksum mismatch: "
            f"dataset={actual_dataset_checksum}, manifest={manifest.dataset_checksum}"
        )

    instances = load_split(dataset_path)
    instance_by_id = {instance.question_id: instance for instance in instances}
    if len(instance_by_id) != len(instances):
        raise ValueError("dataset contains duplicate question_id values")
    missing_instances = [q.question_id for q in manifest.questions if q.question_id not in instance_by_id]
    if missing_instances:
        raise ValueError(
            "dataset is missing manifest question IDs: "
            f"{missing_instances[:5]}" + ("..." if len(missing_instances) > 5 else "")
        )

    questions = list(manifest.questions)
    if question_types:
        questions = [q for q in questions if q.question_type in question_types]
    if limit is not None:
        questions = questions[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path = telemetry_path or output_path.with_suffix(output_path.suffix + ".telemetry.jsonl")
    summary_path = summary_path or output_path.with_suffix(output_path.suffix + ".summary.json")
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if resume:
        hypothesis_ids: set[str] = set()
        if output_path.exists():
            with output_path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = row.get("question_id")
                    if qid:
                        hypothesis_ids.add(qid)
        successful_telemetry_ids: set[str] = set()
        if telemetry_path.exists():
            with telemetry_path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("status") == "ok"
                        and row.get("question_id")
                        and row.get("snapshot_name") == snapshot_name
                        and row.get("dataset_checksum") == manifest.dataset_checksum
                        and row.get("manifest_identity") == _manifest_identity(manifest)
                        and row.get("retrieval_as_of") == manifest.retrieval_as_of
                    ):
                        successful_telemetry_ids.add(row["question_id"])
        completed_ids = hypothesis_ids & successful_telemetry_ids
        logger.info("snapshot-reader resume: skipping %d completed questions", len(completed_ids))

    owns_pool = pool is None
    if pool is None:
        pool = await _make_pool()
    if embedder is None:
        embedder = _make_embedder()
    if reader is None:
        reader = Reader()

    # The identity gate is read-only and runs before any Reader billing.
    live_identity = await _snapshot_live_identity(pool, manifest)
    manifest_identity = _manifest_identity(manifest)
    as_of = datetime.fromisoformat(manifest.retrieval_as_of)
    started_at = time.monotonic()
    records: list[dict] = []
    questions_failed = 0
    questions_done = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0

    try:
        with output_path.open("a", encoding="utf-8") as hypotheses, telemetry_path.open(
            "a", encoding="utf-8"
        ) as telemetry_out:
            for index, q in enumerate(questions):
                if q.question_id in completed_ids:
                    continue
                instance = instance_by_id[q.question_id]
                policy = policy_for(q.question_type)
                if top_k > policy.top_k:
                    from benchmarks.longmemeval.router import RetrievalPolicy
                    policy = RetrievalPolicy(
                        top_k=top_k,
                        overfetch_multiplier=policy.overfetch_multiplier,
                    )

                diagnostics = RetrievalDiagnostics()
                indexed_turn_ids = list(q.turn_ids)
                gold_session_turn_ids = [
                    tid for tid in indexed_turn_ids
                    if q.turn_session_map.get(tid) in set(q.gold_session_ids)
                ]
                diagnostics.indexed_turn_ids = indexed_turn_ids
                diagnostics.gold_session_turn_ids = gold_session_turn_ids
                question_started = time.perf_counter()
                provenance = {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "project_id": q.project_id,
                    "snapshot_name": manifest.snapshot_name,
                    "manifest_identity": manifest_identity,
                    "dataset_checksum": manifest.dataset_checksum,
                    "retrieval_as_of": manifest.retrieval_as_of,
                    "indexed_turn_count": len(indexed_turn_ids),
                    "gold_session_turn_count": len(gold_session_turn_ids),
                    "retrieval_tier": "turns",
                    "reader_top_k": policy.top_k,
                }
                try:
                    memories = await retrieve(
                        pool,
                        embedder,
                        question=instance.question,
                        question_type=instance.question_type,
                        project_id=q.project_id,
                        policy=policy,
                        tier="turns",
                        user_id=BENCHMARK_USER_ID,
                        expected_turn_count=q.row_count,
                        as_of=as_of,
                        gold_turn_ids=gold_session_turn_ids,
                        diagnostics=diagnostics,
                        turn_session_map=q.turn_session_map,
                    )
                    response = await reader.read_answer(
                        question=instance.question,
                        question_date=instance.question_date,
                        question_type=instance.question_type,
                        memories=memories,
                        top_k=policy.top_k,
                    )
                except Exception as exc:
                    questions_failed += 1
                    failed = {
                        **provenance,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_ms": (time.perf_counter() - question_started) * 1000.0,
                        "retrieved_turn_ids": [],
                        "retrieval_diagnostics": diagnostics.to_dict(
                            gold_turn_ids=gold_session_turn_ids,
                            top_k=policy.top_k,
                            recall_hit=False,
                        ),
                    }
                    telemetry_out.write(json.dumps(failed) + "\n")
                    telemetry_out.flush()
                    records.append(failed)
                    logger.exception("snapshot-reader question %s failed", q.question_id)
                    continue

                retrieved_turn_ids = [memory.memory.id for memory in memories[:10]]
                retrieved_sessions = {
                    q.turn_session_map[turn_id]
                    for turn_id in retrieved_turn_ids
                    if turn_id in q.turn_session_map
                }
                recall_hit = bool(retrieved_sessions & set(q.gold_session_ids))
                record = {
                    **provenance,
                    "status": "ok",
                    "hypothesis": response.hypothesis,
                    "reader_model": response.model,
                    "input_tokens": response.input_tokens,
                    "cached_input_tokens": response.cached_input_tokens,
                    "output_tokens": response.output_tokens,
                    "n_recalled": len(memories),
                    "retrieved_turn_ids": retrieved_turn_ids,
                    "retrieved_session_ids": sorted(retrieved_sessions),
                    "gold_session_ids": list(q.gold_session_ids),
                    "recall_at_10_hit": recall_hit,
                    "retrieval_diagnostics": diagnostics.to_dict(
                        gold_turn_ids=gold_session_turn_ids,
                        top_k=policy.top_k,
                        recall_hit=recall_hit,
                    ),
                    "elapsed_ms": (time.perf_counter() - question_started) * 1000.0,
                }
                hypotheses.write(json.dumps({
                    "question_id": q.question_id,
                    "hypothesis": response.hypothesis,
                }) + "\n")
                hypotheses.flush()
                telemetry_out.write(json.dumps(record) + "\n")
                telemetry_out.flush()
                records.append(record)
                questions_done += 1
                input_tokens += response.input_tokens
                cached_tokens += response.cached_input_tokens
                output_tokens += response.output_tokens
                if (index + 1) % 25 == 0:
                    logger.info("snapshot-reader: %d/%d", index + 1, len(questions))
    finally:
        if owns_pool:
            await pool.close()

    # A second read-only gate catches accidental state drift while the Reader
    # was running. It never restores or mutates the snapshot.
    final_identity = await _snapshot_live_identity(pool, manifest) if not owns_pool else None
    if owns_pool:
        # Reopen only for the final identity query; this remains read-only.
        check_pool = await _make_pool()
        try:
            final_identity = await _snapshot_live_identity(check_pool, manifest)
        finally:
            await check_pool.close()
    if final_identity != live_identity:
        raise RuntimeError(
            "fixed-materialization identity drifted during snapshot-reader run: "
            f"before={live_identity}, after={final_identity}"
        )

    successful_total = sum(1 for record in records if record.get("status") == "ok")
    expected_ids = {q.question_id for q in questions}
    output_ids: list[str] = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("question_id") in expected_ids:
                    output_ids.append(row["question_id"])
    unique_output_ids = set(output_ids)
    duplicate_question_ids = sorted({qid for qid in output_ids if output_ids.count(qid) > 1})
    summary = {
        "status": (
            "complete"
            if len(unique_output_ids) == len(expected_ids) and not duplicate_question_ids
            else "incomplete"
        ),
        "questions_requested": len(expected_ids),
        "questions_done_this_run": questions_done,
        "questions_failed_this_run": questions_failed,
        "questions_complete": len(unique_output_ids),
        "questions_missing": sorted(expected_ids - unique_output_ids),
        "duplicate_question_ids": duplicate_question_ids,
        "input_tokens_this_run": input_tokens,
        "cached_input_tokens_this_run": cached_tokens,
        "output_tokens_this_run": output_tokens,
        "elapsed_seconds": time.monotonic() - started_at,
        "dataset": str(dataset_path),
        "dataset_checksum": manifest.dataset_checksum,
        "snapshot_name": manifest.snapshot_name,
        "manifest_identity": manifest_identity,
        "retrieval_as_of": manifest.retrieval_as_of,
        "live_identity_before": live_identity,
        "live_identity_after": final_identity,
        "hypotheses_path": str(output_path),
        "telemetry_path": str(telemetry_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _dataset_checksum_for_reader(dataset_path: Path) -> str:
    """Hash the dataset without importing snapshot helpers into the CLI path."""
    digest = hashlib.sha256()
    with dataset_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    type=click.Choice(["belief", "turns", "auto", "belief-view", "replay"]),
    default="belief",
    show_default=True,
    help=(
        "Retrieval tier. 'belief' (default) hits hybrid recall over memories. "
        "'turns' queries episode_turns directly (use with --mode turns). "
        "'auto' routes multi-session and temporal-reasoning to turns; "
        "everything else stays on belief. 'belief-view' materializes claims "
        "from the turns substrate then reads the supersession-collapsed "
        "belief_claims view, falling back to turn recall on a miss (use with "
        "--mode turns; targets knowledge-update questions). 'replay' does "
        "everything belief-view does AND drives the recall-gap replay loop "
        "(enqueue + drain the aggregate detector) per question, so multi-turn "
        "enumeration claims land before recall — the A/B partner to belief-view "
        "for measuring the replay substrate (use with --mode turns)."
    ),
)
@click.option(
    "--replay-executor",
    type=click.Choice(["inline", "batch"]),
    default="inline",
    show_default=True,
    help=(
        "Only with --tier replay. 'inline' makes synchronous Haiku/Sonnet calls "
        "(deterministic, no batch polling). 'batch' drives the Anthropic Batch "
        "API — exactly the production consolidate() path (50%% cheaper) but "
        "polls up to ~300s per question. Both produce IDENTICAL claims; the flag "
        "trades fidelity vs speed, not result."
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
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Stable hypotheses JSONL path (enables detached runs and resume).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip question_ids already present in --output-file.",
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
    "--warm-boost-rounds",
    type=int,
    default=0,
    show_default=True,
    help=(
        "P1.A5: pre-warm the turn-tier boost loop with N rounds of recall + "
        "access-log + boost before each question's actual recall query. 0 "
        "(default) is the cold-DB baseline path. Only active in "
        "--mode turns --tier turns."
    ),
)
@click.option(
    "--warm-boost-queries-per-round",
    type=int,
    default=10,
    show_default=True,
    help=(
        "Queries per warmup round. Each query is a randomly-sampled turn's "
        "content prefix — recall-driven, not gold-keyed."
    ),
)
@click.option(
    "--snapshot-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("benchmarks/longmemeval/snapshots"),
    show_default=True,
    help="Directory for snapshots (manifest + COPY files).",
)
@click.option(
    "--snapshot-name",
    default=None,
    help="Name for the snapshot (used as subdirectory under --snapshot-dir).",
)
@click.option(
    "--materialize-only",
    is_flag=True,
    default=False,
    help=(
        "Ingest all questions, persist a manifest + snapshot, then exit. "
        "Does NOT run recall, Reader, or cleanup. Creates the fixed-"
        "materialization state for A/B comparisons. Use with --mode turns "
        "--tier turns."
    ),
)
@click.option(
    "--recall-only",
    is_flag=True,
    default=False,
    help=(
        "Assume a snapshot is already loaded in the DB. Load the manifest "
        "for side-maps, run recall-only (no ingest, no Reader), record "
        "recall@k. Does NOT cleanup. Use with --snapshot-name to specify "
        "which manifest to load."
    ),
)
@click.option(
    "--snapshot-reader-only",
    is_flag=True,
    default=False,
    help=(
        "Read from an already-restored fixed snapshot with retrieve()+Reader "
        "only. Never ingests, materializes, restores, or cleans up. Requires "
        "--snapshot-name and --output-file; supports --resume."
    ),
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
    output_file: Path | None,
    resume: bool,
    question_types: tuple[str, ...],
    stratified_frac: float | None,
    sample_seed: int,
    no_cleanup: bool,
    warm_boost_rounds: int,
    warm_boost_queries_per_round: int,
    replay_executor: str,
    snapshot_dir: Path,
    snapshot_name: str | None,
    materialize_only: bool,
    recall_only: bool,
    snapshot_reader_only: bool,
    log_level: str,
) -> None:
    """Run Weft against the LongMemEval benchmark, write hypotheses JSONL."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # belief-view / replay guard rails — both tiers materialize claims from the
    # turns substrate, so both share the same two misconfigurations that burn
    # real Reader spend producing garbage. Hard-fail (before any env/key checks)
    # instead of warning per question. 'replay' inherits both because it runs
    # the belief-view path plus the replay loop on top.
    claim_tiers = ("belief-view", "replay")
    if tier in claim_tiers and mode != "turns":
        raise click.UsageError(
            f"--tier {tier} requires --mode turns: the materializer reads "
            f"episode_turns, which mode={mode!r} never writes — every recall "
            "would be empty while the Reader still bills per question."
        )
    if tier in claim_tiers and no_cleanup:
        raise click.UsageError(
            f"--tier {tier} is incompatible with --no-cleanup: "
            "belief_claims has no project_id scoping, so claims from one "
            "question leak into every later question's claim recall "
            "and silently corrupt the run."
        )

    # Match Weft's config loader: load ~/.weft/.env before checking env.
    # Without this, keys placed in the standard Weft .env would be rejected
    # because the check runs before load_config() triggers load_dotenv().
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")

    selected_modes = sum(
        bool(flag) for flag in (materialize_only, recall_only, snapshot_reader_only)
    )
    if selected_modes > 1:
        raise click.UsageError(
            "--materialize-only, --recall-only, and --snapshot-reader-only "
            "are mutually exclusive"
        )

    # --materialize-only and --recall-only don't need the Reader (no LLM calls).
    # Only the full pipeline and fixed-snapshot Reader require ANTHROPIC_API_KEY.
    if not materialize_only and not recall_only:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException(
                "ANTHROPIC_API_KEY is required for the Reader stage. "
                "Set it in your shell or add it to ~/.weft/.env."
            )

    if snapshot_reader_only:
        _assert_local_benchmark_target()
        if not snapshot_name:
            raise click.UsageError(
                "--snapshot-reader-only requires --snapshot-name"
            )
        if output_file is None:
            raise click.UsageError(
                "--snapshot-reader-only requires --output-file so the run is "
                "stable, resumable, and judgeable"
            )
        if no_cleanup:
            logger.warning("--no-cleanup is ignored by --snapshot-reader-only")

    # --materialize-only requires --mode turns
    if materialize_only and mode != "turns":
        raise click.UsageError(
            "--materialize-only requires --mode turns: only turn-mode "
            "ingest produces episode_turns rows for the snapshot."
        )

    # --recall-only requires --snapshot-name
    if recall_only and not snapshot_name:
        raise click.UsageError(
            "--recall-only requires --snapshot-name: specify the manifest "
            "to load for the recall-only run."
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
    # Suffix warm-boost config so A/B runs (rerank-on/off, varying rounds)
    # don't collide. Also captures whether WEFT_TURN_RERANK_DISABLE was set
    # for run-time disambiguation.
    warm_slug = f"_warm{warm_boost_rounds}" if warm_boost_rounds > 0 else ""
    if os.environ.get("WEFT_TURN_RERANK_DISABLE") == "1":
        warm_slug = f"{warm_slug}_rerankoff"
    # Suffix the replay executor so inline-vs-batch A/B runs don't collide.
    replay_slug = f"_{replay_executor}" if tier == "replay" else ""
    output_path = output_file or (
        output_dir / f"{split_name}_{mode}{tier_slug}{replay_slug}{slug}{warm_slug}_{timestamp}.jsonl"
    )
    if resume and output_file is None:
        raise click.UsageError("--resume requires --output-file")

    # --materialize-only: ingest all questions, persist manifest + snapshot, exit.
    if materialize_only:
        snap_name = snapshot_name or f"{split_name}_{timestamp}"
        asyncio.run(
            _run_materialize_only(
                dataset_path=dataset_path,
                snapshot_dir=snapshot_dir,
                snapshot_name=snap_name,
                limit=limit,
                question_types=qt_set,
                stratified_frac=stratified_frac,
                sample_seed=sample_seed,
            )
        )
        click.echo(f"\nMaterialized snapshot: {snapshot_dir / snap_name}")
        return

    # --recall-only: load manifest, run recall-only, record recall@k, exit.
    if recall_only:
        recall_summary = asyncio.run(
            _run_recall_only(
                dataset_path=dataset_path,
                snapshot_dir=snapshot_dir,
                snapshot_name=snapshot_name,
                output_dir=output_dir,
                split_name=split_name,
                timestamp=timestamp,
                limit=limit,
                question_types=qt_set,
                top_k=top_k,
            )
        )
        click.echo(f"\nRecall-only complete:")
        click.echo(f"  recall@{recall_summary['k']}: {recall_summary['recall_at_k']:.3f} "
                   f"({recall_summary['n_hits']}/{recall_summary['n_questions']})")
        click.echo(f"  summary: {recall_summary['summary_path']}")
        return

    if snapshot_reader_only:
        snapshot_output = output_file
        assert snapshot_output is not None
        snapshot_telemetry = snapshot_output.with_suffix(
            snapshot_output.suffix + ".telemetry.jsonl"
        )
        snapshot_summary = snapshot_output.with_suffix(
            snapshot_output.suffix + ".summary.json"
        )
        reader_summary = asyncio.run(
            _run_snapshot_reader_only(
                dataset_path=dataset_path,
                snapshot_dir=snapshot_dir,
                snapshot_name=snapshot_name,
                output_path=snapshot_output,
                telemetry_path=snapshot_telemetry,
                summary_path=snapshot_summary,
                limit=limit,
                question_types=qt_set,
                top_k=top_k,
                resume=resume,
            )
        )
        click.echo("\\nFixed-snapshot Reader complete:")
        click.echo(
            f"  status={reader_summary['status']} "
            f"complete={reader_summary['questions_complete']}/"
            f"{reader_summary['questions_requested']} "
            f"failed_this_run={reader_summary['questions_failed_this_run']}"
        )
        click.echo(f"  hypotheses: {snapshot_output}")
        click.echo(f"  telemetry: {snapshot_telemetry}")
        click.echo(f"  summary: {snapshot_summary}")
        return

    stats = asyncio.run(
        run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            mode=mode,
            top_k=top_k,
            cleanup=not no_cleanup,
            limit=limit,
            resume=resume,
            question_types=qt_set,
            stratified_frac=stratified_frac,
            sample_seed=sample_seed,
            tier=tier,
            warm_boost_rounds=warm_boost_rounds,
            warm_boost_queries_per_round=warm_boost_queries_per_round,
            replay_executor=replay_executor,  # type: ignore[arg-type]
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
    if stats.recall_at_k is not None:
        click.echo(
            f"  recall@{stats.recall_k}: {stats.recall_at_k:.3f} "
            f"({stats.recall_n_hits}/{stats.recall_n_questions})"
        )
        click.echo(f"  recall jsonl: {stats.recall_jsonl_path}")
        click.echo(f"  recall summary: {stats.recall_summary_path}")
    if tier in ("belief-view", "replay"):
        click.echo(
            f"  materialize: turns={stats.materialize_turns_total} "
            f"claims={stats.materialize_claims_written} "
            f"superseded={stats.materialize_claims_superseded} "
            f"abstentions={stats.materialize_abstentions} "
            f"errors={stats.materialize_errors}"
        )
        if stats.materialize_errors:
            click.echo(
                f"  WARNING: {stats.materialize_errors} materialization "
                "errors — the claim view under-read those turns; treat the "
                "score as a lower bound, not a gate result.",
            )
    if tier == "replay":
        click.echo(
            f"  replay ({replay_executor}): enqueued={stats.replay_rows_enqueued} "
            f"processed={stats.replay_rows_processed} "
            f"done={stats.replay_rows_done} failed={stats.replay_rows_failed} "
            f"agg_claims={stats.replay_claims_written} "
            f"superseded={stats.replay_claims_superseded}"
        )
        if stats.questions_failed and stats.questions_done == 0:
            # Every question errored before recall — the replay loop never
            # actually ran (e.g. a missing replay_queue table). agg_claims=0
            # here means "crashed", NOT "no enumerations found". Do not print
            # the reassuring null-result note: the results file is invalid.
            click.echo(
                f"  ERROR: all {stats.questions_failed} questions FAILED — the "
                "replay loop did not run (see the traceback above; a missing "
                "replay_queue table means migrations v53/v55 are unapplied). "
                "agg_claims=0 is a crash, not a null result; the hypotheses "
                "file is empty/invalid and must NOT be judged."
            )
        elif stats.replay_claims_written == 0:
            click.echo(
                "  NOTE: replay wrote 0 aggregate claims — the loop RAN but the "
                "detector found no multi-turn enumerations. This is a real null "
                "result (replay had no effect here), not an inert path; compare "
                "the score against a --tier belief-view run to confirm.",
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
# 3b. BELIEF-VIEW GATE (loom-1fe75d00) — knowledge-update accuracy ≥ 0.85.
#     Requires --mode turns (the substrate the materializer reads) and
#     --tier belief-view (materialize per question, then read the claim
#     view). Start with the knowledge-update subset to bound cost: the
#     materializer makes ONE Haiku call per haystack turn, so a full M-tier
#     run is materially more expensive than a Reader-only run — scope first,
#     widen only if the subset clears the gate.
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_m.json \
#        --mode turns --tier belief-view \
#        --question-type knowledge-update \
#        --question-type knowledge-update_abs
#     Then score with the judge (step 4) and record in tests/baselines.md.
#
# 3c. REPLAY-LOOP A/B (recall-gap substrate, loom-dcfaf656) — does the multi-turn
#     aggregate detector lift multi-session / temporal recall? Run the SAME
#     subset twice and diff the judge scores: the only delta between the two is
#     the replay loop (enqueue + drain) running on top of identical per-turn
#     materialization. If --tier replay does not beat --tier belief-view on
#     these classes, the loop's dead-tell fired (route back to the E2 detector
#     prompt / E1 linkage). The replay claims ARE on the read path here — unlike
#     a default run, where the substrate is inert (see replay_drive.py).
#       # Baseline (no aggregate claims):
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_m.json \
#        --mode turns --tier belief-view \
#        --question-type multi-session --question-type temporal-reasoning
#       # Treatment (aggregate claims via the replay loop):
#    uv run python -m benchmarks.longmemeval.adapter \
#        --dataset ../LongMemEval/data/longmemeval_m.json \
#        --mode turns --tier replay --replay-executor inline \
#        --question-type multi-session --question-type temporal-reasoning
#     Swap --replay-executor batch for a production-exact (consolidate()) pass —
#     identical claims, Batch-API dispatch, ~300s/question polling. Watch the
#     run summary's "replay: agg_claims=N" line: N=0 across all questions means
#     the loop ran but found no enumerations (a real null, not an inert path).
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
