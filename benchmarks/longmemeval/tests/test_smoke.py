#!/usr/bin/env python3
"""
test_smoke.py — End-to-end smoke test for the LongMemEval adapter.

Builds a tiny synthetic LongMemEval-shaped dataset, runs both ingest modes
through the full pipeline against a real Postgres (via testcontainers — same
fixture the rest of Weft's tests use), and asserts the JSONL contract holds.

The Reader is stubbed out so this test does not require an Anthropic API key
or paid LLM calls. The real Reader is exercised separately when running the
benchmark for real.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-04-30
Python:  >= 3.12

Dependencies:
    pytest, pytest-asyncio, testcontainers (already in dev group)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from weft.embeddings import get_provider

from benchmarks.longmemeval.adapter import run_benchmark
from benchmarks.longmemeval.dataset import load_split
from benchmarks.longmemeval.ingest import project_id_for
from benchmarks.longmemeval.reader import Reader, ReaderResponse


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _tiny_dataset() -> list[dict]:
    """Two synthetic instances mimicking the LongMemEval schema."""
    return [
        {
            "question_id": "smoke-001",
            "question_type": "single-session-user",
            "question": "What is the user's favorite color?",
            "answer": "blue",
            "question_date": "2024/01/15",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/01/10"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "My favorite color is blue."},
                    {"role": "assistant", "content": "Got it!"},
                ]
            ],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "smoke-002",
            "question_type": "multi-session",
            "question": "What hobby has the user mentioned across sessions?",
            "answer": "rock climbing",
            "question_date": "2024/02/20",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/01/05", "2024/02/01"],
            "haystack_sessions": [
                [{"role": "user", "content": "I started rock climbing last week."}],
                [{"role": "user", "content": "Rock climbing is going great."}],
            ],
            "answer_session_ids": ["s1", "s2"],
        },
    ]


@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_lme.json"
    path.write_text(json.dumps(_tiny_dataset()), encoding="utf-8")
    return path


def _stub_reader(canned: str = "blue") -> Reader:
    """Reader whose .read_answer is a no-network AsyncMock returning ``canned``."""
    reader = Reader.__new__(Reader)  # bypass __init__ (no client needed)
    reader.read_answer = AsyncMock(  # type: ignore[method-assign]
        return_value=ReaderResponse(
            hypothesis=canned,
            model="stub",
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
        )
    )
    return reader


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_dataset_loader_parses_tiny_split(tiny_dataset_path: Path) -> None:
    instances = load_split(tiny_dataset_path)
    assert len(instances) == 2
    assert instances[0].question_id == "smoke-001"
    assert instances[0].sessions[0].has_answer is True
    assert instances[1].question_type == "multi-session"
    assert len(instances[1].sessions) == 2


@pytest.mark.asyncio
async def test_raw_mode_produces_one_jsonl_per_question(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    output = tmp_path / "out.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub-answer")

    stats = await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="raw",
        top_k=5,
        cleanup=True,
        pool=pool,
        embedder=embedder,
        reader=reader,
    )

    assert stats.questions_done == 2
    assert stats.questions_failed == 0
    assert stats.sessions_ingested == 3  # 1 + 2

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {row["question_id"] for row in parsed} == {"smoke-001", "smoke-002"}
    assert all(row["hypothesis"] == "stub-answer" for row in parsed)

    # Cleanup actually deleted memories — verify the projects are empty.
    for qid in ("smoke-001", "smoke-002"):
        n = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE project_id = $1",
            project_id_for(qid),
        )
        assert n == 0


@pytest.mark.asyncio
async def test_no_cleanup_leaves_memories(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    output = tmp_path / "out_keep.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub")

    await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="raw",
        top_k=5,
        cleanup=False,
        pool=pool,
        embedder=embedder,
        reader=reader,
    )

    n = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE project_id LIKE 'lme_%'",
    )
    assert n == 3  # one memory per session in raw mode


@pytest.mark.asyncio
async def test_question_type_filter_processes_only_matching(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    """--question-type filter restricts the run to matching instances only."""
    output = tmp_path / "filtered.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub")

    stats = await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="raw",
        top_k=5,
        cleanup=True,
        question_types=frozenset({"multi-session"}),
        pool=pool,
        embedder=embedder,
        reader=reader,
    )

    # Only the multi-session question (smoke-002) should be processed.
    assert stats.questions_total == 1
    assert stats.questions_done == 1

    parsed = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question_id"] for row in parsed] == ["smoke-002"]

    # Stats sidecar records the filter for traceability.
    stats_payload = json.loads(
        output.with_suffix(output.suffix + ".stats.json").read_text()
    )
    assert stats_payload["question_types"] == ["multi-session"]


@pytest.mark.asyncio
async def test_question_type_filter_with_no_matches_runs_zero(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    """An unknown question_type filter yields a zero-question run, not an error."""
    output = tmp_path / "empty.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub")

    stats = await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="raw",
        top_k=5,
        cleanup=True,
        question_types=frozenset({"does-not-exist"}),
        pool=pool,
        embedder=embedder,
        reader=reader,
    )
    assert stats.questions_total == 0
    assert stats.questions_done == 0
    assert not output.read_text().strip()


@pytest.mark.asyncio
async def test_warm_boost_diverges_turn_usefulness_scores(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    """P1.A5: --warm-boost-rounds N must move at least one turn's
    ``usefulness_score`` off the v46 ingest default of 0.7.

    Without warmup, every turn would sit at 0.7 forever and the P1.A3
    rerank would have no signal to work with (the whole reason the
    cold-DB harness can't measure Track A's compounding contribution).
    """
    output = tmp_path / "warm.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub")

    await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="turns",
        tier="turns",
        top_k=5,
        cleanup=False,
        warm_boost_rounds=2,
        warm_boost_queries_per_round=4,
        pool=pool,
        embedder=embedder,
        reader=reader,
    )

    boosted_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM episode_turns t
         JOIN episodes e ON e.id = t.episode_id
         WHERE e.project_id LIKE 'lme_%'
           AND t.last_boosted_at IS NOT NULL
        """
    )
    assert boosted_count > 0, (
        "warm-boost should have written last_boosted_at on at least "
        "one turn"
    )

    # Stats sidecar should carry the warm_boost aggregates (P1.A5
    # diagnostic surface for M-tier A/B comparison).
    stats_payload = json.loads(
        output.with_suffix(output.suffix + ".stats.json").read_text()
    )
    wb = stats_payload["warm_boost"]
    assert wb is not None
    assert wb["config_rounds"] == 2
    assert wb["config_queries_per_round"] == 4
    assert wb["total_rounds_ran"] >= 2
    assert wb["total_queries"] > 0
    assert wb["total_boosted_turns"] > 0
    assert wb["rerank_disabled"] is False

    # Spot-check that some turn moved off the 0.7 default.
    diverged = await pool.fetchval(
        """
        SELECT COUNT(*) FROM episode_turns t
         JOIN episodes e ON e.id = t.episode_id
         WHERE e.project_id LIKE 'lme_%'
           AND t.usefulness_score > 0.7
        """
    )
    assert diverged > 0, (
        "at least one turn's usefulness_score should be above the "
        "0.7 ingest default after warmup"
    )


@pytest.mark.asyncio
async def test_recall_finds_evidence_session_in_raw_mode(
    pool, tmp_path: Path, tiny_dataset_path: Path
) -> None:
    """Raw-mode ingest should make the evidence session retrievable.

    This is the structural readiness check: if hybrid recall can't surface
    the right session for a softball question, the benchmark will fail no
    matter how good the Reader is.
    """
    from weft.store import search_hybrid

    output = tmp_path / "recall_check.jsonl"
    embedder = get_provider("fastembed", dimensions=768)
    reader = _stub_reader("stub")

    await run_benchmark(
        dataset_path=tiny_dataset_path,
        output_path=output,
        mode="raw",
        top_k=5,
        cleanup=False,
        pool=pool,
        embedder=embedder,
        reader=reader,
        limit=1,  # only the favorite-color question
    )

    query = "What is the user's favorite color?"
    embedding = await embedder.embed(query)
    results = await search_hybrid(
        pool, query, embedding,
        limit=5, project_id=project_id_for("smoke-001"),
    )
    assert len(results) >= 1
    assert "blue" in results[0].memory.content.lower()


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# Requires Docker running (testcontainers spins up Postgres-pgvector + Redis).
#
# Run only the smoke tests:
#    uv run pytest benchmarks/longmemeval/tests/ -v
#
# Run a single test:
#    uv run pytest benchmarks/longmemeval/tests/test_smoke.py::test_recall_finds_evidence_session_in_raw_mode -v
#
# ═══════════════════════════════════════════════════════════════
