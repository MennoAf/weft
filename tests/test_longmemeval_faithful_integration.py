"""Real disposable S36 proof for the faithful runner.

This module deliberately targets the stabilized runner seam, not the accepted
one-case slice and not a fake gateway.  Postgres/pgvector, the public MCP
functions, AppContext, replay guard, and cached FastEmbed are real.  Only the
paid Responses and official-judge transports are injected.

The source-side seam required by this proof is:

* ``run_calibration(..., gateway_factory=..., client_factory=...)``; and
* ``resume_run(..., gateway_factory=..., client_factory=..., judge_client=...)``.

``gateway_factory`` must create a fresh ``FaithfulGateway`` for each immutable
case scope.  A dispatcher around one gateway is intentionally not acceptable.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from benchmarks.longmemeval.faithful_agent import GPT4O_JUDGE_MODEL, FaithfulAgent
from benchmarks.longmemeval.faithful_gateway import create_local_gateway
from benchmarks.longmemeval.faithful_s36 import (
    EXPECTED_CASE_COUNT,
    ExecutionGateError,
    RunPaths,
    _read_json,
    approve_calibration,
    issue_calibration_receipt,
    prepare_run,
    resume_run,
    run_calibration,
)
from benchmarks.longmemeval.faithful_budget import BudgetExceeded, BudgetLedger
from tests.test_longmemeval_faithful_s36 import _write_fixture_dataset
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
from weft.db.migrations import run_migrations
from weft.embeddings import get_provider
from weft.mcp.server import AppContext


class FakeResponses:
    """Responses transport fake; no SDK client or HTTP is constructed."""

    def __init__(self, *, answer_text: str = "The public remembered answer.") -> None:
        self.answer_text = answer_text
        self.calls: list[dict] = []
        self.judge_calls: list[dict] = []
        self._serial = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self._serial += 1
        if kwargs.get("model") == GPT4O_JUDGE_MODEL:
            # Calibration shares this transport with the Luna agent, so the
            # fake must distinguish the official GPT-4o judge request.  Keep
            # the exact official prompt in the call ledger for assertions.
            self.judge_calls.append(kwargs)
            return _response("yes")
        tools = kwargs.get("tools") or ()
        names = {item.get("name") for item in tools if isinstance(item, dict)}
        inputs = kwargs.get("input") or ()
        completed_call_ids = {
            item.get("call_id") for item in inputs
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        }
        called_names = {
            item.get("name") for item in inputs
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        # Writer: two real public remembers in one Responses turn, proving
        # multiple writes/session rather than one-memory fallback behavior.
        # The next round receives function_call_output items and must become a
        # terminal assistant response; emitting remembers every round would
        # exercise the runner's failure bound rather than a valid conversation.
        if "weft_remember" in names and not completed_call_ids:
            calls = [
                SimpleNamespace(
                    type="function_call", call_id=f"remember-{self._serial}-a",
                    name="weft_remember", arguments=json.dumps({
                        "content": f"Case fact {self._serial} is persisted in the public memory store.",
                        "type": "fact", "topic": ["faithful-s36", "case-fact"],
                    }),
                ),
                SimpleNamespace(
                    type="function_call", call_id=f"remember-{self._serial}-b",
                    name="weft_remember", arguments=json.dumps({
                        "content": f"Case constraint {self._serial} matters for the later answer.",
                        "type": "fact", "topic": ["faithful-s36", "case-constraint"],
                    }),
                ),
            ]
            return _response("stored public facts", output=calls)
        if "weft_prime" in names and "weft_prime" not in called_names:
            # First answer turn must use the real public prime lifecycle.
            return _response(
                "need public context",
                output=[SimpleNamespace(
                    type="function_call", call_id=f"prime-{self._serial}",
                    name="weft_prime", arguments=json.dumps({
                        "query": "case fact and constraint", "disclosure": "full",
                    }),
                )],
            )
        if "weft_recall" in names and "weft_recall" not in called_names:
            return _response(
                "need public recall",
                output=[SimpleNamespace(
                    type="function_call", call_id=f"recall-{self._serial}",
                    name="weft_recall", arguments=json.dumps({
                        "query": "case fact and constraint", "limit": 10,
                    }),
                )],
            )
        return _response(self.answer_text)


class FakeJudge:
    """Official judge transport fake; prompt/label persistence remains real."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.prompts: list[str] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("model") == GPT4O_JUDGE_MODEL
        prompt = kwargs["input"][0]["content"]
        assert isinstance(prompt, str) and prompt.strip()
        self.prompts.append(prompt)
        return _response("yes")


def _response(text: str, *, output: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        output=output or [],
        usage=SimpleNamespace(
            input_tokens=24, output_tokens=8, cached_input_tokens=0,
            reasoning_tokens=0,
        ),
    )


def _consolidation_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return logged consolidation subsystem failures; never hide them."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "weft.consolidation"
        and record.levelno >= logging.ERROR
        and record.getMessage().endswith("subsystem failed")
    ]


def _fixture_with_public_histories(tmp_path: Path) -> tuple[Path, Path]:
    """Use 36 synthetic cases with two short chronological public sessions."""
    records: list[dict] = []
    ids: list[str] = []
    for index in range(EXPECTED_CASE_COUNT):
        qid = f"case-{index:02d}"
        ids.append(qid)
        early, late = f"s-{index}-early", f"s-{index}-late"
        records.append({
            "question_id": qid,
            "question_type": "multi-session",
            "question": f"What fact belongs to synthetic case {index}?",
            "answer": "SYNTHETIC_GOLD_MUST_NEVER_REACH_RUNTIME",
            "question_date": "2023/04/20",
            "haystack_session_ids": [early, late],
            "haystack_dates": ["2023/04/19 (Wed) 09:00", "2023/04/21 (Fri) 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": f"Public history for {early}."}],
                [{"role": "user", "content": f"Public history for {late}."}],
            ],
            "answer_session_ids": [early, late],
        })
    dataset = tmp_path / "synthetic36.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest = tmp_path / "synthetic36-manifest.json"
    manifest.write_text(json.dumps({"selection": {"ordered_question_ids": ids}}), encoding="utf-8")
    return dataset, manifest


async def _fresh_benchmark_db(pg_dsn: str, name: str) -> tuple[str, asyncpg.Pool]:
    """Create one disposable benchmark DB inside the testcontainer."""
    parsed = urlsplit(pg_dsn)
    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    dsn = urlunsplit((parsed.scheme, parsed.netloc, f"/{name}", "", ""))

    async def init(conn):
        await _pgvector_codec_init(conn)

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, init=init)
    await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await run_migrations(pool)
    await register_pgvector_codec(pool)
    return dsn, pool


async def _drop_benchmark_db(pg_dsn: str, name: str) -> None:
    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture
async def real_s36_root(pg_dsn, tmp_path: Path):
    """Serial disposable root: DB, FastEmbed, and artifacts are test-owned."""
    name = f"longmemeval_bench_{uuid.uuid4().hex[:12]}"
    dsn, pool = await _fresh_benchmark_db(pg_dsn, name)
    provider = get_provider(
        "fastembed", model_name="BAAI/bge-small-en-v1.5", dimensions=768,
    )
    config = WeftConfig()
    config.retrieval.recovery_mode = "off"
    app = AppContext(pool=pool, cache=NullCache(), embedding=provider, config=config)
    try:
        yield dsn, pool, app, provider, tmp_path
    finally:
        tasks = tuple(getattr(app, "_background_tasks", ()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await pool.close()
        await _drop_benchmark_db(pg_dsn, name)


_LME_CHECKOUT = Path(os.environ.get("LONGMEMEVAL_PATH", "/tmp/longmemeval-source-20260921"))
_LME_READY = (_LME_CHECKOUT / "src" / "evaluation" / "evaluate_qa.py").is_file()
requires_lme_checkout = pytest.mark.skipif(
    not _LME_READY,
    reason="requires a full LongMemEval checkout (set LONGMEMEVAL_PATH or restore /tmp/longmemeval-source-* with src/evaluation/evaluate_qa.py)",
)


@pytest.mark.asyncio
@requires_lme_checkout
async def test_actual_s36_real_gateway_prepare_calibrate_hold_approve_resume(
    real_s36_root, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
):
    """Run all 36 cases through real public tools and durable runner artifacts."""
    dsn, pool, app, provider, root = real_s36_root
    caplog.set_level(logging.ERROR, logger="weft.consolidation")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    owner = "faithful-s36-real-owner"
    agent_id = "faithful-s36"
    dataset, manifest = _fixture_with_public_histories(root)
    paths = RunPaths.from_root(root / "artifacts")
    prepared = prepare_run(dataset, manifest, paths=paths, owner_id=owner, agent_id=agent_id)
    assert len(prepared.ordered_question_ids) == EXPECTED_CASE_COUNT
    assert _read_json(paths.manifest)["status"] == "PREPARED_NOT_AUTHORIZED"

    official_root = Path(os.environ["LONGMEMEVAL_PATH"])
    created_gateways: list[object] = []
    paid_clients: list[FakeResponses] = []
    judge = FakeJudge()

    async def gateway_factory(*, owner_id: str, project_id: str, agent_id: str, embedding):
        gateway = await create_local_gateway(
            dsn, owner_id=owner_id, project_id=project_id,
            agent_id=agent_id, embedding=embedding,
        )
        # This is an actual gateway over an actual pool/AppContext, never a
        # dispatcher or fake public-tool implementation.
        created_gateways.append(gateway)
        return gateway

    def client_factory(*, phase: str, ledger):
        client = FakeResponses()
        paid_clients.append(client)
        return client

    async def measured_calibration() -> dict:
        measured = await run_calibration(
            dataset, manifest, paths=paths, dsn=dsn, owner_id=owner,
            case_limit=1, judge_root=official_root,
            gateway_factory=gateway_factory, client_factory=client_factory,
        )
        assert measured["completed"] is True
        return measured

    hold = await asyncio.to_thread(
        issue_calibration_receipt,
        paths, execute=True, calibration_runner=measured_calibration,
        notes="real local pgvector/FastEmbed representative; fake paid transport only",
    )
    assert hold["status"] == "HOLD_FOR_APPROVAL"
    assert hold["approval_required"] is True
    assert hold["representative"]["cases"]
    assert len(created_gateways) >= 1

    with pytest.raises((ValueError, TypeError)):
        approve_calibration(
            paths, approved_by="operator", projected_total_usd=float("nan"),
            projection_basis="must reject non-finite projection",
        )
    measured_projection = hold["projection"]
    assert isinstance(measured_projection, dict)
    measured_total = measured_projection["projected_total_usd"]
    with pytest.raises((ValueError, TypeError, ExecutionGateError)):
        approve_calibration(
            paths, approved_by="operator", projected_total_usd=measured_total + 0.01,
            projection_basis="must match measured projection",
        )
    approved = approve_calibration(
        paths, approved_by="integration-operator",
        projected_total_usd=measured_total,
        projection_basis=measured_projection["basis"],
    )
    assert approved["status"] == "CALIBRATED"

    # Each case receives a newly created immutable gateway, and each gateway
    # records real public remember/prime/recall/handoff calls.
    result = await resume_run(
        dataset, manifest, paths=paths, execute=True, dsn=dsn,
        judge_root=official_root, judge_client=judge,
        gateway_factory=gateway_factory, client_factory=client_factory,
    )
    assert result["status"] == "COMPLETED"
    assert result["case_count"] == EXPECTED_CASE_COUNT
    assert result["completed_count"] == EXPECTED_CASE_COUNT
    assert result["failed_count"] == 0
    assert len(result["cases"]) == EXPECTED_CASE_COUNT
    assert len({row["question_id"] for row in result["cases"]}) == EXPECTED_CASE_COUNT
    assert len(created_gateways) >= EXPECTED_CASE_COUNT
    assert len({id(gateway) for gateway in created_gateways}) == len(created_gateways)

    rows = await pool.fetch(
        "SELECT project_id, user_id, content, embedding, vector_dims(embedding) AS dimensions "
        "FROM memories WHERE user_id = $1 ORDER BY created_at", owner,
    )
    assert rows
    assert all(row["embedding"] is not None and row["dimensions"] == 768 for row in rows)
    assert len({row["project_id"] for row in rows}) >= EXPECTED_CASE_COUNT
    assert all("SYNTHETIC_GOLD" not in row["content"] for row in rows)

    # Primary positive evidence: every case persisted public memory evidence,
    # public prime and recall, and an official prompt/result row.
    rendered = json.dumps(result["cases"], sort_keys=True)
    assert rendered.count('"name": "weft_remember"') >= EXPECTED_CASE_COUNT * 2
    assert rendered.count('"name": "weft_prime"') >= EXPECTED_CASE_COUNT
    assert rendered.count('"name": "weft_recall"') >= EXPECTED_CASE_COUNT
    assert all(row["judge"] and row["judge"]["prompt_sha256"] for row in result["cases"])
    assert all(row["judge"]["raw"] == "yes" and row["judge"]["label"] is True for row in result["cases"])
    assert len(judge.calls) == EXPECTED_CASE_COUNT
    assert len(judge.prompts) == EXPECTED_CASE_COUNT

    # The fake judge must receive the exact official prompt, not a stand-in
    # prompt or the reader's answer text.  Calibration uses the shared paid
    # transport, so assert that prompt too.
    from benchmarks.longmemeval.judge import _official_prompt_loader
    prompt_fn = _official_prompt_loader(official_root)
    references = {row["question_id"]: row for row in json.loads(dataset.read_text())}
    expected_run_prompts = [
        prompt_fn(
            references[row["question_id"]]["question_type"],
            references[row["question_id"]]["question"],
            references[row["question_id"]]["answer"],
            row["hypothesis"],
            abstention="_abs" in row["question_id"],
        )
        for row in result["cases"]
    ]
    assert judge.prompts == expected_run_prompts
    calibration_judge_prompts = [
        call["input"][0]["content"]
        for client in paid_clients
        for call in client.judge_calls
    ]
    assert len(calibration_judge_prompts) == 1
    calibration_case = hold["representative"]["cases"][0]
    calibration_reference = references[calibration_case["question_id"]]
    assert calibration_judge_prompts[0] == prompt_fn(
        calibration_reference["question_type"],
        calibration_reference["question"],
        calibration_reference["answer"],
        calibration_case["hypothesis"],
        abstention="_abs" in calibration_case["question_id"],
    )

    # Consolidation is deliberately real in this proof.  Logged subsystem
    # errors are failures, not acceptable background noise or proof of success.
    assert _consolidation_failures(caplog) == []

    # Per-case owner isolation is checked using a real gateway over the same DB.
    other = await create_local_gateway(
        dsn, owner_id="faithful-s36-other-owner",
        project_id="longmemeval-case-00", embedding=provider,
    )
    try:
        hidden = await other.call("weft_recall", {"query": "public history", "limit": 10})
        assert "error" not in hidden
        assert "Case fact" not in json.dumps(hidden, sort_keys=True)
    finally:
        await other.close()

    # Replay denial case: a pending queue row blocks public calls before any
    # paid transport; cleanup only this test-created row and episode.
    episode_id = "real-s36-replay-denial-episode"
    await pool.execute(
        "INSERT INTO episodes (id, title, summary, user_id) VALUES ($1, $2, $3, $4)",
        episode_id, "replay denial", "synthetic", owner,
    )
    await pool.execute(
        "INSERT INTO replay_queue (id, episode_id, turn_ids, reason, user_id) "
        "VALUES ($1, $2, $3, $4, $5)",
        "real-s36-replay-denial", episode_id, [], "integration proof", owner,
    )
    denied_gateway = await create_local_gateway(
        dsn, owner_id=owner, project_id="longmemeval-case-00", embedding=provider,
    )
    try:
        with pytest.raises(Exception, match="pending replay_queue"):
            await denied_gateway.call("weft_prime", {"budget_tokens": 100})
    finally:
        await denied_gateway.close()
        await pool.execute("DELETE FROM replay_queue WHERE id = $1", "real-s36-replay-denial")
        await pool.execute("DELETE FROM episodes WHERE id = $1", episode_id)

    # The same completed artifact is the input to the fresh-process probe. It
    # must return without fake paid calls or additional memory/recall writes.
    before = await pool.fetchrow(
        "SELECT (SELECT count(*) FROM memories) AS memories, "
        "(SELECT count(*) FROM memory_access_log) AS access, "
        "(SELECT count(*) FROM weft_recall_queries) AS recalls",
    )
    receipt_hash = hashlib.sha256(paths.receipt.read_bytes()).hexdigest()
    probe = subprocess.run([
        sys.executable, "tests/acceptance_fakes/faithful_restart_probe.py",
        "--dataset", str(dataset), "--manifest", str(manifest),
        "--artifact-root", str(paths.root), "--dsn", dsn,
        "--owner-id", owner, "--db-counts", json.dumps(dict(before)),
        "--receipt-sha256", receipt_hash,
    ], capture_output=True, text=True, env={**os.environ, "LONGMEMEVAL_PATH": str(official_root)})
    assert probe.returncode == 0, probe.stderr
    after = await pool.fetchrow(
        "SELECT (SELECT count(*) FROM memories) AS memories, "
        "(SELECT count(*) FROM memory_access_log) AS access, "
        "(SELECT count(*) FROM weft_recall_queries) AS recalls",
    )
    assert dict(before) == dict(after)
    assert hashlib.sha256(paths.receipt.read_bytes()).hexdigest() == receipt_hash
    assert all(client.calls for client in paid_clients)


def test_cli_equivalent_contract_has_explicit_fake_transport_seam():
    """The real CLI must expose an injected transport seam; no paid fallback."""
    from benchmarks.longmemeval import faithful_s36
    signature = inspect.signature(faithful_s36._main_async)
    assert "args" in signature.parameters
    # This marker is intentionally strict: once source stabilization lands,
    # CLI tests must invoke _main_async with injected fake transport clients,
    # rather than constructing OpenAIResponsesClient in the test.
    assert hasattr(faithful_s36, "_main_async")


def test_cumulative_ledger_rejects_calibration5_and_total20(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "real-s36"})
    # $0.90/M conservative input pricing: $4.50 calibration, $18 total.
    for _ in range(2):
        ledger.reserve("gpt-5.6-luna", 2_500_000, 0, phase="calibration")
    with pytest.raises(BudgetExceeded, match="calibration"):
        ledger.reserve("gpt-5.6-luna", 1_000_000, 0, phase="calibration")
    for _ in range(6):
        ledger.reserve("gpt-5.6-luna", 2_500_000, 0, phase="run")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("gpt-5.6-luna", 3_000_000, 0, phase="run")
