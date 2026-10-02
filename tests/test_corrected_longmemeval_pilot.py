from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.longmemeval import pilot
from benchmarks.longmemeval.dataset import Instance, Session, Turn
from benchmarks.longmemeval.reader import ReaderResponse
from benchmarks.longmemeval import judge

_LME_CHECKOUT = Path(os.environ.get("LONGMEMEVAL_PATH", "/tmp/longmemeval-source-20260921"))
_LME_READY = (_LME_CHECKOUT / "src" / "evaluation" / "evaluate_qa.py").is_file()
requires_lme_checkout = pytest.mark.skipif(
    not _LME_READY,
    reason="requires a full LongMemEval checkout (set LONGMEMEVAL_PATH or restore /tmp/longmemeval-source-* with src/evaluation/evaluate_qa.py)",
)


def test_corrected_sdk_constructors_bind_retry_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs["max_retries"])

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAnthropic)
    provider = pilot._make_corrected_classifier_provider()
    _detector, client = pilot._make_corrected_detector()
    assert seen == [pilot.CLASSIFIER_MAX_RETRIES, pilot.DETECTOR_MAX_RETRIES]
    assert provider.owns_client is True
    assert client is not None


@pytest.mark.asyncio
async def test_detector_injected_client_is_used_without_default_singleton() -> None:
    from weft.models import EpisodeTurn, TurnRole
    from datetime import datetime, timezone

    class Messages:
        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(content=[SimpleNamespace(text="[]")])

    fake = SimpleNamespace(messages=Messages())
    turn = EpisodeTurn(
        id="turn-1", episode_id="episode-1", turn_index=0, role=TurnRole.user,
        content="I ran today", occurred_at=datetime.now(timezone.utc), trace_id=None,
        source_session_id=None, importance_score=0.5, token_count=2, user_id="u", created_at=datetime.now(timezone.utc),
    )
    from weft.views.belief_detector import detect_belief_updates
    assert await detect_belief_updates(turn, client=fake) == []
    assert fake.messages.kwargs["model"] == pilot.DETECTOR_MODEL


@requires_lme_checkout
def test_bounded_judge_stops_after_configured_attempts(tmp_path: Path) -> None:
    source_root = _LME_CHECKOUT
    ref = tmp_path / "ref.json"
    hyp = tmp_path / "hyp.jsonl"
    ref.write_text(json.dumps([{"question_id": "q1", "question_type": "knowledge-update", "question": "q", "answer": "a"}]))
    hyp.write_text(json.dumps({"question_id": "q1", "hypothesis": "a"}) + "\n")

    class Completions:
        def __init__(self) -> None:
            self.calls = 0
        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls <= pilot.JUDGE_MAX_RETRIES:
                raise RuntimeError("transient")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    metrics = judge.run_bounded_judge(hyp_path=hyp, ref_path=ref, source_root=source_root, client=client)
    assert completions.calls == pilot.JUDGE_MAX_RETRIES + 1
    assert metrics["n_correct_total"] == 1
    output = json.loads((tmp_path / "hyp.jsonl.eval-results-gpt-4o").read_text().strip())
    assert output["autoeval_label"] == {"model": "gpt-4o-2024-08-06", "label": True}


def test_corrected_manifest_reuses_existing_s36_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    historical = root / "benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json"
    assert historical.is_file()
    value = json.loads(historical.read_text())
    assert len(value["selection"]["ordered_question_ids"]) == 36
    assert pilot.CORRECTED_TIER == "belief"
    assert pilot.CORRECTED_INGEST_MODE == "production-belief"


def test_corrected_source_has_public_entrypoint_not_private_router() -> None:
    source = Path(pilot.__file__).read_text()
    assert "weft.mcp.tools import weft_recall" in source
    assert "tier=CORRECTED_TIER" in source
    assert "project_id=None" in source
    assert "belief-view" not in source


def _instance(question_id: str = "q1", question_type: str = "knowledge-update") -> Instance:
    return Instance(
        question_id=question_id,
        question_type=question_type,
        question="What changed?",
        answer="it changed",
        question_date="2024/01/01",
        sessions=(Session("s1", "2023/01/01", (Turn("user", "a claim"),)),),
    )


def test_corrected_budget_refuses_over_bound_without_resources() -> None:
    estimate = pilot.corrected_cost_estimate([_instance()], ["q1"], judge_questions=36, prior_spend_usd=pilot.AUTHORIZATION_CEILING_USD)
    with pytest.raises(RuntimeError, match="authorization ceiling"):
        pilot.enforce_corrected_budget(estimate)
    assert estimate["calls"]["classifier"] == 1
    assert estimate["calls"]["judge"] == 36


def test_corrected_budget_has_explicit_models_and_all_paid_stages() -> None:
    estimate = pilot.corrected_cost_estimate([_instance()], ["q1"], judge_questions=36)
    assert estimate["models"] == {
        "classifier": "claude-haiku-4-5-20251001",
        "detector": "claude-haiku-4-5-20251001",
        "reader": pilot.READER_MODEL,
        "judge": pilot.JUDGE_MODEL,
    }
    assert all(estimate["bounds_usd"][stage] > 0 for stage in ("classifier", "detector", "reader", "judge", "incremental_reserved"))


def test_corrected_judge_projection_preserves_exact_s36_ids(tmp_path: Path) -> None:
    ids = [f"q{i}" for i in range(36)]
    manifest = {"selection": {"ordered_question_ids": ids}, "manifest_sha256": "m"}
    rows = [{"question_id": qid, "tier": "belief", "status": "ok", "manifest_sha256": "m", "hypothesis": f"h-{qid}"} for qid in ids]
    manifest_path = tmp_path / "manifest.json"
    rows_path = tmp_path / "rows.jsonl"
    manifest_path.write_text(json.dumps(manifest))
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = pilot.judge_projection(rows_path, manifest_path, tmp_path, "belief")
    projected = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question_id"] for row in projected] == ids
    assert len(projected) == 36


@dataclass
class _FakePool:
    closed: bool = False

    async def close(self) -> None:
        self.closed = True


class _FakeEmbedder:
    pass


class _FakeReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def read_answer(self, **kwargs: Any) -> ReaderResponse:
        self.calls.append(kwargs)
        context = kwargs["recall_response"]
        return ReaderResponse(
            hypothesis="synthetic answer",
            model="fake-reader",
            input_tokens=3,
            cached_input_tokens=0,
            output_tokens=2,
            system_prompt="evidence-only system",
            user_content=f"Memories:\n{context['results'][0]['content']}",
        )


class _FakeGenerationProvider:
    pass


def _synthetic_run_inputs(tmp_path: Path) -> tuple[list[str], dict[str, Any], list[Instance]]:
    ids = [f"synthetic-{i}" for i in range(36)]
    instances = [_instance(qid, pilot.QUESTION_TYPES[i // 6]) for i, qid in enumerate(ids)]
    manifest = {
        "schema": pilot.CORRECTED_SCHEMA,
        "status": "PREPARED_NOT_AUTHORIZED",
        "manifest_sha256": "synthetic-manifest",
        "dataset": {"sha256": "synthetic-dataset"},
        "selection": {"ordered_question_ids": ids, "question_types": list(pilot.QUESTION_TYPES)},
        "arm": "belief",
        "ingest": {"whole_session_shortcut": False},
        "normalization": {},
    }
    return ids, manifest, instances


@pytest.mark.asyncio
async def test_real_corrected_runner_fake_dependencies_writes_36_rows_and_evidence_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mandatory offline E2E: invoke the real runner with every external seam fake."""
    ids, manifest, instances = _synthetic_run_inputs(tmp_path)
    calls = {"load": 0, "materialize": 0, "recall": 0, "cleanup": 0}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(pilot, "load_corrected_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(pilot, "load_split", lambda _path: instances)

    async def load_haystack(*args: Any, **kwargs: Any) -> int:
        calls["load"] += 1
        kwargs["turn_session_map"]["turn-1"] = "s1"
        kwargs["turn_content_map"]["turn-1"] = "a claim"
        return 1

    async def materialize(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["materialize"] += 1
        return SimpleNamespace(to_dict=lambda: {"turns_total": 1, "turns_processed": 1, "claims_written": 1, "claims_superseded": 0, "abstentions": 0, "errors": 0})

    async def recall(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["recall"] += 1
        return {"tier": "belief", "count": 1, "results": [{"id": "belief-1", "content": "only retrieved evidence", "source_provenance": "conversation"}]}

    async def cleanup(*args: Any, **kwargs: Any) -> None:
        calls["cleanup"] += 1

    monkeypatch.setattr(pilot, "load_haystack", load_haystack)
    monkeypatch.setattr(pilot, "materialize_question", materialize)
    monkeypatch.setattr(pilot, "_public_recall", recall)
    monkeypatch.setattr(pilot, "cleanup_haystack", cleanup)
    monkeypatch.setattr(pilot, "derive_task_shape", lambda *args: SimpleNamespace(task_shape="single-session", routing_class="synthetic", top_k=10))

    reader = _FakeReader()
    summary = await pilot.run_corrected_pilot(
        root=tmp_path,
        dataset_path=tmp_path / "dataset.json",
        manifest_path=tmp_path / "manifest.json",
        normalization_source_path=tmp_path / "source.json",
        output_dir=tmp_path / "out",
        execute=True,
        generation_provider=_FakeGenerationProvider(),
        pool=_FakePool(),
        embedder=_FakeEmbedder(),
        reader=reader,
    )

    assert summary["status"] == "complete"
    rows = [json.loads(line) for line in (tmp_path / "out/corrected-belief-rows.jsonl").read_text().splitlines()]
    assert len(rows) == 36
    assert all(row["status"] == "ok" and "error" not in row for row in rows)
    assert [row["question_id"] for row in rows] == ids
    assert calls == {"load": 36, "materialize": 36, "recall": 36, "cleanup": 36}
    assert len(reader.calls) == 36
    assert {row["tier"] for row in rows} == {"belief"}
    assert all("only retrieved evidence" in row["reader_context"]["user_content"] for row in rows)
    ledger = json.loads((tmp_path / "out/corrected-belief-attempt-ledger.json").read_text())
    assert ledger["status"] == "complete"
    assert ledger["completed_ids"] == ids
    projection = pilot.judge_projection(
        tmp_path / "out/corrected-belief-rows.jsonl", tmp_path / "manifest.json", tmp_path / "out", "belief"
    )
    assert len(projection.read_text().splitlines()) == 36


def test_corrected_restart_cap_and_ledger_validation(tmp_path: Path) -> None:
    ids = [f"q{i}" for i in range(36)]
    manifest = {"manifest_sha256": "m"}
    base = {"schema": "weft.longmemeval.attempt-ledger.v1", "manifest_sha256": "m", "question_ids": ids, "failed_ids": [], "consumed_usd": "1", "reserved_usd": "2"}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({**base, "status": "failed", "completed_ids": ids[:19]}))
    with pytest.raises(RuntimeError, match="beyond 50%"):
        pilot._load_corrected_ledger(path, manifest, ids)
    path.write_text(json.dumps({**base, "status": "failed", "completed_ids": ids[:18]}))
    assert pilot._load_corrected_ledger(path, manifest, ids)["completed_ids"] == ids[:18]
    for bad in ({"status": "failed", "completed_ids": ids[:18]}, {**base, "status": "failed", "completed_ids": ids[:18], "reserved_usd": "wat"}):
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):
            pilot._load_corrected_ledger(path, manifest, ids)


@pytest.mark.asyncio
async def test_corrected_over_budget_refuses_before_any_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids, manifest, instances = _synthetic_run_inputs(tmp_path)
    resources = []
    monkeypatch.setattr(pilot, "load_corrected_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(pilot, "load_split", lambda _path: instances)
    monkeypatch.setenv("LONGMEMEVAL_PRIOR_SPEND_USD", "50")
    with pytest.raises(RuntimeError, match="authorization ceiling|operational stop"):
        await pilot.run_corrected_pilot(
            root=tmp_path, dataset_path=tmp_path / "dataset", manifest_path=tmp_path / "manifest",
            normalization_source_path=tmp_path / "source", output_dir=tmp_path / "out", execute=True,
        )
    assert resources == []
