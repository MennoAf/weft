"""Real local integration for one faithful multi-session vertical slice.

Only paid boundaries are faked. PostgreSQL/pgvector, public MCP memory tools,
contradiction checks, and cached FastEmbed are real.  The test is intentionally
one owner of one case scope and never uses a configured MCP server.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import asyncpg

import pytest

from benchmarks.longmemeval.faithful_slice import (
    AmbiguousSliceError,
    BudgetExceeded,
    QUESTION,
    REFERENCE_ANSWER,
    SliceLedger,
    SliceStore,
    StrictGateway,
    load_official_judge_prompt,
    run_slice,
)
from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.embeddings import get_provider
from weft.mcp.server import AppContext


class FakeResponses:
    """Responses-shaped fake; no HTTP or provider SDK is used."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("restart attempted a paid call")
        return self.responses.pop(0)


class FakeJudge:
    def __init__(self, text: str = "yes") -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.text = text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))]
        )


class PaidNetworkDenied:
    """Any accidental real paid client construction/HTTP use fails loudly."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("paid OpenAI/Anthropic client construction is forbidden")

    def request(self, *args, **kwargs):
        raise AssertionError("paid provider HTTP is forbidden")


def response(text: str = "ack", *, call: tuple[str, str, dict] | None = None, calls: list[tuple[str, str, dict]] | None = None) -> SimpleNamespace:
    output = []
    for name, call_id, args in ([call] if call else []) + (calls or []):
        output.append(SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=json.dumps(args)))
    return SimpleNamespace(
        output_text=text,
        output=output,
        usage=SimpleNamespace(input_tokens=20, output_tokens=8, cached_input_tokens=0, reasoning_tokens=0),
    )


@pytest.fixture
def real_app(pool):
    """Real pgvector pool + cached FastEmbed BAAI/bge-small padded to 768."""
    provider = get_provider("fastembed", model_name="BAAI/bge-small-en-v1.5", dimensions=768)
    assert provider.provider_name == "fastembed"
    assert provider.dimensions == 768
    config = WeftConfig()
    config.retrieval.recovery_mode = "off"
    return AppContext(pool=pool, cache=NullCache(), embedding=provider, config=config)


@pytest.mark.asyncio
async def test_synthetic_slice_real_public_tools_and_restart(real_app, pool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=PaidNetworkDenied, AsyncOpenAI=PaidNetworkDenied))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=PaidNetworkDenied, AsyncAnthropic=PaidNetworkDenied))
    owner = "faithful-slice-owner-a"
    project = "longmemeval-faithful-slice-a"
    token = current_user_id.set(owner)
    try:
        official_root = Path(os.environ["LONGMEMEVAL_PATH"])
        contradiction_calls: list[bool] = []
        import weft.consolidation as consolidation
        original_contradictions = consolidation.check_contradictions_on_store

        async def contradiction_spy(*args, **kwargs):
            contradiction_calls.append(True)
            return await original_contradictions(*args, **kwargs)

        monkeypatch.setattr(consolidation, "check_contradictions_on_store", contradiction_spy)
        paid = FakeResponses([
            response(calls=[
                ("weft_remember", "w1", {"content": "The user keeps a paper atlas in the study.", "type": "fact", "topic": ["atlas"]}),
                ("weft_remember", "w2", {"content": "The user plans a coastal train trip in October.", "type": "fact", "topic": ["travel"]}),
            ]),
            response("stored early"),
            response(calls=[
                ("weft_remember", "w3", {"content": "The user dislikes overnight flights.", "type": "fact", "topic": ["travel-preference"]}),
                ("weft_remember", "w4", {"content": "The user prefers daytime rail travel.", "type": "fact", "topic": ["travel-preference"]}),
            ]),
            response("stored late"),
            response(call=("weft_recall", "r1", {"query": "planned October trip and overnight flight constraint", "limit": 10})),
            response("The trip is a coastal train trip in October; overnight flights are disliked."),
        ])
        judge = FakeJudge("yes")
        state = await run_slice(app=real_app, root=tmp_path, owner_id=owner, project_id=project, client=paid, judge_client=judge, official_source_root=official_root)
        assert state["status"] == "complete"
        assert len(state["sessions"]) == 2
        assert len(state["sessions"][0]["tool_results"]) == 2
        assert state["answer"]["text"]
        assert state["answer_tool_results"]
        assert all("error" not in call["result"] for call in state["answer_tool_results"])
        assert "overnight flights" in json.dumps(state["answer_tool_results"], sort_keys=True)
        assert state["question"] == QUESTION
        assert state["reference_answer"] == REFERENCE_ANSWER
        assert state["hypothesis"] == state["answer"]["text"]
        assert state["judge"]["prompt"] == load_official_judge_prompt(official_root, question_type="multi-session", question=QUESTION, answer=REFERENCE_ANSWER, hypothesis=state["hypothesis"])
        assert state["judge"]["raw"] == "yes"  # protocol evidence only, not an accuracy claim
        assert contradiction_calls and all(contradiction_calls)
        assert state["ledger"]["calibration_reserved_usd"] <= 5.0
        assert state["ledger"]["total_reserved_usd"] <= 20.0
        assert len(paid.calls) == 6
        assert len(judge.calls) == 1
        assert judge.calls[0]["messages"][0]["content"] == state["judge"]["prompt"]

        rows = await pool.fetch("SELECT content, embedding, vector_dims(embedding) AS dimensions FROM memories WHERE project_id = $1 AND user_id = $2 ORDER BY created_at", project, owner)
        assert len(rows) >= 5  # four remembers plus the handoff memory
        assert all(row["embedding"] is not None and row["dimensions"] == 768 for row in rows)
        owner_gateway = StrictGateway(real_app, owner_id=owner, project_id=project)
        owner_recall = await owner_gateway.call("weft_recall", {"query": "overnight flights", "limit": 10})
        assert "error" not in owner_recall
        assert any("overnight flights" in item.get("content", "") for item in owner_recall.get("results", []))

        # A different owner must receive a non-error empty result, not an
        # infrastructure error; only returned evidence is used for isolation.
        other = StrictGateway(real_app, owner_id="faithful-slice-owner-b", project_id=project)
        hidden = await other.call("weft_recall", {"query": "paper atlas coastal train October", "limit": 10})
        assert "error" not in hidden
        rendered = json.dumps(hidden.get("results", []), sort_keys=True)
        assert "paper atlas" not in rendered
        assert "coastal train" not in rendered

        # Restart must not write DB rows or call either fake provider.
        await asyncio.gather(*tuple(getattr(real_app, "_background_tasks", ())), return_exceptions=True)
        before = await pool.fetchrow("SELECT (SELECT count(*) FROM memories) AS memories, (SELECT count(*) FROM memory_access_log) AS access, (SELECT count(*) FROM weft_recall_queries) AS recalls")
        restart_paid = FakeResponses([])
        restart_judge = FakeJudge()
        restarted = await run_slice(app=real_app, root=tmp_path, owner_id=owner, project_id=project, client=restart_paid, judge_client=restart_judge, official_source_root=official_root)
        after = await pool.fetchrow("SELECT (SELECT count(*) FROM memories) AS memories, (SELECT count(*) FROM memory_access_log) AS access, (SELECT count(*) FROM weft_recall_queries) AS recalls")
        assert restarted == state
        assert dict(before) == dict(after)
        assert restart_paid.calls == []
        assert restart_judge.calls == []
    finally:
        current_user_id.reset(token)
        # AppContext owns request-triggered background tasks; settle them before
        # the fixture closes the pool, preserving the one-owner teardown order.
        tasks = tuple(getattr(real_app, "_background_tasks", ()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_inflight_state_fails_closed_and_scope_override_is_rejected(real_app, tmp_path: Path):
    binding = {"owner_id": "owner", "project_id": "project", "case": "synthetic-faithful-slice-v1"}
    store = SliceStore(tmp_path / "case.json", binding=binding)
    store.save({"schema": "faithful-slice.v1", "status": "running", "in_flight": "answer"})
    with pytest.raises(Exception, match="in-flight"):
        store.load()
    with pytest.raises(Exception, match="binding mismatch"):
        SliceStore(tmp_path / "case.json", binding={**binding, "owner_id": "attacker"}).load()
    ledger = SliceLedger(tmp_path / "ledger.json", binding=binding)
    with pytest.raises(Exception, match="mismatched ledger"):
        SliceLedger(tmp_path / "ledger.json", binding={**binding, "project_id": "attacker"})
    gateway = StrictGateway(real_app, owner_id="owner", project_id="project")
    with pytest.raises(Exception, match="project_id override"):
        await gateway.call("weft_recall", {"query": "x", "project_id": "other"})


def test_same_ledger_enforces_calibration_and_total_caps(tmp_path: Path):
    binding = {"owner_id": "owner", "project_id": "project", "case": "synthetic-faithful-slice-v1"}
    ledger = SliceLedger(tmp_path / "ledger.json", binding=binding)
    for _ in range(2):
        ledger.reserve("gpt-4o", 1_000_000, 0, phase="calibration")
    with pytest.raises(BudgetExceeded, match=r"\$5"):
        ledger.reserve("gpt-4o", 1_000_000, 0, phase="calibration")
    # No request above either cap is admitted, even after switching phase.
    for _ in range(6):
        ledger.reserve("gpt-4o", 1_000_000, 0, phase="run")
    assert ledger.summary()["calibration_reserved_usd"] <= 5.0
    assert ledger.summary()["total_reserved_usd"] <= 20.0
    with pytest.raises(BudgetExceeded, match=r"\$20"):
        for _ in range(100):
            ledger.reserve("gpt-4o", 1_000_000, 0, phase="run")


def test_official_judge_prompt_adapter_and_fake_response_semantics():
    root = Path(os.environ["LONGMEMEVAL_PATH"])
    source = root / "src" / "evaluation" / "evaluate_qa.py"
    assert source.exists()
    hypothesis = "The trip is a coastal train trip in October; overnight flights are disliked."
    prompt = load_official_judge_prompt(root, question_type="multi-session", question=QUESTION, answer=REFERENCE_ANSWER, hypothesis=hypothesis)
    assert isinstance(prompt, str) and prompt.strip()
    assert QUESTION in prompt
    assert REFERENCE_ANSWER in prompt
    assert hypothesis in prompt
    fake = FakeJudge("yes")
    completion = fake.chat.completions.create(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": prompt}], n=1, temperature=0, max_tokens=10)
    assert completion.choices[0].message.content == "yes"
