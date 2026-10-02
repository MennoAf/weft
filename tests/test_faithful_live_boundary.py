"""Offline regressions for the faithful live calibration boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.dataset import Session, Turn
from benchmarks.longmemeval.faithful_agent import AgentExecutionError, AgentPolicy, FaithfulAgent, _tool_definition
from benchmarks.longmemeval.faithful_budget import BudgetLedger
from benchmarks.longmemeval.faithful_gateway import FaithfulGateway, GatewayError
from benchmarks.longmemeval.faithful_s36 import _checkpoint, prepare_run, RunPaths
from tests.test_longmemeval_faithful_s36 import _write_fixture_dataset


class ResponsesShape:
    """Responses-shaped fake with one tool call followed by a terminal reply."""

    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments
        self.calls: list[dict[str, object]] = []
        self._round = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self._round += 1
        if self._round == 1:
            return SimpleNamespace(
                output_text="",
                output=[SimpleNamespace(
                    type="function_call",
                    call_id="remember-1",
                    name="weft_remember",
                    arguments=json.dumps(self.arguments),
                )],
                usage=SimpleNamespace(input_tokens=10, output_tokens=2, cached_input_tokens=0, reasoning_tokens=0),
            )
        return SimpleNamespace(
            output_text="ack",
            output=[],
            usage=SimpleNamespace(input_tokens=10, output_tokens=2, cached_input_tokens=0, reasoning_tokens=0),
        )


class RecordingGateway:
    """No-DB gateway double that records the exact post-boundary arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def call(self, name: str, arguments: dict[str, object]):
        self.calls.append({"name": name, "arguments": dict(arguments)})
        return {"id": "memory-1"}


def _session() -> Session:
    return Session("s1", "2023/04/19", (Turn("user", "Remember this public fact."),))


def test_model_scope_is_host_owned_and_null_defaults_are_injected(tmp_path: Path) -> None:
    """Omitted/null model scope gets fixed identity, while schemas expose no scope."""
    writer_schema = _tool_definition("weft_remember")["parameters"]
    assert "project_id" not in writer_schema["properties"]
    assert "agent_id" not in writer_schema["properties"]
    assert writer_schema["required"] == ["content"]

    client = ResponsesShape({"content": "durable fact", "project_id": None, "agent_id": None})
    gateway = RecordingGateway()
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "boundary"})
    agent = FaithfulAgent(client, ledger, tools=gateway, policy=AgentPolicy(max_tool_rounds=2))

    result = asyncio.run(agent.write_session(_session(), project_id="fixed-project", agent_id="faithful-s36"))

    assert result.text == "ack"
    assert gateway.calls == [{
        "name": "weft_remember",
        "arguments": {
            "content": "durable fact",
            "project_id": "fixed-project",
            "agent_id": "faithful-s36",
        },
    }]


def test_non_null_model_scope_override_still_reaches_gateway_and_is_rejected(tmp_path: Path) -> None:
    """Removing scope from schemas must not weaken immutable gateway enforcement."""
    client = ResponsesShape({
        "content": "must not write",
        "project_id": "attacker-project",
        "agent_id": "attacker-agent",
    })
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "boundary"})
    gateway = FaithfulGateway(
        app=None, ctx=None, owner_id="owner", project_id="fixed-project", agent_id="faithful-s36"
    )
    agent = FaithfulAgent(client, ledger, tools=gateway, policy=AgentPolicy(max_tool_rounds=2))

    with pytest.raises(GatewayError, match="cannot override"):
        asyncio.run(agent.write_session(_session(), project_id="fixed-project", agent_id="faithful-s36"))

    assert len(client.calls) == 1
    reservations = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["reservations"]
    assert reservations[0]["status"] == "completed"


def test_checkpoint_accepts_durable_failure_marker_without_replay(tmp_path: Path) -> None:
    """A failed tool attempt remains operator-visible as in-flight evidence."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint.update({
        "in_flight": "case-00",
        "in_flight_stage": "session:s-0-early",
        "last_error": "GatewayError: agent_id cannot override the faithful run scope",
        "evidence": {
            "case-00": {
                "question_id": "case-00",
                "status": "failed",
                "tool_results": [{
                    "name": "weft_remember",
                    "arguments": {"content": "durable fact", "agent_id": "attacker-agent"},
                    "error": "GatewayError: agent_id cannot override the faithful run scope",
                }],
            },
        },
    })
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")

    checked = _checkpoint(paths.checkpoint, [f"case-{index:02d}" for index in range(36)])

    assert checked["in_flight"] == "case-00"
    assert checked["evidence"]["case-00"]["status"] == "failed"
    assert checked["evidence"]["case-00"]["tool_results"][0]["error"].startswith("GatewayError:")
