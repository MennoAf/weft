"""Offline regressions for fail-closed Responses incomplete output handling."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.dataset import Session, Turn
from benchmarks.longmemeval.faithful_agent import (
    MAX_OUTPUT_TOKENS,
    AgentExecutionError,
    AgentPolicy,
    BoundedJudge,
    FaithfulAgent,
    FRESH_GPT6_MAX_OUTPUT_TOKENS,
    IncompleteResponseError,
)
from benchmarks.longmemeval.faithful_budget import BudgetLedger


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class RecordingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, arguments: dict[str, object]) -> dict[str, str]:
        self.calls.append((name, dict(arguments)))
        return {"id": "must-not-be-called"}


def _session() -> Session:
    return Session("s1", "2023/04/19", (Turn("user", "Remember this fact."),))


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=MAX_OUTPUT_TOKENS,
        cached_input_tokens=0,
        reasoning_tokens=0,
    )


def _agent(response: object, tmp_path: Path) -> tuple[FaithfulAgent, FakeResponses, RecordingGateway, Path]:
    client = FakeResponses(response)
    gateway = RecordingGateway()
    ledger_path = tmp_path / "ledger.json"
    ledger = BudgetLedger(ledger_path, binding={"run": "incomplete"})
    return (
        FaithfulAgent(client, ledger, tools=gateway, policy=AgentPolicy(max_tool_rounds=2)),
        client,
        gateway,
        ledger_path,
    )


def test_incomplete_status_precedes_partial_tool_json_and_never_calls_gateway(tmp_path: Path) -> None:
    partial = '{"content":"private partial payload that must not be persisted'
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="",
        output=[SimpleNamespace(
            type="function_call",
            call_id="call-1",
            name="weft_remember",
            arguments=partial,
        )],
        usage=_usage(),
    )
    agent, client, gateway, ledger_path = _agent(response, tmp_path)

    with pytest.raises(IncompleteResponseError) as caught:
        asyncio.run(agent.write_session(_session(), project_id="p", agent_id="a"))

    assert len(client.calls) == 1
    assert gateway.calls == []
    assert caught.value.metadata == {
        "response_status": "incomplete",
        "incomplete_reason": "max_output_tokens",
        "output_item_types": ["function_call"],
        "tool_calls": [{
            "call_id": "call-1",
            "name": "weft_remember",
            "arguments_sha256": (
                "6a022e7f7a179a6170bfb7840ba6dd376b961fe23652250d70dc79fe88dcc53e"
            ),
        }],
    }
    error_text = str(caught.value)
    assert partial not in error_text
    assert "private partial payload" not in error_text
    rows = json.loads(ledger_path.read_text(encoding="utf-8"))["reservations"]
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_incomplete_response_is_not_retried_and_cap_is_bounded(tmp_path: Path) -> None:
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details={"reason": "content_filter"},
        output_text="partial text",
        output=[],
        usage=_usage(),
    )
    agent, client, gateway, _ledger_path = _agent(response, tmp_path)

    with pytest.raises(IncompleteResponseError):
        asyncio.run(agent.answer(
            question="What is remembered?",
            question_date="2023/04/20",
            task_shape=None,
            recalled_context=None,
            project_id="p",
            agent_id="a",
        ))

    assert len(client.calls) == 1
    assert client.calls[0]["max_output_tokens"] == 512
    assert gateway.calls == []


def test_fresh_output_cap_is_2048_reserved_and_incomplete_tool_call_fails_closed(tmp_path: Path) -> None:
    partial = '{"content":"cut off before the tool arguments completed'
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="",
        output=[SimpleNamespace(
            type="function_call", call_id="call-fresh", name="weft_remember",
            arguments=partial,
        )],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=FRESH_GPT6_MAX_OUTPUT_TOKENS,
            cached_input_tokens=0, reasoning_tokens=0,
        ),
    )
    client = FakeResponses(response)
    gateway = RecordingGateway()
    ledger_path = tmp_path / "fresh-cap-ledger.json"
    ledger = BudgetLedger(ledger_path, binding={"run": "fresh-output-cap"})
    agent = FaithfulAgent(
        client, ledger, tools=gateway, model="gpt-6-luna",
        policy=AgentPolicy(max_output_tokens=FRESH_GPT6_MAX_OUTPUT_TOKENS,
                           allow_final_response_after_tool_rounds=True),
    )

    with pytest.raises(IncompleteResponseError):
        asyncio.run(agent.write_session(_session(), project_id="p", agent_id="a"))

    assert client.calls[0]["max_output_tokens"] == 2048
    assert len(client.calls) == 1
    assert gateway.calls == []
    reservation = json.loads(ledger_path.read_text(encoding="utf-8"))["reservations"][0]
    assert reservation["estimated_usd"] >= 2048 * 0.50 / 1_000_000
    assert reservation["status"] == "completed"


def test_incomplete_judge_binary_text_is_unknown_and_never_accepted(tmp_path: Path) -> None:
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="yes",
        output=[],
        usage=_usage(),
    )
    client = FakeResponses(response)
    ledger_path = tmp_path / "judge-ledger.json"
    ledger = BudgetLedger(ledger_path, binding={"run": "incomplete-judge"})
    judge = BoundedJudge(client, ledger)

    from benchmarks.longmemeval.faithful_agent import IncompleteResponseError
    with pytest.raises(IncompleteResponseError) as caught:
        asyncio.run(judge.judge("official judge prompt"))

    assert caught.value.metadata["response_status"] == "incomplete"
    assert len(client.calls) == 1
    rows = json.loads(ledger_path.read_text(encoding="utf-8"))["reservations"]
    assert len(rows) == 1
    assert rows[0]["status"] == "unknown"
    assert rows[0]["actual_usd"] is not None
    assert ledger.summary()["unknown_count"] == 1


def test_completed_response_malformed_arguments_remain_fail_closed_without_content(tmp_path: Path) -> None:
    partial = '{"content":"do not disclose this malformed content'
    response = SimpleNamespace(
        status="completed",
        output_text="",
        output=[SimpleNamespace(
            type="function_call",
            call_id="call-2",
            name="weft_remember",
            arguments=partial,
        )],
        usage=_usage(),
    )
    agent, _client, gateway, _ledger_path = _agent(response, tmp_path)

    with pytest.raises(AgentExecutionError) as caught:
        asyncio.run(agent.write_session(_session(), project_id="p", agent_id="a"))

    assert gateway.calls == []
    assert partial not in str(caught.value)
    assert "arguments_sha256=" in str(caught.value)
