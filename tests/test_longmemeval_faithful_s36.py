#!/usr/bin/env python3
"""Offline contract tests for the faithful S36 runner.

These tests deliberately avoid provider, database, embedding, and container
execution.  They exercise the approved seams and keep safety assertions strict.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.dataset import Session, Turn
from benchmarks.longmemeval.full_s_profile import FULL_S_CASE_COUNT, FULL_S_PROFILE, prepare_full_s_manifest
from benchmarks.longmemeval.faithful_agent import (
    AgentExecutionError,
    AgentPolicy,
    BoundedJudge,
    FaithfulAgent,
    NoAnthropicExecutionError,
    OpenAIResponsesClient,
    assert_no_anthropic_execution,
)
from benchmarks.longmemeval.faithful_budget import (
    BudgetError, BudgetLedger, FreshRunPricing, LedgerBindingError,
    TotalBudgetExceeded, canonical_hash,
)
from benchmarks.longmemeval.faithful_gateway import FaithfulGateway, GatewayError
from benchmarks.longmemeval.faithful_s36 import (
    EXPECTED_CASE_COUNT,
    ExecutionGateError,
    FaithfulRunError,
    INFLIGHT_RECOVERY_JOURNAL_SCHEMA,
    RunPaths,
    _checkpoint,
    _manifest_checkpoint_state,
    _sort_sessions,
    _validate_full_s_source_hashes,
    approve_calibration,
    issue_calibration_receipt,
    main as faithful_main,
    prepare_run,
    GPT6_SELECTED35_EXCLUDED_ID,
    GPT6_SELECTED35_ARTIFACT_NAMESPACE,
    GPT6_FULL_S_ARTIFACT_NAMESPACE,
    FRESH_RUN_PROFILE,
    GPT6_LUNA_MODEL,
    SAFE_SKIP_CASE_ID,
    _agent_policy_for_profile,
    _profile_from_binding,
    recover_inflight_case,
    recover_safe_skip,
    resume_run,
    run_calibration,
)
from benchmarks.longmemeval.task_shape import TaskShape


class FakeResponses:
    """Responses-shaped fake returning deterministic text and usage."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeGateway:
    """Pure in-memory public tool gateway."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "weft_recall":
            return {"results": [{"content": "A public remembered fact."}]}
        return {"id": f"memory-{len(self.calls)}", "content": arguments.get("content", "")}


def _response(text: str = "done", output: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        output=output or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cached_input_tokens=0, reasoning_tokens=0),
    )


def _session(day: str, sid: str, content: str | None = None) -> Session:
    return Session(sid, day, (Turn("user", content or f"Fact from {sid}"),))


def _write_fixture_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a 36-case fixture with multiple public sessions per case."""
    records = []
    ids = []
    for index in range(EXPECTED_CASE_COUNT):
        qid = f"case-{index:02d}"
        early = f"s-{index}-early"
        late = f"s-{index}-late"
        ids.append(qid)
        records.append({
            "question_id": qid,
            "question_type": "multi-session",
            "question": f"What fact belongs to case {index}?",
            "answer": "SECRET_GOLD_MUST_NEVER_REACH_RUNTIME",
            "question_date": "2023/04/20",
            "haystack_session_ids": [early, late],
            "haystack_dates": ["2023/04/19 (Wed) 09:00", "2023/04/21 (Fri) 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": f"Fact from {early}"}],
                [{"role": "user", "content": f"Fact from {late}"}],
            ],
            "answer_session_ids": [early, late],
        })
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selection": {"ordered_question_ids": ids}}), encoding="utf-8")
    return dataset, manifest


def _write_full_s_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Build a 500-row source with one duplicate session in the first row."""
    source_rows = []
    ids = [f"full-s-{index:03d}" for index in range(FULL_S_CASE_COUNT)]
    for index, question_id in enumerate(ids):
        session_ids = [f"session-{index}-first", f"session-{index}-second"]
        dates = ["2024/01/01", "2024/01/02"]
        sessions = [
            [{"role": "user", "content": f"first-version-{question_id}"}],
            [{"role": "user", "content": f"second-{question_id}"}],
        ]
        if index == 0:
            session_ids.append(session_ids[0])
            dates.append("2024/01/03")
            sessions.append([{"role": "user", "content": "duplicate-version-must-drop"}])
        source_rows.append({
            "question_id": question_id,
            "question_type": "multi-session",
            "question": f"Question {question_id}?",
            "answer": "gold stays out of runtime",
            "question_date": "2024/01/04",
            "haystack_session_ids": session_ids,
            "haystack_dates": dates,
            "haystack_sessions": sessions,
            "answer_session_ids": [session_ids[0]],
            "custom_metadata": {"row": index},
        })
    source = tmp_path / "cleaned-synthetic.json"
    normalized = tmp_path / "normalized-synthetic.json"
    manifest = tmp_path / "full-s-manifest.json"
    source.write_text(json.dumps(source_rows), encoding="utf-8")
    prepared = prepare_full_s_manifest(
        source, normalized, manifest,
        # Pin a real tracked runtime file so full-S execution paths validate
        # the pinned hash against the repository working tree.
        source_hash_paths=("benchmarks/longmemeval/full_s_profile.py",),
    )
    assert prepared["profile"] == FULL_S_PROFILE
    return normalized, manifest, ids


def _write_fake_judge_root(tmp_path: Path) -> Path:
    root = tmp_path / "official-judge"
    source = root / "src" / "evaluation" / "evaluate_qa.py"
    source.parent.mkdir(parents=True)
    source.write_text("# offline judge fixture\n", encoding="utf-8")
    return root


def _approve_measured_calibration(paths: RunPaths) -> dict:
    """Create the measured HOLD receipt and explicitly approve it."""

    async def runner() -> dict:
        binding = json.loads(paths.manifest.read_text(encoding="utf-8"))["binding"]
        ledger = {
            "max_budget_usd": 20.0,
            "calibration_budget_usd": 5.0,
            "reserved_usd": 0.001,
            "calibration_reserved_usd": 0.001,
            "remaining_usd": 19.999,
            "attempt_count": 1,
            "unknown_count": 0,
            "completed_count": 1,
            "binding": binding,
        }
        return {
            "completed": True,
            "case_count": 1,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "measured_usd": 0.001,
            "ledger": ledger,
            "projection": {
                "calibration_cases": 1,
                "measured_reserved_usd": ledger["reserved_usd"],
                "measured_calibration_reserved_usd": ledger["calibration_reserved_usd"],
                "projected_total_usd": 0.036,
                "basis": "same cumulative ledger and measured per-case provider reservations",
                "ledger_sha256": canonical_hash(ledger),
            },
        }

    hold = issue_calibration_receipt(
        paths,
        approved_by="operator-request-is-not-authorization",
        execute=True,
        calibration_runner=runner,
    )
    assert hold["status"] == "HOLD_FOR_APPROVAL"
    assert hold["approval_required"] is True
    assert hold["representative"]["usage"]["input_tokens"] == 100
    projection = hold["projection"]
    assert isinstance(projection, dict)
    assert projection["projected_total_usd"] == pytest.approx(0.036)
    with pytest.raises((ValueError, TypeError, ExecutionGateError)):
        approve_calibration(
            paths,
            approved_by="test-operator",
            projected_total_usd=1.24,
            projection_basis="must match measured projection",
        )
    return approve_calibration(
        paths,
        approved_by="test-operator",
        projected_total_usd=projection["projected_total_usd"],
        projection_basis=projection["basis"],
    )


def test_no_anthropic_guard_rejects_injected_client() -> None:
    """The runtime guard rejects an Anthropic-looking client."""
    AnthropicFake = type("AnthropicFake", (), {})
    AnthropicFake.__module__ = "anthropic.client"
    with pytest.raises(NoAnthropicExecutionError):
        assert_no_anthropic_execution(AnthropicFake())


def test_writer_allows_multiple_memories_across_chronological_sessions(tmp_path: Path) -> None:
    """A run may persist more than one memory; there is no one-memory/session gate."""
    calls = [
        [SimpleNamespace(type="function_call", call_id="c1", name="weft_remember", arguments=json.dumps({"content": "Fact one", "project_id": "p", "agent_id": "a"}))],
        [],
        [SimpleNamespace(type="function_call", call_id="c2", name="weft_remember", arguments=json.dumps({"content": "Fact two", "project_id": "p", "agent_id": "a"}))],
        [],
    ]
    client = FakeResponses([_response(output=calls[0]), _response("ack-1"), _response(output=calls[2]), _response("ack-2")])
    gateway = FakeGateway()
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "x"})
    agent = FaithfulAgent(client, ledger, tools=gateway, policy=AgentPolicy(max_tool_rounds=2))

    first = asyncio.run(agent.write_session(_session("2023/04/19", "early"), project_id="p", agent_id="a"))
    second = asyncio.run(agent.write_session(_session("2023/04/21", "late"), project_id="p", agent_id="a"))

    assert first.text == "ack-1"
    assert second.text == "ack-2"
    assert [name for name, _ in gateway.calls] == ["weft_remember", "weft_remember"]
    assert [args["content"] for _, args in gateway.calls] == ["Fact one", "Fact two"]


def test_agent_policy_output_cap_is_fresh_only() -> None:
    fresh = _agent_policy_for_profile(FRESH_RUN_PROFILE)
    legacy = _agent_policy_for_profile("legacy")

    assert fresh.max_output_tokens == 2048
    assert fresh.max_tool_rounds == 3
    assert fresh.allow_final_response_after_tool_rounds is True
    assert legacy.max_output_tokens == 512
    assert legacy.max_tool_rounds == 3
    assert legacy.allow_final_response_after_tool_rounds is False


def test_fresh_agent_allows_three_tool_rounds_then_final_text(tmp_path: Path) -> None:
    tool_call = lambda index: [SimpleNamespace(
        type="function_call", call_id=f"c{index}", name="weft_remember",
        arguments=json.dumps({"content": f"Fact {index}"}),
    )]
    client = FakeResponses([
        _response(output=tool_call(1)),
        _response(output=tool_call(2)),
        _response(output=tool_call(3)),
        _response("final acknowledgement"),
    ])
    gateway = FakeGateway()
    ledger = BudgetLedger(tmp_path / "fresh-ledger.json", binding={"run": "fresh-loop"})
    agent = FaithfulAgent(
        client, ledger, tools=gateway, model="gpt-6-luna",
        policy=AgentPolicy(allow_final_response_after_tool_rounds=True),
    )

    result = asyncio.run(agent.write_session(_session("2023/04/19", "ultrachat_423307"), project_id="p", agent_id="a"))

    assert result.text == "final acknowledgement"
    assert result.calls == 4
    assert result.tool_calls == 3
    assert len(result.reservations) == 4
    assert len(client.calls) == 4
    assert all(client.calls[index]["tools"] for index in range(3))
    assert client.calls[3]["tools"] == []
    assert "This is the final response" in client.calls[3]["instructions"]
    assert len(gateway.calls) == 3
    assert ledger.summary()["attempt_count"] == 4
    assert ledger.summary()["unknown_count"] == 0


def test_fresh_agent_rejects_tools_from_final_turn_before_side_effects(tmp_path: Path) -> None:
    tool_call = lambda index: [SimpleNamespace(
        type="function_call", call_id=f"c{index}", name="weft_remember",
        arguments=json.dumps({"content": f"Fact {index}"}),
    )]
    client = FakeResponses([_response(output=tool_call(index)) for index in range(1, 5)])
    gateway = FakeGateway()
    ledger = BudgetLedger(tmp_path / "fresh-ledger.json", binding={"run": "fresh-loop-cap"})
    agent = FaithfulAgent(
        client, ledger, tools=gateway, model="gpt-6-luna",
        policy=AgentPolicy(allow_final_response_after_tool_rounds=True),
    )

    with pytest.raises(AgentExecutionError, match="final turn requested tools"):
        asyncio.run(agent.write_session(_session("2023/04/19", "ultrachat_423307"), project_id="p", agent_id="a"))

    assert len(client.calls) == 4
    assert all(client.calls[index]["tools"] for index in range(3))
    assert client.calls[3]["tools"] == []
    assert "Do not call tools" in client.calls[3]["instructions"]
    assert len(gateway.calls) == 3
    rows = json.loads((tmp_path / "fresh-ledger.json").read_text(encoding="utf-8"))["reservations"]
    assert len(rows) == 4
    assert all(row["status"] == "completed" for row in rows)
    assert ledger.summary()["attempt_count"] == 4


def test_legacy_agent_still_fails_after_three_tool_rounds(tmp_path: Path) -> None:
    tool_call = [SimpleNamespace(
        type="function_call", call_id="c", name="weft_remember", arguments="{}",
    )]
    client = FakeResponses([_response(output=tool_call) for _ in range(4)])
    gateway = FakeGateway()
    ledger = BudgetLedger(tmp_path / "legacy-ledger.json", binding={"run": "legacy-loop"})
    agent = FaithfulAgent(client, ledger, tools=gateway, model="gpt-5.6-luna")

    with pytest.raises(AgentExecutionError, match="exceeded bounded rounds"):
        asyncio.run(agent.write_session(_session("2023/04/19", "legacy"), project_id="p", agent_id="a"))

    assert len(client.calls) == 3
    assert len(gateway.calls) == 3
    assert ledger.summary()["attempt_count"] == 3


def test_agent_fails_closed_when_tool_loop_exceeds_bound(tmp_path: Path) -> None:
    """Repeated tool requests cannot create an unbounded loop."""
    call = [SimpleNamespace(type="function_call", call_id="c", name="weft_remember", arguments="{}")]
    client = FakeResponses([_response(output=call), _response(output=call)])
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "x"})
    agent = FaithfulAgent(client, ledger, tools=FakeGateway(), policy=AgentPolicy(max_tool_rounds=2))
    with pytest.raises(AgentExecutionError, match="exceeded bounded rounds"):
        asyncio.run(agent.write_session(_session("2023/04/19", "s1"), project_id="p", agent_id="a"))


def test_openai_adapter_sets_zero_retries_on_sdk_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry policy is an SDK constructor option, never a Responses request field."""
    instances: list[object] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.responses = FakeResponses([_response("ok")])
            instances.append(self)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    client = OpenAIResponsesClient()
    result = asyncio.run(client.create(model="gpt-5.6-luna", input=[], max_output_tokens=8))

    assert result.output_text == "ok"
    assert instances[0].constructor_kwargs["max_retries"] == 0
    assert instances[0].constructor_kwargs["timeout"] > 0
    request = instances[0].responses.calls[0]
    assert "max_retries" not in request
    assert request["max_output_tokens"] == 8


def test_invalid_provider_usage_is_recorded_unknown_and_not_replayed(tmp_path: Path) -> None:
    client = FakeResponses([SimpleNamespace(
        output_text="partial", output=[],
        usage=SimpleNamespace(input_tokens=-1, output_tokens=1),
    )])
    ledger = BudgetLedger(
        tmp_path / "invalid-usage-ledger.json", binding={"run": "invalid-usage"},
    )
    agent = FaithfulAgent(client, ledger)
    from benchmarks.longmemeval.faithful_agent import AmbiguousExecutionError
    with pytest.raises(AmbiguousExecutionError, match="usage was invalid"):
        asyncio.run(agent.answer(
            question="question", question_date="2024-01-01", task_shape=None,
            recalled_context=None, project_id="p", agent_id="a",
        ))
    summary = ledger.summary()
    assert summary["unknown_count"] == 1
    assert summary["reserved_usd"] > 0


def test_responses_usage_preserves_nested_cached_tokens_and_unknown_write_count() -> None:
    from benchmarks.longmemeval.faithful_agent import _usage_from_response

    usage = _usage_from_response(SimpleNamespace(usage=SimpleNamespace(
        input_tokens=20, output_tokens=5,
        input_tokens_details=SimpleNamespace(cached_tokens=7),
    )))
    assert usage is not None
    assert usage.as_dict() == {
        "input_tokens": 20, "output_tokens": 5,
        "cached_input_tokens": 7, "cache_write_input_tokens": None,
        "reasoning_tokens": 0,
    }


def test_bounded_judge_uses_gpt4o_once_without_request_retry_keyword(tmp_path: Path) -> None:
    """The judge is injectable, bounded, and does not retry malformed calls."""
    client = FakeResponses([_response("yes")])
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "x"})
    judge = BoundedJudge(client, ledger)
    label, raw, reservation = asyncio.run(judge.judge("judge prompt"))
    assert label is True
    assert raw == "yes"
    assert reservation
    assert client.calls[0]["model"] == "gpt-4o"
    assert "max_retries" not in client.calls[0]


def test_answer_prompt_is_gold_blind_and_uses_only_public_recall(tmp_path: Path) -> None:
    """Question answering receives public recall, never answer labels or gold text."""
    client = FakeResponses([_response("public answer")])
    ledger = BudgetLedger(tmp_path / "ledger.json", binding={"run": "x"})
    agent = FaithfulAgent(client, ledger, tools=None)
    asyncio.run(agent.answer(
        question="Which city did I visit?",
        question_date="2023/04/20",
        task_shape=TaskShape("single-session", "belief", 10),
        recalled_context='{"results":[{"content":"PUBLIC_CITY"}]}',
        project_id="p",
        agent_id="a",
    ))
    payload = json.dumps(client.calls[0], sort_keys=True)
    assert "Which city did I visit?" in payload
    assert "PUBLIC_CITY" in payload
    assert "SECRET_GOLD_MUST_NEVER_REACH_RUNTIME" not in payload
    assert "answer_session_ids" not in payload
    assert "question_type" not in payload


def test_gateway_rejects_malicious_scope_overrides() -> None:
    """The public gateway cannot be redirected to another project, agent, or workspace."""
    gateway = FaithfulGateway(app=None, ctx=None, owner_id="owner", project_id="expected-project", agent_id="faithful-s36")
    attacks = [
        ("weft_recall", {"query": "x", "project_id": "attacker-project"}),
        ("weft_recall", {"query": "x", "agent_id": "attacker-agent"}),
        ("weft_recall", {"query": "x", "user_id": "attacker-owner"}),
        ("weft_remember", {"content": "x", "workspace_id": "attacker-workspace"}),
    ]
    for name, arguments in attacks:
        with pytest.raises(GatewayError):
            asyncio.run(gateway.call(name, arguments))


def test_prepare_binds_exact_36_cases_and_multiple_sessions_without_gold(tmp_path: Path) -> None:
    """Preparation is offline, hash-bound, and records all public sessions only."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepared = prepare_run(dataset, manifest, paths=paths)
    assert len(prepared.ordered_question_ids) == EXPECTED_CASE_COUNT
    document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert document["case_count"] == EXPECTED_CASE_COUNT
    assert all(len(row["session_ids"]) == 2 for row in document["sessions"])
    assert checkpoint["in_flight"] is None
    assert json.dumps(document) .find("SECRET_GOLD_MUST_NEVER_REACH_RUNTIME") == -1
    assert json.loads(paths.ledger.read_text(encoding="utf-8"))["binding"] == document["binding"]


def test_full_s_source_hash_validation_refuses_tampered_pinned_file(tmp_path: Path) -> None:
    """Runtime recompute mirrors pilot.py: clean tree passes; tampering refuses."""
    pinned = tmp_path / "benchmarks" / "longmemeval" / "full_s_profile.py"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"full-s pinned source v1\n")
    manifest = {
        "profile": FULL_S_PROFILE,
        "source_hashes": {
            "benchmarks/longmemeval/full_s_profile.py": hashlib.sha256(pinned.read_bytes()).hexdigest(),
        },
    }

    _validate_full_s_source_hashes(manifest, root=tmp_path)

    pinned.write_bytes(b"full-s pinned source v2 (tampered)\n")
    with pytest.raises(LedgerBindingError, match="full-S pinned source hash mismatch"):
        _validate_full_s_source_hashes(manifest, root=tmp_path)


def test_fresh_profile_prepares_bound_selected35_without_touching_legacy_root(tmp_path: Path) -> None:
    dataset, manifest = _write_fixture_dataset(tmp_path)
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    records = json.loads(dataset.read_text(encoding="utf-8"))
    canonical_ids = [
        GPT6_SELECTED35_EXCLUDED_ID if index == 0 else f"canonical-{index:02d}"
        for index in range(EXPECTED_CASE_COUNT)
    ]
    for index, row in enumerate(records):
        row["question_id"] = canonical_ids[index]
    manifest_doc["selection"]["ordered_question_ids"] = canonical_ids
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest.write_text(json.dumps(manifest_doc), encoding="utf-8")
    fresh_paths = RunPaths.from_root(tmp_path / "fresh-run")
    prepared = prepare_run(
        dataset, manifest, paths=fresh_paths, owner_id="fresh-owner",
        profile=FRESH_RUN_PROFILE, writer_model="gpt-6-luna", max_budget_usd=50.0,
    )
    document = json.loads(fresh_paths.manifest.read_text(encoding="utf-8"))
    binding = document["binding"]
    assert len(prepared.ordered_question_ids) == EXPECTED_CASE_COUNT - 1
    assert GPT6_SELECTED35_EXCLUDED_ID not in prepared.ordered_question_ids
    assert binding["run_profile"] == FRESH_RUN_PROFILE
    assert binding["writer_model"] == "gpt-6-luna"
    assert binding["max_budget_usd"] == "50.0"
    assert json.loads(binding["pricing_json"])["gpt6_input_usd_per_million"] == 0.10
    assert json.loads(binding["tool_round_policy_json"]) == {
        "max_tool_rounds": 3,
        "allow_final_response_after_tool_rounds": True,
        "max_responses": 4,
        "max_output_tokens": 2048,
        "final_response_tools": False,
        "final_response_instruction": "This is the final response. Do not call tools; finish with a concise text response or acknowledgement.",
    }
    mutated_binding = dict(binding)
    mutated_binding["tool_round_policy_json"] = json.dumps({
        "max_tool_rounds": 4,
        "allow_final_response_after_tool_rounds": True,
        "max_responses": 5,
    }, sort_keys=True, separators=(",", ":"))
    with pytest.raises(LedgerBindingError, match="tool-round policy"):
        _profile_from_binding(mutated_binding)
    mutated_cap_binding = dict(binding)
    mutated_cap_policy = json.loads(binding["tool_round_policy_json"])
    mutated_cap_policy["max_output_tokens"] = 512
    mutated_cap_binding["tool_round_policy_json"] = json.dumps(
        mutated_cap_policy, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(LedgerBindingError, match="tool-round policy"):
        _profile_from_binding(mutated_cap_binding)
    assert json.loads(fresh_paths.ledger.read_text(encoding="utf-8"))["max_budget_usd"] == 50.0
    assert RunPaths.from_root().root != fresh_paths.root


def test_legacy_profile_keeps_default_binding_and_budget(tmp_path: Path) -> None:
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "legacy-run")
    prepared = prepare_run(dataset, manifest, paths=paths)
    document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert len(prepared.ordered_question_ids) == EXPECTED_CASE_COUNT
    assert "run_profile" not in document["binding"]
    assert "writer_model" not in document["binding"]
    assert "tool_round_policy_json" not in document["binding"]
    assert json.loads(paths.ledger.read_text(encoding="utf-8"))["max_budget_usd"] == 20.0


def test_fresh_profile_rejects_model_cap_or_selection_override(tmp_path: Path) -> None:
    dataset, manifest = _write_fixture_dataset(tmp_path)
    with pytest.raises(Exception, match=r"GPT-6 Luna and the \$50 cap"):
        prepare_run(
            dataset, manifest, paths=RunPaths.from_root(tmp_path / "bad-model"),
            profile=FRESH_RUN_PROFILE, writer_model="gpt-5.6-luna", max_budget_usd=50.0,
        )
    with pytest.raises(Exception, match="fixed to selected-35"):
        prepare_run(
            dataset, manifest, paths=RunPaths.from_root(tmp_path / "bad-selection"),
            profile=FRESH_RUN_PROFILE, writer_model="gpt-6-luna", max_budget_usd=50.0,
            exclude_question_ids=["case-01"],
        )


def test_bare_float_calibration_projection_is_rejected(tmp_path: Path) -> None:
    """Approval requires the finite mapping emitted by measured calibration."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)

    async def runner() -> dict:
        return {
            "completed": True,
            "cases": [{"question_id": "case-00"}],
            "projection": 1.25,
        }

    hold = issue_calibration_receipt(paths, execute=True, calibration_runner=runner)
    assert hold["status"] == "HOLD_FOR_APPROVAL"
    with pytest.raises(ExecutionGateError, match="projection"):
        approve_calibration(
            paths, approved_by="operator", projected_total_usd=1.25,
            projection_basis="bare float must not authorize approval",
        )


def test_non_finite_approval_projection_is_rejected(tmp_path: Path) -> None:
    """Approval rejects NaN even when the measured hold is otherwise valid."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)

    async def runner() -> dict:
        binding = json.loads(paths.manifest.read_text(encoding="utf-8"))["binding"]
        ledger = {
            "max_budget_usd": 20.0,
            "calibration_budget_usd": 5.0,
            "reserved_usd": 0.001,
            "calibration_reserved_usd": 0.001,
            "remaining_usd": 19.999,
            "attempt_count": 1,
            "unknown_count": 0,
            "completed_count": 1,
            "binding": binding,
        }
        return {
            "completed": True,
            "ledger": ledger,
            "projection": {
                "calibration_cases": 1,
                "measured_reserved_usd": 0.001,
                "measured_calibration_reserved_usd": 0.001,
                "projected_total_usd": 0.036,
                "basis": "same cumulative ledger and measured per-case provider reservations",
                "ledger_sha256": canonical_hash(ledger),
            },
        }

    issue_calibration_receipt(paths, execute=True, calibration_runner=runner)
    with pytest.raises(ValueError, match="finite"):
        approve_calibration(
            paths,
            approved_by="operator",
            projected_total_usd=float("nan"),
            projection_basis="must reject non-finite projection",
        )


def test_calibration_requires_execute_measured_runner_then_explicit_approval(tmp_path: Path) -> None:
    """A name alone is not authorization; measured HOLD must be separately approved."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)
    with pytest.raises(ExecutionGateError, match="explicit --execute"):
        issue_calibration_receipt(paths, approved_by="operator")
    with pytest.raises(ExecutionGateError, match="real calibration gateway/provider"):
        issue_calibration_receipt(paths, approved_by="operator", execute=True)

    receipt = _approve_measured_calibration(paths)
    assert receipt["status"] == "CALIBRATED"
    assert receipt["approval_required"] is False
    assert receipt["approved_by"] == "test-operator"
    assert receipt["projected_total_usd"] == pytest.approx(0.036)


def test_resume_requires_explicit_execute_approved_calibration_and_local_dsn(tmp_path: Path) -> None:
    """Resume rejects prepare-only, HOLD, and non-disposable DSN attempts."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    judge_root = _write_fake_judge_root(tmp_path)
    prepare_run(dataset, manifest, paths=paths, judge_root=judge_root)
    with pytest.raises(ExecutionGateError, match="prepare-only"):
        asyncio.run(resume_run(dataset, manifest, paths=paths, execute=False, dsn=None))

    async def runner() -> dict:
        binding = json.loads(paths.manifest.read_text(encoding="utf-8"))["binding"]
        ledger = {
            "max_budget_usd": 20.0,
            "calibration_budget_usd": 5.0,
            "reserved_usd": 0.001,
            "calibration_reserved_usd": 0.001,
            "remaining_usd": 19.999,
            "attempt_count": 1,
            "unknown_count": 0,
            "completed_count": 1,
            "binding": binding,
        }
        return {
            "completed": True,
            "projection": {
                "calibration_cases": 1,
                "measured_reserved_usd": ledger["reserved_usd"],
                "measured_calibration_reserved_usd": ledger["calibration_reserved_usd"],
                "projected_total_usd": 0.036,
                "basis": "same cumulative ledger and measured per-case provider reservations",
                "ledger_sha256": canonical_hash(ledger),
            },
            "ledger": ledger,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    issue_calibration_receipt(paths, execute=True, calibration_runner=runner)
    with pytest.raises(ExecutionGateError, match="still HOLD_FOR_APPROVAL"):
        asyncio.run(resume_run(
            dataset, manifest, paths=paths, execute=True,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            judge_root=judge_root,
        ))
    approve_calibration(
        paths,
        approved_by="operator",
        projected_total_usd=0.036,
        projection_basis="same cumulative ledger and measured per-case provider reservations",
    )
    with pytest.raises(GatewayError, match="disposable|production|ordinary"):
        asyncio.run(resume_run(
            dataset, manifest, paths=paths, execute=True,
            dsn="postgresql://prod@localhost:5432/production",
            judge_root=judge_root,
        ))


def test_resume_api_requires_judge_root_before_lock_or_factories(tmp_path: Path) -> None:
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)
    _approve_measured_calibration(paths)
    factory_calls: list[str] = []

    async def gateway_factory(**kwargs):
        factory_calls.append("gateway")
        raise AssertionError("gateway must not be constructed")

    def client_factory(**kwargs):
        factory_calls.append("client")
        raise AssertionError("provider client must not be constructed")

    with pytest.raises(ExecutionGateError, match="judge_root is required"):
        asyncio.run(resume_run(
            dataset, manifest, paths=paths, execute=True,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            gateway_factory=gateway_factory, client_factory=client_factory,
        ))

    assert not paths.lock.exists()
    assert factory_calls == []
    assert not paths.receipt.exists()


def test_interrupted_in_flight_checkpoint_is_not_replayed(tmp_path: Path) -> None:
    """An interrupted case remains operator-owned and cannot be auto-replayed."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    judge_root = _write_fake_judge_root(tmp_path)
    prepare_run(dataset, manifest, paths=paths, judge_root=judge_root)
    _approve_measured_calibration(paths)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["in_flight"] = "case-00"
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    # A valid binding must reach the explicit interrupted-write safety gate.
    with pytest.raises(ExecutionGateError, match="unknown in-flight case"):
        asyncio.run(resume_run(
            dataset, manifest, paths=paths, execute=True,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            judge_root=judge_root,
        ))


def test_checkpoint_validator_accepts_operator_visible_in_flight_marker(tmp_path: Path) -> None:
    """The checkpoint schema preserves an interrupted case for inspection."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    prepare_run(dataset, manifest, paths=paths)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["in_flight"] = "case-00"
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    checked = _checkpoint(paths.checkpoint, [f"case-{index:02d}" for index in range(EXPECTED_CASE_COUNT)])
    assert checked["in_flight"] == "case-00"
    assert checked["completed_question_ids"] == []
    assert checked["failed_question_ids"] == []


def _recovery_fixture(tmp_path: Path) -> tuple[RunPaths, BudgetLedger, str, dict]:
    """Build isolated run artifacts in the exact safe-skip precondition state."""
    dataset, manifest_path = _write_fixture_dataset(tmp_path)
    records = json.loads(dataset.read_text(encoding="utf-8"))
    records[0]["question_id"] = SAFE_SKIP_CASE_ID
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_doc["selection"]["ordered_question_ids"][0] = SAFE_SKIP_CASE_ID
    manifest_path.write_text(json.dumps(manifest_doc), encoding="utf-8")
    paths = RunPaths.from_root(tmp_path / "safe-skip-artifacts")
    prepare_run(dataset, manifest_path, paths=paths, owner_id="offline-test")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    binding = manifest["binding"]
    profile, _, maximum, _, _, pricing, _ = _profile_from_binding(binding)
    assert profile == "legacy"
    ledger = BudgetLedger(paths.ledger, max_budget_usd=maximum, pricing=pricing, binding=binding)
    reservations = [
        ledger.reserve("gpt-5.6-luna", 100, 10, phase="run")
        for _ in range(11)
    ]
    for reservation in reservations[:-1]:
        ledger.finalize(
            reservation.reservation_id,
            usage={"input_tokens": 1, "output_tokens": 1},
        )
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["completed_question_ids"] = [f"case-{index:02d}" for index in range(1, 12)]
    checkpoint["in_flight"] = SAFE_SKIP_CASE_ID
    checkpoint["in_flight_stage"] = "session:ultrachat_236018"
    checkpoint["sessions"] = {
        SAFE_SKIP_CASE_ID: {
            f"session-{index}": "completed" for index in range(6)
        }
    }
    checkpoint["evidence"] = {
        SAFE_SKIP_CASE_ID: {
            "question_id": SAFE_SKIP_CASE_ID,
            "status": "failed",
            "error": None,
            "sessions": [{"session_id": f"session-{index}"} for index in range(6)],
        }
    }
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    return paths, ledger, reservations[-1].reservation_id, checkpoint


def test_safe_skip_recovery_updates_only_target_and_is_idempotent(tmp_path: Path) -> None:
    paths, ledger, reservation_id, original = _recovery_fixture(tmp_path)
    original_rows = json.loads(paths.ledger.read_text(encoding="utf-8"))["reservations"]
    result = recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id)
    repeated = recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id)

    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    rows = json.loads(paths.ledger.read_text(encoding="utf-8"))["reservations"]
    target = next(row for row in rows if row["reservation_id"] == reservation_id)
    assert result["changed"] is True
    assert repeated["changed"] is False
    assert checkpoint["completed_question_ids"] == original["completed_question_ids"]
    assert checkpoint["failed_question_ids"] == [SAFE_SKIP_CASE_ID]
    assert checkpoint["in_flight"] is None
    assert checkpoint["in_flight_stage"] is None
    assert checkpoint["evidence"][SAFE_SKIP_CASE_ID]["status"] == "failed"
    assert target["status"] == "unknown"
    assert target["actual_usd"] is None
    assert target["estimated_usd"] > 0
    assert ledger.summary()["reserved_usd"] >= target["estimated_usd"]
    assert rows[:-1] == original_rows[:-1]


def test_safe_skip_recovery_rejects_case_and_reservation_mismatches(tmp_path: Path) -> None:
    paths, ledger, reservation_id, original = _recovery_fixture(tmp_path)
    with pytest.raises(FaithfulRunError, match="only permits case"):
        recover_safe_skip(paths, case_id="case-00", reservation_id=reservation_id)
    with pytest.raises(BudgetError, match="reservation ID must identify exactly one"):
        recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id="not-the-reservation")
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert checkpoint == original
    assert ledger.summary()["unknown_count"] == 0
    assert not paths.checkpoint.with_suffix(paths.checkpoint.suffix + ".safe-skip.json").exists()


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("in_flight", None),
        ("completed", None),
        ("failed", "ProviderError: partial failure"),
    ],
)
def test_safe_skip_recovery_rejects_other_evidence_states(
    tmp_path: Path, status: str, error: str | None,
) -> None:
    paths, ledger, reservation_id, _ = _recovery_fixture(tmp_path)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["evidence"][SAFE_SKIP_CASE_ID]["status"] = status
    checkpoint["evidence"][SAFE_SKIP_CASE_ID]["error"] = error
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(FaithfulRunError, match="must be failed with no case error"):
        recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id)

    current = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert current == checkpoint
    assert ledger.summary()["unknown_count"] == 0
    assert not paths.checkpoint.with_suffix(paths.checkpoint.suffix + ".safe-skip.json").exists()


def _general_recovery_fixture(
    tmp_path: Path,
    *,
    case_id: str = "gpt4_9a159967",
    prior_failed_ids: tuple[str, ...] = (SAFE_SKIP_CASE_ID,),
    completed_count: int = 21,
    in_flight_stage: str = "session:ultrachat_134591",
    prefinalized_timeout: bool = False,
) -> tuple[RunPaths, BudgetLedger, str, dict]:
    """Build a binding-valid selected-35 run in one generic recovery state."""
    if case_id in prior_failed_ids:
        raise ValueError("target case must not also be a prior failure")
    dataset, manifest_path = _write_fixture_dataset(tmp_path)
    records = json.loads(dataset.read_text(encoding="utf-8"))
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [f"case-{index:02d}" for index in range(EXPECTED_CASE_COUNT)]
    ids[0] = case_id
    ids[1 : 1 + len(prior_failed_ids)] = prior_failed_ids
    ids[1 + len(prior_failed_ids)] = GPT6_SELECTED35_EXCLUDED_ID
    for record, question_id in zip(records, ids, strict=True):
        record["question_id"] = question_id
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest_doc["selection"]["ordered_question_ids"] = ids
    manifest_path.write_text(json.dumps(manifest_doc), encoding="utf-8")

    paths = RunPaths.from_root(tmp_path / "generic-recovery-artifacts")
    prepare_run(
        dataset, manifest_path, paths=paths, owner_id="offline-test",
        profile=FRESH_RUN_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=50.0,
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    binding = manifest["binding"]
    profile, _, maximum, _, _, pricing, _ = _profile_from_binding(binding)
    assert profile == FRESH_RUN_PROFILE
    ledger = BudgetLedger(paths.ledger, max_budget_usd=maximum, pricing=pricing, binding=binding)
    completed = [
        selected_id for selected_id in manifest["ordered_question_ids"]
        if selected_id not in {case_id, *prior_failed_ids}
    ][:completed_count]
    reservations = [
        ledger.reserve(GPT6_LUNA_MODEL, 100, 10, phase="run")
        for _ in range(10)
    ]
    for reservation in reservations:
        ledger.finalize(
            reservation.reservation_id,
            usage={"input_tokens": 1, "output_tokens": 1},
        )
    outstanding = ledger.reserve(GPT6_LUNA_MODEL, 24_278_512, 0, phase="run")
    if prefinalized_timeout:
        ledger.finalize(
            outstanding.reservation_id,
            error="Request timed out.",
            unknown=True,
        )

    selected_rows = {row["question_id"]: row for row in manifest["sessions"]}
    target_sessions = selected_rows[case_id]["session_ids"]
    prior_sessions = {
        failed_id: selected_rows[failed_id]["session_ids"]
        for failed_id in prior_failed_ids
    }
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["completed_question_ids"] = completed
    checkpoint["failed_question_ids"] = list(prior_failed_ids)
    checkpoint["in_flight"] = case_id
    checkpoint["in_flight_stage"] = in_flight_stage
    checkpoint["sessions"] = {
        case_id: {session_id: "completed" for session_id in target_sessions},
        **{
            failed_id: {session_id: "completed" for session_id in session_ids}
            for failed_id, session_ids in prior_sessions.items()
        },
    }
    checkpoint["evidence"] = {
        **{
            failed_id: {"question_id": failed_id, "status": "failed"}
            for failed_id in prior_failed_ids
        },
        case_id: {
            "question_id": case_id,
            "status": "in_flight" if prefinalized_timeout else "failed",
            "error": (
                "AmbiguousExecutionError: provider outcome is unknown; "
                "reservation retained and replay is forbidden"
                if prefinalized_timeout else None
            ),
            "sessions": [{"session_id": session_id} for session_id in target_sessions],
        },
    }
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    return paths, ledger, outstanding.reservation_id, checkpoint


def test_generic_inflight_recovery_handles_prior_failure_and_is_idempotent(tmp_path: Path) -> None:
    paths, ledger, reservation_id, original = _general_recovery_fixture(tmp_path)
    before = json.loads(paths.ledger.read_text(encoding="utf-8"))
    result = recover_inflight_case(
        paths, case_id="gpt4_9a159967", reservation_id=reservation_id,
        expected_completed_count=21,
    )
    repeated = recover_inflight_case(
        paths, case_id="gpt4_9a159967", reservation_id=reservation_id,
        expected_completed_count=21,
    )
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    after = json.loads(paths.ledger.read_text(encoding="utf-8"))
    target = next(row for row in after["reservations"] if row["reservation_id"] == reservation_id)

    assert result["changed"] is True
    assert repeated["changed"] is False
    assert checkpoint["completed_question_ids"] == original["completed_question_ids"]
    assert checkpoint["failed_question_ids"] == ["95228167", "gpt4_9a159967"]
    assert checkpoint["in_flight"] is None
    assert checkpoint["in_flight_stage"] is None
    assert checkpoint["evidence"]["gpt4_9a159967"]["status"] == "failed"
    assert checkpoint["evidence"]["95228167"] == {"question_id": "95228167", "status": "failed"}
    assert target["status"] == "unknown"
    assert target["estimated_usd"] == before["reservations"][-1]["estimated_usd"]
    assert target["estimated_usd"] == pytest.approx(6.069628)
    assert target["actual_usd"] is None
    assert after["reservations"][:-1] == before["reservations"][:-1]


def test_generic_inflight_recovery_accepts_prefinalized_timeout_without_ledger_write(tmp_path: Path) -> None:
    case_id = "6222b6eb"
    prior_failed_ids = ("95228167", "gpt4_9a159967")
    paths, ledger, reservation_id, original = _general_recovery_fixture(
        tmp_path,
        case_id=case_id,
        prior_failed_ids=prior_failed_ids,
        completed_count=30,
        in_flight_stage="session:0254514e_2",
        prefinalized_timeout=True,
    )
    ledger_before = paths.ledger.read_bytes()
    reservation_before = ledger.inspect_reservation(reservation_id)

    result = recover_inflight_case(
        paths,
        case_id=case_id,
        reservation_id=reservation_id,
        expected_completed_count=30,
    )
    recovered_checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    repeated = recover_inflight_case(
        paths,
        case_id=case_id,
        reservation_id=reservation_id,
        expected_completed_count=30,
    )

    assert result["changed"] is True
    assert repeated["changed"] is False
    assert recovered_checkpoint["completed_question_ids"] == original["completed_question_ids"]
    assert recovered_checkpoint["failed_question_ids"] == sorted([*prior_failed_ids, case_id])
    assert recovered_checkpoint["in_flight"] is None
    assert recovered_checkpoint["in_flight_stage"] is None
    assert recovered_checkpoint["evidence"][case_id]["status"] == "failed"
    assert recovered_checkpoint["evidence"][case_id]["error"] == (
        "AmbiguousExecutionError: provider outcome is unknown; "
        "reservation retained and replay is forbidden"
    )
    assert recovered_checkpoint["evidence"][case_id]["safe_skip_recovery"]["ledger_already_unknown"] is True
    assert paths.ledger.read_bytes() == ledger_before
    assert ledger.inspect_reservation(reservation_id) == reservation_before
    assert reservation_before["status"] == "unknown"
    assert reservation_before["error"] == "Request timed out."
    assert reservation_before["estimated_usd"] == pytest.approx(6.069628)
    assert not any(row["status"] == "reserved" for row in json.loads(ledger_before)["reservations"])


def test_generic_timeout_recovery_rejects_mismatched_error_status_count_and_reserved_rows(tmp_path: Path) -> None:
    case_id = "6222b6eb"
    args = {
        "case_id": case_id,
        "reservation_id": "placeholder",
        "expected_completed_count": 30,
    }
    paths, ledger, reservation_id, original = _general_recovery_fixture(
        tmp_path,
        case_id=case_id,
        prior_failed_ids=("95228167", "gpt4_9a159967"),
        completed_count=30,
        prefinalized_timeout=True,
    )
    args["reservation_id"] = reservation_id
    original_ledger = paths.ledger.read_bytes()
    original_checkpoint = paths.checkpoint.read_bytes()

    checkpoint = json.loads(original_checkpoint)
    checkpoint["evidence"][case_id]["error"] = "AmbiguousExecutionError: provider omitted usage"
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(FaithfulRunError, match="matching in-flight ambiguous evidence"):
        recover_inflight_case(paths, **args)
    paths.checkpoint.write_bytes(original_checkpoint)

    checkpoint = json.loads(original_checkpoint)
    checkpoint["evidence"][case_id]["status"] = "failed"
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(FaithfulRunError, match="matching in-flight ambiguous evidence"):
        recover_inflight_case(paths, **args)
    paths.checkpoint.write_bytes(original_checkpoint)

    with pytest.raises(FaithfulRunError, match="expected 29 completed cases"):
        recover_inflight_case(paths, **{**args, "expected_completed_count": 29})
    with pytest.raises(BudgetError, match="reservation ID must identify exactly one"):
        recover_inflight_case(paths, **{**args, "reservation_id": "wrong-timeout-reservation"})

    ledger_doc = json.loads(paths.ledger.read_bytes())
    target_row = next(row for row in ledger_doc["reservations"] if row["reservation_id"] == reservation_id)
    target_row["error"] = "Request timed out after retry."
    paths.ledger.write_text(json.dumps(ledger_doc), encoding="utf-8")
    ledger_with_mismatched_error = paths.ledger.read_bytes()
    with pytest.raises(BudgetError, match="not entirely outstanding or identically recovered"):
        recover_inflight_case(paths, **args)
    assert json.loads(paths.checkpoint.read_bytes()) == original
    assert paths.ledger.read_bytes() == ledger_with_mismatched_error

    # Restore the exact timeout in this temporary artifact before introducing
    # an unrelated outstanding reservation for the next fail-closed check.
    ledger_doc = json.loads(ledger_with_mismatched_error)
    target_row = next(row for row in ledger_doc["reservations"] if row["reservation_id"] == reservation_id)
    target_row["error"] = "Request timed out."
    paths.ledger.write_text(json.dumps(ledger_doc), encoding="utf-8")
    outstanding = ledger.reserve(GPT6_LUNA_MODEL, 100, 10, phase="run")
    with pytest.raises(BudgetError, match="zero reserved ledger rows"):
        recover_inflight_case(paths, **args)
    assert ledger.inspect_reservation(outstanding.reservation_id)["status"] == "reserved"
    assert json.loads(paths.checkpoint.read_bytes()) == original
    assert not list(paths.root.glob("session-checkpoint.json.recover-*.json"))
    assert ledger.inspect_reservation(reservation_id)["status"] == "unknown"
    assert ledger.inspect_reservation(reservation_id)["error"] == "Request timed out."
    assert original_ledger != paths.ledger.read_bytes()


def test_generic_inflight_recovery_rejects_case_reservation_and_count_mismatch(tmp_path: Path) -> None:
    paths, ledger, reservation_id, original = _general_recovery_fixture(tmp_path)
    with pytest.raises(FaithfulRunError, match="expected 20 completed cases"):
        recover_inflight_case(
            paths, case_id="gpt4_9a159967", reservation_id=reservation_id,
            expected_completed_count=20,
        )
    with pytest.raises(FaithfulRunError, match="outside the prepared selection"):
        recover_inflight_case(
            paths, case_id="not-selected", reservation_id=reservation_id,
            expected_completed_count=21,
        )
    with pytest.raises(BudgetError, match="reservation ID must identify exactly one"):
        recover_inflight_case(
            paths, case_id="gpt4_9a159967", reservation_id="wrong-reservation",
            expected_completed_count=21,
        )
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8")) == original
    assert ledger.summary()["unknown_count"] == 0
    assert not list(paths.root.glob("session-checkpoint.json.recover-*.json"))


def test_generic_inflight_recovery_requires_explicit_count_and_excludes_pinned_case(tmp_path: Path) -> None:
    paths, _, reservation_id, _ = _general_recovery_fixture(tmp_path)
    with pytest.raises(TypeError):
        recover_inflight_case(paths, case_id="gpt4_9a159967", reservation_id=reservation_id)  # type: ignore[call-arg]
    with pytest.raises(FaithfulRunError, match="must use the pinned safe-skip"):
        recover_inflight_case(
            paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id,
            expected_completed_count=21,
        )


def _terminated_pre_provider_fixture(
    tmp_path: Path, *, evidence_status: str = "in_flight",
) -> tuple[RunPaths, str, dict]:
    """Build a selected-35 run killed during session ingest (zero reservations).

    Mirrors an external timeout while a case was ingesting sessions: the case
    is the in-flight marker, its evidence holds ingested session rows with no
    answer or judge result, and every ledger reservation is finalized — no
    reservation row references the case at all. ``evidence_status`` covers
    both live kill shapes: the evidence row still in-flight, or finalized as
    failed by the runner's signal handler (no error key) before the marker
    cleared.
    """
    case_id = "ad7109d1"
    prior_failed_id = "5d3d2817"
    dataset, manifest_path = _write_fixture_dataset(tmp_path)
    records = json.loads(dataset.read_text(encoding="utf-8"))
    ids = [f"case-{index:02d}" for index in range(EXPECTED_CASE_COUNT)]
    ids[0] = case_id
    ids[1] = prior_failed_id
    ids[2] = GPT6_SELECTED35_EXCLUDED_ID
    for record, question_id in zip(records, ids, strict=True):
        record["question_id"] = question_id
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_doc["selection"]["ordered_question_ids"] = ids
    manifest_path.write_text(json.dumps(manifest_doc), encoding="utf-8")

    paths = RunPaths.from_root(tmp_path / "terminated-pre-provider-artifacts")
    prepare_run(
        dataset, manifest_path, paths=paths, owner_id="offline-test",
        profile=FRESH_RUN_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=50.0,
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    binding = manifest["binding"]
    _, _, maximum, _, _, pricing, _ = _profile_from_binding(binding)
    ledger = BudgetLedger(paths.ledger, max_budget_usd=maximum, pricing=pricing, binding=binding)
    reservations = [
        ledger.reserve(GPT6_LUNA_MODEL, 100, 10, phase="run")
        for _ in range(10)
    ]
    for reservation in reservations:
        ledger.finalize(
            reservation.reservation_id,
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    selected_rows = {row["question_id"]: row for row in manifest["sessions"]}
    target_sessions = selected_rows[case_id]["session_ids"]
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["completed_question_ids"] = [
        selected_id for selected_id in manifest["ordered_question_ids"]
        if selected_id not in {case_id, prior_failed_id}
    ][:16]
    checkpoint["failed_question_ids"] = [prior_failed_id]
    checkpoint["in_flight"] = case_id
    checkpoint["in_flight_stage"] = f"session:{target_sessions[0]}"
    checkpoint["sessions"] = {
        case_id: {session_id: "completed" for session_id in target_sessions[:2]},
        prior_failed_id: {},
    }
    checkpoint["evidence"] = {
        prior_failed_id: {"question_id": prior_failed_id, "status": "failed"},
        case_id: {
            "question_id": case_id,
            "status": evidence_status,
            "sessions": [
                {
                    "session_id": session_id,
                    "result": {"mode": "dual", "ingested": True},
                    "tool_results": {"tools": []},
                }
                for session_id in target_sessions[:2]
            ],
        },
    }
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    return paths, case_id, checkpoint


@pytest.mark.parametrize("evidence_status", ["in_flight", "failed"])
def test_recover_inflight_cli_handles_externally_terminated_pre_provider_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], evidence_status: str,
) -> None:
    paths, case_id, original = _terminated_pre_provider_fixture(
        tmp_path, evidence_status=evidence_status,
    )
    ledger_before = paths.ledger.read_bytes()
    journal_path = paths.checkpoint.with_suffix(
        paths.checkpoint.suffix + f".recover-{canonical_hash([case_id, ''])[:16]}.json"
    )
    base_args = [
        "recover-inflight", "--artifact-root", str(paths.root), "--case-id", case_id,
    ]

    # Refusal: the confirmed completed count must match exactly.
    assert faithful_main([*base_args, "--expected-completed-count", "15"]) == 2
    assert "expected 15 completed cases" in capsys.readouterr().err
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8")) == original
    assert not journal_path.exists()

    # Success through the real CLI path: case failed, marker cleared, journal.
    assert faithful_main([*base_args, "--expected-completed-count", "16"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] is True
    assert result["status"] == "recovered"

    recovered = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    case_after = recovered["evidence"][case_id]
    assert recovered["failed_question_ids"] == sorted({"5d3d2817", case_id})
    assert recovered["in_flight"] is None
    assert recovered["in_flight_stage"] is None
    assert case_after["status"] == "failed"
    assert "externally terminated before provider dispatch" in case_after["error"]
    assert case_after["safe_skip_recovery"]["reservation_id"] == ""
    assert case_after["safe_skip_recovery"]["ledger_already_unknown"] is False
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["schema"] == INFLIGHT_RECOVERY_JOURNAL_SCHEMA
    assert journal["operation_id"] == case_after["safe_skip_recovery"]["operation_id"]
    assert journal["checkpoint_before_sha256"] == canonical_hash(original)
    assert journal["checkpoint_after_sha256"] == canonical_hash(recovered)
    assert journal["checkpoint_after"] == recovered
    assert paths.ledger.read_bytes() == ledger_before

    # Second invocation refuses: the externally terminated case is terminal.
    assert faithful_main([*base_args, "--expected-completed-count", "16"]) == 2
    assert "already recovered" in capsys.readouterr().err
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8")) == recovered


def test_recover_inflight_cli_refuses_failed_evidence_with_case_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A finalized failure carrying a case error is a genuine case failure."""
    paths, case_id, _ = _terminated_pre_provider_fixture(
        tmp_path, evidence_status="failed",
    )
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["evidence"][case_id]["error"] = (
        "ContextWindowError: prompt exceeds the bound model context"
    )
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    ledger_before = paths.ledger.read_bytes()
    journal_path = paths.checkpoint.with_suffix(
        paths.checkpoint.suffix + f".recover-{canonical_hash([case_id, ''])[:16]}.json"
    )

    assert faithful_main([
        "recover-inflight", "--artifact-root", str(paths.root), "--case-id", case_id,
        "--expected-completed-count", "16",
    ]) == 2
    assert "no provider outcome recorded" in capsys.readouterr().err
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8")) == checkpoint
    assert paths.ledger.read_bytes() == ledger_before
    assert not journal_path.exists()


def test_safe_skip_journal_recovers_after_ledger_checkpoint_crash(tmp_path: Path, monkeypatch) -> None:
    paths, ledger, reservation_id, original = _recovery_fixture(tmp_path)
    real_atomic_json = __import__("benchmarks.longmemeval.faithful_s36", fromlist=["_atomic_json"])._atomic_json
    failed_once = False

    def crash_before_checkpoint(path: Path, value: dict) -> None:
        nonlocal failed_once
        if path == paths.checkpoint and not failed_once:
            failed_once = True
            raise OSError("simulated process crash before checkpoint replace")
        real_atomic_json(path, value)

    monkeypatch.setattr("benchmarks.longmemeval.faithful_s36._atomic_json", crash_before_checkpoint)
    with pytest.raises(OSError, match="simulated process crash"):
        recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id)
    assert ledger.summary()["unknown_count"] == 1
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8")) == original

    monkeypatch.setattr("benchmarks.longmemeval.faithful_s36._atomic_json", real_atomic_json)
    result = recover_safe_skip(paths, case_id=SAFE_SKIP_CASE_ID, reservation_id=reservation_id)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert result["changed"] is True
    assert checkpoint["in_flight"] is None
    assert checkpoint["failed_question_ids"] == [SAFE_SKIP_CASE_ID]
    assert ledger.summary()["unknown_count"] == 1


def test_dataset_hash_mutation_after_prepare_is_rejected(tmp_path: Path) -> None:
    """Changing public source bytes after preparation invalidates the binding."""
    dataset, manifest = _write_fixture_dataset(tmp_path)
    paths = RunPaths.from_root(tmp_path / "artifacts")
    judge_root = _write_fake_judge_root(tmp_path)
    prepare_run(dataset, manifest, paths=paths, judge_root=judge_root)
    _approve_measured_calibration(paths)
    records = json.loads(dataset.read_text(encoding="utf-8"))
    records[0]["haystack_sessions"][0][0]["content"] = "MUTATED_PUBLIC_HISTORY"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(LedgerBindingError, match="binding"):
        asyncio.run(resume_run(
            dataset, manifest, paths=paths, execute=True,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            judge_root=judge_root,
        ))


def test_full_s_profile_prepares_all_rows_with_dedupe_metadata_and_own_namespace(tmp_path: Path) -> None:
    dataset, manifest, ordered_ids = _write_full_s_fixture(tmp_path)
    normalized_rows = json.loads(dataset.read_text(encoding="utf-8"))
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))

    assert len(normalized_rows) == FULL_S_CASE_COUNT == 500
    assert [row["question_id"] for row in normalized_rows] == ordered_ids
    assert normalized_rows[0]["haystack_session_ids"] == ["session-0-first", "session-0-second"]
    assert normalized_rows[0]["haystack_dates"] == ["2024/01/01", "2024/01/02"]
    assert normalized_rows[0]["haystack_sessions"][0][0]["content"] == "first-version-full-s-000"
    assert normalized_rows[0]["custom_metadata"] == {"row": 0}
    assert manifest_doc["normalization"]["duplicate_session_id_group_count"] == 1
    assert manifest_doc["normalization"]["duplicate_groups"] == [{
        "question_id": "full-s-000", "session_id": "session-0-first",
        "kept_source_index": 0,
        "occurrences": [{"source_index": 0}, {"source_index": 2}],
        "discarded_source_indexes": [2],
    }]
    assert manifest_doc["selection"]["ordered_question_ids"] == ordered_ids
    assert manifest_doc["arms"] == ["turns"]
    assert manifest_doc["retrieval"]["tier"] == "turns"

    paths = RunPaths.from_root(tmp_path / "full-s-run")
    prepared = prepare_run(
        dataset, manifest, paths=paths, owner_id="full-s-test",
        profile=FULL_S_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=150.0,
    )
    assert list(prepared.ordered_question_ids) == ordered_ids
    assert prepared.binding["retrieval_tier"] == "turns"
    assert prepared.binding["operational_stop_usd"] == "150.0"
    assert GPT6_FULL_S_ARTIFACT_NAMESPACE != GPT6_SELECTED35_ARTIFACT_NAMESPACE
    assert RunPaths.from_root(GPT6_FULL_S_ARTIFACT_NAMESPACE).root != RunPaths.from_root(
        GPT6_SELECTED35_ARTIFACT_NAMESPACE
    ).root


def test_full_s_turns_tier_is_injected_into_recall_and_selected35_remains_auto(tmp_path: Path) -> None:
    for tier in ("turns", "auto"):
        client = FakeResponses([
            _response(output=[SimpleNamespace(
                type="function_call", call_id=f"recall-{tier}", name="weft_recall",
                arguments=json.dumps({"query": "public fact"}),
            )]),
            _response("done"),
        ])
        gateway = FakeGateway()
        agent = FaithfulAgent(
            client, BudgetLedger(tmp_path / f"{tier}.json", binding={"tier": tier}),
            tools=gateway, model="gpt-6-luna", retrieval_tier=tier,
        )
        result = asyncio.run(agent.answer(
            question="What fact?", question_date="2024/01/04", task_shape=None,
            recalled_context=None, project_id="p", agent_id="a",
        ))
        assert result.text == "done"
        assert gateway.calls == [("weft_recall", {
            "query": "public fact", "tier": tier, "project_id": "p", "agent_id": "a",
        })]


def test_manifest_checkpoint_state_uses_manifest_order_and_shell_quotes_resume_filters(tmp_path: Path) -> None:
    ordered_ids = ["q-1", "q-2", "q-3", "q-4"]
    manifest_path = tmp_path / "actual manifest.json"
    dataset_path = tmp_path / "long mem eval dataset.json"
    artifact_root = tmp_path / "artifact root"
    manifest = {
        "ordered_question_ids": ordered_ids,
        "manifest_path": "stale embedded path.json",
        "binding": {
            "run_profile": FULL_S_PROFILE,
            "writer_model": "gpt-6-luna",
            "max_budget_usd": "150.0",
            "selection_policy_include_question_ids": json.dumps(["q-1", "q-3"]),
            "selection_policy_exclude_question_ids": json.dumps(["q-2"]),
        },
    }
    state = _manifest_checkpoint_state(
        manifest,
        {"completed_question_ids": ["q-2", "q-1"], "failed_question_ids": ["q-3"]},
        dataset_path=dataset_path, manifest_path=manifest_path, artifact_root=artifact_root,
    )

    assert state["completed_question_ids"] == ["q-1", "q-2"]
    assert state["failed_question_ids"] == ["q-3"]
    assert state["pending_question_ids"] == ["q-4"]
    assert state["complete"] is False
    assert state["last_completed_question_id"] == "q-2"
    command = shlex.split(state["resume_command"])
    assert "--dataset" in command and command[command.index("--dataset") + 1] == str(dataset_path)
    assert "--manifest" in command and command[command.index("--manifest") + 1] == str(manifest_path)
    assert "stale embedded path.json" not in command
    assert command.count("--include-question-id") == 2
    assert command[command.index("--include-question-id") + 1] == "q-1"
    assert command.count("--exclude-question-id") == 1
    assert command[command.index("--exclude-question-id") + 1] == "q-2"


def test_budget_stop_checkpoint_returns_partial_receipt_idempotently_without_execution(tmp_path: Path, monkeypatch) -> None:
    dataset, manifest, ordered_ids = _write_full_s_fixture(tmp_path)
    paths = RunPaths.from_root(tmp_path / "full-s-partial")
    prepared = prepare_run(
        dataset, manifest, paths=paths, owner_id="full-s-test",
        profile=FULL_S_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=150.0,
    )
    ledger = BudgetLedger(
        paths.ledger, max_budget_usd=150.0, calibration_budget_usd=10.0,
        operational_stop_usd=150.0, pricing=FreshRunPricing(), binding=prepared.binding,
    )
    reservation = ledger.reserve("gpt-6-luna", 599_996_000, 0)
    ledger.finalize(
        reservation.reservation_id,
        usage={"input_tokens": 599_996_000, "output_tokens": 0},
    )
    with pytest.raises(TotalBudgetExceeded) as refusal:
        ledger.reserve("gpt-6-luna", 8_000, 0)
    stop = {
        "used_usd": refusal.value.used_usd,
        "next_estimate_usd": refusal.value.estimate_usd,
        "ceiling_usd": refusal.value.ceiling_usd,
        "last_completed_question_id": ordered_ids[1],
    }
    assert stop["used_usd"] == pytest.approx(149.999)
    assert stop["next_estimate_usd"] == pytest.approx(0.001)
    assert stop["ceiling_usd"] == pytest.approx(150.0)
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint.update({
        "completed_question_ids": ordered_ids[:2],
        "failed_question_ids": [],
        "in_flight": None,
        "budget_stop": stop,
        "budget_stopped_question_id": ordered_ids[2],
        "evidence": {
            ordered_ids[0]: {"question_id": ordered_ids[0], "status": "completed"},
            ordered_ids[1]: {"question_id": ordered_ids[1], "status": "completed"},
            ordered_ids[2]: {"question_id": ordered_ids[2], "status": "budget_stopped"},
        },
    })
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    paths.calibration.write_text(json.dumps({
        "status": "CALIBRATED", "approval_required": False, "binding": prepared.binding,
    }), encoding="utf-8")
    monkeypatch.setattr("benchmarks.longmemeval.faithful_s36.require_fastembed", lambda: pytest.fail("must not initialize embedding"))

    judge_root = _write_fake_judge_root(tmp_path)
    first = asyncio.run(resume_run(
        dataset, manifest, paths=paths, execute=True,
        dsn="postgresql://bench@localhost:5432/longmemeval_bench",
        judge_root=judge_root,
    ))
    second = asyncio.run(resume_run(
        dataset, manifest, paths=paths, execute=True,
        dsn="postgresql://bench@localhost:5432/longmemeval_bench",
        judge_root=judge_root,
    ))

    assert first == second
    assert first["status"] == "PARTIAL_BUDGET_STOP"
    assert first["selected_denominator"] == FULL_S_CASE_COUNT
    assert first["scored_count"] == 2
    assert first["scored_denominator"] == FULL_S_CASE_COUNT
    assert first["last_completed_question_id"] == ordered_ids[1]
    assert first["budget_estimate_vs_actual"]["estimated_usd"] == pytest.approx(reservation.estimated_usd)
    assert first["budget_estimate_vs_actual"]["measured_actual_usd"] is not None
    assert first["budget_estimate_vs_actual"]["refused_next_estimate_usd"] == pytest.approx(0.001)
    assert first["budget_stop"] == stop
    assert first["budget_estimate_vs_actual"]["actual_vs_estimate_usd"] == pytest.approx(0.0)
    assert first["budget_estimate_vs_actual"]["actual_usage_complete"] is True
    assert str(dataset) in first["resume_command"]


def test_sessions_are_processed_chronologically_without_answer_labels() -> None:
    """Date ordering is deterministic and independent of answer labels."""
    instance = SimpleNamespace(sessions=(_session("2023/04/21 (Fri) 09:00", "late", "late"), _session("2023/04/19 (Wed) 09:00", "early", "early")))
    assert [item.session_id for item in _sort_sessions(instance)] == ["early", "late"]


def test_full_s_calibrate_refuses_pinned_source_mismatch_through_real_call_path(tmp_path: Path) -> None:
    """Through the real calibrate path, a pinned source whose bytes no longer
    match the preparation manifest refuses with the named-file mismatch."""
    dataset, _manifest, _ids = _write_full_s_fixture(tmp_path)
    pinned = "benchmarks/longmemeval/full_s_profile.py"
    prep_manifest_path = tmp_path / "full-s-pinned-manifest.json"
    prepare_full_s_manifest(
        tmp_path / "cleaned-synthetic.json", tmp_path / "normalized-pinned.json",
        prep_manifest_path, source_hash_paths=(pinned,),
    )
    doc = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
    doc["source_hashes"][pinned] = "0" * 64  # source changed since preparation
    prep_manifest_path.write_text(json.dumps(doc), encoding="utf-8")
    paths = RunPaths.from_root(tmp_path / "full-s-tamper-run")
    prepare_run(
        dataset, prep_manifest_path, paths=paths, owner_id="full-s-test",
        profile=FULL_S_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=150.0,
    )
    with pytest.raises(LedgerBindingError) as refusal:
        asyncio.run(run_calibration(
            dataset, prep_manifest_path, paths=paths,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            owner_id="full-s-test",
        ))
    message = str(refusal.value)
    assert message.startswith("full-S pinned source hash mismatch: ")
    assert pinned in message


def test_full_s_calibrate_refuses_preparation_manifest_without_source_hashes(tmp_path: Path) -> None:
    """A full-S preparation manifest without source_hashes is a hard refusal
    through the real calibrate path — the former silent skip is gone."""
    dataset, _manifest, _ids = _write_full_s_fixture(tmp_path)
    prep_manifest_path = tmp_path / "full-s-nosource-manifest.json"
    prepare_full_s_manifest(
        tmp_path / "cleaned-synthetic.json", tmp_path / "normalized-nosource.json",
        prep_manifest_path, source_hash_paths=(),
    )
    doc = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
    del doc["source_hashes"]
    prep_manifest_path.write_text(json.dumps(doc), encoding="utf-8")
    paths = RunPaths.from_root(tmp_path / "full-s-nosource-run")
    prepare_run(
        dataset, prep_manifest_path, paths=paths, owner_id="full-s-test",
        profile=FULL_S_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=150.0,
    )
    with pytest.raises(LedgerBindingError, match="source_hashes must be a non-empty object"):
        asyncio.run(run_calibration(
            dataset, prep_manifest_path, paths=paths,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            owner_id="full-s-test",
        ))


def test_calibrate_refuses_invalid_owner_id_before_any_write(tmp_path: Path) -> None:
    """An owner_id the storage layer cannot validate refuses before any
    gateway or write exists, never landing NULL-user_id global-scope rows."""
    dataset, manifest, _ids = _write_full_s_fixture(tmp_path)
    paths = RunPaths.from_root(tmp_path / "full-s-owner-run")
    prepare_run(
        dataset, manifest, paths=paths, owner_id="full-s-test",
        profile=FULL_S_PROFILE, writer_model=GPT6_LUNA_MODEL, max_budget_usd=150.0,
    )
    with pytest.raises(ExecutionGateError, match="owner_id"):
        asyncio.run(run_calibration(
            dataset, manifest, paths=paths,
            dsn="postgresql://bench@localhost:5432/longmemeval_bench",
            owner_id="faithful gpt6.selected35",
        ))
    assert not paths.calibration.exists()


# uv run pytest tests/test_longmemeval_faithful_s36.py -q
