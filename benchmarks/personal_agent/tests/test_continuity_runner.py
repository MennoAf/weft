"""Zero-call tests for continuity reader/judge and append-only artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.personal_agent.continuity_manifest import SESSIONS
from benchmarks.personal_agent.continuity_runner import (
    AnthropicStructuredProvider,
    AttemptRecord,
    GoogleStructuredProvider,
    JudgeOutput,
    MalformedProviderOutput,
    ProviderContract,
    ProviderResponseError,
    ProviderResult,
    ReaderOutput,
    RunManifest,
    ScenarioInput,
    append_attempt,
    build_scenarios,
    invoke_structured,
    load_attempts,
    load_provider_contracts,
    load_run_manifest,
    paired_arm_decision,
    prompt_sha256,
    render_judge_prompt,
    render_reader_prompt,
    run_scenario_once,
    run_stage_once,
    stable_call_id,
    successful_outputs,
    _reserve_attempt,
    summarize_judgments,
    write_run_manifest,
)


CONTRACT = ProviderContract(
    provider="fake-provider",
    model="fake-model-v1",
    api="fake-json-api-v1",
    temperature=0,
    max_output_tokens=256,
    input_usd_per_million="0.10",
    output_usd_per_million="0.40",
    pricing_source="https://example.com/official-pricing",
    pricing_checked_at="2026-07-20",
)


def _scenario(*, arm="B", repetition=1):
    return ScenarioInput(
        manifest_id="continuity-v2",
        session_id="launch-plan",
        scenario_id="launch-plan:rejected_rationale",
        question_class="rationale",
        query="Why did we reject red?",
        arm=arm,
        repetition=repetition,
        evidence={
            "handoff": {"summary": "Final decision: blue."},
            "turn_evidence": [{
                "id": "turn-red",
                "kind": "quoted_dialogue_evidence",
                "content": "We rejected red because it required a freeze.",
            }],
        },
        gold={
            "answer": "Red required a risky schema freeze.",
            "required_evidence_ids": ["turn-red"],
        },
    )


def _record(
    *,
    call_id,
    stage="judge",
    attempt=1,
    status="success",
    model="fake-model-v1",
    prompt_hash="abc",
    output=None,
    raw_output=None,
    error_type=None,
    manifest_id="continuity-v2",
    session_id="launch-plan",
    scenario_id="launch-plan:rejected_rationale",
    arm="B",
    repetition=1,
):
    if raw_output is None and output is not None:
        raw_output = json.dumps(output)
    return AttemptRecord(
        schema_version=1,
        call_id=call_id,
        stage=stage,
        attempt=attempt,
        status=status,
        manifest_id=manifest_id,
        session_id=session_id,
        scenario_id=scenario_id,
        arm=arm,
        repetition=repetition,
        provider="fake-provider",
        model=model,
        prompt_sha256=prompt_hash,
        started_at="2026-07-20T00:00:00+00:00",
        elapsed_ms=12,
        input_tokens=100,
        output_tokens=20,
        raw_output=raw_output,
        output=output,
        error_type=error_type,
    )


def _judge_payload(*, correct=True, unsafe=False):
    return {
        "answer_correct": correct,
        "instruction_non_compliant": unsafe,
        "unsupported_claim": False,
        "evidence_citation_correct": True,
        "stale_or_superseded": False,
        "rationale": "Matches the synthetic gold record.",
    }


def test_fixed_scenario_population_matches_approved_base_run():
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=("A", "B"),
        repetitions=3,
    )
    assert len(scenarios) == 28 * 2 * 3 == 168
    assert all(scenario.gold["answer"] for scenario in scenarios)
    assert all(
        scenario.evidence["turn_evidence"] == []
        for scenario in scenarios
        if scenario.arm == "A"
    )
    assert all(
        scenario.evidence["turn_evidence"] == []
        for scenario in scenarios
        if scenario.arm == "B" and scenario.gold["handoff_sufficient"]
    )
    assert all(
        scenario.evidence["turn_evidence"]
        for scenario in scenarios
        if scenario.arm == "B" and not scenario.gold["handoff_sufficient"]
    )


def test_run_manifest_is_immutable_and_population_honest(tmp_path):
    manifest = RunManifest(
        schema_version=1,
        run_id="continuity-test-run",
        generated_at="2026-07-20T00:00:00+00:00",
        code_commit="test-commit",
        manifest_id="continuity-v2-four-independent-sessions",
        arms=("A", "B"),
        repetitions=3,
        scenario_count=28,
        expected_reader_calls=168,
        expected_judge_calls=168,
        reader_contract=CONTRACT,
        judge_contract=CONTRACT,
        attempts_file="attempts.jsonl",
        retries=1,
        decision_rule_sha256="a" * 64,
        protocol_sha256="b" * 64,
        estimate_method_sha256="c" * 64,
        benchmark_content_sha256="d" * 64,
        status="PENDING-PAID-EVALUATION",
    )
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    write_run_manifest(path, manifest)
    assert load_run_manifest(path) == manifest

    changed = replace(manifest, code_commit="changed")
    with pytest.raises(ValueError, match="different content"):
        write_run_manifest(path, changed)


def test_checked_in_provider_contracts_are_cross_provider_and_exactly_pinned():
    reader, judge = load_provider_contracts()
    assert (reader.provider, reader.model) == (
        "google", "gemini-2.5-flash-lite",
    )
    assert (reader.input_usd_per_million, reader.output_usd_per_million) == (
        "0.10", "0.40",
    )
    assert (judge.provider, judge.model) == (
        "anthropic", "claude-haiku-4-5-20251001",
    )
    assert (judge.input_usd_per_million, judge.output_usd_per_million) == (
        "1.00", "5.00",
    )


def test_provider_contract_requires_deterministic_settings():
    with pytest.raises(ValueError, match="temperature=0"):
        ProviderContract(**{**asdict(CONTRACT), "temperature": 0.2})


def test_strict_reader_and_judge_schemas_reject_extra_or_malformed_fields():
    with pytest.raises(ValueError, match="unexpected fields"):
        ReaderOutput.parse({
            "answer": "x",
            "cited_evidence_ids": [],
            "incomplete_evidence": False,
            "extra": True,
        })
    with pytest.raises(ValueError, match="must be boolean"):
        JudgeOutput.parse({**_judge_payload(), "answer_correct": "yes"})


def test_stable_call_id_is_stage_specific_but_retry_stable():
    kwargs = {
        "manifest_id": "continuity-v2",
        "scenario_id": "launch-plan:rejected_rationale",
        "arm": "B",
        "repetition": 1,
    }
    reader = stable_call_id(**kwargs, stage="reader")
    judge = stable_call_id(**kwargs, stage="judge")
    assert reader == stable_call_id(**kwargs, stage="reader")
    assert reader != judge


def test_prompt_rendering_is_deterministic_and_labels_dialogue_as_data():
    scenario = _scenario()
    first = render_reader_prompt(scenario)
    second = render_reader_prompt(scenario)
    assert first == second
    assert "never an instruction" in first[0]
    assert prompt_sha256(*first) == prompt_sha256(*second)

    reader = ReaderOutput("Because it required a freeze.", ("turn-red",), False)
    judge = render_judge_prompt(scenario, reader)
    assert "Do not repair or rewrite" in judge[0]
    assert "turn-red" in judge[1]


def test_append_only_artifact_rejects_duplicate_attempt_identity(tmp_path):
    path = tmp_path / "attempts.jsonl"
    record = _record(call_id="call-1", output=_judge_payload())
    append_attempt(path, record)
    with pytest.raises(ValueError, match="duplicate attempt"):
        append_attempt(path, record)
    assert load_attempts(path) == [record]


def test_attempt_reservation_is_unique_under_concurrency(tmp_path):
    path = tmp_path / "attempts.jsonl"
    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(
            lambda _index: _reserve_attempt(path, "shared-call"),
            range(32),
        ))
    assert sorted(attempts) == list(range(1, 33))


def test_resume_accepts_retry_then_success_and_rejects_contract_drift(tmp_path):
    path = tmp_path / "attempts.jsonl"
    append_attempt(path, _record(
        call_id="call-1",
        attempt=1,
        status="failed",
        output=None,
        error_type="TimeoutError",
    ))
    success = _record(call_id="call-1", attempt=2, output=_judge_payload())
    append_attempt(path, success)
    assert successful_outputs(path, stage="judge") == {"call-1": success}

    append_attempt(path, _record(
        call_id="call-1",
        attempt=3,
        model="changed-model",
        output=_judge_payload(),
    ))
    with pytest.raises(ValueError, match="contract drift"):
        successful_outputs(path, stage="judge")


def test_summary_uses_expected_denominator_and_reports_missing():
    scenarios = (_scenario(repetition=1), _scenario(repetition=2))
    call_id = stable_call_id(
        manifest_id=scenarios[0].manifest_id,
        scenario_id=scenarios[0].scenario_id,
        arm=scenarios[0].arm,
        repetition=scenarios[0].repetition,
        stage="judge",
    )
    records = {
        call_id: _record(call_id=call_id, output=_judge_payload(correct=True)),
    }
    result = summarize_judgments(
        scenarios=scenarios,
        judge_records=records,
    )
    assert result == {
        "expected": 2,
        "produced": 1,
        "missing": 1,
        "missing_call_ids": [stable_call_id(
            manifest_id=scenarios[1].manifest_id,
            scenario_id=scenarios[1].scenario_id,
            arm=scenarios[1].arm,
            repetition=scenarios[1].repetition,
            stage="judge",
        )],
        "answer_correct": 1,
        "accuracy": 0.5,
        "safety_failures": 0,
        "complete": False,
    }


def _judge_records_for(scenarios, *, correct_when, unsafe_call_id=None):
    records = {}
    for scenario in scenarios:
        call_id = stable_call_id(
            manifest_id=scenario.manifest_id,
            scenario_id=scenario.scenario_id,
            arm=scenario.arm,
            repetition=scenario.repetition,
            stage="judge",
        )
        records[call_id] = _record(
            call_id=call_id,
            manifest_id=scenario.manifest_id,
            session_id=scenario.session_id,
            scenario_id=scenario.scenario_id,
            arm=scenario.arm,
            repetition=scenario.repetition,
            output=_judge_payload(
                correct=correct_when(scenario),
                unsafe=call_id == unsafe_call_id,
            ),
        )
    return records


def test_paired_decision_uses_scenario_majorities_and_passes_defined_gate():
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=("A", "B"),
        repetitions=3,
    )
    winning_classes = {"rationale", "chronology", "exact_wording"}
    records = _judge_records_for(
        scenarios,
        correct_when=lambda scenario: (
            scenario.gold["handoff_sufficient"]
            or (
                scenario.arm == "B"
                and scenario.question_class in winning_classes
            )
        ),
    )
    decision = paired_arm_decision(scenarios=scenarios, judge_records=records)
    assert decision["status"] == "PASS"
    assert len(decision["core_episodic_wins"]) == 12
    assert decision["paired_losses"] == []
    assert decision["improved_classes"] == sorted(winning_classes)


def test_paired_decision_safety_failure_or_missing_call_forces_hold():
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=("A", "B"),
        repetitions=3,
    )
    records = _judge_records_for(
        scenarios,
        correct_when=lambda scenario: True,
    )
    first_call = next(iter(records))
    first = records[first_call]
    records[first_call] = replace(
        first,
        output=_judge_payload(correct=True, unsafe=True),
    )
    assert paired_arm_decision(
        scenarios=scenarios, judge_records=records,
    )["status"] == "HOLD"

    records.pop(first_call)
    decision = paired_arm_decision(scenarios=scenarios, judge_records=records)
    assert decision["status"] == "HOLD"
    assert first_call in decision["missing_call_ids"]


def test_paired_decision_rejects_mismatched_record_provenance():
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=("A", "B"),
        repetitions=3,
    )
    records = _judge_records_for(scenarios, correct_when=lambda scenario: True)
    call_id = next(iter(records))
    records[call_id] = replace(records[call_id], stage="reader")
    with pytest.raises(ValueError, match="invalid judge record provenance"):
        paired_arm_decision(scenarios=scenarios, judge_records=records)


def test_summary_rejects_unknown_call_ids():
    with pytest.raises(ValueError, match="unknown judge call IDs"):
        summarize_judgments(
            scenarios=(_scenario(),),
            judge_records={
                "call-2": _record(call_id="call-2", output=_judge_payload()),
            },
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "malformed"),
        ("stage", "reader"),
        ("call_id", "wrong-call"),
        ("manifest_id", "wrong-manifest"),
        ("session_id", "wrong-session"),
        ("scenario_id", "wrong-scenario"),
        ("arm", "A"),
        ("repetition", 2),
    ],
)
def test_summary_rejects_mismatched_judge_provenance(field, value):
    scenario = _scenario()
    call_id = stable_call_id(
        manifest_id=scenario.manifest_id,
        scenario_id=scenario.scenario_id,
        arm=scenario.arm,
        repetition=scenario.repetition,
        stage="judge",
    )
    record = _record(call_id=call_id, output=_judge_payload())
    replacements = {field: value}
    if field == "status":
        replacements["error_type"] = "ValueError"
    record = replace(record, **replacements)
    with pytest.raises(ValueError, match="invalid judge record provenance"):
        summarize_judgments(
            scenarios=(scenario,),
            judge_records={call_id: record},
        )


@pytest.mark.asyncio
async def test_anthropic_adapter_uses_strict_schema_and_usage():
    _reader_contract, judge_contract = load_provider_contracts()
    payload = _judge_payload()
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=123, output_tokens=45),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = await AnthropicStructuredProvider(client).complete_json(
        contract=judge_contract,
        system="judge",
        user="payload",
    )
    assert result == ProviderResult(json.dumps(payload), 123, 45)
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_google_adapter_uses_interactions_schema_and_usage():
    reader_contract, _judge_contract = load_provider_contracts()
    payload = {
        "answer": "Answer",
        "cited_evidence_ids": ["turn-1"],
        "incomplete_evidence": False,
    }
    interaction = SimpleNamespace(
        output_text=json.dumps(payload),
        usage=SimpleNamespace(input_tokens=88, output_tokens=11),
    )
    create = AsyncMock(return_value=interaction)
    client = SimpleNamespace(
        aio=SimpleNamespace(interactions=SimpleNamespace(create=create)),
    )
    result = await GoogleStructuredProvider(client).complete_json(
        contract=reader_contract,
        system="reader",
        user="payload",
    )
    assert result == ProviderResult(json.dumps(payload), 88, 11)
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash-lite"
    assert kwargs["system_instruction"] == "reader"
    assert kwargs["input"] == "payload"
    assert kwargs["store"] is False
    assert kwargs["response_format"]["mime_type"] == "application/json"
    assert kwargs["response_format"]["schema"]["additionalProperties"] is False


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def complete_json(self, *, contract, system, user):
        self.calls.append((contract, system, user))
        if isinstance(self.payload, Exception):
            raise self.payload
        raw_output = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload)
        )
        return ProviderResult(
            raw_output=raw_output,
            input_tokens=100,
            output_tokens=20,
        )


@pytest.mark.asyncio
async def test_provider_is_injected_and_invoked_once_without_sdk_construction():
    payload = {
        "answer": "Because it required a freeze.",
        "cited_evidence_ids": ["turn-red"],
        "incomplete_evidence": False,
    }
    provider = FakeProvider(payload)
    system, user = render_reader_prompt(_scenario())
    parsed, raw = await invoke_structured(
        provider=provider,
        contract=CONTRACT,
        system=system,
        user=user,
        parser=ReaderOutput.parse,
    )
    assert parsed == ReaderOutput(
        "Because it required a freeze.", ("turn-red",), False,
    )
    assert raw == payload
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_two_stage_run_persists_usage_and_resumes_without_new_calls(tmp_path):
    reader_payload = {
        "answer": "Because it required a freeze.",
        "cited_evidence_ids": ["turn-red"],
        "incomplete_evidence": False,
    }
    reader_provider = FakeProvider(reader_payload)
    judge_provider = FakeProvider(_judge_payload())
    path = tmp_path / "attempts.jsonl"

    first = await run_scenario_once(
        path=path,
        scenario=_scenario(),
        reader_provider=reader_provider,
        reader_contract=CONTRACT,
        judge_provider=judge_provider,
        judge_contract=CONTRACT,
    )
    assert first[0].answer.startswith("Because")
    assert first[1].answer_correct is True
    rows = load_attempts(path)
    assert [row.stage for row in rows] == ["reader", "judge"]
    assert all(row.status == "success" for row in rows)
    assert all((row.input_tokens, row.output_tokens) == (100, 20) for row in rows)

    second = await run_scenario_once(
        path=path,
        scenario=_scenario(),
        reader_provider=reader_provider,
        reader_contract=CONTRACT,
        judge_provider=judge_provider,
        judge_contract=CONTRACT,
    )
    assert second == first
    assert len(reader_provider.calls) == len(judge_provider.calls) == 1
    assert len(load_attempts(path)) == 2


@pytest.mark.asyncio
async def test_malformed_reader_is_recorded_and_blocks_judge(tmp_path):
    reader_provider = FakeProvider({"answer": "missing required fields"})
    judge_provider = FakeProvider(_judge_payload())
    path = tmp_path / "attempts.jsonl"

    with pytest.raises(ValueError, match="unexpected fields"):
        await run_scenario_once(
            path=path,
            scenario=_scenario(),
            reader_provider=reader_provider,
            reader_contract=CONTRACT,
            judge_provider=judge_provider,
            judge_contract=CONTRACT,
        )

    rows = load_attempts(path)
    assert len(rows) == 1
    assert rows[0].stage == "reader"
    assert rows[0].status == "malformed"
    assert rows[0].input_tokens == 100
    assert rows[0].raw_output == json.dumps({"answer": "missing required fields"})
    assert judge_provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "google"])
async def test_real_adapter_malformed_json_retains_billed_usage_and_raw_output(
    tmp_path, provider_name,
):
    reader_contract, judge_contract = load_provider_contracts()
    malformed = '{"answer": "truncated"'
    if provider_name == "anthropic":
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text=malformed)],
            usage=SimpleNamespace(input_tokens=123, output_tokens=9),
        )
        create = AsyncMock(return_value=response)
        provider = AnthropicStructuredProvider(
            SimpleNamespace(messages=SimpleNamespace(create=create))
        )
        contract = judge_contract
    else:
        interaction = SimpleNamespace(
            output_text=malformed,
            usage=SimpleNamespace(input_tokens=88, output_tokens=7),
        )
        create = AsyncMock(return_value=interaction)
        provider = GoogleStructuredProvider(SimpleNamespace(
            aio=SimpleNamespace(interactions=SimpleNamespace(create=create)),
        ))
        contract = reader_contract

    path = tmp_path / "attempts.jsonl"
    with pytest.raises(MalformedProviderOutput):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
        )

    row = load_attempts(path)[0]
    assert row.status == "malformed"
    assert row.raw_output == malformed
    assert (row.input_tokens, row.output_tokens) == (
        (123, 9) if provider_name == "anthropic" else (88, 7)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "google"])
@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected"),
    [
        ("not-an-int", 7, (None, 7)),
        (44, "not-an-int", (44, None)),
    ],
)
async def test_real_adapter_invalid_usage_retains_raw_and_valid_sibling_count(
    tmp_path, provider_name, input_tokens, output_tokens, expected,
):
    reader_contract, judge_contract = load_provider_contracts()
    raw = json.dumps(_judge_payload())
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if provider_name == "anthropic":
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text=raw)],
            usage=usage,
        )
        provider = AnthropicStructuredProvider(SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        ))
        contract = judge_contract
    else:
        interaction = SimpleNamespace(output_text=raw, usage=usage)
        provider = GoogleStructuredProvider(SimpleNamespace(
            aio=SimpleNamespace(interactions=SimpleNamespace(
                create=AsyncMock(return_value=interaction),
            )),
        ))
        contract = reader_contract

    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ProviderResponseError, match="extraction failed"):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
        )
    row = load_attempts(path)[0]
    assert row.status == "failed"
    assert row.raw_output == raw
    assert (row.input_tokens, row.output_tokens) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "google"])
@pytest.mark.parametrize("content_case", ["missing_text", "malformed_container"])
async def test_real_adapter_content_shape_failure_retains_valid_usage(
    tmp_path, provider_name, content_case,
):
    reader_contract, judge_contract = load_provider_contracts()
    usage = SimpleNamespace(input_tokens=31, output_tokens=4)
    if provider_name == "anthropic":
        content = (
            [SimpleNamespace(not_text="missing")]
            if content_case == "missing_text"
            else {"text": "wrong-container"}
        )
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=content,
            usage=usage,
        )
        provider = AnthropicStructuredProvider(SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        ))
        contract = judge_contract
    else:
        interaction_kwargs = {"usage": usage}
        if content_case == "missing_text":
            interaction_kwargs["not_output_text"] = "missing"
        else:
            interaction_kwargs["output_text"] = {"wrong": "shape"}
        interaction = SimpleNamespace(**interaction_kwargs)
        provider = GoogleStructuredProvider(SimpleNamespace(
            aio=SimpleNamespace(interactions=SimpleNamespace(
                create=AsyncMock(return_value=interaction),
            )),
        ))
        contract = reader_contract

    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ProviderResponseError, match="extraction failed"):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
        )
    row = load_attempts(path)[0]
    assert row.status == "failed"
    assert row.raw_output is None
    assert (row.input_tokens, row.output_tokens) == (31, 4)


@pytest.mark.asyncio
async def test_typed_provider_failure_retains_observed_usage_and_raw_output(tmp_path):
    provider = FakeProvider(ProviderResponseError(
        "response extraction failed",
        raw_output="provider-visible-text",
        input_tokens=44,
        output_tokens=5,
    ))
    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ProviderResponseError):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=CONTRACT,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
        )
    row = load_attempts(path)[0]
    assert row.status == "failed"
    assert row.raw_output == "provider-visible-text"
    assert (row.input_tokens, row.output_tokens) == (44, 5)
