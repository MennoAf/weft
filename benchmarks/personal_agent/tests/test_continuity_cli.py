"""Zero-call and hard-ceiling tests for the continuity paid-run boundary."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json

import pytest

from benchmarks.personal_agent import continuity_cli, continuity_runner
from benchmarks.personal_agent.continuity_cli import (
    _assert_protocol_current,
    _run_manifest,
    build_estimate,
    main,
    run_paid,
    validate_paid_authority,
    write_estimate,
)
from benchmarks.personal_agent.continuity_runner import (
    AttemptRecord,
    CostCeilingExceeded,
    DECISION_RULE,
    PAID_APPROVAL_PHRASE,
    ProviderContract,
    ProviderResult,
    ReaderOutput,
    SpendGuard,
    append_attempt,
    load_provider_contracts,
    estimate_text_tokens,
    projected_call_cost,
    run_stage_once,
)
from benchmarks.personal_agent.tests.test_continuity_runner import (
    CONTRACT,
    _scenario,
)


def test_estimate_renders_fixed_population_without_provider_construction(tmp_path):
    output = tmp_path / "estimate.json"
    estimate = write_estimate(output, retries=1)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == estimate
    assert estimate["mode"] == "estimate-only-zero-call"
    assert estimate["provider_clients_constructed"] is False
    assert estimate["network_calls"] == 0
    assert estimate["expected_reader_calls"] == 168
    assert estimate["expected_judge_calls"] == 168
    assert len(estimate["reader_prompts"]) == 168
    assert len(estimate["judge_prompt_envelopes"]) == 168
    assert len({item["call_id"] for item in estimate["reader_prompts"]}) == 168
    assert len({item["call_id"] for item in estimate["judge_prompt_envelopes"]}) == 168
    assert "maximum output tokens charged" in estimate["cost_method"]
    assert all(item["system"] and item["user"] for item in estimate["reader_prompts"])
    assert all(
        item["kind"] == "conservative_pre_reader_envelope"
        for item in estimate["judge_prompt_envelopes"]
    )
    assert estimate["production_wiring_enabled"] is False
    assert estimate["materializer_automatic"] is False


def test_estimate_cli_reports_zero_calls(tmp_path, capsys):
    output = tmp_path / "estimate.json"
    assert main(["estimate", "--output", str(output), "--retries", "0"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["provider_clients_constructed"] is False
    assert rendered["network_calls"] == 0
    assert output.exists()


def test_token_estimate_is_fail_safe_utf8_byte_bound():
    assert estimate_text_tokens("abcd") == 4
    assert estimate_text_tokens("é") == 2
    assert estimate_text_tokens("") == 1


def test_retry_inclusive_worst_case_scales_from_one_attempt():
    no_retry = build_estimate(retries=0)
    two_retries = build_estimate(retries=2)
    one = Decimal(no_retry["expected_one_attempt_cost_usd"])
    assert Decimal(no_retry["retry_inclusive_worst_case_cost_usd"]) == one
    assert Decimal(two_retries["retry_inclusive_worst_case_cost_usd"]) == one * 3
    assert one > 0


def test_estimate_pins_precommitted_decision_rule_hash():
    estimate = build_estimate()
    expected_hash = hashlib.sha256(
        json.dumps(
            DECISION_RULE,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert estimate["decision_rule"] == DECISION_RULE
    assert estimate["decision_rule_sha256"] == expected_hash


def test_protocol_hash_changes_with_retry_or_scenario_content(monkeypatch):
    baseline = build_estimate(retries=0)
    assert (
        build_estimate(retries=1)["protocol_sha256"]
        != baseline["protocol_sha256"]
    )
    original_builder = continuity_cli.build_scenarios

    def changed_builder(**kwargs):
        scenarios = original_builder(**kwargs)
        return (replace(scenarios[0], query=scenarios[0].query + " changed"), *scenarios[1:])

    monkeypatch.setattr(continuity_cli, "build_scenarios", changed_builder)
    changed = build_estimate(retries=0)
    assert changed["protocol_sha256"] != baseline["protocol_sha256"]


def test_protocol_revalidation_rejects_content_byte_drift(tmp_path, monkeypatch):
    content = tmp_path / "protocol.py"
    content.write_text("version one", encoding="utf-8")
    monkeypatch.setattr(continuity_cli, "BENCHMARK_CONTENT_PATHS", (content,))
    estimate = build_estimate(retries=0)
    manifest = _run_manifest(
        run_id="test-run",
        attempts_file="attempts.jsonl",
        retries=0,
        estimate=estimate,
    )
    content.write_text("version two", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark protocol drift"):
        _assert_protocol_current(manifest=manifest, retries=0)


def test_protocol_revalidation_rejects_decision_rule_drift(monkeypatch):
    estimate = build_estimate(retries=0)
    manifest = _run_manifest(
        run_id="test-run",
        attempts_file="attempts.jsonl",
        retries=0,
        estimate=estimate,
    )
    monkeypatch.setitem(DECISION_RULE, "minimum_core_episodic_wins", 999)
    with pytest.raises(ValueError, match="benchmark protocol drift"):
        _assert_protocol_current(manifest=manifest, retries=0)


@pytest.mark.parametrize("approval", ["", "yes", "I approve"])
def test_paid_authority_requires_exact_phrase(approval):
    with pytest.raises(ValueError, match="approval must exactly equal"):
        validate_paid_authority(
            approval=approval,
            max_cost_usd="1.00",
            retries=0,
        )


@pytest.mark.parametrize("ceiling", ["0", "-1", "NaN", "Infinity", "not-money"])
def test_paid_authority_requires_positive_finite_ceiling(ceiling):
    with pytest.raises(ValueError, match="positive finite"):
        validate_paid_authority(
            approval=PAID_APPROVAL_PHRASE,
            max_cost_usd=ceiling,
            retries=0,
        )


@pytest.mark.asyncio
async def test_below_estimate_refuses_before_provider_factories(tmp_path):
    constructions = []

    def factory():
        constructions.append(True)
        raise AssertionError("provider factory must not run")

    with pytest.raises(RuntimeError, match="below the retry-inclusive"):
        await run_paid(
            run_dir=tmp_path / "run",
            approval=PAID_APPROVAL_PHRASE,
            max_cost_usd="0.000001",
            retries=1,
            reader_factory=factory,
            judge_factory=factory,
        )
    assert constructions == []
    assert not (tmp_path / "run" / "manifest.json").exists()


class CountingProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete_json(self, **_kwargs):
        self.calls += 1
        return ProviderResult(json.dumps(self.payload), 10, 5)


@pytest.mark.asyncio
async def test_stage_ceiling_checked_before_provider_invocation(tmp_path):
    provider = CountingProvider({
        "answer": "ok",
        "cited_evidence_ids": [],
        "incomplete_evidence": False,
    })
    contract = ProviderContract(**asdict(CONTRACT))
    system = "system"
    user = "user"
    projected = projected_call_cost(
        contract=contract,
        system=system,
        user=user,
    )
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=projected - Decimal("0.0000000001"),
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, projected),),
    )
    with pytest.raises(CostCeilingExceeded):
        await run_stage_once(
            path=tmp_path / "attempts.jsonl",
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system=system,
            user=user,
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_each_retry_rechecks_ceiling_before_provider_invocation(tmp_path):
    provider = CountingProvider({"answer": "missing required fields"})
    contract = ProviderContract(**asdict(CONTRACT))
    system = "system"
    user = "user"
    projected = projected_call_cost(
        contract=contract,
        system=system,
        user=user,
    )
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=projected + Decimal("0.000001"),
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, projected),),
    )
    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ValueError):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system=system,
            user=user,
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )
    assert provider.calls == 1
    with pytest.raises(CostCeilingExceeded):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system=system,
            user=user,
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_concurrent_workers_allow_only_one_provider_call_under_one_call_ceiling(
    tmp_path,
):
    payload = {
        "answer": "ok",
        "cited_evidence_ids": [],
        "incomplete_evidence": False,
    }
    provider = CountingProvider(payload)
    contract = ProviderContract(**asdict(CONTRACT))
    system = "system"
    user = "user"
    fallback = projected_call_cost(
        contract=contract,
        system=system,
        user=user,
    )
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=fallback,
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, fallback),),
    )
    path = tmp_path / "attempts.jsonl"

    async def worker(repetition):
        scenario = replace(_scenario(), repetition=repetition)
        return await run_stage_once(
            path=path,
            scenario=scenario,
            stage="reader",
            provider=provider,
            contract=contract,
            system=system,
            user=user,
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )

    results = await asyncio.gather(worker(1), worker(2), return_exceptions=True)
    assert provider.calls == 1
    assert sum(isinstance(item, CostCeilingExceeded) for item in results) == 1
    assert sum(isinstance(item, ReaderOutput) for item in results) == 1


def test_unresolved_crash_reservation_remains_charged_on_resume(tmp_path):
    contract = ProviderContract(**asdict(CONTRACT))
    fallback = Decimal("0.02")
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=fallback,
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, fallback),),
    )
    path = tmp_path / "attempts.jsonl"
    attempt, reservation_id = guard.reserve_next(
        attempts_path=path,
        call_id="crashed-call",
        contract=contract,
        system="system",
        user="user",
    )
    assert attempt == 1
    assert reservation_id == "crashed-call:1"
    assert not path.exists()
    with pytest.raises(CostCeilingExceeded):
        guard.reserve_next(
            attempts_path=path,
            call_id="resume-call",
            contract=contract,
            system="system",
            user="user",
        )


@pytest.mark.asyncio
async def test_append_failure_leaves_reservation_charged(tmp_path, monkeypatch):
    provider = CountingProvider({
        "answer": "ok",
        "cited_evidence_ids": [],
        "incomplete_evidence": False,
    })
    contract = ProviderContract(**asdict(CONTRACT))
    fallback = projected_call_cost(
        contract=contract,
        system="system",
        user="user",
    )
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=fallback,
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, fallback),),
    )
    path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(
        continuity_runner,
        "_append_reserved_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fsync failed")),
    )
    with pytest.raises(OSError, match="fsync failed"):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )
    assert provider.calls == 1
    with pytest.raises(CostCeilingExceeded):
        guard.reserve_next(
            attempts_path=path,
            call_id="after-fsync-failure",
            contract=contract,
            system="system",
            user="user",
        )


@pytest.mark.asyncio
async def test_post_append_ledger_failure_reconciles_once_on_resume(
    tmp_path, monkeypatch,
):
    provider = CountingProvider({
        "answer": "ok",
        "cited_evidence_ids": [],
        "incomplete_evidence": False,
    })
    contract = ProviderContract(**asdict(CONTRACT))
    fallback = projected_call_cost(
        contract=contract,
        system="system",
        user="user",
    )
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=fallback * 2,
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, fallback),),
    )
    path = tmp_path / "attempts.jsonl"
    real_write = continuity_runner._write_spend_ledger
    calls = 0

    def fail_second_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("ledger fsync failed")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(continuity_runner, "_write_spend_ledger", fail_second_write)
    with pytest.raises(OSError, match="ledger fsync failed"):
        await run_stage_once(
            path=path,
            scenario=_scenario(),
            stage="reader",
            provider=provider,
            contract=contract,
            system="system",
            user="user",
            parser=ReaderOutput.parse,
            spend_guard=guard,
        )
    assert provider.calls == 1
    monkeypatch.setattr(continuity_runner, "_write_spend_ledger", real_write)
    attempt, reservation_id = guard.reserve_next(
        attempts_path=path,
        call_id="second-call",
        contract=contract,
        system="system",
        user="user",
    )
    assert attempt == 1
    assert reservation_id == "second-call:1"
    ledger = continuity_runner._load_spend_ledger(path)
    scenario = _scenario()
    first_call_id = continuity_runner.stable_call_id(
        manifest_id=scenario.manifest_id,
        scenario_id=scenario.scenario_id,
        arm=scenario.arm,
        repetition=scenario.repetition,
        stage="reader",
    )
    assert ledger["reservations"][f"{first_call_id}:1"]["state"] == "reconciled"
    with pytest.raises(CostCeilingExceeded):
        guard.reserve_next(
            attempts_path=path,
            call_id="third-call",
            contract=contract,
            system="system",
            user="user",
        )


def test_reconciliation_rejects_attempt_contract_mismatch(tmp_path):
    contract = ProviderContract(**asdict(CONTRACT))
    fallback = Decimal("0.02")
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=Decimal("1"),
        contracts=(contract,),
        fallback_costs=((contract.provider, contract.model, fallback),),
    )
    path = tmp_path / "attempts.jsonl"
    guard.reserve_next(
        attempts_path=path,
        call_id="corrupt-call",
        contract=contract,
        system="system",
        user="user",
    )
    append_attempt(path, AttemptRecord(
        schema_version=1,
        call_id="corrupt-call",
        stage="reader",
        attempt=1,
        status="failed",
        manifest_id="continuity-v2",
        session_id="launch-plan",
        scenario_id="launch-plan:rejected_rationale",
        arm="B",
        repetition=1,
        provider="wrong-provider",
        model="wrong-model",
        prompt_sha256="abc",
        started_at="2026-07-20T00:00:00+00:00",
        elapsed_ms=1,
        input_tokens=None,
        output_tokens=None,
        raw_output=None,
        output=None,
        error_type="TimeoutError",
    ))
    with pytest.raises(ValueError, match="contract mismatch"):
        guard.reserve_next(
            attempts_path=path,
            call_id="next-call",
            contract=contract,
            system="system",
            user="user",
        )


def _attempt(*, provider, model, input_tokens, output_tokens, attempt=1):
    return AttemptRecord(
        schema_version=1,
        call_id=f"call-{provider}-{attempt}",
        stage="reader",
        attempt=attempt,
        status="failed",
        manifest_id="continuity-v2",
        session_id="launch-plan",
        scenario_id="launch-plan:rejected_rationale",
        arm="B",
        repetition=1,
        provider=provider,
        model=model,
        prompt_sha256="abc",
        started_at="2026-07-20T00:00:00+00:00",
        elapsed_ms=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_output=None,
        output=None,
        error_type="TimeoutError",
    )


def test_global_ceiling_counts_both_providers_and_unknown_usage(tmp_path):
    reader, judge = load_provider_contracts()
    path = tmp_path / "attempts.jsonl"
    append_attempt(path, _attempt(
        provider=reader.provider,
        model=reader.model,
        input_tokens=100,
        output_tokens=10,
    ))
    append_attempt(path, _attempt(
        provider=judge.provider,
        model=judge.model,
        input_tokens=None,
        output_tokens=None,
    ))
    fallback_reader = Decimal("0.01")
    fallback_judge = Decimal("0.02")
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=Decimal("0.0201"),
        contracts=(reader, judge),
        fallback_costs=(
            (reader.provider, reader.model, fallback_reader),
            (judge.provider, judge.model, fallback_judge),
        ),
    )
    with pytest.raises(CostCeilingExceeded, match="spent_or_reserved="):
        guard.reserve_next(
            attempts_path=path,
            call_id="next-reader-call",
            contract=reader,
            system="system",
            user="user",
        )


@pytest.mark.asyncio
async def test_fake_paid_run_writes_complete_artifacts_and_stays_disabled(
    tmp_path, monkeypatch,
):
    scenarios = tuple(
        replace(
            _scenario(arm=arm),
            gold={
                **_scenario(arm=arm).gold,
                "handoff_sufficient": False,
            },
        )
        for arm in ("A", "B")
    )
    monkeypatch.setattr(continuity_cli, "build_scenarios", lambda **_kwargs: scenarios)
    estimate = build_estimate(retries=0)
    monkeypatch.setattr(
        continuity_cli,
        "build_estimate",
        lambda **_kwargs: {
            **estimate,
            "retry_inclusive_worst_case_cost_usd": "0.01",
        },
    )
    monkeypatch.setattr(continuity_cli, "_git_commit", lambda: "test-commit")

    reader_payload = {
        "answer": "Red required a risky schema freeze.",
        "cited_evidence_ids": ["turn-red"],
        "incomplete_evidence": False,
    }
    judge_payload = {
        "answer_correct": True,
        "instruction_non_compliant": False,
        "unsupported_claim": False,
        "evidence_citation_correct": True,
        "stale_or_superseded": False,
        "rationale": "Matches gold.",
    }
    reader = CountingProvider(reader_payload)
    judge = CountingProvider(judge_payload)
    run_dir = tmp_path / "fake-run"
    result = await run_paid(
        run_dir=run_dir,
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd="1.00",
        retries=0,
        reader_factory=lambda: reader,
        judge_factory=lambda: judge,
    )
    assert result["status"] == "COMPLETE"
    assert result["decision"]["status"] == "HOLD"
    assert result["summary"]["complete"] is True
    assert result["production_wiring_enabled"] is False
    assert result["materializer_automatic"] is False
    assert result["attempt_count"] == 4
    assert reader.calls == judge.calls == 2
    manifest = json.loads((run_dir / "manifest.json").read_text())
    persisted = json.loads((run_dir / "results.json").read_text())
    assert manifest["production_wiring_enabled"] is False
    assert persisted == result
    assert persisted["protocol_sha256"] == manifest["protocol_sha256"]
    assert persisted["decision_rule_sha256"] == manifest["decision_rule_sha256"]
    assert (
        persisted["estimate_method_sha256"]
        == manifest["estimate_method_sha256"]
    )
    assert (
        persisted["benchmark_content_sha256"]
        == manifest["benchmark_content_sha256"]
    )


def test_guard_rejects_attempt_from_unapproved_contract(tmp_path):
    reader, judge = load_provider_contracts()
    path = tmp_path / "attempts.jsonl"
    append_attempt(path, _attempt(
        provider="unknown-provider",
        model="unknown-model",
        input_tokens=1,
        output_tokens=1,
    ))
    guard = SpendGuard(
        approval=PAID_APPROVAL_PHRASE,
        max_cost_usd=Decimal("1"),
        contracts=(reader, judge),
        fallback_costs=(
            (reader.provider, reader.model, Decimal("0.01")),
            (judge.provider, judge.model, Decimal("0.02")),
        ),
    )
    with pytest.raises(ValueError, match="unapproved provider contract"):
        guard.reserve_next(
            attempts_path=path,
            call_id="next-reader-call",
            contract=reader,
            system="system",
            user="user",
        )
