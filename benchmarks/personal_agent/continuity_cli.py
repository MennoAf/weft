"""Zero-call estimator and explicitly approved paid continuity runner.

Estimate mode is pure: it renders the fixed corpus and pricing projection without
constructing provider SDK clients. Paid mode validates an exact approval phrase,
positive cost ceiling, immutable run manifest, and retry policy before constructing
providers. Every provider attempt is re-authorized inside ``run_stage_once``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

from benchmarks.personal_agent.continuity_manifest import (
    CONTINUITY_MANIFEST_ID,
    SESSIONS,
)
from benchmarks.personal_agent.continuity_runner import (
    AnthropicStructuredProvider,
    CostCeilingExceeded,
    DECISION_RULE,
    GoogleStructuredProvider,
    MalformedProviderOutput,
    ProviderResponseError,
    ReaderOutput,
    RunManifest,
    ScenarioInput,
    SpendGuard,
    build_scenarios,
    load_attempts,
    load_provider_contracts,
    load_run_manifest,
    paired_arm_decision,
    projected_call_cost,
    prompt_sha256,
    render_judge_prompt,
    render_reader_prompt,
    run_scenario_once,
    stable_call_id,
    successful_outputs,
    summarize_judgments,
    utc_now,
    write_run_manifest,
)

DEFAULT_ARMS = ("A", "B")
DEFAULT_REPETITIONS = 3
DEFAULT_RETRIES = 1
BENCHMARK_CONTENT_PATHS = (
    Path(__file__),
    Path(__file__).with_name("continuity_runner.py"),
    Path(__file__).with_name("continuity_manifest.py"),
    Path(__file__).with_name("continuity_provider_contracts.json"),
)
ESTIMATE_METHOD = {
    "input_token_bound": "one_token_per_utf8_byte",
    "reader_output_tokens": "provider_contract_max_output_tokens",
    "judge_candidate_bytes": "reader_max_output_tokens_times_4_ascii_bytes",
    "judge_output_tokens": "provider_contract_max_output_tokens",
    "judge_prompt_kind": "conservative_pre_reader_envelope",
}


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _benchmark_content_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(BENCHMARK_CONTENT_PATHS, key=lambda item: item.name):
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path.resolve())
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _decision_rule_sha256() -> str:
    return _sha256_json(DECISION_RULE)


def _estimate_method_sha256() -> str:
    return _sha256_json(ESTIMATE_METHOD)


def _judge_envelope(
    scenario: ScenarioInput,
    *,
    reader_contract,
) -> tuple[str, str]:
    """Conservative deterministic judge envelope used only for estimation.

    The actual judge prompt depends on the paid reader response and cannot exist
    before that call. The synthetic candidate deliberately fills the configured
    reader output ceiling so the estimate cannot understate future judge input.
    """
    candidate = ReaderOutput(
        answer="X" * (reader_contract.max_output_tokens * 4),
        cited_evidence_ids=tuple(scenario.gold["required_evidence_ids"]),
        incomplete_evidence=False,
    )
    return render_judge_prompt(scenario, candidate)


def build_estimate(*, retries: int = DEFAULT_RETRIES) -> dict:
    """Build a deterministic, provider-free cost and prompt artifact."""
    if retries < 0:
        raise ValueError("retries must be >= 0")
    reader_contract, judge_contract = load_provider_contracts()
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=DEFAULT_ARMS,
        repetitions=DEFAULT_REPETITIONS,
    )
    reader_prompts = []
    judge_envelopes = []
    reader_cost = Decimal(0)
    judge_cost = Decimal(0)
    for scenario in scenarios:
        reader_system, reader_user = render_reader_prompt(scenario)
        judge_system, judge_user = _judge_envelope(
            scenario,
            reader_contract=reader_contract,
        )
        reader_cost += projected_call_cost(
            contract=reader_contract,
            system=reader_system,
            user=reader_user,
        )
        judge_cost += projected_call_cost(
            contract=judge_contract,
            system=judge_system,
            user=judge_user,
        )
        reader_prompts.append({
            "call_id": stable_call_id(
                manifest_id=scenario.manifest_id,
                scenario_id=scenario.scenario_id,
                arm=scenario.arm,
                repetition=scenario.repetition,
                stage="reader",
            ),
            "scenario_id": scenario.scenario_id,
            "arm": scenario.arm,
            "repetition": scenario.repetition,
            "system": reader_system,
            "user": reader_user,
            "prompt_sha256": prompt_sha256(reader_system, reader_user),
        })
        judge_envelopes.append({
            "call_id": stable_call_id(
                manifest_id=scenario.manifest_id,
                scenario_id=scenario.scenario_id,
                arm=scenario.arm,
                repetition=scenario.repetition,
                stage="judge",
            ),
            "scenario_id": scenario.scenario_id,
            "arm": scenario.arm,
            "repetition": scenario.repetition,
            "system": judge_system,
            "user": judge_user,
            "prompt_sha256": prompt_sha256(judge_system, judge_user),
            "kind": "conservative_pre_reader_envelope",
        })
    expected = reader_cost + judge_cost
    attempts_per_call = retries + 1
    scenario_payload = [asdict(scenario) for scenario in scenarios]
    protocol_payload = {
        "manifest_id": CONTINUITY_MANIFEST_ID,
        "arms": list(DEFAULT_ARMS),
        "repetitions": DEFAULT_REPETITIONS,
        "retries": retries,
        "scenarios": scenario_payload,
        "reader_prompt_hashes": [item["prompt_sha256"] for item in reader_prompts],
        "judge_envelope_hashes": [item["prompt_sha256"] for item in judge_envelopes],
        "reader_contract": asdict(reader_contract),
        "judge_contract": asdict(judge_contract),
        "decision_rule": DECISION_RULE,
        "estimate_method": ESTIMATE_METHOD,
        "benchmark_content_sha256": _benchmark_content_sha256(),
    }
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": "estimate-only-zero-call",
        "provider_clients_constructed": False,
        "network_calls": 0,
        "manifest_id": CONTINUITY_MANIFEST_ID,
        "arms": list(DEFAULT_ARMS),
        "repetitions": DEFAULT_REPETITIONS,
        "scenario_count": len(SESSIONS) * 7,
        "expected_reader_calls": len(scenarios),
        "expected_judge_calls": len(scenarios),
        "retries": retries,
        "attempts_per_call_worst_case": attempts_per_call,
        "reader_contract": asdict(reader_contract),
        "judge_contract": asdict(judge_contract),
        "expected_one_attempt_cost_usd": str(expected),
        "retry_inclusive_worst_case_cost_usd": str(expected * attempts_per_call),
        "cost_method": (
            "exact rendered reader prompts; conservative pre-reader judge envelopes "
            "with candidate answer bytes = reader max_output_tokens * 4; configured "
            "maximum output tokens charged for every attempt"
        ),
        "reader_prompts": reader_prompts,
        "judge_prompt_envelopes": judge_envelopes,
        "decision_rule": dict(DECISION_RULE),
        "decision_rule_sha256": _decision_rule_sha256(),
        "estimate_method": dict(ESTIMATE_METHOD),
        "estimate_method_sha256": _estimate_method_sha256(),
        "protocol_sha256": _sha256_json(protocol_payload),
        "benchmark_content_sha256": protocol_payload["benchmark_content_sha256"],
        "production_wiring_enabled": False,
        "materializer_automatic": False,
        "status": "PENDING-PAID-EVALUATION",
    }


def write_estimate(path: Path, *, retries: int = DEFAULT_RETRIES) -> dict:
    estimate = build_estimate(retries=retries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return estimate


def _parse_positive_decimal(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("max_cost_usd must be a positive finite amount") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("max_cost_usd must be a positive finite amount")
    return amount


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_manifest(
    *,
    run_id: str,
    attempts_file: str,
    retries: int,
    estimate: dict,
) -> RunManifest:
    reader, judge = load_provider_contracts()
    scenario_count = sum(len(session.questions) for session in SESSIONS)
    expected = scenario_count * len(DEFAULT_ARMS) * DEFAULT_REPETITIONS
    return RunManifest(
        schema_version=1,
        run_id=run_id,
        generated_at=utc_now(),
        code_commit=_git_commit(),
        manifest_id=CONTINUITY_MANIFEST_ID,
        arms=DEFAULT_ARMS,
        repetitions=DEFAULT_REPETITIONS,
        scenario_count=scenario_count,
        expected_reader_calls=expected,
        expected_judge_calls=expected,
        reader_contract=reader,
        judge_contract=judge,
        attempts_file=attempts_file,
        retries=retries,
        decision_rule_sha256=estimate["decision_rule_sha256"],
        protocol_sha256=estimate["protocol_sha256"],
        estimate_method_sha256=estimate["estimate_method_sha256"],
        benchmark_content_sha256=estimate["benchmark_content_sha256"],
        status="PENDING-PAID-EVALUATION",
        production_wiring_enabled=False,
        materializer_automatic=False,
    )


def validate_paid_authority(
    *, approval: str, max_cost_usd: str, retries: int,
) -> SpendGuard:
    """Validate all authority fields before any provider factory is called."""
    if retries < 0:
        raise ValueError("retries must be >= 0")
    reader, judge = load_provider_contracts()
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=DEFAULT_ARMS,
        repetitions=DEFAULT_REPETITIONS,
    )
    reader_costs = []
    judge_costs = []
    for scenario in scenarios:
        reader_system, reader_user = render_reader_prompt(scenario)
        judge_system, judge_user = _judge_envelope(
            scenario,
            reader_contract=reader,
        )
        reader_costs.append(projected_call_cost(
            contract=reader,
            system=reader_system,
            user=reader_user,
        ))
        judge_costs.append(projected_call_cost(
            contract=judge,
            system=judge_system,
            user=judge_user,
        ))
    return SpendGuard(
        approval=approval,
        max_cost_usd=_parse_positive_decimal(max_cost_usd),
        contracts=(reader, judge),
        fallback_costs=(
            (reader.provider, reader.model, max(reader_costs)),
            (judge.provider, judge.model, max(judge_costs)),
        ),
    )


def _assert_protocol_current(
    *,
    manifest: RunManifest,
    retries: int,
) -> dict:
    current = build_estimate(retries=retries)
    expected = {
        "decision_rule_sha256": manifest.decision_rule_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "estimate_method_sha256": manifest.estimate_method_sha256,
        "benchmark_content_sha256": manifest.benchmark_content_sha256,
    }
    actual = {key: current[key] for key in expected}
    drift = [key for key in expected if actual[key] != expected[key]]
    if drift:
        raise ValueError(f"benchmark protocol drift: {drift}")
    return current


async def run_paid(
    *,
    run_dir: Path,
    approval: str,
    max_cost_usd: str,
    retries: int = DEFAULT_RETRIES,
    reader_factory: Callable[[], object] = GoogleStructuredProvider,
    judge_factory: Callable[[], object] = AnthropicStructuredProvider,
) -> dict:
    """Execute the fixed A/B run only after immutable authority validation."""
    guard = validate_paid_authority(
        approval=approval,
        max_cost_usd=max_cost_usd,
        retries=retries,
    )
    estimate = build_estimate(retries=retries)
    if Decimal(estimate["retry_inclusive_worst_case_cost_usd"]) > guard.max_cost_usd:
        raise RuntimeError(
            "max_cost_usd is below the retry-inclusive preflight estimate"
        )

    attempts_path = run_dir / "attempts.jsonl"
    manifest_path = run_dir / "manifest.json"
    run_id = run_dir.name
    if manifest_path.exists():
        manifest = load_run_manifest(manifest_path)
        expected = _run_manifest(
            run_id=run_id,
            attempts_file=attempts_path.name,
            retries=retries,
            estimate=estimate,
        )
        fixed_fields = (
            "run_id", "code_commit", "manifest_id", "arms", "repetitions",
            "scenario_count", "expected_reader_calls", "expected_judge_calls",
            "reader_contract", "judge_contract", "attempts_file",
            "retries", "decision_rule_sha256", "protocol_sha256",
            "estimate_method_sha256", "benchmark_content_sha256",
            "production_wiring_enabled",
            "materializer_automatic",
        )
        drift = [
            field for field in fixed_fields
            if getattr(manifest, field) != getattr(expected, field)
        ]
        if drift:
            raise ValueError(f"existing run manifest drift: {drift}")
    else:
        manifest = _run_manifest(
            run_id=run_id,
            attempts_file=attempts_path.name,
            retries=retries,
            estimate=estimate,
        )
        write_run_manifest(manifest_path, manifest)

    _assert_protocol_current(manifest=manifest, retries=retries)

    # Provider construction is deliberately after approval, positive ceiling,
    # retry-inclusive preflight, and immutable manifest validation.
    reader_provider = reader_factory()
    judge_provider = judge_factory()
    reader_contract, judge_contract = load_provider_contracts()
    scenarios = build_scenarios(
        sessions=SESSIONS,
        arms=DEFAULT_ARMS,
        repetitions=DEFAULT_REPETITIONS,
    )
    failures: list[dict] = []
    for scenario in scenarios:
        for attempt_index in range(retries + 1):
            try:
                await run_scenario_once(
                    path=attempts_path,
                    scenario=scenario,
                    reader_provider=reader_provider,
                    reader_contract=reader_contract,
                    judge_provider=judge_provider,
                    judge_contract=judge_contract,
                    spend_guard=guard,
                )
                break
            except CostCeilingExceeded:
                raise
            except (ProviderResponseError, MalformedProviderOutput, TimeoutError) as exc:
                if attempt_index == retries:
                    failures.append({
                        "scenario_id": scenario.scenario_id,
                        "arm": scenario.arm,
                        "repetition": scenario.repetition,
                        "error_type": type(exc).__name__,
                    })
                else:
                    continue

    _assert_protocol_current(manifest=manifest, retries=retries)
    judge_records = successful_outputs(attempts_path, stage="judge")
    summary = summarize_judgments(
        scenarios=scenarios,
        judge_records=judge_records,
    )
    decision = paired_arm_decision(
        scenarios=scenarios,
        judge_records=judge_records,
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_id": run_id,
        "attempt_count": len(load_attempts(attempts_path)),
        "failures": failures,
        "summary": summary,
        "decision": decision,
        "decision_rule_sha256": manifest.decision_rule_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "estimate_method_sha256": manifest.estimate_method_sha256,
        "benchmark_content_sha256": manifest.benchmark_content_sha256,
        "retries": manifest.retries,
        "production_wiring_enabled": False,
        "materializer_automatic": False,
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate = subparsers.add_parser("estimate", help="zero-call cost/prompt estimate")
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    paid = subparsers.add_parser("run", help="explicitly approved paid run")
    paid.add_argument("--run-dir", type=Path, required=True)
    paid.add_argument("--approval", required=True)
    paid.add_argument("--max-cost-usd", required=True)
    paid.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "estimate":
            estimate = write_estimate(args.output, retries=args.retries)
            print(json.dumps({
                "output": str(args.output),
                "expected_one_attempt_cost_usd": estimate["expected_one_attempt_cost_usd"],
                "retry_inclusive_worst_case_cost_usd": estimate[
                    "retry_inclusive_worst_case_cost_usd"
                ],
                "provider_clients_constructed": False,
                "network_calls": 0,
            }, indent=2, sort_keys=True))
            return 0
        asyncio.run(run_paid(
            run_dir=args.run_dir,
            approval=args.approval,
            max_cost_usd=args.max_cost_usd,
            retries=args.retries,
        ))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
