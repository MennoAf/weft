"""Adversarial, provider-free RC-FL-20 preparation contract tests."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_rc_qualification.py"


@pytest.fixture
def prep():
    spec = importlib.util.spec_from_file_location("rc_qualification_prep", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone(value):
    return json.loads(json.dumps(value))


def recompute_controls_hash(manifest, prep):
    manifest["hashes"]["manifest_controls_sha256"] = prep._sha256_json(manifest["controls"])


def readiness(prep, **overrides):
    value = {
        "dataset_checksum": "a" * 64,
        "question_ids_and_order_digest": None,
        "question_count": prep.M_QUESTIONS,
        "ordered_question_ids": [f"q-{i}" for i in range(prep.M_QUESTIONS)],
        "ordered_question_ids_sha256": None,
        "blockers": [],
        "authorization": {
            "operator_id": "operator-1",
            "decision_id": "decision-rc-fl-20-1",
            "purpose": prep.QUALIFICATION_PURPOSE,
            "approved_scope": prep.QUALIFICATION_PURPOSE,
            "approved_arms": list(prep.CANONICAL_HOSTED_ARMS),
            "approved_max_spend_usd": 90.0,
            "currency": "USD",
            "authorized_at": "2026-05-04T12:00:00+00:00",
            "model_profile_ids": {
                arm["arm_id"]: {"model": arm["model"], "profile_id": arm["profile_id"]}
                for arm in prep.EXPECTED_ARMS[1:]
            },
            "record_sha256": None,
        },
        "prices": {
            name: {"amount": amount, "unit": "per 1M tokens", "currency": "USD", "source": "synthetic-test", "as_of": "2026-05-04T12:00:00+00:00"}
            for name, amount in (("embedding", 0.0), ("reader", 1.0), ("judge", 1.0))
        },
        "embedding_item_count": 1000,
        "environment_keys": [],
        "state": "READY",
        "execution_ready": True,
    }
    value["ordered_question_ids_sha256"] = prep._sha256_json(value["ordered_question_ids"])
    value["question_ids_and_order_digest"] = prep.question_ids_and_order_digest(value["dataset_checksum"], value["ordered_question_ids"])
    value["authorization"]["prices_sha256"] = prep.prices_digest(value["prices"])
    value["authorization"]["record_sha256"] = prep.authorization_record_digest(value["authorization"])
    value.update(overrides)
    return value


def test_generation_is_deterministic_and_boundary_is_exact(prep, tmp_path):
    outputs = [(tmp_path / "one.md", tmp_path / "one.json"), (tmp_path / "two.md", tmp_path / "two.json")]
    generated = [prep.generate(root=ROOT, output=md, budget_output=budget) for md, budget in outputs]
    assert generated[0] == generated[1]
    assert outputs[0][0].read_bytes() == outputs[1][0].read_bytes()
    assert outputs[0][1].read_bytes() == outputs[1][1].read_bytes()
    manifest, packet = generated[0]
    assert manifest["status"] == prep.STATUS
    assert manifest["execution_boundary"] == prep.EXPECTED_BOUNDARY
    assert packet["budget_is_authorization"] is False
    strings = prep._walk_strings(manifest)
    assert "OPENAI_API_KEY" in strings
    assert not any("OPENAI_API_KEY=" in value for value in strings)


def test_canonical_matrix_rejects_every_identity_and_extra_or_missing_arm(prep):
    manifest = prep.build_qualification(ROOT)
    for field, replacement in (("arm_id", "A-tampered"), ("profile_id", "profile-tampered"), ("snapshot_id", "snapshot-tampered"), ("role", "judge"), ("provider", "other"), ("model", "other-model"), ("native_signal_dimensions", 1), ("signal_dimensions", 1), ("storage_output_dimensions", 1), ("output_dimensions", 1), ("local", False), ("hosted", True), ("execution_state", "READY"), ("execution", "hosted"), ("credentials_required", True), ("comparison", "optional"), ("semantics", "optional")):
        tampered = clone(manifest)
        tampered["arms"][0][field] = replacement
        with pytest.raises(ValueError, match="arm matrix"):
            prep.validate_qualification(tampered, root=ROOT)
    for arms in (manifest["arms"][:-1], manifest["arms"] + [clone(manifest["arms"][0])]):
        tampered = clone(manifest)
        tampered["arms"] = arms
        with pytest.raises(ValueError, match="arm matrix"):
            prep.validate_qualification(tampered, root=ROOT)


def test_recomputed_arm_and_control_hashes_do_not_bypass_canonical_specs(prep):
    manifest = prep.build_qualification(ROOT)
    tampered = clone(manifest)
    tampered["arms"][1]["model"] = "attacker-model"
    recompute_controls_hash(tampered, prep)
    with pytest.raises(ValueError, match="arm matrix"):
        prep.validate_qualification(tampered, root=ROOT)
    tampered = clone(manifest)
    tampered["controls"]["routing"]["recall_k"] = 99
    recompute_controls_hash(tampered, prep)
    with pytest.raises(ValueError, match="controls canonical"):
        prep.validate_qualification(tampered, root=ROOT)


def test_boundary_schema_rejects_alterations_and_extra_keys(prep):
    manifest = prep.build_qualification(ROOT)
    for key, value in (("authorization_required_before_execution", False), ("credentials_may_not_appear_in_manifest_or_cli", False), ("preparation_only", False), ("hosted_arms_enabled", True), ("explicit_authorization_placeholder_absent", False)):
        tampered = clone(manifest)
        tampered["execution_boundary"][key] = value
        with pytest.raises(ValueError, match="execution boundary"):
            prep.validate_qualification(tampered, root=ROOT)
    tampered = clone(manifest)
    tampered["execution_boundary"]["unexpected"] = False
    with pytest.raises(ValueError, match="execution boundary"):
        prep.validate_qualification(tampered, root=ROOT)


def test_source_provenance_rereads_explicit_current_root_and_rejects_stale_or_tampered(prep, tmp_path):
    for relative in prep.SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    manifest = prep.build_qualification(tmp_path)
    prep.validate_qualification(manifest, root=tmp_path)
    (tmp_path / prep.SOURCE_FILES[0]).write_bytes(b"tampered source")
    with pytest.raises(ValueError, match="source provenance"):
        prep.validate_qualification(manifest, root=tmp_path)
    with pytest.raises((FileNotFoundError, ValueError), match="source"):
        prep.validate_qualification(manifest, root=ROOT / "benchmarks")


def test_source_path_set_and_digest_tampering_fail_even_when_hashes_recomputed(prep):
    manifest = prep.build_qualification(ROOT)
    tampered = clone(manifest)
    tampered["controls"]["source_inputs"]["repository_relative"] = list(prep.SOURCE_FILES[:-1])
    recompute_controls_hash(tampered, prep)
    with pytest.raises(ValueError, match="controls canonical"):
        prep.validate_qualification(tampered, root=ROOT)
    tampered = clone(manifest)
    tampered["code_provenance"]["source_boundary_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="source boundary"):
        prep.validate_qualification(tampered, root=ROOT)


def test_dataset_pending_sentinels_are_exact_and_readiness_is_separate(prep):
    manifest = prep.build_qualification(ROOT)
    for key, value in (("checksum", {"algorithm": "sha256", "value": "arbitrary"}), ("checksum", {"algorithm": "sha256", "value": "pending"})):
        tampered = clone(manifest)
        tampered["controls"]["dataset"][key] = value
        recompute_controls_hash(tampered, prep)
        with pytest.raises(ValueError, match="controls canonical|exact pending"):
            prep.validate_qualification(tampered, root=ROOT)
    for override in ({"dataset_checksum": prep.PENDING}, {"dataset_checksum": "not-hex"}, {"question_ids_and_order_digest": prep.PENDING_IDS}, {"question_ids_and_order_digest": "F" * 64}, {"ordered_question_ids_sha256": "not-hex"}, {"ordered_question_ids": []}, {"question_count": 1}, {"blockers": ["dataset"]}):
        with pytest.raises(ValueError):
            prep.validate_execution_readiness(manifest, readiness(prep, **override), root=ROOT)


def test_execution_readiness_requires_materialized_identity_authorization_prices_and_clean_env(prep):
    manifest = prep.build_qualification(ROOT)
    prep.validate_execution_readiness(manifest, readiness(prep), root=ROOT)
    with pytest.raises(ValueError, match="READY"):
        prep.validate_execution_readiness(manifest, readiness(prep, state="PENDING_EXTERNAL_DATASET", execution_ready=False), root=ROOT)
    with pytest.raises(ValueError, match="READY"):
        prep.validate_execution_readiness(manifest, readiness(prep, execution_ready=False), root=ROOT)
    for key in prep.FORBIDDEN_ENV_KEYS:
        with pytest.raises(ValueError, match="environment"):
            prep.validate_execution_readiness(manifest, readiness(prep, environment_keys=[key]), root=ROOT)
    for bad_auth in ({"approved_by": "operator"}, {"approved_by": "operator", "approved_scope": "A", "authorized_at": "not-a-date", "record_sha256": "c" * 64}):
        with pytest.raises(ValueError, match="authorization"):
            prep.validate_execution_readiness(manifest, readiness(prep, authorization=bad_auth), root=ROOT)
    with pytest.raises(ValueError, match="schema"):
        prep.validate_execution_readiness(manifest, {**readiness(prep), "extra": True}, root=ROOT)


def test_budget_counts_both_tiers_and_unknown_totals_fail_closed(prep):
    manifest = prep.build_qualification(ROOT)
    budget = prep.build_budget(manifest)
    prep.validate_budget(budget)
    assert budget["guardrails"] == {"maximum_spend_guardrail_usd": 100.0, "guardrail_is_a_hard_stop_not_approval": True, "per_arm_guardrail_must_be_recorded_before_execution": True, "actual_cost_must_be_recorded_after_each_repetition": True}
    assert budget["assumptions"]["questions_per_repetition"] == prep.S_QUESTIONS + prep.M_QUESTIONS == 256
    assert budget["assumptions"]["reader_calls_per_repetition"] == 256
    assert budget["assumptions"]["judge_calls_per_repetition"] == 256
    assert budget["assumptions"]["reader_calls_all_repetitions"] == 768
    assert budget["assumptions"]["judge_calls_all_repetitions"] == 768
    assert budget["totals"]["local_embedding_provider_cost_usd"] == 0.0
    assert budget["totals"]["local_all_stage_total_usd"] is None
    assert budget["totals"]["all_arms_total_usd"] is None
    for arm in budget["arms"]:
        assert arm["estimated_units_per_repetition"]["reader_calls"] == 256
        assert arm["estimated_units_per_repetition"]["judge_calls"] == 256
        assert arm["estimated_cost_usd"] is None
    tampered = clone(budget)
    tampered["assumptions"]["reader_calls_per_repetition"] = 251
    with pytest.raises(ValueError, match="256"):
        prep.validate_budget(tampered)
    tampered = clone(budget)
    tampered["guardrails"]["maximum_authorized_spend_usd"] = 100
    with pytest.raises(ValueError, match="maximum_spend"):
        prep.validate_budget(tampered)


def test_rendered_artifacts_truthfully_explain_blockers_controls_and_budget(prep, tmp_path):
    md = tmp_path / "qualification.md"
    budget_path = tmp_path / "budget.json"
    manifest, budget = prep.generate(root=ROOT, output=md, budget_output=budget_path)
    text = md.read_text(encoding="utf-8")
    assert prep.STATUS in text
    assert "256 questions per repetition" in text
    assert "768 over three repetitions" in text
    assert "unknown" in text
    assert "$0.00" in text
    assert "maximum_spend_guardrail_usd" not in text
    assert "PENDING_EXTERNAL_DATASET_SHA256" in text
    assert "WEFT_TURN_RERANK_DISABLE" in text
    assert "no environment override" not in text
    parsed = json.loads(budget_path.read_text(encoding="utf-8"))
    assert parsed["qualification"]["hashes"]["manifest_controls_sha256"] == prep._sha256_json(parsed["qualification"]["controls"])
    assert parsed["budget"]["totals"]["all_arms_total_usd"] is None
    assert budget["status"] == manifest["status"]


def test_no_provider_or_secret_reads_are_needed(prep):
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "load_dotenv" not in source
    assert "httpx" not in source
    assert "asyncpg" not in source
    assert hashlib.sha256((ROOT / prep.SOURCE_FILES[0]).read_bytes()).hexdigest() == prep.source_provenance(ROOT)[prep.SOURCE_FILES[0]]
