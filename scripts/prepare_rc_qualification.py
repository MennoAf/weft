"""Prepare and validate the RC-FL-20 qualification packet.

This module is deliberately a preparation-only boundary. It reads only tracked
source bytes supplied by an explicit repository root and writes deterministic
artifacts. It never reads environment values, loads ``.env``, constructs a
provider client, opens a database, or executes a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

ACCEPTANCE = "RC-FL-20"
SCHEMA = "rc-fl-20.qualification.v2"
BUDGET_SCHEMA = "rc-fl-20.budget.v2"
STATUS = "PREPARED — NOT AUTHORIZED"
HASH_ALGORITHM = "sha256"
PENDING = "PENDING_EXTERNAL_DATASET_SHA256"
PENDING_IDS = "PENDING_EXTERNAL_QUESTION_IDS_SHA256"
S_QUESTIONS = 5
M_QUESTIONS = 251
TOTAL_QUESTIONS_PER_REPETITION = S_QUESTIONS + M_QUESTIONS
REPETITIONS = 3
MAXIMUM_SPEND_GUARDRAIL_USD = 100.0
QUALIFICATION_PURPOSE = "RC-FL-20 LongMemEval M qualification"
CANONICAL_HOSTED_ARMS = ("B-hosted-controlled", "C-hosted-ceiling")
ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
PLACEHOLDER_RE = re.compile(r"^(?:PENDING|PLACEHOLDER|TODO|TBD|UNKNOWN|NONE|NULL)", re.IGNORECASE)
SOURCE_FILES = (
    "benchmarks/longmemeval/adapter.py",
    "benchmarks/longmemeval/ingest.py",
    "benchmarks/longmemeval/judge.py",
    "benchmarks/longmemeval/reader.py",
    "benchmarks/longmemeval/router.py",
    "benchmarks/longmemeval/task_shape.py",
    "weft/config/__init__.py",
    "weft/embeddings/fastembed_provider.py",
    "weft/embeddings/openai.py",
)
FORBIDDEN_ENV_KEYS = (
    "ANTHROPIC_API_KEY", "DATABASE_URL", "LONGMEMEVAL_DATABASE_PASSWORD",
    "LONGMEMEVAL_DATABASE_URL", "LONGMEMEVAL_PATH", "OPENAI_API_KEY",
    "WEFT_API_KEY", "WEFT_DATABASE_CA_CERT", "WEFT_DATABASE_CA_CERT_FILE",
    "WEFT_DATABASE_URL", "WEFT_EMBEDDING_MODEL", "WEFT_EMBEDDING_PROVIDER",
    "WEFT_EPISODE_EMBEDDER", "WEFT_HIERARCHICAL", "WEFT_OPENAI_EMBED_RETRIES",
    "WEFT_OPENAI_EMBED_TIMEOUT", "WEFT_RETRIEVAL_RECOVERY_MODE",
    "WEFT_RETRIEVAL_RECOVERY_PLANNER_ENABLED", "WEFT_TEXT_MODEL_JUDGE",
    "WEFT_TEXT_MODEL_READER", "WEFT_TEXT_MODEL_REPLAY_AGGREGATE",
    "WEFT_TEXT_PROVIDER", "WEFT_TURN_RERANK_DISABLE",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


EXPECTED_BOUNDARY = _freeze({
    "preparation_only": True, "provider_calls": False, "paid_calls": False,
    "database_writes": False, "hosted_arms_enabled": False,
    "authorization_required_before_execution": True,
    "credentials_may_not_appear_in_manifest_or_cli": True,
    "explicit_authorization_placeholder_absent": True,
})
EXPECTED_ARMS = _freeze((
    {"arm_id": "A-local-primary", "comparison": "primary", "semantics": "required primary", "role": "embedding", "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "local": True, "hosted": False, "native_dimensions": 384, "signal_dimensions": 384, "storage_dimensions": 768, "output_dimensions": 768, "native_signal_dimensions": 384, "storage_output_dimensions": 768, "profile_id": "fastembed-bge-small-native384-output768-v1", "snapshot_id": "rc-fl20-a-fastembed-bge-small-v1", "execution_state": "ENABLED_LOCAL_ONLY_IN_FUTURE_AUTHORIZED_RUN", "execution": "provider-free local only", "credentials_required": False, "privacy_data_flow": "embedding stays local; no hosted data transfer", "reembedding": "changing profile_id or snapshot_id requires explicit re-embedding; zero padding is not a conversion"},
    {"arm_id": "B-hosted-controlled", "comparison": "optional controlled comparison", "semantics": "optional", "role": "embedding", "provider": "openai", "model": "text-embedding-3-small", "local": False, "hosted": True, "native_dimensions": 1536, "signal_dimensions": 1536, "storage_dimensions": 768, "output_dimensions": 768, "native_signal_dimensions": 1536, "storage_output_dimensions": 768, "profile_id": "openai-text-embedding-3-small-output768-v1", "snapshot_id": "rc-fl20-b-openai-small-output768-v1", "execution_state": "DISABLED_HOSTED_UNTIL_SEPARATE_AUTHORIZATION", "execution": "hosted and paid; forbidden until separately authorized", "credentials_required": True, "privacy_data_flow": "question/ingest text would be sent to OpenAI; retention and residency require operator confirmation", "reembedding": "requires explicit re-embedding when selected; vectors are profile-specific"},
    {"arm_id": "C-hosted-ceiling", "comparison": "optional ceiling comparison", "semantics": "optional", "role": "embedding", "provider": "openai", "model": "text-embedding-3-large", "local": False, "hosted": True, "native_dimensions": 3072, "signal_dimensions": 3072, "storage_dimensions": 3072, "output_dimensions": 3072, "native_signal_dimensions": 3072, "storage_output_dimensions": 3072, "profile_id": "openai-text-embedding-3-large-output3072-v1", "snapshot_id": "rc-fl20-c-openai-large-output3072-v1", "execution_state": "DISABLED_HOSTED_UNTIL_SEPARATE_AUTHORIZATION", "execution": "hosted and paid; forbidden until separately authorized", "credentials_required": True, "privacy_data_flow": "question/ingest text would be sent to OpenAI; retention and residency require operator confirmation", "reembedding": "requires schema/storage capacity and explicit re-embedding when selected; vectors are profile-specific"},
))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _repo_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ValueError(f"source path is not repository-relative: {relative!r}")
    raw, current = root / relative, root.resolve()
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source path contains a symlink component: {relative}")
    candidate, resolved_root = raw.resolve(), root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {relative}") from exc
    return candidate


def _source_map(root: Path, paths: tuple[str, ...] = SOURCE_FILES) -> dict[str, str]:
    if tuple(paths) != SOURCE_FILES or len(set(paths)) != len(SOURCE_FILES):
        raise ValueError("source path set does not exactly match the canonical source boundary")
    result = {}
    for relative in paths:
        path = _repo_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"source provenance file is missing or symlinked: {relative}")
        result[relative] = _sha256_bytes(path.read_bytes())
    return result


def source_provenance(root: Path) -> dict[str, str]:
    return _source_map(root)


def _controls(provenance: Mapping[str, str]) -> dict[str, Any]:
    return {
        "dataset": {"name": "LongMemEval cleaned", "split": "longmemeval_m_cleaned.json", "checksum": {"algorithm": HASH_ALGORITHM, "value": PENDING}, "state": "PENDING_EXTERNAL_DATASET", "execution_ready": False, "question_ids_and_order": {"algorithm": HASH_ALGORITHM, "value": PENDING_IDS, "order": "exact source order after deterministic stratified sampling"}, "sample": {"strategy": "stratified by question_type", "fraction": 0.5, "seed": 0, "expected_questions": M_QUESTIONS, "full_population_expected_questions": 500}},
        "ingest": {"mode": "turns", "representation": "one episode per question; every user/assistant/system/tool turn as one episode_turns row; unknown roles skipped by runtime contract", "embedding_batch_size": 100},
        "routing": {"task_shape": "derived from question and allowed sessions, label/gold blind", "task_shape_derivation": "derive_task_shape(question, sessions); task_shape and routing_class use only question text, allowed sessions, and ordinary metadata; labels/gold are forbidden", "task_shape_candidate_width": {"single-session": 10, "temporal": 10, "multi-session": 30, "temporal-multi": 30}, "reader_candidate_width": "TaskShape.top_k after explicit caller widening via max(requested_top_k, derived_top_k)", "widening_semantics": "caller top_k may widen Reader/retrieval candidates but never narrows the derived TaskShape width", "tier": "turns", "top_k": 10, "recall_k": 10, "recall_metric_cutoff": 10, "overfetch_multiplier": 4, "warm_boost_rounds": 0, "rerank": "enabled; WEFT_TURN_RERANK_DISABLE is forbidden and must be absent at execution"},
        "reader": {"role": "Reader", "provider": "anthropic", "model": "claude-sonnet-4-6", "abstention_model": "claude-sonnet-4-6", "max_output_tokens": 256, "prompt_source": "benchmarks/longmemeval/reader.py:_BASE_INSTRUCTIONS,_system_prompt_for,_format_memories", "prompt_hash_source": "tracked source hash in code_provenance; no mutable prompt text is copied into the card"},
        "judge": {"role": "judge", "provider": "LongMemEval evaluator via OpenAI", "model": "gpt-4o", "scoring": "boolean autoeval label against the complete reference population", "denominator": "reference question IDs; missing hypothesis or judge result is incorrect and separately categorized", "retrieval_only_isolation": "recall metrics are not combined with answer correctness"},
        "retry_behavior": {"embedding_batch_attempts": 3, "embedding_backoff_seconds": [2.0, 4.0], "per_turn_fallback_delay_seconds": 0.1, "provider_sdk_retries": "profile-specific provider setting must be recorded; WEFT_OPENAI_EMBED_TIMEOUT and WEFT_OPENAI_EMBED_RETRIES are forbidden overrides", "missing_stage_policy": "fail closed; retain failure category and count"},
        "runtime_environment": {"values_read_during_preparation": False, "dotenv_loading": "forbidden; preparation does not load .env or ~/.weft/.env", "forbidden_key_presence": list(FORBIDDEN_ENV_KEYS), "required_unset_at_execution": list(FORBIDDEN_ENV_KEYS), "sanitized_input": "future readiness accepts only a set/list of key names, never values"},
        "source_inputs": {"repository_relative": list(SOURCE_FILES), "source_boundary_sha256": _sha256_json(dict(provenance))},
    }


def build_qualification(root: Path) -> dict[str, Any]:
    root = root.resolve()
    provenance = source_provenance(root)
    controls = _controls(provenance)
    manifest = {"schema": SCHEMA, "acceptance": ACCEPTANCE, "status": STATUS, "hashes": {"canonical_serialization": "JSON sort_keys=true, separators=(',', ':'), ensure_ascii=false, UTF-8", "algorithm": HASH_ALGORITHM, "manifest_controls_sha256": _sha256_json(controls)}, "execution_boundary": dict(EXPECTED_BOUNDARY), "arms": [dict(arm) for arm in EXPECTED_ARMS], "controls": controls, "metrics": {"retrieval_only": ["recall@10", "gold rank (or not retrieved)", "final empty rate", "initial empty rate", "retry/fallback rescue rate", "per question_type recall and empty rate"], "end_to_end_answer_quality": ["reference-denominator answer correctness", "per question_type correctness", "unsupported answer rate", "preference compliance", "enumeration accuracy", "infrastructure/missing-stage failures"], "operational": ["ingest/retrieval/reader/judge latency distributions", "input/cached/output tokens", "embedding request and item counts", "actual provider cost after execution (if authorized)", "storage footprint and re-embedding work"]}, "repetition_plan": {"repetitions_per_enabled_arm": REPETITIONS, "tiers": [{"id": "S_smoke", "questions": S_QUESTIONS, "included": True, "purpose": "evidence only"}, {"id": "M_stratified", "questions": M_QUESTIONS, "included": True, "purpose": "qualification candidate"}], "questions_per_repetition": TOTAL_QUESTIONS_PER_REPETITION, "same_question_ids_across_arms": True, "seeds": {"sampling": 0, "other_randomness": "record explicitly; no unrecorded randomness"}, "raw_artifacts_required_per_repetition": ["hypotheses JSONL", "retrieval telemetry", "judge results", "metrics", "cost/latency ledger", "failure log"], "summary_rule": "report per-repetition distributions and stage-separated failures; never promote a single aggregate to lift"}, "evidence": {"provider_free_contract": "tests/test_rc_benchmark_contract.py and tests/test_rc_benchmark_summary.py; evidence of contracts only, no lift claim", "provider_free_metamorphic": "tests/test_rc_benchmark_metamorphic.py; evidence of label-blind runtime trace only", "local_docker": "evidence/rc-finish-line/mcp-journey/local-docker-acceptance.json; local MCP persistence evidence only, not benchmark lift", "labeled_smoke": "preparation placeholder; if later run, label S/M smoke and do not treat as qualification lift", "historical_results": "retained as historical and excluded from current lift claims"}, "code_provenance": {"source_files": provenance, "source_boundary_sha256": _sha256_json(provenance), "revision_binding": "current source bytes under explicit repository root; no mutable timestamp or HEAD-only binding"}, "authorization": {"status": "NOT_AUTHORIZED", "placeholder": "OPERATOR AUTHORIZATION REQUIRED: record owner, scope, approved arms, maximum spend, date, frozen prices, and immutable artifact destination before any hosted or paid execution", "approved_by": None, "approved_scope": None, "approved_max_usd": None, "authorization_record_sha256": None}, "fail_closed_blockers": ["external dataset file and exact lowercase checksum are not present in this checkout", "exact sampled question ID/order digest and materialized ordered ID manifest are not present in this checkout", "no dated operator authorization record exists", "Reader/judge prices and dataset-dependent embedding item count are unknown", "hosted profiles require credentials and paid execution, which this task forbids", "all listed runtime/provider/database/credential environment keys must be absent at execution"]}
    validate_qualification(manifest, root=root)
    return manifest


def _assert_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match canonical specification")


def validate_qualification(manifest: dict[str, Any], root: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1] if root is None else root
    if manifest.get("schema") != SCHEMA or manifest.get("acceptance") != ACCEPTANCE:
        raise ValueError("wrong qualification schema or acceptance identifier")
    if manifest.get("status") != STATUS:
        raise ValueError("qualification status must remain PREPARED — NOT AUTHORIZED")
    _assert_exact(manifest.get("execution_boundary"), EXPECTED_BOUNDARY, "execution boundary")
    _assert_exact(manifest.get("arms"), [dict(arm) for arm in EXPECTED_ARMS], "arm matrix")
    if set(manifest.get("arms", [{}])[0]) != set(EXPECTED_ARMS[0]):
        raise ValueError("arm schema contains missing or extra fields")
    authorization = manifest.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("status") != "NOT_AUTHORIZED":
        raise ValueError("authorization placeholder is missing or already approved")
    if any(authorization.get(key) is not None for key in ("approved_by", "approved_scope", "authorization_record_sha256")):
        raise ValueError("authorization placeholder is missing or already approved")
    controls = manifest.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("controls are missing")
    current = source_provenance(root)
    code = manifest.get("code_provenance", {})
    _assert_exact(code.get("source_files"), current, "source provenance")
    _assert_exact(code.get("source_boundary_sha256"), _sha256_json(current), "source boundary digest")
    _assert_exact(controls, _controls(current), "controls canonical specification")
    _assert_exact(manifest.get("hashes", {}).get("manifest_controls_sha256"), _sha256_json(controls), "manifest controls hash")
    if controls.get("dataset", {}).get("checksum") != {"algorithm": HASH_ALGORITHM, "value": PENDING}:
        raise ValueError("dataset checksum must use the exact pending sentinel")
    if controls.get("dataset", {}).get("question_ids_and_order", {}).get("value") != PENDING_IDS:
        raise ValueError("question IDs/order digest must use the exact pending sentinel")
    if controls.get("routing", {}).get("recall_k") != 10 or controls.get("routing", {}).get("recall_metric_cutoff") != 10:
        raise ValueError("retrieval recall cutoff must remain 10")
    _validate_no_credentials(manifest)


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _walk_strings(child)]
    return [value] if isinstance(value, str) else []


def _validate_no_credentials(value: Any) -> None:
    for text in _walk_strings(value):
        upper = text.upper()
        if "OPENAI_API_KEY=" in upper or "ANTHROPIC_API_KEY=" in upper or "AUTHORIZATION: BEARER" in upper:
            raise ValueError("credential-bearing input in qualification manifest")


def _aware_iso(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or PLACEHOLDER_RE.match(value):
        raise ValueError(f"{label} must be concrete")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware ISO-8601")
    return value


def question_ids_and_order_digest(dataset_checksum: str, ordered_ids: list[str]) -> str:
    return _sha256_json({"dataset_sha256": dataset_checksum, "ordered_question_ids": ordered_ids})


def prices_digest(prices: Mapping[str, Any]) -> str:
    return _sha256_json(prices)


def authorization_record_digest(authorization: Mapping[str, Any]) -> str:
    return _sha256_json({key: authorization[key] for key in authorization if key != "record_sha256"})


def _validate_price_records(prices: Any) -> None:
    if not isinstance(prices, dict) or set(prices) != {"embedding", "reader", "judge"}:
        raise ValueError("all embedding, Reader, and judge prices must be frozen")
    for name, price in prices.items():
        if not isinstance(price, dict) or set(price) != {"amount", "unit", "currency", "source", "as_of"}:
            raise ValueError(f"{name} price requires amount, unit, currency, source, and as_of")
        amount = price["amount"]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{name} price amount must be finite and nonnegative")
        if any(not isinstance(price[key], str) or not price[key] or PLACEHOLDER_RE.match(price[key]) for key in ("unit", "currency", "source")):
            raise ValueError(f"{name} price unit, currency, and source must be concrete")
        _aware_iso(price["as_of"], f"{name} price as_of")


def validate_execution_readiness(manifest: dict[str, Any], readiness: Mapping[str, Any], root: Path | None = None) -> None:
    validate_qualification(manifest, root=root)
    required = {"dataset_checksum", "question_ids_and_order_digest", "question_count", "ordered_question_ids", "ordered_question_ids_sha256", "blockers", "authorization", "prices", "embedding_item_count", "environment_keys", "state", "execution_ready"}
    if set(readiness) != required:
        raise ValueError("execution readiness schema has missing or extra fields")
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    for key in ("dataset_checksum", "ordered_question_ids_sha256", "question_ids_and_order_digest"):
        if not isinstance(readiness[key], str) or not hex64.fullmatch(readiness[key]):
            raise ValueError(f"{key} must be a lowercase 64-hex digest")
    ids = readiness["ordered_question_ids"]
    if not isinstance(ids, list) or len(ids) != M_QUESTIONS or any(not isinstance(item, str) or not item.strip() for item in ids) or len(set(ids)) != M_QUESTIONS:
        raise ValueError("ordered question IDs must contain exactly 251 unique nonempty strings")
    if readiness["question_count"] != M_QUESTIONS:
        raise ValueError("question count must be exactly the sampled M count")
    if readiness["ordered_question_ids_sha256"] != _sha256_json(ids):
        raise ValueError("ordered question ID manifest hash does not match materialized IDs")
    if readiness["dataset_checksum"] in (PENDING, PENDING_IDS) or readiness["question_ids_and_order_digest"] != question_ids_and_order_digest(readiness["dataset_checksum"], ids):
        raise ValueError("question IDs/order digest is not canonically bound to dataset and ordered IDs")
    if readiness["state"] != "READY" or readiness["execution_ready"] is not True or readiness["blockers"] != []:
        raise ValueError("execution readiness must be explicitly READY, true, and unblocked")
    auth = readiness["authorization"]
    auth_keys = {"operator_id", "decision_id", "purpose", "approved_scope", "approved_arms", "approved_max_spend_usd", "currency", "authorized_at", "model_profile_ids", "prices_sha256", "record_sha256"}
    if not isinstance(auth, dict) or set(auth) != auth_keys:
        raise ValueError("dated canonical authorization record is required")
    for key in ("operator_id", "decision_id", "purpose", "approved_scope", "currency"):
        if not isinstance(auth[key], str) or not auth[key] or PLACEHOLDER_RE.match(auth[key]):
            raise ValueError("authorization identifiers and scope must be concrete")
    if auth["purpose"] != QUALIFICATION_PURPOSE or auth["approved_scope"] != QUALIFICATION_PURPOSE or auth["approved_arms"] != list(CANONICAL_HOSTED_ARMS):
        raise ValueError("authorization purpose/scope/arms do not match canonical qualification")
    if not isinstance(auth["model_profile_ids"], dict) or set(auth["model_profile_ids"]) != set(CANONICAL_HOSTED_ARMS):
        raise ValueError("authorization model/profile IDs must name canonical hosted arms")
    for arm_id in CANONICAL_HOSTED_ARMS:
        arm = next(x for x in EXPECTED_ARMS if x["arm_id"] == arm_id)
        if auth["model_profile_ids"][arm_id] != {"model": arm["model"], "profile_id": arm["profile_id"]}:
            raise ValueError("authorization model/profile binding is not canonical")
    _aware_iso(auth["authorized_at"], "authorization authorized_at")
    _validate_price_records(readiness["prices"])
    if auth["prices_sha256"] != prices_digest(readiness["prices"]):
        raise ValueError("authorization price digest does not match frozen prices")
    if isinstance(auth["approved_max_spend_usd"], bool) or not isinstance(auth["approved_max_spend_usd"], (int, float)) or not math.isfinite(auth["approved_max_spend_usd"]) or not 0 <= auth["approved_max_spend_usd"] <= MAXIMUM_SPEND_GUARDRAIL_USD:
        raise ValueError("approved maximum spend must be finite, nonnegative, and within hard-stop guardrail")
    if not hex64.fullmatch(auth["record_sha256"]) or auth["record_sha256"] != authorization_record_digest(auth):
        raise ValueError("authorization record digest does not match canonical fields")
    if not isinstance(readiness["embedding_item_count"], int) or isinstance(readiness["embedding_item_count"], bool) or readiness["embedding_item_count"] <= 0:
        raise ValueError("embedding item count must be a positive materialized count")
    env_keys = readiness["environment_keys"]
    if not isinstance(env_keys, list) or len(set(env_keys)) != len(env_keys) or any(not isinstance(key, str) or not ENV_KEY_RE.fullmatch(key) or key in FORBIDDEN_ENV_KEYS or any(part in key for part in ("SECRET", "PASSWORD", "TOKEN", "KEY", "PATH", "URL")) for key in env_keys):
        raise ValueError("environment_keys must be unique sanitized non-secret key names")


def build_budget(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_qualification(manifest)
    arms = []
    for arm in manifest["arms"]:
        hosted = arm["hosted"]
        arms.append({"arm_id": arm["arm_id"], "profile_id": arm["profile_id"], "model": arm["model"], "native_signal_dimensions": arm["native_signal_dimensions"], "storage_output_dimensions": arm["storage_output_dimensions"], "repetitions": REPETITIONS, "tier_inclusion": {"S_smoke": True, "M_stratified": True}, "sample_counts": {"S_smoke": S_QUESTIONS, "M_stratified": M_QUESTIONS, "total_per_repetition": TOTAL_QUESTIONS_PER_REPETITION}, "estimated_units_per_repetition": {"question_stage_calls": TOTAL_QUESTIONS_PER_REPETITION, "reader_calls": TOTAL_QUESTIONS_PER_REPETITION, "judge_calls": TOTAL_QUESTIONS_PER_REPETITION, "embedding_items": "dataset-dependent; count from frozen ingest manifest"}, "estimated_units_all_repetitions": {"question_stage_calls": TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS, "reader_calls": TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS, "judge_calls": TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS, "embedding_items": "dataset-dependent; count from frozen ingest manifests"}, "pricing": {"embedding_usd_per_million_tokens": 0.0 if not hosted else None, "reader_usd_per_million_tokens": None, "judge_usd_per_million_tokens": None, "source": "local computation; no provider price" if not hosted else "PLACEHOLDER—operator must supply source and as_of before authorization", "as_of": None}, "estimated_cost_usd": None, "notes": "cost is not authorization; Reader/judge pricing and dataset-dependent embedding item count remain unknown; local embedding provider cost is separately known as $0"})
    return {"schema": BUDGET_SCHEMA, "acceptance": ACCEPTANCE, "status": STATUS, "budget_is_authorization": False, "assumptions": {"S_questions": S_QUESTIONS, "M_questions": M_QUESTIONS, "questions_per_repetition": TOTAL_QUESTIONS_PER_REPETITION, "repetitions_per_enabled_arm": REPETITIONS, "tier_inclusion": {"S_smoke": True, "M_stratified": True}, "reader_calls_per_question": 1, "judge_calls_per_question": 1, "reader_calls_per_repetition": TOTAL_QUESTIONS_PER_REPETITION, "judge_calls_per_repetition": TOTAL_QUESTIONS_PER_REPETITION, "reader_calls_all_repetitions": TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS, "judge_calls_all_repetitions": TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS, "retrieval_metric": "recall@10, independent of Reader/judge candidate width", "cost_units": "USD; hosted prices intentionally unset until sourced and dated"}, "arms": arms, "guardrails": {"maximum_spend_guardrail_usd": MAXIMUM_SPEND_GUARDRAIL_USD, "guardrail_is_a_hard_stop_not_approval": True, "per_arm_guardrail_must_be_recorded_before_execution": True, "actual_cost_must_be_recorded_after_each_repetition": True}, "totals": {"local_embedding_provider_cost_usd": 0.0, "local_all_stage_total_usd": None, "hosted_total_usd": None, "all_arms_total_usd": None, "reason": "Reader/judge prices are unknown and embedding item count is dataset-dependent; no known total of $0 is emitted."}, "authorization": manifest["authorization"], "fail_closed": ["Do not execute if any hosted/paid flag is true without a separate authorization record.", "Do not execute with credential-bearing CLI/runtime inputs or forbidden environment keys.", "Do not execute until dataset checksum, exact question ID/order digest, and materialized ordered IDs are filled and verified.", "Do not treat this budget or historical results as a lift claim."]}


def validate_budget(budget: dict[str, Any]) -> None:
    if budget.get("schema") != BUDGET_SCHEMA or budget.get("status") != STATUS or budget.get("budget_is_authorization") is not False:
        raise ValueError("budget must remain preparation-only and not authorized")
    if budget.get("authorization", {}).get("status") != "NOT_AUTHORIZED":
        raise ValueError("budget authorization must remain NOT_AUTHORIZED")
    expected = TOTAL_QUESTIONS_PER_REPETITION
    assumptions = budget.get("assumptions", {})
    if assumptions.get("S_questions") + assumptions.get("M_questions") != expected:
        raise ValueError("S+M must equal 256 questions per repetition")
    if assumptions.get("reader_calls_per_repetition") != expected or assumptions.get("judge_calls_per_repetition") != expected:
        raise ValueError("Reader and judge must each have 256 question-stage calls per repetition")
    if assumptions.get("reader_calls_all_repetitions") != expected * REPETITIONS or assumptions.get("judge_calls_all_repetitions") != expected * REPETITIONS:
        raise ValueError("Reader and judge must each have 768 calls over three repetitions")
    guardrails = budget.get("guardrails", {})
    if "maximum_authorized_spend_usd" in guardrails or guardrails.get("maximum_spend_guardrail_usd", 0) <= 0:
        raise ValueError("non-authorizing maximum_spend_guardrail_usd is required")
    totals = budget.get("totals", {})
    if totals.get("local_embedding_provider_cost_usd") != 0.0 or totals.get("local_all_stage_total_usd") is not None or totals.get("all_arms_total_usd") is not None:
        raise ValueError("unknown Reader/judge or embedding item costs must keep all-stage totals unknown")
    for arm in budget.get("arms", []):
        units = arm.get("estimated_units_per_repetition", {})
        if units.get("reader_calls") != expected or units.get("judge_calls") != expected or units.get("question_stage_calls") != expected:
            raise ValueError("per-arm question-stage arithmetic is inconsistent")
        if arm.get("estimated_cost_usd") is not None:
            raise ValueError("per-arm total must remain unknown while required prices/items are unknown")


def render_markdown(manifest: dict[str, Any], budget: dict[str, Any]) -> str:
    controls = manifest["controls"]
    lines = ["# RC-FL-20 qualification run card", "", f"**Status: {STATUS}**", "", "This is deterministic preparation evidence only. It authorizes no provider, database, hosted, or paid execution and makes no lift claim.", "", "## Frozen arm matrix", "", "| Arm | Role | Provider/model | Local/hosted | Native/signal | Storage/output | Profile ID | Snapshot ID | Semantics | Execution |", "|---|---|---|---|---:|---:|---|---|---|---|"]
    for arm in manifest["arms"]:
        lines.append(f"| {arm['arm_id']} | {arm['role']} | `{arm['provider']}` / `{arm['model']}` | {str(arm['local']).lower()}/{str(arm['hosted']).lower()} | {arm['native_signal_dimensions']} | {arm['storage_output_dimensions']} | `{arm['profile_id']}` | `{arm['snapshot_id']}` | {arm['semantics']} | {arm['execution']} |")
    lines.extend(["", "Arm A is the required local primary. Arms B and C are optional hosted comparisons and remain disabled until separately authorized.", "", "## Frozen inputs and controls", "", f"- Dataset split: `{controls['dataset']['split']}`; checksum: `{controls['dataset']['checksum']['algorithm']}` `{controls['dataset']['checksum']['value']}`.", f"- Exact sampled question IDs/order: `{controls['dataset']['question_ids_and_order']['algorithm']}` `{controls['dataset']['question_ids_and_order']['value']}`; stratified fraction `{controls['dataset']['sample']['fraction']}`, seed `{controls['dataset']['sample']['seed']}`.", f"- Ingest: `{controls['ingest']['mode']}`; {controls['ingest']['representation']}.", f"- Routing: label/gold-blind `{controls['routing']['task_shape_derivation']}`; recall cutoff `recall_k={controls['routing']['recall_k']}`. Reader candidate width is derived TaskShape `top_k` (10 or 30), with widening only by explicit `max`.", f"- Reader: `{controls['reader']['provider']}` / `{controls['reader']['model']}`; prompt source `{controls['reader']['prompt_source']}`.", f"- Judge: `{controls['judge']['provider']}` / `{controls['judge']['model']}`; complete reference denominator, missing stages incorrect.", f"- Runtime environment: preparation reads no values; `.env` loading is forbidden; listed provider/model/credential/database/rerank/timeout/retry keys (including `WEFT_TURN_RERANK_DISABLE`, `WEFT_OPENAI_EMBED_TIMEOUT`, and `WEFT_OPENAI_EMBED_RETRIES`) must be absent at execution.", "- Canonical hashes: SHA-256 over UTF-8 JSON with sorted keys, compact separators, and no ASCII escaping; validator re-reads current bytes under an explicit root.", "", "## Repetition and measurement plan", "", "Both tiers are explicitly included: S smoke (5) + M stratified (251) = **256 questions per repetition**. Reader and judge each make 256 question-stage calls per repetition, or 768 over three repetitions. Embedding item count remains dataset-dependent and pending.", "Run 3 independent repetitions per enabled arm. Preserve hypotheses, retrieval telemetry, judge results, metrics, latency/token/cost ledgers, and failures per repetition. Report distributions, not a single aggregate.", "", "## Budget summary (not authorization)", "", f"- S: {S_QUESTIONS}; M: {M_QUESTIONS}; total per repetition: {TOTAL_QUESTIONS_PER_REPETITION}; Reader/judge each: {TOTAL_QUESTIONS_PER_REPETITION} per repetition and {TOTAL_QUESTIONS_PER_REPETITION * REPETITIONS} over 3 repetitions.", f"- Maximum spend guardrail: **${budget['guardrails']['maximum_spend_guardrail_usd']:.2f}**, not approval; it is a hard stop only.", "- Local embedding provider cost is separately known as $0.00. Local all-stage and all-arm totals are **unknown**, not $0.00, because Reader/judge prices and dataset-dependent embedding item count are unknown.", "", "## Evidence, blockers, and authorization", "", "Provider-free contract, metamorphic, summary, and local Docker receipts are contract evidence only; historical results remain historical and cannot become current lift claims.", "", f"**Authorization:** `{manifest['authorization']['status']}`. {manifest['authorization']['placeholder']}", "", "Execution remains blocked until external dataset/checksum, exact ordered ID materialization/hash, cleared blockers, dated authorization, frozen prices, and a sanitized environment key-presence set are supplied to the separate execution-readiness validator.", "", "## Content provenance", "", f"- Source boundary SHA-256: `{manifest['code_provenance']['source_boundary_sha256']}`.", f"- Controls SHA-256: `{manifest['hashes']['manifest_controls_sha256']}`.", f"- Exact source set: {', '.join(f'`{p}`' for p in SOURCE_FILES)}.", "- No timestamp or HEAD-only binding is used; current source bytes are re-read and compared.", ""])
    return "\n".join(lines)


def generate(*, root: Path, output: Path, budget_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = build_qualification(root)
    budget = build_budget(manifest)
    validate_budget(budget)
    output.parent.mkdir(parents=True, exist_ok=True)
    budget_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(manifest, budget), encoding="utf-8")
    budget_output.write_text(json.dumps({"qualification": manifest, "budget": budget}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RC-FL-20 artifacts without executing a benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    generate(root=root, output=args.output, budget_output=args.budget)
    print(f"{ACCEPTANCE}: {STATUS}")
    print(f"qualification: {args.output}")
    print(f"budget: {args.budget}")


if __name__ == "__main__":
    main()
