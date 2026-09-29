#!/usr/bin/env python3
"""faithful_s36.py — Prepare and safely resume the faithful LongMemEval S36 run.

This benchmark-local runner keeps preparation, calibration, and execution
separate. Preparation is offline and writes only hash-bound artifacts. Resume
requires an explicit calibration receipt, a disposable local benchmark DSN,
FastEmbed availability, and an explicit execute flag. The runner selects the
approved 36 cases in manifest order, processes each case's sessions
chronologically, and counts failures in the exact 36-case denominator.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-09-21
Python:  >= 3.12

Dependencies:
    fastembed>=0.4.0 — required local embedding provider for execution.
    openai>=1.0.0 — lazy Luna/GPT-4o adapter for explicit execution only.

Usage:
    See bottom of file for prepare/calibrate/resume commands.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import shlex
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from benchmarks.longmemeval.agent_workload import EXPECTED_CASE_COUNT, load_manifest, sha256_file
from benchmarks.longmemeval.full_s_profile import (
    FULL_S_CASE_COUNT,
    FULL_S_MAX_BUDGET_USD,
    FULL_S_OPERATIONAL_STOP_USD,
    FULL_S_PROFILE,
    FULL_S_WRITER_MODEL,
)
from benchmarks.longmemeval.dataset import Instance, load_split
from benchmarks.longmemeval.faithful_agent import (
    GPT4O_JUDGE_MODEL,
    GPT6_LUNA_MODEL,
    FRESH_GPT6_MAX_OUTPUT_TOKENS,
    MAX_TOOL_ROUNDS,
    LUNA_MODEL,
    AmbiguousExecutionError,
    AgentExecutionError,
    AgentPolicy,
    BoundedJudge,
    FaithfulAgent,
    OpenAIResponsesClient,
    assert_no_anthropic_execution,
)
from benchmarks.longmemeval.faithful_budget import (
    BudgetExceeded,
    BudgetLedger,
    DEFAULT_CALIBRATION_BUDGET_USD,
    DEFAULT_TOTAL_BUDGET_USD,
    GPT6_TOTAL_BUDGET_USD,
    FreshRunPricing,
    LedgerBindingError,
    Pricing,
    TotalBudgetExceeded,
    canonical_hash,
    import_prior_ledger,
)


ARTIFACT_NAMESPACE = Path("artifacts/belief-recall-0bk79s/faithful")
GPT6_SELECTED35_ARTIFACT_NAMESPACE = Path("artifacts/longmemeval-gpt6-selected35/faithful")
GPT6_FULL_S_ARTIFACT_NAMESPACE = Path("artifacts/longmemeval-gpt6-full-s-turns/faithful")
GPT6_SELECTED35_EXCLUDED_ID = "7527f7e2"
FRESH_RUN_PROFILE = "gpt6-luna-selected35-v1"
FULL_S_CALIBRATION_BUDGET_USD = 10.0
CHECKPOINT_SCHEMA = "weft.longmemeval.faithful-s36-checkpoint.v1"
CALIBRATION_SCHEMA = "weft.longmemeval.faithful-s36-calibration.v1"
EXECUTION_SCHEMA = "weft.longmemeval.faithful-s36-execution.v1"
SAFE_SKIP_CASE_ID = "95228167"
SAFE_SKIP_JOURNAL_SCHEMA = "weft.longmemeval.faithful-s36-safe-skip.v1"
INFLIGHT_RECOVERY_JOURNAL_SCHEMA = "weft.longmemeval.faithful-s36-inflight-recovery.v1"
PROVIDER_TIMEOUT_LEDGER_ERROR = "Request timed out."
PROVIDER_TIMEOUT_EVIDENCE_ERROR = (
    "AmbiguousExecutionError: provider outcome is unknown; "
    "reservation retained and replay is forbidden"
)


class FaithfulRunError(RuntimeError):
    """Raised when preparation or execution safety invariants fail."""


class ExecutionGateError(FaithfulRunError):
    """Raised when explicit execution authorization is missing."""


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Durable artifact paths for one prepared run."""

    root: Path
    manifest: Path
    checkpoint: Path
    ledger: Path
    calibration: Path
    receipt: Path
    lock: Path

    @classmethod
    def from_root(cls, root: Path = ARTIFACT_NAMESPACE) -> "RunPaths":
        """Construct standard paths beneath an artifact namespace."""
        root = Path(root)
        return cls(
            root=root,
            manifest=root / "run-manifest.json",
            checkpoint=root / "session-checkpoint.json",
            ledger=root / "budget-ledger.json",
            calibration=root / "calibration-receipt.json",
            receipt=root / "execution-receipt.json",
            lock=root / "run.lock",
        )


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Offline preparation metadata bound to source artifacts."""

    paths: RunPaths
    dataset_path: Path
    manifest_path: Path
    ordered_question_ids: tuple[str, ...]
    binding: Mapping[str, str]


def _canonical_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _resolve_run_profile(
    *,
    profile: str = "legacy",
    writer_model: str | None = None,
    max_budget_usd: float | None = None,
    selection_policy: Mapping[str, Sequence[str] | None] | None = None,
) -> tuple[str, str, float, float, Any, str, dict[str, list[str] | None] | None]:
    """Validate the model, ceilings, retrieval tier, and cohort for a profile."""
    if profile == "legacy":
        model = writer_model or LUNA_MODEL
        budget = DEFAULT_TOTAL_BUDGET_USD if max_budget_usd is None else max_budget_usd
        if model != LUNA_MODEL or budget != DEFAULT_TOTAL_BUDGET_USD:
            raise FaithfulRunError("legacy runs remain fixed to GPT-5.6 Luna and the $20 cap")
        return profile, model, budget, DEFAULT_CALIBRATION_BUDGET_USD, Pricing(), "auto", None
    if profile == FULL_S_PROFILE:
        if writer_model != FULL_S_WRITER_MODEL or max_budget_usd != FULL_S_MAX_BUDGET_USD:
            raise FaithfulRunError("full-S profile requires --writer-model gpt-6-luna and --max-budget-usd 150")
        if selection_policy is not None and (
            selection_policy.get("include_question_ids") is not None
            or selection_policy.get("exclude_question_ids") is not None
        ):
            raise FaithfulRunError("full-S profile is fixed to all 500 manifest cases")
        return profile, writer_model, max_budget_usd, FULL_S_CALIBRATION_BUDGET_USD, FreshRunPricing(), "turns", {
            "include_question_ids": None,
            "exclude_question_ids": None,
        }
    if profile != FRESH_RUN_PROFILE:
        raise FaithfulRunError(f"unsupported run profile: {profile!r}")
    if writer_model is None or max_budget_usd is None:
        raise FaithfulRunError("fresh profile requires explicit writer_model and max_budget_usd options")
    model = writer_model
    budget = max_budget_usd
    if model != GPT6_LUNA_MODEL or budget != GPT6_TOTAL_BUDGET_USD:
        raise FaithfulRunError("fresh profile requires explicit GPT-6 Luna and the $50 cap")
    if selection_policy is not None:
        include_ids = selection_policy.get("include_question_ids")
        exclude_ids = selection_policy.get("exclude_question_ids")
        if include_ids is not None or exclude_ids != [GPT6_SELECTED35_EXCLUDED_ID]:
            raise FaithfulRunError("fresh profile requires selected-35 excluding 7527f7e2")
    return profile, model, budget, DEFAULT_CALIBRATION_BUDGET_USD, FreshRunPricing(), "auto", {
        "include_question_ids": None,
        "exclude_question_ids": [GPT6_SELECTED35_EXCLUDED_ID],
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FaithfulRunError(f"{path} must contain a JSON object")
    return value


def _manifest_hash(path: Path) -> str:
    """Hash a manifest as bytes for exact artifact binding."""
    return sha256_file(path)


def _dataset_hash(path: Path) -> str:
    """Hash a dataset as bytes without exposing its content in receipts."""
    return sha256_file(path)


def _normalize_selection_policy(
    ordered_ids: Sequence[str],
    *,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, list[str]] | None]:
    """Validate an explicit opt-in subset/exclusion over canonical manifest IDs."""
    if include_question_ids is None and exclude_question_ids is None:
        return list(ordered_ids), None

    def validate_ids(values: Sequence[str] | None, label: str) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise FaithfulRunError(f"{label} must be a sequence of manifest question IDs")
        result = list(values)
        if any(not isinstance(value, str) or not value.strip() for value in result):
            raise FaithfulRunError(f"{label} must contain non-empty question IDs")
        if len(result) != len(set(result)):
            raise FaithfulRunError(f"{label} contains duplicate question IDs")
        unknown = [value for value in result if value not in set(ordered_ids)]
        if unknown:
            raise FaithfulRunError(f"{label} contains IDs outside the canonical manifest: {unknown[:3]}")
        return result

    included = validate_ids(include_question_ids, "include_question_ids")
    excluded = validate_ids(exclude_question_ids, "exclude_question_ids")
    overlap = set(included) & set(excluded)
    if overlap:
        raise FaithfulRunError(f"selection policy both includes and excludes IDs: {sorted(overlap)[:3]}")
    selected_set = set(included) if include_question_ids is not None else set(ordered_ids)
    selected_set.difference_update(excluded)
    selected = [question_id for question_id in ordered_ids if question_id in selected_set]
    if not selected:
        raise FaithfulRunError("explicit selection policy selected no manifest cases")
    return selected, {
        "include_question_ids": (
            [question_id for question_id in ordered_ids if question_id in set(included)]
            if include_question_ids is not None else None
        ),
        "exclude_question_ids": (
            [question_id for question_id in ordered_ids if question_id in set(excluded)]
            if exclude_question_ids is not None else None
        ),
    }


def _select_instances(
    dataset_path: Path,
    manifest_path: Path,
    *,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
) -> tuple[list[Instance], dict[str, Any]]:
    """Select manifest IDs in order; any subset/exclusion must be explicit."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("selection"), dict):
        raise FaithfulRunError("source manifest selection is malformed")
    ordered = manifest["selection"].get("ordered_question_ids")
    if not isinstance(ordered, list) or any(not isinstance(item, str) or not item for item in ordered):
        raise FaithfulRunError("source manifest must contain ordered string question IDs")
    if len(ordered) != len(set(ordered)):
        raise FaithfulRunError("source manifest contains duplicate question IDs")
    if manifest.get("profile") == FULL_S_PROFILE:
        if len(ordered) != FULL_S_CASE_COUNT or manifest.get("arms") != ["turns"]:
            raise FaithfulRunError("full-S manifest must bind exactly 500 cases and the turns-only arm")
        if manifest.get("retrieval", {}).get("tier") != "turns":
            raise FaithfulRunError("full-S manifest retrieval tier must be turns")
        if manifest.get("ingest", {}).get("mode") != "dual":
            raise FaithfulRunError("full-S manifest must retain dual ingestion")
        if manifest.get("dataset", {}).get("sha256") != _dataset_hash(dataset_path):
            raise FaithfulRunError("full-S normalized dataset hash does not match manifest")
    elif len(ordered) != EXPECTED_CASE_COUNT:
        raise FaithfulRunError(f"standard faithful manifest must preserve exactly {EXPECTED_CASE_COUNT} cases")
    selected_ids, _ = _normalize_selection_policy(
        ordered,
        include_question_ids=include_question_ids,
        exclude_question_ids=exclude_question_ids,
    )
    instances = load_split(dataset_path)
    by_id = {instance.question_id: instance for instance in instances}
    if len(by_id) != len(instances):
        raise FaithfulRunError("dataset contains duplicate question_id values")
    missing = [question_id for question_id in selected_ids if question_id not in by_id]
    if missing:
        raise FaithfulRunError(f"dataset is missing selected IDs: {missing[:3]}")
    selected = [by_id[question_id] for question_id in selected_ids]
    if not selected or (
        include_question_ids is None and exclude_question_ids is None
        and len(selected) != len(ordered)
    ):
        raise FaithfulRunError("faithful runner selection does not match the complete source manifest")
    return selected, manifest


def _validate_full_s_source_hashes(
    manifest: Mapping[str, Any], *, root: Path = Path.cwd()
) -> None:
    """Reject full-S execution when any manifest-pinned source file changed."""
    if manifest.get("profile") != FULL_S_PROFILE:
        return
    pinned = manifest.get("source_hashes")
    if not isinstance(pinned, Mapping) or not pinned:
        raise LedgerBindingError("full-S manifest source_hashes must be a non-empty object")
    for relative, expected in sorted(pinned.items()):
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise LedgerBindingError(f"full-S manifest has an invalid source hash path: {relative!r}")
        path = Path(root) / relative
        if not path.is_file():
            raise LedgerBindingError(f"full-S pinned source file is missing: {relative}")
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise LedgerBindingError(f"full-S pinned source hash mismatch: {relative}")


def _validate_run_source_hashes(
    run_document: Mapping[str, Any], *, root: Path = Path.cwd()
) -> None:
    """Validate the preparation manifest that actually carries source_hashes.

    The prepared execution manifest (``paths.manifest``) records the binding
    and the preparation manifest path but carries no top-level ``profile`` or
    ``source_hashes`` fields, so handing it straight to the full-S validator
    silently skipped validation. For a full-S run, re-read the full-S
    preparation manifest fresh from the recorded ``manifest_path`` and
    validate it; absence of ``source_hashes`` is a hard refusal. Non-full-S
    profiles keep skipping legitimately.
    """
    binding = run_document.get("binding")
    if not isinstance(binding, Mapping) or binding.get("run_profile") != FULL_S_PROFILE:
        return
    manifest_path_text = run_document.get("manifest_path")
    if not isinstance(manifest_path_text, str) or not manifest_path_text.strip():
        raise LedgerBindingError("full-S run manifest does not record its preparation manifest_path")
    preparation_path = Path(manifest_path_text)
    if not preparation_path.is_file():
        raise LedgerBindingError(f"full-S preparation manifest is missing: {manifest_path_text}")
    preparation_manifest = json.loads(preparation_path.read_text(encoding="utf-8"))
    _validate_full_s_source_hashes(preparation_manifest, root=root)


def _validate_owner_id(owner_id: str) -> None:
    """Refuse owner identities the storage layer would silently downgrade.

    ``weft.db.connection.set_user_context_value`` validates the identity with
    the same alphanumeric-plus-hyphen/underscore predicate and, on rejection,
    skips the ``SET LOCAL app.user_id`` GUC; ``weft.store`` then stamps row
    ``user_id`` from ``current_setting('app.user_id', true)``, so a rejected
    owner would land NULL-user_id rows in RLS global scope. Mirror the
    storage predicate here and refuse before any gateway or write.
    """
    value = owner_id.strip() if isinstance(owner_id, str) else ""
    if not value or not value.replace("-", "").replace("_", "").isalnum():
        raise ExecutionGateError(
            f"owner_id must be a non-empty [A-Za-z0-9_-] identity; refused: {owner_id!r}"
        )


def _case_history_hash(instance: Instance) -> str:
    """Hash only public haystack history, excluding question and gold fields."""
    history = {
        "question_id": instance.question_id,
        "sessions": [
            {"session_id": session.session_id, "date": session.date,
             "turns": [{"role": turn.role, "content": turn.content} for turn in session.turns]}
            for session in instance.sessions
        ],
    }
    return canonical_hash(history)


FASTEMBED_PROVIDER = "fastembed"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_DIMENSIONS = 768


def _judge_source_hash(judge_root: Path | None) -> str:
    """Hash the exact preserved evaluator source used for official judging."""
    if judge_root is None:
        return ""
    source = Path(judge_root) / "src" / "evaluation" / "evaluate_qa.py"
    if not source.is_file():
        raise ExecutionGateError(f"preserved evaluator not found: {source}")
    return sha256_file(source)


def _selection_policy_from_binding(binding: Mapping[str, Any]) -> dict[str, list[str] | None]:
    """Load the explicit selection fields persisted in a prepared binding."""
    include_ids = json.loads(str(binding.get("selection_policy_include_question_ids", "null")))
    exclude_ids = json.loads(str(binding.get("selection_policy_exclude_question_ids", "null")))
    if include_ids is not None and not isinstance(include_ids, list):
        raise LedgerBindingError("prepared include selection policy is invalid")
    if exclude_ids is not None and not isinstance(exclude_ids, list):
        raise LedgerBindingError("prepared exclude selection policy is invalid")
    return {"include_question_ids": include_ids, "exclude_question_ids": exclude_ids}


def _effective_selection_policy(
    binding: Mapping[str, Any],
    ordered_ids: Sequence[str],
    *,
    include_question_ids: Sequence[str] | None,
    exclude_question_ids: Sequence[str] | None,
) -> dict[str, list[str] | None]:
    """Use the bound policy by default; explicit options must match it."""
    if include_question_ids is None and exclude_question_ids is None:
        policy = _selection_policy_from_binding(binding)
    else:
        _, normalized = _normalize_selection_policy(
            ordered_ids,
            include_question_ids=include_question_ids,
            exclude_question_ids=exclude_question_ids,
        )
        policy = normalized or {"include_question_ids": None, "exclude_question_ids": None}
    if canonical_hash(policy) != binding.get("selection_policy_sha256"):
        raise LedgerBindingError("selection policy differs from prepared binding")
    return policy


def _operational_stop_for_profile(profile: str, max_budget_usd: float) -> float:
    """Resolve the early run stop without changing any profile's hard cap."""
    return FULL_S_OPERATIONAL_STOP_USD if profile == FULL_S_PROFILE else max_budget_usd


def _operational_stop_from_binding(binding: Mapping[str, Any]) -> float:
    """Resolve the immutable operational stop recorded by a prepared profile."""
    profile = str(binding.get("run_profile", "legacy"))
    if profile == "legacy":
        return DEFAULT_TOTAL_BUDGET_USD
    expected = FULL_S_OPERATIONAL_STOP_USD if profile == FULL_S_PROFILE else GPT6_TOTAL_BUDGET_USD
    try:
        actual = float(binding.get("operational_stop_usd", "nan"))
    except (TypeError, ValueError) as exc:
        raise LedgerBindingError("prepared operational stop is malformed") from exc
    if actual != expected:
        raise LedgerBindingError("prepared operational stop does not match its profile")
    return actual


def _binding(
    dataset_path: Path,
    manifest_path: Path,
    selected: Sequence[Instance],
    *,
    owner_id: str = "",
    agent_id: str = "faithful-s36",
    judge_root: Path | None = None,
    carry_forward: Mapping[str, Any] | None = None,
    selection_policy: Mapping[str, Sequence[str] | None] | None = None,
    profile: str = "legacy",
    writer_model: str = LUNA_MODEL,
    max_budget_usd: float = DEFAULT_TOTAL_BUDGET_USD,
    pricing: Any | None = None,
) -> dict[str, str]:
    """Build a run binding from sources, selection, model, pricing, and scope."""
    ids = [instance.question_id for instance in selected]
    policy = {
        "include_question_ids": (selection_policy or {}).get("include_question_ids"),
        "exclude_question_ids": (selection_policy or {}).get("exclude_question_ids"),
    }
    pricing = pricing or Pricing()
    legacy_policy = {
        "writer_model": LUNA_MODEL,
        "judge_model": GPT4O_JUDGE_MODEL,
        "anthropic_execution": False,
        "background_tools": False,
        "tool_schemas": {
            "writer": ["weft_remember"],
            "answerer": ["weft_prime", "weft_recall"],
        },
        "judge_prompt": "LongMemEval official get_anscheck_prompt",
    }
    profile_policy = dict(legacy_policy)
    retrieval_tier = "turns" if profile == FULL_S_PROFILE else "auto"
    calibration_budget = FULL_S_CALIBRATION_BUDGET_USD if profile == FULL_S_PROFILE else DEFAULT_CALIBRATION_BUDGET_USD
    operational_stop = _operational_stop_for_profile(profile, max_budget_usd)
    if profile != "legacy":
        profile_policy.update({
            "writer_model": writer_model,
            "retrieval_tier": retrieval_tier,
            "calibration_budget_usd": calibration_budget,
            "operational_stop_usd": operational_stop,
            "max_tool_rounds": MAX_TOOL_ROUNDS,
            "allow_final_response_after_tool_rounds": True,
            "max_responses": MAX_TOOL_ROUNDS + 1,
            "max_output_tokens": FRESH_GPT6_MAX_OUTPUT_TOKENS,
            "final_response_tools": False,
            "final_response_instruction": "This is the final response. Do not call tools; finish with a concise text response or acknowledgement.",
        })
    pricing_document = asdict(pricing)
    binding = {
        "dataset_sha256": _dataset_hash(dataset_path),
        "manifest_sha256": _manifest_hash(manifest_path),
        "selection_sha256": canonical_hash(ids),
        "selection_policy_sha256": canonical_hash(policy),
        "selection_policy_include_question_ids": json.dumps(
            policy["include_question_ids"], separators=(",", ":")
        ),
        "selection_policy_exclude_question_ids": json.dumps(
            policy["exclude_question_ids"], separators=(",", ":")
        ),
        "history_sha256": canonical_hash([_case_history_hash(instance) for instance in selected]),
        "owner_id": owner_id,
        "agent_id": agent_id,
        "scope_sha256": canonical_hash({"owner_id": owner_id, "agent_id": agent_id}),
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "execution_schema": EXECUTION_SCHEMA,
        "policy_sha256": canonical_hash(legacy_policy if profile == "legacy" else profile_policy),
        "embedding_provider": FASTEMBED_PROVIDER,
        "embedding_model": FASTEMBED_MODEL,
        "embedding_dimensions": str(FASTEMBED_DIMENSIONS),
        "judge_source_sha256": _judge_source_hash(judge_root),
        "pricing_sha256": canonical_hash(pricing_document),
        "prior_ledger_path": str((carry_forward or {}).get("source_path", "")),
        "prior_ledger_sha256": str((carry_forward or {}).get("source_sha256", "")),
        "prior_ledger_path_digest": str((carry_forward or {}).get("source_path_digest", "")),
        "prior_ledger_import_sum_usd": str((carry_forward or {}).get("imported_usd", 0.0)),
        "prior_ledger_import_calibration_usd": str((carry_forward or {}).get("imported_calibration_usd", 0.0)),
        "prior_ledger_reservation_count": str((carry_forward or {}).get("reservation_count", 0)),
    }
    if profile != "legacy":
        binding.update({
            "run_profile": profile,
            "writer_model": writer_model,
            "max_budget_usd": str(max_budget_usd),
            "calibration_budget_usd": str(calibration_budget),
            "operational_stop_usd": str(operational_stop),
            "retrieval_tier": retrieval_tier,
            "pricing_json": json.dumps(pricing_document, sort_keys=True, separators=(",", ":")),
            "tool_round_policy_json": json.dumps({
                "max_tool_rounds": MAX_TOOL_ROUNDS,
                "allow_final_response_after_tool_rounds": True,
                "max_responses": MAX_TOOL_ROUNDS + 1,
                "max_output_tokens": FRESH_GPT6_MAX_OUTPUT_TOKENS,
                "final_response_tools": False,
                "final_response_instruction": "This is the final response. Do not call tools; finish with a concise text response or acknowledgement.",
            }, sort_keys=True, separators=(",", ":")),
            "tool_round_policy_sha256": canonical_hash({
                "max_tool_rounds": MAX_TOOL_ROUNDS,
                "allow_final_response_after_tool_rounds": True,
                "max_responses": MAX_TOOL_ROUNDS + 1,
                "max_output_tokens": FRESH_GPT6_MAX_OUTPUT_TOKENS,
                "final_response_tools": False,
                "final_response_instruction": "This is the final response. Do not call tools; finish with a concise text response or acknowledgement.",
            }),
        })
    return binding


def _agent_policy_for_profile(profile: str) -> AgentPolicy:
    """Return the bounded writer policy pinned by the selected run profile."""
    if profile in {FRESH_RUN_PROFILE, FULL_S_PROFILE}:
        return AgentPolicy(
            max_tool_rounds=MAX_TOOL_ROUNDS,
            max_output_tokens=FRESH_GPT6_MAX_OUTPUT_TOKENS,
            allow_final_response_after_tool_rounds=True,
        )
    if profile == "legacy":
        return AgentPolicy()
    raise FaithfulRunError(f"unsupported agent policy profile: {profile!r}")


def _profile_from_binding(binding: Mapping[str, Any]) -> tuple[str, str, float, float, float, Any, str]:
    """Resolve and verify the exact model, ceilings, tier, and pricing in a run."""
    profile = str(binding.get("run_profile", "legacy"))
    if profile == "legacy":
        if any(key in binding for key in (
            "writer_model", "max_budget_usd", "pricing_json", "tool_round_policy_json",
            "tool_round_policy_sha256", "calibration_budget_usd", "operational_stop_usd", "retrieval_tier",
        )):
            raise LedgerBindingError("legacy binding unexpectedly contains profile-specific options")
        return "legacy", LUNA_MODEL, DEFAULT_TOTAL_BUDGET_USD, DEFAULT_CALIBRATION_BUDGET_USD, DEFAULT_TOTAL_BUDGET_USD, Pricing(), "auto"
    if profile not in {FRESH_RUN_PROFILE, FULL_S_PROFILE}:
        raise LedgerBindingError(f"unsupported prepared run profile: {profile!r}")
    model = binding.get("writer_model")
    expected_model = FULL_S_WRITER_MODEL if profile == FULL_S_PROFILE else GPT6_LUNA_MODEL
    expected_budget = FULL_S_MAX_BUDGET_USD if profile == FULL_S_PROFILE else GPT6_TOTAL_BUDGET_USD
    expected_calibration_budget = FULL_S_CALIBRATION_BUDGET_USD if profile == FULL_S_PROFILE else DEFAULT_CALIBRATION_BUDGET_USD
    expected_operational_stop = FULL_S_OPERATIONAL_STOP_USD if profile == FULL_S_PROFILE else GPT6_TOTAL_BUDGET_USD
    expected_tier = "turns" if profile == FULL_S_PROFILE else "auto"
    try:
        budget = float(binding.get("max_budget_usd", "nan"))
        calibration_budget = float(binding.get("calibration_budget_usd", "nan"))
        operational_stop = float(binding.get("operational_stop_usd", "nan"))
        pricing_data = json.loads(str(binding.get("pricing_json", "")))
        round_policy = json.loads(str(binding.get("tool_round_policy_json", "")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LedgerBindingError("profile binding options are malformed") from exc
    expected_pricing = asdict(FreshRunPricing())
    expected_round_policy = {
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "allow_final_response_after_tool_rounds": True,
        "max_responses": MAX_TOOL_ROUNDS + 1,
        "max_output_tokens": FRESH_GPT6_MAX_OUTPUT_TOKENS,
        "final_response_tools": False,
        "final_response_instruction": "This is the final response. Do not call tools; finish with a concise text response or acknowledgement.",
    }
    if model != expected_model or budget != expected_budget:
        raise LedgerBindingError(f"{profile} binding has an unexpected writer model or total budget")
    if (
        calibration_budget != expected_calibration_budget
        or operational_stop != expected_operational_stop
        or binding.get("retrieval_tier") != expected_tier
    ):
        raise LedgerBindingError(f"{profile} calibration ceiling, operational stop, or retrieval tier is invalid")
    if pricing_data != expected_pricing or canonical_hash(pricing_data) != binding.get("pricing_sha256"):
        raise LedgerBindingError("profile pricing snapshot is invalid")
    if (
        round_policy != expected_round_policy
        or canonical_hash(round_policy) != binding.get("tool_round_policy_sha256")
    ):
        raise LedgerBindingError("profile tool-round policy is invalid")
    return profile, model, budget, calibration_budget, operational_stop, FreshRunPricing(), expected_tier


def _carry_forward_from_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read and verify the immutable prior ledger recorded in a binding."""
    path = str(binding.get("prior_ledger_path", ""))
    if not path:
        return {}
    carry = import_prior_ledger(Path(path))
    expected = {
        "source_path": path,
        "source_sha256": str(binding.get("prior_ledger_sha256", "")),
        "source_path_digest": str(binding.get("prior_ledger_path_digest", "")),
        "imported_usd": float(binding.get("prior_ledger_import_sum_usd", "0")),
        "imported_calibration_usd": float(binding.get("prior_ledger_import_calibration_usd", "0")),
        "reservation_count": int(binding.get("prior_ledger_reservation_count", "0")),
    }
    actual = {key: carry[key] for key in expected}
    if actual != expected:
        raise LedgerBindingError("prior ledger changed since preparation")
    return carry


def prepare_run(
    dataset_path: Path,
    manifest_path: Path,
    *,
    paths: RunPaths | None = None,
    owner_id: str = "",
    agent_id: str = "faithful-s36",
    judge_root: Path | None = None,
    prior_ledger: Path | None = None,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
    profile: str = "legacy",
    writer_model: str | None = None,
    max_budget_usd: float | None = None,
) -> PreparedRun:
    """Prepare a hash-bound run offline; no provider, DB, or embedding calls occur."""
    if not isinstance(owner_id, str):
        raise ExecutionGateError("owner_id must be a string")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ExecutionGateError("agent_id is required to bind the faithful run")
    # API callers may prepare an offline fixture without identity; the CLI
    # requires an explicit owner before it reaches this function.  Keep the
    # deterministic fixture identity for backwards-compatible test seams.
    owner_id = owner_id.strip() or "faithful-s36-offline"
    default_root = (
        GPT6_FULL_S_ARTIFACT_NAMESPACE if profile == FULL_S_PROFILE
        else GPT6_SELECTED35_ARTIFACT_NAMESPACE if profile == FRESH_RUN_PROFILE
        else ARTIFACT_NAMESPACE
    )
    paths = paths or RunPaths.from_root(default_root)
    if profile == FRESH_RUN_PROFILE and (
        include_question_ids is not None
        or exclude_question_ids not in (None, [GPT6_SELECTED35_EXCLUDED_ID])
    ):
        raise FaithfulRunError("fresh profile selection is fixed to selected-35 excluding 7527f7e2")
    if profile == FRESH_RUN_PROFILE and exclude_question_ids is None:
        exclude_question_ids = [GPT6_SELECTED35_EXCLUDED_ID]
    if profile == FULL_S_PROFILE and (include_question_ids is not None or exclude_question_ids is not None):
        raise FaithfulRunError("full-S profile is fixed to all 500 manifest cases")
    (resolved_profile, resolved_model, resolved_budget, calibration_budget,
     pricing, retrieval_tier, _) = _resolve_run_profile(
        profile=profile, writer_model=writer_model, max_budget_usd=max_budget_usd,
    )
    operational_stop = _operational_stop_for_profile(resolved_profile, resolved_budget)
    selected, manifest = _select_instances(
        dataset_path, manifest_path,
        include_question_ids=include_question_ids,
        exclude_question_ids=exclude_question_ids,
    )
    if resolved_profile == FULL_S_PROFILE and manifest.get("profile") != FULL_S_PROFILE:
        raise FaithfulRunError("full-S profile requires its full-S source manifest")
    if resolved_profile != FULL_S_PROFILE and manifest.get("profile") == FULL_S_PROFILE:
        raise FaithfulRunError("full-S source manifest requires the full-S runner profile")
    carry_forward = import_prior_ledger(prior_ledger) if prior_ledger is not None else {}
    _, selection_policy = _normalize_selection_policy(
        manifest["selection"]["ordered_question_ids"],
        include_question_ids=include_question_ids,
        exclude_question_ids=exclude_question_ids,
    )
    if selection_policy is None:
        selection_policy = {"include_question_ids": None, "exclude_question_ids": None}
    expected_selection_policy = (
        {"include_question_ids": None, "exclude_question_ids": None}
        if resolved_profile == FULL_S_PROFILE
        else {"include_question_ids": None, "exclude_question_ids": [GPT6_SELECTED35_EXCLUDED_ID]}
        if resolved_profile == FRESH_RUN_PROFILE
        else selection_policy
    )
    if resolved_profile != "legacy" and selection_policy != expected_selection_policy:
        raise FaithfulRunError(f"{resolved_profile} selection policy does not match its fixed cohort")
    binding = _binding(
        dataset_path, manifest_path, selected,
        owner_id=owner_id, agent_id=agent_id, judge_root=judge_root,
        carry_forward=carry_forward, selection_policy=selection_policy,
        profile=resolved_profile, writer_model=resolved_model,
        max_budget_usd=resolved_budget, pricing=pricing,
    )
    rows = [
        {
            "index": index,
            "question_id": instance.question_id,
            "session_ids": [session.session_id for session in instance.sessions],
            "history_sha256": _case_history_hash(instance),
            "status": "pending",
        }
        for index, instance in enumerate(selected)
    ]
    manifest_doc = {
        "schema": EXECUTION_SCHEMA,
        "status": "PREPARED_NOT_AUTHORIZED",
        "binding": binding,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "case_count": len(selected),
        "ordered_question_ids": [instance.question_id for instance in selected],
        "sessions": rows,
        "provider_policy": {
            "writer_answerer": resolved_model,
            "judge": GPT4O_JUDGE_MODEL,
            "anthropic_execution": False,
            "fastembed_required": True,
            "max_retries": 0,
            "background_tools": False,
            "accounting_basis": "local conservative estimate; not an invoice guarantee",
        },
    }
    # Preparation is idempotent and never destroys progress from a prior run.
    if paths.manifest.exists():
        existing = _read_json(paths.manifest)
        if existing.get("binding") != binding:
            raise LedgerBindingError("prepared manifest binding differs; refusing destructive prepare")
    else:
        _atomic_json(paths.manifest, manifest_doc)
    if not paths.checkpoint.exists():
        _atomic_json(paths.checkpoint, {
            "schema": CHECKPOINT_SCHEMA,
            "binding": binding,
            "case_count": len(selected),
            "completed_question_ids": [],
            "failed_question_ids": [],
            "in_flight": None,
            "sessions": {},
            "carry_forward": {
                "prior_ledger_path": binding["prior_ledger_path"],
                "prior_ledger_sha256": binding["prior_ledger_sha256"],
                "prior_ledger_path_digest": binding["prior_ledger_path_digest"],
                "import_sum_usd": binding["prior_ledger_import_sum_usd"],
                "import_calibration_usd": binding["prior_ledger_import_calibration_usd"],
                "reservation_count": binding["prior_ledger_reservation_count"],
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    BudgetLedger(
        paths.ledger, max_budget_usd=resolved_budget,
        calibration_budget_usd=calibration_budget, operational_stop_usd=operational_stop,
        pricing=pricing,
        binding=binding, carry_forward=carry_forward,
    )
    return PreparedRun(paths, dataset_path, manifest_path,
                       tuple(instance.question_id for instance in selected), binding)


def _calibration_selection(selected: Sequence[Instance], limit: int = 4) -> list[Instance]:
    """Select deterministic representatives by preserved history-size order only."""
    if limit < 1:
        raise ValueError("calibration case limit must be positive")
    buckets: dict[str, Instance] = {}
    for instance in selected:
        history_bucket = "small" if len(instance.sessions) <= 1 else "medium" if len(instance.sessions) <= 3 else "large"
        buckets.setdefault(history_bucket, instance)
    representatives = [buckets[key] for key in ("small", "medium", "large") if key in buckets]
    chosen = {instance.question_id for instance in representatives}
    for instance in selected:
        if len(representatives) >= limit:
            break
        if instance.question_id not in chosen:
            representatives.append(instance)
            chosen.add(instance.question_id)
    return representatives[:limit]


async def _close_owned(value: Any) -> None:
    """Close an owned async resource, including the lazy OpenAI wrapper."""
    close = getattr(value, "close", None)
    if not callable(close) and isinstance(value, OpenAIResponsesClient):
        close = getattr(getattr(value, "_client", None), "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


async def run_calibration(
    dataset_path: Path,
    manifest_path: Path,
    *,
    paths: RunPaths,
    dsn: str,
    owner_id: str,
    case_limit: int = 4,
    judge_root: Path | None = None,
    gateway_factory: Callable[..., Awaitable[Any]] | None = None,
    client_factory: Callable[..., Any] | None = None,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run real bounded writer/answer/judge calibration against the public gateway."""
    _validate_owner_id(owner_id)
    manifest = _read_json(paths.manifest)
    _validate_run_source_hashes(manifest)
    binding = manifest["binding"]
    run_profile, writer_model, max_budget_usd, calibration_budget, operational_stop, pricing, retrieval_tier = _profile_from_binding(binding)
    source_manifest = load_manifest(
        manifest_path,
        expected_case_count=FULL_S_CASE_COUNT if run_profile == FULL_S_PROFILE else EXPECTED_CASE_COUNT,
    )
    carry_forward = _carry_forward_from_binding(binding)
    selection_policy = _effective_selection_policy(
        binding,
        source_manifest["selection"]["ordered_question_ids"],
        include_question_ids=include_question_ids,
        exclude_question_ids=exclude_question_ids,
    )
    selected, _ = _select_instances(
        dataset_path,
        manifest_path,
        include_question_ids=selection_policy["include_question_ids"],
        exclude_question_ids=selection_policy["exclude_question_ids"],
    )
    cases = _calibration_selection(selected, case_limit)
    current_binding = _binding(
        dataset_path, manifest_path, selected,
        owner_id=owner_id, agent_id="faithful-s36", judge_root=judge_root,
        carry_forward=carry_forward, selection_policy=selection_policy,
        profile=run_profile, writer_model=writer_model,
        max_budget_usd=max_budget_usd, pricing=pricing,
    )
    if binding != current_binding:
        raise LedgerBindingError("calibration selection/source binding differs from prepared run")
    if binding.get("owner_id") != owner_id:
        raise LedgerBindingError("calibration owner identity does not match prepared binding")
    if binding.get("agent_id") != "faithful-s36":
        raise LedgerBindingError("calibration agent identity does not match prepared binding")
    expected_judge_hash = binding.get("judge_source_sha256", "")
    actual_judge_hash = _judge_source_hash(judge_root)
    if not judge_root or not actual_judge_hash:
        raise ExecutionGateError("judge root is required for official calibration prompt")
    if expected_judge_hash and expected_judge_hash != actual_judge_hash:
        raise LedgerBindingError("official judge source changed since preparation")
    existing = _read_json(paths.calibration) if paths.calibration.exists() else None
    if existing and existing.get("binding") == binding and existing.get("representative"):
        return dict(existing["representative"])
    ledger = BudgetLedger(
        paths.ledger, max_budget_usd=max_budget_usd,
        calibration_budget_usd=calibration_budget, operational_stop_usd=operational_stop,
        pricing=pricing, binding=binding, carry_forward=carry_forward,
    )
    from benchmarks.longmemeval.faithful_gateway import create_local_gateway
    embedder = require_fastembed()
    if gateway_factory is None:
        async def gateway_factory(**kwargs: Any) -> Any:
            return await create_local_gateway(dsn, **kwargs)
    client = client_factory(phase="calibration", ledger=ledger) if client_factory else OpenAIResponsesClient()
    selected_ids = [instance.question_id for instance in selected]
    checkpoint = _checkpoint(paths.checkpoint, selected_ids)
    if checkpoint.get("binding") != binding:
        raise LedgerBindingError("calibration checkpoint binding does not match prepared binding")
    if checkpoint.get("in_flight"):
        raise ExecutionGateError(
            f"checkpoint has unknown in-flight case {checkpoint['in_flight']!r}; inspect before calibration"
        )
    checkpoint_evidence = checkpoint.setdefault("evidence", {})
    measurements: list[dict[str, Any]] = []
    try:
        for instance in cases:
            project = f"longmemeval-{instance.question_id}"
            case_row: dict[str, Any] = checkpoint_evidence.setdefault(instance.question_id, {
                "question_id": instance.question_id, "status": "in_flight", "sessions": [],
            })
            # Persist the marker before constructing/using the writer.  A
            # failed calibration must remain operator-visible and unreplayed.
            checkpoint["in_flight"] = instance.question_id
            checkpoint["in_flight_stage"] = "gateway"
            _write_checkpoint(paths.checkpoint, checkpoint)
            gateway = None
            try:
                gateway = await gateway_factory(
                    owner_id=owner_id, project_id=project,
                    agent_id=binding["agent_id"], embedding=embedder,
                )
                # Rebuild the agent after every gateway scope change; it must
                # never retain a closed/previous immutable gateway.
                agent = FaithfulAgent(
                    client, ledger, tools=gateway, phase="calibration", model=writer_model,
                    policy=_agent_policy_for_profile(run_profile),
                    retrieval_tier=retrieval_tier,
                )
                judge = BoundedJudge(client, ledger, phase="calibration")
                for session in _sort_sessions(instance):
                    checkpoint["in_flight_stage"] = f"session:{session.session_id}"
                    _write_checkpoint(paths.checkpoint, checkpoint)
                    if run_profile == FULL_S_PROFILE:
                        # Ingest exactly like the run loop: ingest_session_dual
                        # is clear-then-insert per session, so a calibration
                        # attempt and a resumed run converge on the same rows
                        # instead of leaving LLM-chosen raw memories the
                        # deterministic per-session delete cannot match.
                        await _ingest_full_s_session(
                            instance, session,
                            pool=getattr(gateway, "_pool", None),
                            embedder=embedder,
                            owner_id=owner_id,
                            project_id=project,
                        )
                        case_row.setdefault("sessions", []).append({
                            "session_id": session.session_id, "date": session.date,
                            "result": {"mode": "dual", "ingested": True},
                            "tool_results": _gateway_evidence(gateway),
                        })
                    else:
                        session_result = await agent.write_session(
                            session, project_id=project, agent_id=binding["agent_id"]
                        )
                        case_row.setdefault("sessions", []).append({
                            "session_id": session.session_id, "date": session.date,
                            "result": _agent_result_evidence(session_result),
                            "tool_results": _gateway_evidence(gateway),
                        })
                    _write_checkpoint(paths.checkpoint, checkpoint)
                checkpoint["in_flight_stage"] = "answer"
                _write_checkpoint(paths.checkpoint, checkpoint)
                answer = await agent.answer(
                    question=instance.question,
                    question_date=instance.question_date,
                    task_shape=None,
                    recalled_context=None,
                    project_id=project,
                    agent_id=binding["agent_id"],
                )
                from benchmarks.longmemeval.judge import _official_prompt_loader
                prompt = _official_prompt_loader(judge_root)(
                    instance.question_type, instance.question, instance.answer, answer.text,
                    abstention=instance.is_abstention,
                )
                label, raw, reservation_id = await judge.judge(prompt)
                measurement = {
                    "question_id": instance.question_id,
                    "question_type": instance.question_type,
                    "history_sessions": len(instance.sessions),
                    "hypothesis": answer.text,
                    "judge_label": label,
                    "judge_raw": raw,
                    "judge_reservation_id": reservation_id,
                    "writer_answer_reservations": list(answer.reservations),
                }
                measurements.append(measurement)
                case_row.update({
                    "status": "completed", "result": measurement,
                    "tool_results": _gateway_evidence(gateway),
                })
                checkpoint["in_flight"] = None
                checkpoint["in_flight_stage"] = None
                _write_checkpoint(paths.checkpoint, checkpoint)
            except Exception as exc:
                case_row.update({
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "tool_results": _gateway_evidence(gateway),
                })
                checkpoint["last_error"] = case_row["error"]
                # Deliberately retain in_flight: the provider/tool attempt may
                # have committed, so calibration cannot silently replay it.
                _write_checkpoint(paths.checkpoint, checkpoint)
                raise
            finally:
                if gateway is not None:
                    await gateway.close()
    finally:
        await _close_owned(client)
    summary = ledger.summary()
    projection = {
        "calibration_cases": len(measurements),
        "measured_reserved_usd": summary["reserved_usd"],
        "measured_calibration_reserved_usd": summary["calibration_reserved_usd"],
        "projected_total_usd": summary["reserved_usd"] * (len(selected) / len(measurements)),
        "basis": "same cumulative ledger and measured per-case provider reservations",
        "ledger_sha256": canonical_hash(summary),
    }
    return {"completed": len(measurements) == len(cases), "cases": measurements, "ledger": summary, "projection": projection}


def issue_calibration_receipt(
    paths: RunPaths,
    *,
    approved_by: str = "",
    calibration_budget_usd: float = DEFAULT_CALIBRATION_BUDGET_USD,
    notes: str = "",
    execute: bool = False,
    calibration_runner: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run a bounded calibration and emit a measured approval hold.

    A name or free-form approval string never turns calibration into an
    authorization.  The representative runner must execute against the same
    budget ledger and return measured usage/case evidence; the resulting
    receipt remains ``HOLD_FOR_APPROVAL`` until a separate operator gate
    records measured projection and approval.
    """
    run_manifest = _read_json(paths.manifest)
    if run_manifest.get("schema") != EXECUTION_SCHEMA:
        raise FaithfulRunError("unsupported prepared manifest schema")
    _, _, _, bound_calibration_budget, _, _, _ = _profile_from_binding(run_manifest.get("binding", {}))
    if calibration_budget_usd <= 0 or calibration_budget_usd != bound_calibration_budget:
        raise ValueError("calibration budget must match the prepared profile binding")
    if not execute:
        raise ExecutionGateError("calibration requires explicit --execute; no receipt was issued")
    if calibration_runner is None:
        raise ExecutionGateError("real calibration gateway/provider is required; no fake calibration")
    result = asyncio.run(calibration_runner())
    if not isinstance(result, Mapping) or not result.get("completed"):
        raise ExecutionGateError("representative calibration did not complete successfully")
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "status": "HOLD_FOR_APPROVAL",
        "approved_by": approved_by.strip() or None,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "calibration_budget_usd": calibration_budget_usd,
        "notes": notes,
        "binding": run_manifest["binding"],
        "representative": dict(result),
        "projection": result.get("projection"),
        "approval_required": True,
    }
    _atomic_json(paths.calibration, receipt)
    return receipt


def approve_calibration(
    paths: RunPaths,
    *,
    approved_by: str,
    projected_total_usd: float,
    projection_basis: str,
) -> dict[str, Any]:
    """Approve a measured calibration hold with explicit projection evidence."""
    if not approved_by.strip() or not projection_basis.strip():
        raise ValueError("approved_by and projection_basis are required")
    if not isinstance(projected_total_usd, (int, float)) or isinstance(projected_total_usd, bool) or not math.isfinite(float(projected_total_usd)):
        raise ValueError("projected total must be finite")
    receipt = _read_json(paths.calibration)
    bound_profile, _, bound_budget, _, _, _, _ = _profile_from_binding(receipt.get("binding", {}))
    if projected_total_usd <= 0 or projected_total_usd > bound_budget:
        raise ValueError("projected total must be positive and within the prepared run budget")
    if receipt.get("schema") != CALIBRATION_SCHEMA or receipt.get("status") != "HOLD_FOR_APPROVAL":
        raise ExecutionGateError("calibration is not a measured hold awaiting approval")
    if receipt.get("approval_required") is not True or not receipt.get("representative"):
        raise ExecutionGateError("calibration lacks representative evidence")
    projection = receipt.get("projection")
    if not isinstance(projection, Mapping):
        raise ExecutionGateError("calibration lacks measured projection evidence")
    measured = projection.get("projected_total_usd")
    if not isinstance(measured, (int, float)) or isinstance(measured, bool) or not math.isfinite(float(measured)):
        raise ExecutionGateError("calibration projection is not finite")
    if abs(float(projected_total_usd) - float(measured)) > 1e-9:
        raise ExecutionGateError("approved projection does not match measured projection")
    receipt.update({
        "status": "CALIBRATED",
        "approved_by": approved_by.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "projected_total_usd": projected_total_usd,
        "projection_basis": projection_basis.strip(),
        "approval_required": False,
    })
    _atomic_json(paths.calibration, receipt)
    return receipt


def _require_execution_gate(paths: RunPaths, *, execute: bool, dsn: str | None) -> dict[str, Any]:
    """Require explicit execution, approved measured calibration, and local DSN."""
    if not execute:
        raise ExecutionGateError("resume is prepare-only by default; pass --execute explicitly")
    from benchmarks.longmemeval.faithful_gateway import validate_local_dsn
    validate_local_dsn(dsn or "")
    calibration = _read_json(paths.calibration)
    if calibration.get("status") != "CALIBRATED" or calibration.get("approval_required"):
        raise ExecutionGateError("calibration is measured but still HOLD_FOR_APPROVAL")
    manifest = _read_json(paths.manifest)
    if calibration.get("binding") != manifest.get("binding"):
        raise ExecutionGateError("calibration receipt binding does not match manifest")
    return calibration


def require_fastembed() -> object:
    """Verify the required local FastEmbed provider can be imported."""
    try:
        from weft.embeddings.fastembed_provider import FastEmbedProvider
    except ImportError as exc:
        raise ExecutionGateError("FastEmbed is required for faithful execution") from exc
    return FastEmbedProvider()


def _checkpoint(path: Path, selected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Read and validate a checkpoint object and its selected-case algebra."""
    value = _read_json(path)
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise FaithfulRunError("unsupported checkpoint schema")
    completed = value.get("completed_question_ids")
    failed = value.get("failed_question_ids")
    if not isinstance(completed, list) or not isinstance(failed, list):
        raise FaithfulRunError("checkpoint completed/failed fields must be lists")
    if any(not isinstance(item, str) or not item for item in completed + failed):
        raise FaithfulRunError("checkpoint IDs must be non-empty strings")
    if len(set(completed)) != len(completed) or len(set(failed)) != len(failed):
        raise FaithfulRunError("checkpoint IDs must not contain duplicates")
    if set(completed) & set(failed):
        raise FaithfulRunError("checkpoint completed and failed IDs must be disjoint")
    if selected_ids is not None:
        selected = set(selected_ids)
        if (set(completed) | set(failed)) - selected:
            raise FaithfulRunError("checkpoint contains IDs outside the selected manifest")
    in_flight = value.get("in_flight")
    if in_flight is not None and (not isinstance(in_flight, str) or not in_flight):
        raise FaithfulRunError("checkpoint in_flight must be null or a non-empty ID")
    if selected_ids is not None and in_flight is not None and in_flight not in set(selected_ids):
        raise FaithfulRunError("checkpoint in_flight is outside the selected manifest")
    sessions = value.get("sessions", {})
    if not isinstance(sessions, dict):
        raise FaithfulRunError("checkpoint sessions must be an object")
    for qid, states in sessions.items():
        if not isinstance(qid, str) or not isinstance(states, dict):
            raise FaithfulRunError("checkpoint session map is malformed")
        if selected_ids is not None and qid not in set(selected_ids):
            raise FaithfulRunError("checkpoint session map contains an unselected case")
        if any(not isinstance(sid, str) or state != "completed" for sid, state in states.items()):
            raise FaithfulRunError("checkpoint session states must be completed by session ID")
    return value


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    """Persist a checkpoint with an updated UTC timestamp."""
    updated = dict(value)
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, updated)


def _manifest_checkpoint_state(
    manifest: Mapping[str, Any], checkpoint: Mapping[str, Any], *,
    dataset_path: Path, manifest_path: Path, artifact_root: Path,
) -> dict[str, Any]:
    """Compare terminal checkpoint IDs against manifest order and derive resume CLI."""
    ordered_ids = manifest.get("ordered_question_ids")
    if not isinstance(ordered_ids, list) or any(not isinstance(item, str) or not item for item in ordered_ids):
        raise FaithfulRunError("prepared manifest ordered_question_ids is malformed")
    if len(ordered_ids) != len(set(ordered_ids)):
        raise FaithfulRunError("prepared manifest ordered_question_ids contains duplicates")
    completed = checkpoint.get("completed_question_ids", [])
    failed = checkpoint.get("failed_question_ids", [])
    if not isinstance(completed, list) or not isinstance(failed, list):
        raise FaithfulRunError("checkpoint terminal question IDs are malformed")
    terminal = set(completed) | set(failed)
    if terminal - set(ordered_ids):
        raise FaithfulRunError("checkpoint contains IDs outside manifest ordered_question_ids")
    pending = [question_id for question_id in ordered_ids if question_id not in terminal]
    last_completed = next((qid for qid in reversed(ordered_ids) if qid in set(completed)), None)
    binding = manifest.get("binding", {})
    profile = binding.get("run_profile", "legacy") if isinstance(binding, Mapping) else "legacy"
    model = binding.get("writer_model") if isinstance(binding, Mapping) else None
    budget = binding.get("max_budget_usd") if isinstance(binding, Mapping) else None
    command = [
        "uv", "run", "python", "-m", "benchmarks.longmemeval.faithful_s36", "resume",
        "--dataset", str(dataset_path), "--manifest", str(manifest_path),
        "--artifact-root", str(artifact_root), "--profile", str(profile),
        "--judge-root", "<PRESERVED_LONGMEMEVAL_ROOT>", "--execute", "--dsn", "<LOCAL_DISPOSABLE_DSN>",
    ]
    if model:
        command.extend(("--writer-model", str(model)))
    if budget is not None:
        command.extend(("--max-budget-usd", str(budget)))
    if isinstance(binding, Mapping):
        for field, option in (
            ("selection_policy_include_question_ids", "--include-question-id"),
            ("selection_policy_exclude_question_ids", "--exclude-question-id"),
        ):
            try:
                values = json.loads(str(binding.get(field, "null")))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise FaithfulRunError(f"prepared {field} is malformed") from exc
            if values is not None and (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise FaithfulRunError(f"prepared {field} must be null or a list of non-empty IDs")
            for value in values or ():
                command.extend((option, value))
    return {
        "selected_question_count": len(ordered_ids),
        "completed_question_ids": [qid for qid in ordered_ids if qid in set(completed)],
        "failed_question_ids": [qid for qid in ordered_ids if qid in set(failed)],
        "pending_question_ids": pending,
        "complete": not pending,
        "last_completed_question_id": last_completed,
        "resume_command": shlex.join(command),
        "resume_command_placeholders": {
            "judge_root": "<PRESERVED_LONGMEMEVAL_ROOT>",
            "dsn": "<LOCAL_DISPOSABLE_DSN>",
        },
    }


def _externally_terminated_pre_provider(
    checkpoint: Mapping[str, Any],
    ledger_path: Path,
    *,
    case_id: str,
) -> bool:
    """True only when a case was killed before any provider dispatch.

    Requires the case to be the checkpoint's in-flight marker, still
    non-terminal, with evidence recording no provider outcome — the row may
    still be in-flight, or finalized as failed by the runner's signal
    handler with no answer, judge result, or case error — and a ledger
    containing no reservation row that references the case at all, i.e.
    nothing outstanding and nothing ambiguous to lose.
    """
    if checkpoint.get("in_flight") != case_id:
        return False
    completed = checkpoint.get("completed_question_ids")
    if isinstance(completed, list) and case_id in completed:
        return False
    failed = checkpoint.get("failed_question_ids")
    if isinstance(failed, list) and case_id in failed:
        return False
    evidence = checkpoint.get("evidence")
    case_row = evidence.get(case_id) if isinstance(evidence, dict) else None
    if not isinstance(case_row, dict):
        return False
    # The invariant is that no provider outcome was recorded, not the
    # specific evidence status string: the kill can land before finalization
    # (status "in_flight") or after a signal handler finalizes the row as
    # "failed". A finalized failure carrying a case error is a genuine case
    # failure, not an external termination.
    if case_row.get("status") not in ("in_flight", "failed"):
        return False
    if case_row.get("answer") or case_row.get("judge"):
        return False
    if case_row.get("status") == "failed" and case_row.get("error") is not None:
        return False
    try:
        ledger_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    rows = ledger_doc.get("reservations") if isinstance(ledger_doc, dict) else None
    if not isinstance(rows, list):
        return False
    return not any(
        isinstance(row, dict) and case_id in json.dumps(row, sort_keys=True)
        for row in rows
    )


def _recover_inflight_case(
    paths: RunPaths,
    *,
    case_id: str,
    reservation_id: str,
    expected_completed_count: int,
    pinned: bool,
) -> dict[str, Any]:
    """Apply one offline, journaled recovery after exact operator confirmation."""
    if not isinstance(expected_completed_count, int) or isinstance(expected_completed_count, bool):
        raise FaithfulRunError("expected_completed_count must be an explicit integer")
    if expected_completed_count < 0:
        raise FaithfulRunError("expected_completed_count cannot be negative")
    if not isinstance(case_id, str) or not case_id.strip():
        raise FaithfulRunError("recovery requires one explicit selected case ID")
    if not isinstance(reservation_id, str) or (reservation_id and not reservation_id.strip()):
        raise FaithfulRunError(
            "recovery requires one explicit reservation ID, or none for a case "
            "externally terminated before any provider dispatch"
        )
    if pinned and case_id != SAFE_SKIP_CASE_ID:
        raise FaithfulRunError(f"safe-skip recovery only permits case {SAFE_SKIP_CASE_ID}")
    if not pinned and case_id == SAFE_SKIP_CASE_ID:
        raise FaithfulRunError(f"case {SAFE_SKIP_CASE_ID} must use the pinned safe-skip recovery command")
    if not all(path.is_file() for path in (paths.manifest, paths.checkpoint, paths.ledger)):
        raise FaithfulRunError("safe-skip recovery requires existing manifest, checkpoint, and ledger files")
    if paths.ledger.stat().st_size == 0:
        raise FaithfulRunError("safe-skip recovery refuses an empty budget ledger")

    if pinned:
        error = f"operator safe-skip recovery for case {case_id} reservation {reservation_id}"
    elif reservation_id:
        error = f"operator in-flight recovery for case {case_id} reservation {reservation_id}"
    else:
        error = (
            "externally terminated before provider dispatch "
            f"(operator recovery) for case {case_id}"
        )
    journal_schema = SAFE_SKIP_JOURNAL_SCHEMA if pinned else INFLIGHT_RECOVERY_JOURNAL_SCHEMA
    journal_suffix = ".safe-skip.json" if pinned else f".recover-{canonical_hash([case_id, reservation_id])[:16]}.json"
    journal_path = paths.checkpoint.with_suffix(paths.checkpoint.suffix + journal_suffix)
    with _exclusive_run_lock(paths.lock):
        manifest = _read_json(paths.manifest)
        binding = manifest.get("binding")
        selected_ids = manifest.get("ordered_question_ids")
        if not isinstance(binding, dict) or not isinstance(selected_ids, list):
            raise FaithfulRunError("prepared manifest binding or selected IDs are malformed")
        if (
            not selected_ids
            or any(not isinstance(item, str) or not item for item in selected_ids)
            or len(selected_ids) != len(set(selected_ids))
            or manifest.get("case_count") != len(selected_ids)
            or manifest.get("schema") != EXECUTION_SCHEMA
            or canonical_hash(selected_ids) != binding.get("selection_sha256")
        ):
            raise FaithfulRunError("prepared manifest case selection is invalid or does not match its binding")
        checkpoint = _checkpoint(paths.checkpoint, selected_ids)
        if checkpoint.get("binding") != binding:
            raise LedgerBindingError("checkpoint binding does not match prepared manifest")
        if checkpoint.get("case_count") != len(selected_ids):
            raise FaithfulRunError("checkpoint case count differs from prepared manifest")
        if case_id not in selected_ids:
            raise FaithfulRunError("recovery case is outside the prepared selection")

        _, writer_model, max_budget_usd, calibration_budget, operational_stop, pricing, _ = _profile_from_binding(binding)
        carry_forward = _carry_forward_from_binding(binding)
        ledger = BudgetLedger(
            paths.ledger, max_budget_usd=max_budget_usd,
            calibration_budget_usd=calibration_budget, operational_stop_usd=operational_stop,
            pricing=pricing,
            binding=binding, carry_forward=carry_forward,
        )
        terminated_pre_provider = not pinned and not reservation_id
        if terminated_pre_provider:
            # No reservation can exist: the run died during tool/session work
            # before any provider dispatch, so the ledger must not change.
            target_reservation = {}
            expected_model = writer_model
            ledger_row = None
            ledger_state = {"reservation_ids": [], "status": "none", "changed": False}
        else:
            if not reservation_id:
                raise FaithfulRunError(
                    "recovery requires one explicit reservation ID, or none for a case "
                    "externally terminated before any provider dispatch"
                )
            target_reservation = ledger.inspect_reservation(reservation_id)
            expected_model = target_reservation.get("model")
            if expected_model not in {writer_model, GPT4O_JUDGE_MODEL}:
                raise FaithfulRunError("reservation model is outside the bound writer/judge models")
        operation_id = canonical_hash({
            "binding": binding,
            "case_id": case_id,
            "reservation_id": reservation_id,
            "expected_model": expected_model,
            "error": error,
            "expected_completed_count": expected_completed_count,
            "journal_schema": journal_schema,
        })
        current_hash = canonical_hash(checkpoint)
        journal: dict[str, Any] | None = None
        if journal_path.exists():
            journal = _read_json(journal_path)
            expected_journal = {
                "schema": journal_schema,
                "operation_id": operation_id,
                "binding": binding,
                "case_id": case_id,
                "reservation_id": reservation_id,
                "expected_model": expected_model,
                "error": error,
                "expected_completed_count": expected_completed_count,
            }
            if any(journal.get(key) != value for key, value in expected_journal.items()):
                raise FaithfulRunError("safe-skip journal does not match the requested recovery")
            checkpoint_after = journal.get("checkpoint_after")
            if not isinstance(checkpoint_after, dict) or canonical_hash(checkpoint_after) != journal.get("checkpoint_after_sha256"):
                raise FaithfulRunError("safe-skip journal target checkpoint is invalid")
            before_hash = journal.get("checkpoint_before_sha256")
            after_hash = journal["checkpoint_after_sha256"]
        else:
            journal = None
            before_hash = current_hash
            after_hash = None
            checkpoint_after = None

        target_is_prefinalized_timeout = (
            target_reservation.get("status") == "unknown"
            and target_reservation.get("error") == PROVIDER_TIMEOUT_LEDGER_ERROR
        )
        if terminated_pre_provider:
            pass  # ledger_row/ledger_state established above; nothing to inspect
        elif not pinned and target_is_prefinalized_timeout:
            ledger_row = ledger.inspect_prefinalized_timeout_reservation(
                reservation_id,
                expected_model=expected_model,
                error=PROVIDER_TIMEOUT_LEDGER_ERROR,
            )
            ledger_state = {
                "reservation_ids": [reservation_id], "status": "unknown", "changed": False,
            }
        else:
            ledger_row = None
            ledger_state = ledger.inspect_unknown_batch(
                [reservation_id], expected_model=expected_model, error=error,
            )
        if journal is None:
            completed = checkpoint.get("completed_question_ids", [])
            failed = checkpoint.get("failed_question_ids", [])
            evidence = checkpoint.get("evidence", {})
            case_row = evidence.get(case_id) if isinstance(evidence, dict) else None
            if len(completed) != expected_completed_count:
                raise FaithfulRunError(
                    f"recovery expected {expected_completed_count} completed cases, found {len(completed)}"
                )
            if checkpoint.get("in_flight") != case_id:
                raise FaithfulRunError("checkpoint in_flight does not match the requested case")
            if case_id in completed or case_id in failed:
                raise FaithfulRunError("recovery case is already terminal in the checkpoint")
            if not isinstance(case_row, dict):
                raise FaithfulRunError("target case evidence is malformed before recovery")
            if target_is_prefinalized_timeout:
                if (
                    case_row.get("status") != "in_flight"
                    or case_row.get("error") != PROVIDER_TIMEOUT_EVIDENCE_ERROR
                    or ledger_row is None
                    or ledger_row.get("error") != PROVIDER_TIMEOUT_LEDGER_ERROR
                ):
                    raise FaithfulRunError(
                        "pre-finalized timeout recovery requires matching in-flight ambiguous evidence"
                    )
                if ledger_state != {
                    "reservation_ids": [reservation_id], "status": "unknown", "changed": False,
                }:
                    raise FaithfulRunError("provider timeout reservation is not already unknown")
            elif terminated_pre_provider:
                if not _externally_terminated_pre_provider(checkpoint, paths.ledger, case_id=case_id):
                    raise FaithfulRunError(
                        "recovery without a reservation ID requires a case externally terminated "
                        "before any provider dispatch: evidence with no provider outcome recorded "
                        "(no answer, judge result, or case error) and zero ledger reservations "
                        "for the case"
                    )
            else:
                if case_row.get("status") != "failed" or case_row.get("error") is not None:
                    raise FaithfulRunError(
                        "target case evidence must be failed with no case error before recovery"
                    )
                if ledger_state != {
                    "reservation_ids": [reservation_id], "status": "reserved", "changed": True,
                }:
                    raise FaithfulRunError("reservation is not an unrecovered outstanding target")

            checkpoint_after = json.loads(json.dumps(checkpoint))
            checkpoint_after["failed_question_ids"] = sorted(set(failed) | {case_id})
            checkpoint_after["in_flight"] = None
            checkpoint_after["in_flight_stage"] = None
            checkpoint_after["updated_at"] = datetime.now(timezone.utc).isoformat()
            case_after = checkpoint_after["evidence"][case_id]
            case_after["status"] = "failed"
            case_after["error"] = (
                case_row["error"] if target_is_prefinalized_timeout else error
            )
            case_after["safe_skip_recovery"] = {
                "operation_id": operation_id,
                "reservation_id": reservation_id,
                "reason": error,
                "ledger_already_unknown": target_is_prefinalized_timeout,
            }
            after_hash = canonical_hash(checkpoint_after)
            journal = {
                "schema": journal_schema,
                "operation_id": operation_id,
                "binding": binding,
                "case_id": case_id,
                "reservation_id": reservation_id,
                "expected_model": expected_model,
                "error": error,
                "expected_completed_count": expected_completed_count,
                "checkpoint_before_sha256": before_hash,
                "checkpoint_after": checkpoint_after,
                "checkpoint_after_sha256": after_hash,
            }
            _atomic_json(journal_path, journal)
        else:
            if current_hash not in {before_hash, after_hash}:
                raise FaithfulRunError("checkpoint changed since safe-skip journal; refusing recovery")
            if current_hash == after_hash:
                if terminated_pre_provider:
                    raise FaithfulRunError(
                        "checkpoint is already recovered; the externally terminated case is terminal"
                    )
                if ledger_state["status"] != "unknown":
                    raise FaithfulRunError("checkpoint is recovered but reservation is not unknown")
                return {
                    "status": "recovered", "case_id": case_id,
                    "reservation_id": reservation_id, "changed": False,
                    "operation_id": operation_id,
                }

        if terminated_pre_provider:
            pass  # nothing was dispatched; there is no reservation to finalize
        elif target_is_prefinalized_timeout:
            verified_row = ledger.inspect_prefinalized_timeout_reservation(
                reservation_id,
                expected_model=expected_model,
                error=PROVIDER_TIMEOUT_LEDGER_ERROR,
            )
            if verified_row.get("estimated_usd") != target_reservation.get("estimated_usd"):
                raise FaithfulRunError("provider timeout reservation estimate changed during recovery")
        else:
            ledger.mark_unknown_batch(
                [reservation_id], expected_model=expected_model, error=error,
            )
            verified_ledger_state = ledger.inspect_unknown_batch(
                [reservation_id], expected_model=expected_model, error=error,
            )
            if verified_ledger_state["status"] != "unknown":
                raise FaithfulRunError("ledger did not persist unknown reservation state")
        assert checkpoint_after is not None
        _atomic_json(paths.checkpoint, checkpoint_after)
        return {
            "status": "recovered", "case_id": case_id,
            "reservation_id": reservation_id, "changed": True,
            "operation_id": operation_id,
        }


def recover_safe_skip(
    paths: RunPaths,
    *,
    case_id: str,
    reservation_id: str,
) -> dict[str, Any]:
    """Run the legacy pinned safe-skip recovery for case 95228167 (11 done)."""
    return _recover_inflight_case(
        paths,
        case_id=case_id,
        reservation_id=reservation_id,
        expected_completed_count=11,
        pinned=True,
    )


def recover_inflight_case(
    paths: RunPaths,
    *,
    case_id: str,
    reservation_id: str = "",
    expected_completed_count: int,
) -> dict[str, Any]:
    """Offline recovery for another explicitly selected in-flight case.

    The completed count must be confirmed by the caller. With a reservation
    ID this recovers the two provider-dispatch shapes (pre-finalized timeout
    or one outstanding reserved reservation). With no reservation ID it
    recovers only a case externally terminated before any provider dispatch:
    evidence recording no provider outcome (in-flight, or finalized failed
    with no answer, judge result, or case error) and zero ledger
    reservations for the case. This generic path cannot recover the case
    reserved for the pinned ``recover_safe_skip`` API.
    """
    return _recover_inflight_case(
        paths,
        case_id=case_id,
        reservation_id=reservation_id,
        expected_completed_count=expected_completed_count,
        pinned=False,
    )


async def _ingest_full_s_session(
    instance: Instance,
    session: Session,
    *,
    pool: Any,
    embedder: Any,
    owner_id: str,
    project_id: str,
) -> None:
    """Idempotently dual-ingest one full-S session (raw memory + episode turns)."""
    from benchmarks.longmemeval.ingest import ingest_session_dual
    from weft.auth import current_user_id

    if pool is None:
        raise ExecutionGateError(
            "full-S dual ingestion requires a database-backed gateway pool"
        )
    token = current_user_id.set(owner_id)
    try:
        await ingest_session_dual(pool, embedder, instance, session, project_id)
    finally:
        current_user_id.reset(token)


def _agent_result_evidence(result: Any) -> dict[str, Any]:
    """Serialize an AgentResult without retaining hidden prompts or gold."""
    return {
        "text": getattr(result, "text", ""),
        "model": getattr(result, "model", ""),
        "calls": getattr(result, "calls", 0),
        "tool_calls": getattr(result, "tool_calls", 0),
        "reservations": list(getattr(result, "reservations", ())),
        "usage": [usage.as_dict() for usage in getattr(result, "usage", ())],
    }


def _gateway_evidence(gateway: Any) -> list[dict[str, Any]]:
    """Return public tool evidence when the gateway exposes an audit log."""
    calls = getattr(gateway, "calls", ())
    return [dict(call) for call in calls] if isinstance(calls, (list, tuple)) else []


@contextmanager
def _exclusive_run_lock(path: Path):
    """Hold an exclusive lock for the whole run, not just ledger updates."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExecutionGateError("another faithful run already holds the run lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sort_sessions(instance: Instance) -> tuple[Any, ...]:
    """Sort sessions by parseable date then original position, without gold."""
    from datetime import datetime
    def key(item: tuple[int, Any]) -> tuple[int, str, int]:
        index, session = item
        raw = session.date.replace("/", "-").split(" (")[0]
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d")
            return (0, parsed.isoformat(), index)
        except ValueError:
            return (1, raw, index)
    return tuple(session for _, session in sorted(enumerate(instance.sessions), key=key))


async def _resume_run_locked(
    dataset_path: Path,
    manifest_path: Path,
    *,
    paths: RunPaths | None = None,
    execute: bool = False,
    dsn: str | None = None,
    agent_factory: Callable[[BudgetLedger], FaithfulAgent] | None = None,
    tool_gateway: object | None = None,
    judge_root: Path | None = None,
    judge_client: Any | None = None,
    gateway_factory: Callable[..., Awaitable[Any]] | None = None,
    client_factory: Callable[..., Any] | None = None,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resume pending cases with explicit authorization and durable checkpoints.

    The default is intentionally fail-closed. Production integration supplies
    a public-tool gateway; this module does not create a pool or inherit a
    production DSN. Unknown provider outcomes remain in-flight and are not
    auto-replayed.
    """
    paths = paths or RunPaths.from_root()
    calibration = _require_execution_gate(paths, execute=execute, dsn=dsn)
    manifest = _read_json(paths.manifest)
    _validate_run_source_hashes(manifest)
    binding = manifest["binding"]
    _validate_owner_id(str(binding.get("owner_id", "")))
    run_profile, writer_model, max_budget_usd, calibration_budget, operational_stop, pricing, retrieval_tier = _profile_from_binding(binding)
    source_manifest = load_manifest(
        manifest_path,
        expected_case_count=FULL_S_CASE_COUNT if run_profile == FULL_S_PROFILE else EXPECTED_CASE_COUNT,
    )
    selection_policy = _effective_selection_policy(
        binding,
        source_manifest["selection"]["ordered_question_ids"],
        include_question_ids=include_question_ids,
        exclude_question_ids=exclude_question_ids,
    )
    selected, source_manifest = _select_instances(
        dataset_path,
        manifest_path,
        include_question_ids=selection_policy["include_question_ids"],
        exclude_question_ids=selection_policy["exclude_question_ids"],
    )
    carry_forward = _carry_forward_from_binding(binding)
    current_binding = _binding(
        dataset_path,
        manifest_path,
        selected,
        owner_id=binding.get("owner_id", ""),
        agent_id=binding.get("agent_id", "faithful-s36"),
        judge_root=judge_root if binding.get("judge_source_sha256") else None,
        carry_forward=carry_forward,
        selection_policy=selection_policy,
        profile=run_profile, writer_model=writer_model,
        max_budget_usd=max_budget_usd, pricing=pricing,
    )
    if binding != current_binding:
        raise LedgerBindingError("source/config/schema binding changed since preparation")
    owner_id = binding.get("owner_id", "").strip()
    bound_agent_id = binding.get("agent_id", "")
    if not owner_id or bound_agent_id != "faithful-s36":
        raise LedgerBindingError("prepared run lacks immutable owner/agent scope binding")
    if judge_root is not None:
        supplied_judge_hash = _judge_source_hash(judge_root)
        if binding.get("judge_source_sha256") and binding.get("judge_source_sha256") != supplied_judge_hash:
            raise LedgerBindingError("official judge source changed since preparation")
    # ``source_manifest`` is the upstream selection manifest; it intentionally
    # has no execution binding. The prepared artifact binding was recomputed
    # above from the current dataset and upstream manifest bytes.
    selected_ids = [instance.question_id for instance in selected]
    if manifest.get("ordered_question_ids") != selected_ids or manifest.get("case_count") != len(selected_ids):
        raise LedgerBindingError("prepared manifest ordered IDs differ from current selected source rows")
    checkpoint = _checkpoint(paths.checkpoint, selected_ids)
    if checkpoint.get("binding") != current_binding:
        raise LedgerBindingError("checkpoint binding does not match current sources")
    checkpoint_state = _manifest_checkpoint_state(
        manifest, checkpoint, dataset_path=dataset_path,
        manifest_path=manifest_path, artifact_root=paths.root,
    )
    if checkpoint.get("budget_stop") is not None:
        if checkpoint.get("in_flight") is not None:
            raise ExecutionGateError("budget-stop checkpoint still has an unsettled in-flight case")
        if paths.receipt.exists():
            receipt = _read_json(paths.receipt)
            if receipt.get("binding") != binding:
                raise LedgerBindingError("partial receipt binding does not match current sources")
            if receipt.get("status") == "PARTIAL_BUDGET_STOP":
                return receipt
        budget_stop = checkpoint["budget_stop"]
        if not isinstance(budget_stop, Mapping):
            raise FaithfulRunError("checkpoint budget-stop evidence is malformed")
        ledger = BudgetLedger(
            paths.ledger, max_budget_usd=max_budget_usd,
            calibration_budget_usd=calibration_budget, operational_stop_usd=operational_stop,
            pricing=pricing, binding=binding, carry_forward=_carry_forward_from_binding(binding),
        )
        budget_summary = ledger.summary()
        rows = [
            checkpoint.get("evidence", {}).get(qid)
            for qid in selected_ids if qid in checkpoint.get("evidence", {})
        ]
        receipt = {
            "schema": EXECUTION_SCHEMA,
            "status": "PARTIAL_BUDGET_STOP",
            "case_count": len(selected_ids),
            "selected_denominator": len(selected_ids),
            "completed_count": len(checkpoint.get("completed_question_ids", [])),
            "scored_count": len(checkpoint.get("completed_question_ids", [])),
            "scored_denominator": len(selected_ids),
            "failed_count": len(checkpoint.get("failed_question_ids", [])),
            "last_completed_question_id": checkpoint_state["last_completed_question_id"],
            "resume_command": checkpoint_state["resume_command"],
            "resume_command_placeholders": checkpoint_state["resume_command_placeholders"],
            "budget_stop": dict(budget_stop),
            "calibration": calibration,
            "binding": binding,
            "budget": budget_summary,
            "budget_estimate_vs_actual": {
                "estimated_usd": budget_summary["estimated_usd"],
                "measured_actual_usd": budget_summary["measured_actual_usd"],
                "actual_vs_estimate_usd": budget_summary["actual_vs_estimate_usd"],
                "actual_usage_complete": budget_summary["actual_usage_complete"],
                "refused_next_estimate_usd": budget_stop.get("next_estimate_usd"),
            },
            "cases": rows,
        }
        _atomic_json(paths.receipt, receipt)
        return receipt
    if checkpoint_state["complete"]:
        if paths.receipt.exists():
            receipt = _read_json(paths.receipt)
            if receipt.get("binding") != binding:
                raise LedgerBindingError("terminal receipt binding does not match current sources")
            return receipt
        terminal_receipt = {
            "schema": EXECUTION_SCHEMA,
            "status": "COMPLETED",
            "case_count": len(selected),
            "selected_denominator": len(selected),
            "scored_count": len(checkpoint.get("completed_question_ids", [])),
            "scored_denominator": len(selected),
            "failed_count": len(checkpoint.get("failed_question_ids", [])),
            "last_completed_question_id": checkpoint_state["last_completed_question_id"],
            "resume_command": checkpoint_state["resume_command"],
            "binding": binding,
            "calibration": calibration,
            "cases": [
                checkpoint.get("evidence", {}).get(instance.question_id)
                for instance in selected if instance.question_id in checkpoint.get("evidence", {})
            ],
        }
        _atomic_json(paths.receipt, terminal_receipt)
        return terminal_receipt
    if checkpoint.get("in_flight"):
        raise ExecutionGateError(
            f"checkpoint has unknown in-flight case {checkpoint['in_flight']!r}; inspect before resume"
        )
    completed = set(checkpoint.get("completed_question_ids", []))
    failed = set(checkpoint.get("failed_question_ids", []))
    evidence = checkpoint.setdefault("evidence", {})
    selected_ids = [instance.question_id for instance in selected]
    terminal_ids = completed | failed
    if terminal_ids == set(selected_ids) and paths.receipt.exists():
        receipt = _read_json(paths.receipt)
        if receipt.get("binding") != binding:
            raise LedgerBindingError("completed receipt binding does not match current sources")
        return receipt

    # Validate all local/provider prerequisites before constructing resources.
    from benchmarks.longmemeval.faithful_gateway import create_local_gateway, validate_local_dsn
    validate_local_dsn(dsn or "")
    embedder = require_fastembed()
    assert_no_anthropic_execution()
    carry_forward = _carry_forward_from_binding(binding)
    ledger = BudgetLedger(
        paths.ledger, max_budget_usd=max_budget_usd,
        calibration_budget_usd=calibration_budget, operational_stop_usd=operational_stop,
        pricing=pricing, binding=binding, carry_forward=carry_forward,
    )
    owned_judge_client = False
    if judge_root is not None and judge_client is None:
        judge_client = OpenAIResponsesClient()
        owned_judge_client = True
    if client_factory is None:
        client_factory = lambda *, phase, ledger: OpenAIResponsesClient()
    if gateway_factory is None and agent_factory is None and tool_gateway is None:
        async def gateway_factory(**kwargs: Any) -> Any:
            return await create_local_gateway(dsn or "", **kwargs)
    session_state = checkpoint.setdefault("sessions", {})
    budget_stop: dict[str, Any] | None = None
    for instance in selected:
        if instance.question_id in terminal_ids:
            continue
        case_project = f"longmemeval-{instance.question_id}"
        # Establish durable operator-visible state before gateway construction
        # or any writer/provider activity.  A failed case remains in-flight so
        # resume cannot silently treat an uncertain attempt as safe to replay.
        checkpoint["in_flight"] = instance.question_id
        checkpoint["in_flight_stage"] = "gateway"
        case_sessions = session_state.setdefault(instance.question_id, {})
        case_row: dict[str, Any] = evidence.setdefault(instance.question_id, {
            "question_id": instance.question_id, "status": "failed",
            "sessions": [], "answer": None, "judge": None,
        })
        _write_checkpoint(paths.checkpoint, checkpoint)
        gateway = tool_gateway
        owned_gateway = False
        client = None
        owned_client = False
        try:
            if gateway_factory is not None:
                gateway = await gateway_factory(
                    owner_id=owner_id, project_id=case_project,
                    agent_id=bound_agent_id, embedding=embedder,
                )
                owned_gateway = True
            elif gateway is not None:
                if getattr(gateway, "project_id", case_project) != case_project:
                    raise ExecutionGateError("tool gateway scope does not match case binding")
            if agent_factory is not None and gateway is tool_gateway:
                agent = agent_factory(ledger)
            else:
                client = client_factory(phase="run", ledger=ledger)
                owned_client = True
                agent = FaithfulAgent(
                    client, ledger, tools=gateway, model=writer_model,
                    policy=_agent_policy_for_profile(run_profile),
                    retrieval_tier=retrieval_tier,
                )
            # Move the durable marker to the first provider stage; it was
            # already persisted before gateway construction above.
            checkpoint["in_flight_stage"] = "writer"
            _write_checkpoint(paths.checkpoint, checkpoint)
            for session in _sort_sessions(instance):
                if case_sessions.get(session.session_id) == "completed":
                    continue
                checkpoint["in_flight_stage"] = f"session:{session.session_id}"
                _write_checkpoint(paths.checkpoint, checkpoint)
                if run_profile == FULL_S_PROFILE:
                    await _ingest_full_s_session(
                        instance, session,
                        pool=getattr(gateway, "_pool", None),
                        embedder=embedder,
                        owner_id=owner_id,
                        project_id=case_project,
                    )
                    case_row.setdefault("sessions", []).append({
                        "session_id": session.session_id, "date": session.date,
                        "result": {"mode": "dual", "ingested": True},
                        "tool_results": _gateway_evidence(gateway),
                    })
                else:
                    session_result = await agent.write_session(session, project_id=case_project, agent_id=bound_agent_id)
                    case_row.setdefault("sessions", []).append({
                        "session_id": session.session_id, "date": session.date,
                        "result": _agent_result_evidence(session_result),
                        "tool_results": _gateway_evidence(gateway),
                    })
                case_sessions[session.session_id] = "completed"
                _write_checkpoint(paths.checkpoint, checkpoint)
            checkpoint["in_flight_stage"] = "answer"
            _write_checkpoint(paths.checkpoint, checkpoint)
            answer = await agent.answer(
                question=instance.question, question_date=instance.question_date,
                task_shape=None, recalled_context=None,
                project_id=case_project, agent_id=bound_agent_id,
            )
            case_row.update({
                "status": "completed", "hypothesis": answer.text,
                "answer": _agent_result_evidence(answer),
                "tool_results": _gateway_evidence(gateway),
            })
            if judge_root is not None:
                from benchmarks.longmemeval.judge import _official_prompt_loader
                prompt = _official_prompt_loader(judge_root)(
                    instance.question_type, instance.question, instance.answer,
                    answer.text, abstention=instance.is_abstention,
                )
                label, raw, reservation_id = await BoundedJudge(judge_client, ledger, phase="run").judge(prompt)
                case_row["judge"] = {
                    "label": label, "raw": raw, "reservation_id": reservation_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "model": GPT4O_JUDGE_MODEL,
                }
            completed.add(instance.question_id)
        except TotalBudgetExceeded as exc:
            # The failed reservation is rejected before dispatch. Keep any
            # completed sessions durable, but do not mark this question failed.
            case_row.update({
                "status": "budget_stopped",
                "budget_stop": {
                    "used_usd": exc.used_usd,
                    "next_estimate_usd": exc.estimate_usd,
                    "ceiling_usd": exc.ceiling_usd,
                    "last_completed_question_id": next((qid for qid in reversed(selected_ids) if qid in completed), None),
                },
            })
            checkpoint["in_flight"] = None
            checkpoint["in_flight_stage"] = None
            checkpoint["budget_stop"] = case_row["budget_stop"]
            checkpoint["budget_stopped_question_id"] = instance.question_id
            budget_stop = dict(case_row["budget_stop"])
            _write_checkpoint(paths.checkpoint, checkpoint)
            break
        except AmbiguousExecutionError as exc:
            checkpoint["last_error"] = f"{type(exc).__name__}: {exc}"
            case_row["status"] = "in_flight"
            case_row["error"] = checkpoint["last_error"]
            case_row["tool_results"] = _gateway_evidence(gateway)
            _write_checkpoint(paths.checkpoint, checkpoint)
            raise
        except Exception as exc:
            case_row["status"] = "failed"
            case_row["error"] = f"{type(exc).__name__}: {exc}"
            case_row["tool_results"] = _gateway_evidence(gateway)
            failed.add(instance.question_id)
        finally:
            checkpoint["completed_question_ids"] = sorted(completed)
            checkpoint["failed_question_ids"] = sorted(failed)
            # Never clear the marker on an uncertain/failed case.  A terminal
            # success is the only state that permits the next case to run.
            if instance.question_id in completed or budget_stop is not None:
                checkpoint["in_flight"] = None
                checkpoint["in_flight_stage"] = None
            else:
                checkpoint["in_flight"] = instance.question_id
            _write_checkpoint(paths.checkpoint, checkpoint)
            if owned_gateway:
                await gateway.close()
            if owned_client:
                await _close_owned(client)
    if owned_judge_client:
        await _close_owned(judge_client)
    rows = [evidence[qid] for qid in selected_ids if qid in evidence]
    budget_summary = ledger.summary()
    latest_checkpoint = _checkpoint(paths.checkpoint, selected_ids)
    completion = _manifest_checkpoint_state(
        manifest, latest_checkpoint, dataset_path=dataset_path,
        manifest_path=manifest_path, artifact_root=paths.root,
    )
    all_terminal = completion["complete"]
    receipt = {
        "schema": EXECUTION_SCHEMA,
        "status": (
            "PARTIAL_BUDGET_STOP" if budget_stop is not None
            else "COMPLETED" if all_terminal else "PARTIAL"
        ),
        "case_count": len(selected_ids),
        "selected_denominator": len(selected_ids),
        "completed_count": len(completed),
        "scored_count": len(completed),
        "scored_denominator": len(selected_ids),
        "failed_count": len(failed),
        "last_completed_question_id": completion["last_completed_question_id"],
        "resume_command": completion["resume_command"],
        "resume_command_placeholders": completion["resume_command_placeholders"],
        "budget_stop": budget_stop,
        "calibration": calibration,
        "binding": binding,
        "budget": budget_summary,
        "budget_estimate_vs_actual": {
            "estimated_usd": budget_summary["estimated_usd"],
            "measured_actual_usd": budget_summary["measured_actual_usd"],
            "actual_vs_estimate_usd": budget_summary["actual_vs_estimate_usd"],
            "actual_usage_complete": budget_summary["actual_usage_complete"],
            "refused_next_estimate_usd": None if budget_stop is None else budget_stop["next_estimate_usd"],
        },
        "cases": rows,
    }
    if budget_stop is not None:
        run_manifest = dict(manifest)
        run_manifest["status"] = "PARTIAL_BUDGET_STOP"
        run_manifest["budget_stop"] = budget_stop
        _atomic_json(paths.manifest, run_manifest)
    _atomic_json(paths.receipt, receipt)
    return receipt


async def resume_run(
    dataset_path: Path,
    manifest_path: Path,
    *,
    paths: RunPaths | None = None,
    execute: bool = False,
    dsn: str | None = None,
    agent_factory: Callable[[BudgetLedger], FaithfulAgent] | None = None,
    tool_gateway: object | None = None,
    judge_root: Path | None = None,
    judge_client: Any | None = None,
    gateway_factory: Callable[..., Awaitable[Any]] | None = None,
    client_factory: Callable[..., Any] | None = None,
    include_question_ids: Sequence[str] | None = None,
    exclude_question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resume one run while holding the process-wide run lock."""
    if execute and judge_root is None:
        raise ExecutionGateError("judge_root is required for live resume and the official GPT-4o judge")
    paths = paths or RunPaths.from_root()
    with _exclusive_run_lock(paths.lock):
        return await _resume_run_locked(
            dataset_path, manifest_path, paths=paths, execute=execute, dsn=dsn,
            agent_factory=agent_factory, tool_gateway=tool_gateway,
            judge_root=judge_root, judge_client=judge_client,
            gateway_factory=gateway_factory, client_factory=client_factory,
            include_question_ids=include_question_ids,
            exclude_question_ids=exclude_question_ids,
        )


async def _public_recall(gateway: object | None, question: str, shape: Any, project_id: str) -> str:
    """Call public recall only through an explicitly supplied gateway."""
    if gateway is None or not hasattr(gateway, "call"):
        raise ExecutionGateError("public recall gateway is required; no hidden/background fallback")
    response = await gateway.call("weft_recall", {
        "query": question, "project_id": project_id, "agent_id": "faithful-s36",
        "limit": shape.top_k, "tier": "auto", "mode": "hybrid", "retrieval_mode": "face",
    })
    if not isinstance(response, Mapping):
        raise AgentExecutionError("weft_recall returned a non-object")
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse prepare, calibrate, resume, and offline recovery commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "calibrate", "resume"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--artifact-root", type=Path)
        child.add_argument("--profile", choices=("legacy", FRESH_RUN_PROFILE, FULL_S_PROFILE), default="legacy")
        child.add_argument("--writer-model", choices=(LUNA_MODEL, GPT6_LUNA_MODEL))
        child.add_argument("--max-budget-usd", type=float)
        child.add_argument("--include-question-id", action="append", dest="include_question_ids")
        child.add_argument("--exclude-question-id", action="append", dest="exclude_question_ids")
        if command == "prepare":
            child.add_argument("--prior-ledger", type=Path)
        if command != "resume":
            child.add_argument("--owner-id", required=True)
        child.add_argument("--judge-root", type=Path, required=True)
    cal = sub.choices["calibrate"]
    cal.add_argument("--approved-by", default="")
    cal.add_argument("--notes", default="")
    cal.add_argument("--execute", action="store_true")
    cal.add_argument("--dsn", default=os.environ.get("LONGMEMEVAL_DATABASE_URL"))
    cal.add_argument("--case-limit", type=int, default=4)
    approve = sub.add_parser("approve")
    approve.add_argument("--artifact-root", type=Path)
    approve.add_argument("--profile", choices=("legacy", FRESH_RUN_PROFILE, FULL_S_PROFILE), default="legacy")
    approve.add_argument("--writer-model", choices=(LUNA_MODEL, GPT6_LUNA_MODEL))
    approve.add_argument("--max-budget-usd", type=float)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--projected-total-usd", type=float, required=True)
    approve.add_argument("--projection-basis", required=True)
    recovery = sub.add_parser(
        "recover-safe-skip",
        help="offline recovery for the specifically pinned in-flight case 95228167",
    )
    recovery.add_argument("--artifact-root", type=Path, required=True)
    recovery.add_argument("--case-id", required=True)
    recovery.add_argument("--reservation-id", required=True)
    inflight = sub.add_parser(
        "recover-inflight",
        help=(
            "offline recovery for another explicitly selected in-flight case "
            "(provider-timeout, outstanding-reservation, or pre-provider termination)"
        ),
    )
    inflight.add_argument("--artifact-root", type=Path, required=True)
    inflight.add_argument("--case-id", required=True)
    inflight.add_argument(
        "--reservation-id", default="",
        help=(
            "reservation ID for provider-dispatch shapes (required for a "
            "pre-finalized provider timeout or one outstanding reserved "
            "reservation); omit it only when the case was externally terminated "
            "before any provider dispatch (evidence recording no provider "
            "outcome — in-flight or finalized failed with no answer, judge "
            "result, or case error — and zero ledger reservations for the case)"
        ),
    )
    inflight.add_argument("--expected-completed-count", type=int, required=True)
    resume = sub.choices["resume"]
    resume.add_argument("--execute", action="store_true")
    resume.add_argument("--dsn", default=os.environ.get("LONGMEMEVAL_DATABASE_URL"))
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    """Execute one CLI command without implicit network behavior."""
    args.profile = getattr(args, "profile", "legacy")
    args.writer_model = getattr(args, "writer_model", None)
    args.max_budget_usd = getattr(args, "max_budget_usd", None)
    if args.command == "recover-safe-skip":
        result = recover_safe_skip(
            RunPaths.from_root(args.artifact_root),
            case_id=args.case_id,
            reservation_id=args.reservation_id,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "recover-inflight":
        result = recover_inflight_case(
            RunPaths.from_root(args.artifact_root),
            case_id=args.case_id,
            reservation_id=args.reservation_id,
            expected_completed_count=args.expected_completed_count,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    artifact_root = args.artifact_root or (
        GPT6_FULL_S_ARTIFACT_NAMESPACE if args.profile == FULL_S_PROFILE
        else GPT6_SELECTED35_ARTIFACT_NAMESPACE if args.profile == FRESH_RUN_PROFILE
        else ARTIFACT_NAMESPACE
    )
    paths = RunPaths.from_root(artifact_root)
    if args.profile == FRESH_RUN_PROFILE and (
        args.writer_model != GPT6_LUNA_MODEL or args.max_budget_usd != GPT6_TOTAL_BUDGET_USD
    ):
        raise ExecutionGateError("fresh profile requires --writer-model gpt-6-luna and --max-budget-usd 50")
    if args.profile == FULL_S_PROFILE and (
        args.writer_model != FULL_S_WRITER_MODEL or args.max_budget_usd != FULL_S_MAX_BUDGET_USD
    ):
        raise ExecutionGateError("full-S profile requires --writer-model gpt-6-luna and --max-budget-usd 150")
    if args.profile == "legacy" and (
        args.writer_model not in (None, LUNA_MODEL)
        or args.max_budget_usd not in (None, DEFAULT_TOTAL_BUDGET_USD)
    ):
        raise ExecutionGateError("legacy profile remains fixed to GPT-5.6 Luna and the $20 cap")
    if args.command in {"calibrate", "resume"} and paths.manifest.exists():
        pinned_binding = _read_json(paths.manifest).get("binding", {})
    elif args.command == "approve" and paths.calibration.exists():
        pinned_binding = _read_json(paths.calibration).get("binding", {})
    else:
        pinned_binding = None
    if pinned_binding is not None:
        bound_profile, bound_model, bound_budget, bound_calibration_budget, _, _, _ = _profile_from_binding(pinned_binding)
        if args.profile != bound_profile:
            raise LedgerBindingError("CLI profile differs from prepared run binding")
        if args.writer_model not in (None, bound_model) or args.max_budget_usd not in (None, bound_budget):
            raise LedgerBindingError("CLI model/budget differs from prepared run binding")
    if args.command == "prepare":
        if not args.owner_id:
            raise ExecutionGateError("prepare requires explicit --owner-id")
        prepared = prepare_run(
            args.dataset, args.manifest, paths=paths,
            owner_id=args.owner_id, judge_root=args.judge_root,
            prior_ledger=args.prior_ledger,
            include_question_ids=args.include_question_ids,
            exclude_question_ids=args.exclude_question_ids,
            profile=args.profile, writer_model=args.writer_model,
            max_budget_usd=args.max_budget_usd,
        )
        output = {
            "status": "prepared", "case_count": len(prepared.ordered_question_ids),
            "root": str(paths.root),
        }
        # Keep lightweight API doubles usable while real preparations expose
        # the complete immutable binding and carried budget evidence.
        if hasattr(prepared, "binding"):
            carry_forward = _carry_forward_from_binding(prepared.binding)
            _, _, bound_budget, bound_calibration_budget, bound_operational_stop, bound_pricing, _ = _profile_from_binding(prepared.binding)
            output.update({
                "binding": prepared.binding,
                "budget": BudgetLedger(
                    paths.ledger, max_budget_usd=bound_budget,
                    calibration_budget_usd=bound_calibration_budget,
                    operational_stop_usd=bound_operational_stop, pricing=bound_pricing,
                    binding=prepared.binding, carry_forward=carry_forward,
                ).summary(),
            })
        print(json.dumps(output, sort_keys=True))
        return 0
    if args.command == "calibrate":
        if not args.execute:
            raise ExecutionGateError("calibrate is prepare-only by default; pass --execute explicitly")
        if not args.dsn or not args.owner_id or args.judge_root is None:
            raise ExecutionGateError("calibrate requires --dsn, --owner-id, and --judge-root before provider construction")
        measured = await run_calibration(
            args.dataset, args.manifest, paths=paths, dsn=args.dsn,
            owner_id=args.owner_id, case_limit=args.case_limit,
            judge_root=args.judge_root,
            include_question_ids=args.include_question_ids,
            exclude_question_ids=args.exclude_question_ids,
        )
        prepared_binding = _read_json(paths.manifest)["binding"]
        _, _, _, bound_calibration_budget, _, _, _ = _profile_from_binding(prepared_binding)
        receipt = {
            "schema": CALIBRATION_SCHEMA,
            "status": "HOLD_FOR_APPROVAL",
            "approved_by": args.approved_by.strip() or None,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "calibration_budget_usd": bound_calibration_budget,
            "notes": args.notes,
            "binding": _read_json(paths.manifest)["binding"],
            "representative": measured,
            "projection": measured["projection"],
            "approval_required": True,
        }
        _atomic_json(paths.calibration, receipt)
        print(json.dumps({"status": receipt["status"], "projection": receipt["projection"]}))
        return 0
    if args.command == "approve":
        receipt = approve_calibration(
            paths, approved_by=args.approved_by,
            projected_total_usd=args.projected_total_usd,
            projection_basis=args.projection_basis,
        )
        print(json.dumps({"status": receipt["status"], "projected_total_usd": receipt["projected_total_usd"]}))
        return 0
    if args.judge_root is None:
        raise ExecutionGateError("resume requires --judge-root for the official GPT-4o judge")
    result = await resume_run(
        args.dataset,
        args.manifest,
        paths=paths,
        execute=args.execute,
        dsn=args.dsn,
        judge_root=args.judge_root,
        include_question_ids=args.include_question_ids,
        exclude_question_ids=args.exclude_question_ids,
    )
    print(json.dumps({"status": result["status"], "case_count": result["case_count"], "failed_count": result["failed_count"]}))
    return 0


def _load_cli_environment() -> None:
    """Load CLI environment defaults without overriding process or cwd values."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path.home() / ".weft" / ".env")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert safety failures to a non-zero status."""
    try:
        _load_cli_environment()
        return asyncio.run(_main_async(_parse_args(argv)))
    except (OSError, ValueError, json.JSONDecodeError, FaithfulRunError) as exc:
        print(f"faithful S36 refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 1. Prepare offline (no DB/provider calls):
#    uv run python -m benchmarks.longmemeval.faithful_s36 prepare \
#      --dataset /path/to/longmemeval_s_pilot_derivative.json \
#      --manifest benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json
# 2. Record the $5 calibration gate:
#    uv run python -m benchmarks.longmemeval.faithful_s36 calibrate \
#      --dataset /path/to/longmemeval_s_pilot_derivative.json \
#      --manifest benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json \
#      --approved-by 'operator'
# 3. Resume only against disposable local Postgres, with explicit execution:
#    LONGMEMEVAL_DATABASE_URL=postgresql://localhost/longmemeval_bench \
#    uv run python -m benchmarks.longmemeval.faithful_s36 resume --execute \
#      --dataset /path/to/longmemeval_s_pilot_derivative.json \
#      --manifest benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json
# 4. Offline safe-skip recovery is pinned to case 95228167 and requires its
#    exact outstanding reservation ID; it does not use dataset, provider, or DB:
#    uv run python -m benchmarks.longmemeval.faithful_s36 recover-safe-skip \
#      --artifact-root /path/to/run-artifacts --case-id 95228167 \
#      --reservation-id <exact-ledger-reservation-id>
# 5. Expected output:
#    Preparation/calibration receipts offline; resume writes a hash-bound
#    checkpoint and a 36-case execution receipt with failures counted.
#
# ═══════════════════════════════════════════════════════════════
