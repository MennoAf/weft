"""Preparation and bounded execution harness for a three-tier LongMemEval pilot.

This module is deliberately additive: it does not alter the legacy adapter's
CLI or the completed baseline artifacts.  ``prepare`` is provider/database-free
and freezes a small, equal-by-question-type population plus a conservative cost
plan.  ``run --execute`` is the only provider/database path and requires an
explicit local benchmark DSN and ``OPENAI_API_KEY``.

The pilot compares ``turns``, ``belief``, and ``auto`` over the same per-question
``dual`` substrate (raw session memories plus episode turns).  The three arms
are not the RC-FL-20 qualification run; this is a bounded, one-repetition
experiment whose output can later be promoted into a separately authorized
qualification packet.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import types
from contextvars import Token

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmarks.longmemeval.dataset import Instance, load_split
from benchmarks.longmemeval.ingest import cleanup_haystack, load_haystack, project_id_for
from benchmarks.longmemeval.materialize import Detector, materialize_question
from benchmarks.longmemeval.reader import Reader
from benchmarks.longmemeval.router import RetrievalDiagnostics, RetrievalPolicy, retrieve
from benchmarks.longmemeval.task_shape import derive_task_shape
from weft.text_generation import (
    AnthropicTextGenerationProvider,
    OpenAITextGenerationProvider,
    TextGenerationProvider,
)

ARMS = ("turns", "belief", "auto")
CORRECTED_SCHEMA = "weft.longmemeval.corrected-belief-pilot.v1"
CORRECTED_ARM = "belief"
CORRECTED_INGEST_MODE = "production-belief"
CORRECTED_TIER = "belief"
CORRECTED_USER_ID = "longmemeval-bench"
CORRECTED_SCOPE = "global"
QUESTION_TYPES = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)
SCHEMA = "weft.longmemeval.pilot.v1"
NORMALIZATION_SCHEMA = "weft.longmemeval.dataset-normalization.v1"
REPETITIONS = 1
RECALL_K = 10
READER_MODEL = "gpt-5.6-luna"
JUDGE_MODEL = "gpt-4o"
READER_MAX_OUTPUT_TOKENS = 256
OPERATIONAL_STOP_USD = Decimal("40")
AUTHORIZATION_CEILING_USD = Decimal("50")
RESERVE_FACTOR = Decimal("1.25")
# Corrected benchmark retry policy. The corrected runner disables SDK retries for
# its own OpenAI Reader construction; the other paid seams are bounded here
# conservatively so preflight never treats a retry-capable call as one attempt.
CLASSIFIER_MAX_RETRIES = 2
DETECTOR_MAX_RETRIES = 2
READER_MAX_RETRIES = 0
JUDGE_MAX_RETRIES = 4
MAX_CORRECTED_ATTEMPTS = 36
MAX_RESUME_COMPLETED = 18
# Pinch inputs: measured from the completed 500-question baseline and
# conservative judge wrapper bounds. These are planning values, not billing.
READER_INPUT_TOKENS_PER_QUESTION = Decimal("4248.438")
READER_OUTPUT_TOKENS_PER_QUESTION = Decimal("52.408")
JUDGE_INPUT_TOKENS_PER_QUESTION = Decimal("10000")
JUDGE_OUTPUT_TOKENS_PER_QUESTION = Decimal("500")
# Corrected BELIEF-only preflight bounds. These are deliberately worst-case:
# classifier receives at most the ingest pipeline's 12,000-character input;
# detector's own documented bound is 800 input tokens; Reader uses its request
# output cap rather than the historical average. Unknown usage is never zero.
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
DETECTOR_MODEL = "claude-haiku-4-5-20251001"
CLASSIFIER_INPUT_TOKENS_PER_CALL = Decimal("12000")
CLASSIFIER_OUTPUT_TOKENS_PER_CALL = Decimal("512")
DETECTOR_INPUT_TOKENS_PER_CALL = Decimal("800")
DETECTOR_OUTPUT_TOKENS_PER_CALL = Decimal("512")
HAIKU_INPUT_USD_PER_MILLION = Decimal("1.00")
HAIKU_OUTPUT_USD_PER_MILLION = Decimal("5.00")
LUNA_INPUT_USD_PER_MILLION = Decimal("0.20")
LUNA_OUTPUT_USD_PER_MILLION = Decimal("1.20")
GPT4O_INPUT_USD_PER_MILLION = Decimal("2.50")
GPT4O_OUTPUT_USD_PER_MILLION = Decimal("10.00")

# Source files whose bytes affect the pilot.  The manifest binds them so a
# later run cannot silently use a different router or Reader contract.
SOURCE_FILES = (
    "benchmarks/longmemeval/pilot.py",
    "benchmarks/longmemeval/dataset.py",
    "benchmarks/longmemeval/ingest.py",
    "benchmarks/longmemeval/router.py",
    "benchmarks/longmemeval/reader.py",
    "benchmarks/longmemeval/task_shape.py",
    "benchmarks/longmemeval/materialize.py",
    "weft/ingest_pipeline.py",
    "weft/views/belief_detector.py",
    "weft/views/belief_query.py",
    "weft/text_generation.py",
)


@dataclass(frozen=True)
class CostEstimate:
    questions: int
    arms: int
    repetitions: int
    reader_calls: int
    judge_calls: int
    reader_cost_usd: Decimal
    judge_cost_usd: Decimal
    total_cost_usd: Decimal
    reserve_factor: Decimal
    operational_stop_usd: Decimal
    authorization_ceiling_usd: Decimal

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Decimal):
                value[key] = str(item.quantize(Decimal("0.000001"), rounding=ROUND_UP))
        return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pilot source is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _session_payload_hash(date: Any, session: Any) -> str:
    """Hash the date/session payload represented by one haystack position."""
    return sha256_bytes(_canonical({"date": date, "session": session}))


def _duplicate_session_groups(row: dict[str, Any]) -> dict[str, list[int]]:
    """Return repeated haystack session IDs and their source positions."""
    ids = row.get("haystack_session_ids")
    dates = row.get("haystack_dates")
    sessions = row.get("haystack_sessions")
    if not isinstance(ids, list) or not isinstance(dates, list) or not isinstance(sessions, list):
        raise ValueError(
            f"Malformed instance {row.get('question_id')}: haystack arrays must be lists"
        )
    if not (len(ids) == len(dates) == len(sessions)):
        raise ValueError(
            f"Malformed instance {row.get('question_id')}: haystack arrays disagree in length"
        )
    positions: dict[str, list[int]] = {}
    for index, session_id in enumerate(ids):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(
                f"Malformed instance {row.get('question_id')}: session IDs must be strings"
            )
        positions.setdefault(session_id, []).append(index)
    return {session_id: indexes for session_id, indexes in positions.items() if len(indexes) > 1}


def _ensure_distinct_paths(source_path: Path, output_paths: tuple[tuple[str, Path], ...]) -> None:
    """Reject output paths that could overwrite the source or each other."""
    source_resolved = source_path.resolve()
    seen: dict[Path, str] = {}
    for label, output_path in output_paths:
        resolved = output_path.resolve()
        if resolved == source_resolved:
            raise ValueError(f"{label} path aliases immutable source path")
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(f"{label} path aliases {previous} output path")
        seen[resolved] = label


def normalize_dataset(
    source_path: Path,
    derived_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Create an auditable pilot-only derivative of a LongMemEval JSON split.

    Identical repeated session payloads keep their first occurrence and are
    recorded as collapsed. Any question containing a repeated session ID with
    conflicting date/session payloads is excluded in full. The source file is
    never modified, and retained rows are validated by the production typed
    ``Instance`` loader before the derivative is written.
    """
    _ensure_distinct_paths(
        source_path,
        (("derived", derived_path), ("report", report_path)),
    )
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LongMemEval source must contain a JSON array")

    normalized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    question_ids: set[str] = set()

    for row_number, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Malformed instance at source index {row_number}: expected object")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Malformed instance at source index {row_number}: question_id must be a string")
        if question_id in question_ids:
            raise ValueError(f"duplicate question_id {question_id!r}")
        question_ids.add(question_id)
        duplicate_groups = _duplicate_session_groups(row)
        conflicts: list[dict[str, Any]] = []
        row_collapsed: list[dict[str, Any]] = []
        for session_id, indexes in duplicate_groups.items():
            payloads = [
                {
                    "source_index": index,
                    "date": row["haystack_dates"][index],
                    "payload_sha256": _session_payload_hash(
                        row["haystack_dates"][index], row["haystack_sessions"][index]
                    ),
                }
                for index in indexes
            ]
            if len({payload["payload_sha256"] for payload in payloads}) > 1:
                conflicts.append({"session_id": session_id, "occurrences": payloads})
            else:
                row_collapsed.append(
                    {
                        "question_id": question_id,
                        "question_type": row.get("question_type"),
                        "session_id": session_id,
                        "kept_source_index": indexes[0],
                        "dropped_source_indices": indexes[1:],
                        "payload_sha256": payloads[0]["payload_sha256"],
                    }
                )
        if conflicts:
            excluded.append(
                {
                    "question_id": question_id,
                    "question_type": row.get("question_type"),
                    "reason": "conflicting_duplicate_haystack_session_ids",
                    "duplicate_groups": conflicts,
                }
            )
            # Do not retain any portion of a question with ambiguous ground truth.
            continue

        collapsed.extend(row_collapsed)
        if duplicate_groups:
            ids = row["haystack_session_ids"]
            keep_indexes = [
                index
                for index, session_id in enumerate(ids)
                if index == duplicate_groups.get(session_id, [index])[0]
            ]
            normalized_row = dict(row)
            normalized_row["haystack_session_ids"] = [ids[index] for index in keep_indexes]
            normalized_row["haystack_dates"] = [row["haystack_dates"][index] for index in keep_indexes]
            normalized_row["haystack_sessions"] = [row["haystack_sessions"][index] for index in keep_indexes]
            normalized.append(normalized_row)
        else:
            normalized.append(row)

    # The typed loader remains the authority for all rows that enter the pilot.
    for row in normalized:
        try:
            Instance.from_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"normalized record {row.get('question_id')} fails typed loader: {exc}"
            ) from exc

    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_bytes = (json.dumps(normalized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    derived_path.write_bytes(derived_bytes)
    derived_hash = sha256_bytes(derived_bytes)
    source_hash = sha256_bytes(source_bytes)
    by_type: dict[str, int] = {}
    for row in normalized:
        question_type = row["question_type"]
        by_type[question_type] = by_type.get(question_type, 0) + 1

    report = {
        "schema": NORMALIZATION_SCHEMA,
        "status": "PILOT_DERIVATIVE_READY",
        "purpose": "pilot_only",
        "official_benchmark": {
            "source_unchanged": True,
            "source_records_remain_authoritative": True,
            "duplicate_id_semantics": "unadjudicated_upstream_data_question",
            "full_suite_disposition": "separate_follow_up_required",
        },
        "source": {
            "filename": source_path.name,
            "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
            "path": str(source_path),
            "sha256": source_hash,
            "record_count": len(raw),
        },
        "derived": {
            "filename": derived_path.name,
            "path": str(derived_path),
            "sha256": derived_hash,
            "record_count": len(normalized),
            "valid_question_type_counts": dict(sorted(by_type.items())),
        },
        "rule": {
            "identical_duplicate_payload": "preserve_first_occurrence",
            "conflicting_duplicate_payload": "exclude_entire_question",
            "payload_definition": "canonical date plus session JSON at each repeated session ID",
        },
        "collapsed_identical_groups": collapsed,
        "exclusions": excluded,
        "exclusion_count": len(excluded),
        "collapsed_group_count": len(collapsed),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _normalization_metadata(
    dataset_path: Path,
    report_path: Path | None,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and summarize a normalization report for manifest binding."""
    if report_path is None:
        raise ValueError("normalized pilot manifest requires a normalization report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != NORMALIZATION_SCHEMA or report.get("status") != "PILOT_DERIVATIVE_READY":
        raise ValueError("invalid pilot normalization report")
    dataset_hash = sha256_file(dataset_path)
    if report.get("derived", {}).get("sha256") != dataset_hash:
        raise ValueError("normalization report derived checksum mismatch")
    if report.get("derived", {}).get("path") and Path(report["derived"]["path"]).name != dataset_path.name:
        raise ValueError("normalization report dataset filename mismatch")
    if source_path is not None:
        if report.get("source", {}).get("sha256") != sha256_file(source_path):
            raise ValueError("normalization report source checksum mismatch")
    return {
        "schema": report["schema"],
        "report_filename": report_path.name,
        "report_sha256": sha256_file(report_path),
        "source_sha256": report["source"]["sha256"],
        "source_record_count": report["source"]["record_count"],
        "derived_sha256": report["derived"]["sha256"],
        "derived_record_count": report["derived"]["record_count"],
        "excluded_question_ids": [item["question_id"] for item in report["exclusions"]],
        "exclusion_count": report["exclusion_count"],
        "collapsed_group_count": report["collapsed_group_count"],
    }


def _manifest_hash(manifest_without_hash: dict[str, Any]) -> str:
    return sha256_bytes(_canonical(manifest_without_hash))


def _non_abstention_instances(dataset_path: Path) -> list[Instance]:
    instances = load_split(dataset_path)
    return [instance for instance in instances if instance.question_type in QUESTION_TYPES]


def select_questions(
    dataset_path: Path,
    *,
    per_type: int = 6,
    seed: int = 0,
) -> list[Instance]:
    """Select an equal, deterministic six-type population in source order."""
    if per_type < 1:
        raise ValueError("per_type must be positive")
    instances = _non_abstention_instances(dataset_path)
    by_type: dict[str, list[Instance]] = {question_type: [] for question_type in QUESTION_TYPES}
    for instance in instances:
        by_type[instance.question_type].append(instance)
    selected_ids: set[str] = set()
    rng = random.Random(seed)
    for question_type in QUESTION_TYPES:
        group = list(by_type[question_type])
        if len(group) < per_type:
            raise ValueError(
                f"question type {question_type!r} has only {len(group)} records; "
                f"cannot select {per_type}"
            )
        rng.shuffle(group)
        selected_ids.update(instance.question_id for instance in group[:per_type])
    selected = [instance for instance in instances if instance.question_id in selected_ids]
    if len(selected) != per_type * len(QUESTION_TYPES):
        raise ValueError("selected question population has an unexpected denominator")
    return selected


def corrected_cost_estimate(
    instances: list[Instance],
    pending_ids: list[str],
    *,
    judge_questions: int | None = None,
    prior_spend_usd: Decimal = Decimal("0"),
) -> dict[str, Any]:
    """Return a fail-closed upper bound for one corrected BELIEF run.

    The bound includes every paid stage: one classifier request per persisted
    source turn, one detector request per eligible turn, one Reader request per
    pending question, and one reserved judge request per S36 question. The
    classifier input bound uses the ingest pipeline's 12,000-character cap,
    converted conservatively at four characters/token; detector pricing and
    its documented 800-token prompt bound come from belief_detector. No
    provider or database object is touched here.
    """
    by_id = {item.question_id: item for item in instances}
    selected = [by_id[question_id] for question_id in pending_ids]
    turns = [
        turn
        for item in selected
        for session in item.sessions
        for turn in session.turns
        if turn.role in {"user", "assistant", "system", "tool"}
    ]
    classifier_input = sum(
        min(12_000, max(1, len(turn.content))) / Decimal("4") for turn in turns
    )
    classifier_output = Decimal(len(turns)) * CLASSIFIER_OUTPUT_TOKENS_PER_CALL
    detector_calls = sum(
        1 for turn in turns if turn.role in {"user", "assistant"}
    )
    detector_input = Decimal(detector_calls) * DETECTOR_INPUT_TOKENS_PER_CALL
    detector_output = Decimal(detector_calls) * DETECTOR_OUTPUT_TOKENS_PER_CALL
    reader_calls = len(pending_ids)
    reader_input = Decimal(reader_calls) * READER_INPUT_TOKENS_PER_QUESTION
    reader_output = Decimal(reader_calls) * Decimal(READER_MAX_OUTPUT_TOKENS)
    judge_calls = len(pending_ids) if judge_questions is None else judge_questions
    judge_input = Decimal(judge_calls) * JUDGE_INPUT_TOKENS_PER_QUESTION
    judge_output = Decimal(judge_calls) * JUDGE_OUTPUT_TOKENS_PER_QUESTION
    classifier = (classifier_input * HAIKU_INPUT_USD_PER_MILLION + classifier_output * HAIKU_OUTPUT_USD_PER_MILLION) / Decimal(1_000_000)
    detector = (detector_input * HAIKU_INPUT_USD_PER_MILLION + detector_output * HAIKU_OUTPUT_USD_PER_MILLION) / Decimal(1_000_000)
    reader = (reader_input * LUNA_INPUT_USD_PER_MILLION + reader_output * LUNA_OUTPUT_USD_PER_MILLION) / Decimal(1_000_000)
    judge = (judge_input * GPT4O_INPUT_USD_PER_MILLION + judge_output * GPT4O_OUTPUT_USD_PER_MILLION) / Decimal(1_000_000)
    # Retry bounds multiply both input and output token envelopes. The reserve
    # factor remains an additional uncertainty margin, not a retry substitute.
    classifier *= CLASSIFIER_MAX_RETRIES + 1
    detector *= DETECTOR_MAX_RETRIES + 1
    reader *= READER_MAX_RETRIES + 1
    judge *= JUDGE_MAX_RETRIES + 1
    incremental = (classifier + detector + reader + judge) * RESERVE_FACTOR
    cumulative = prior_spend_usd + incremental
    return {
        "models": {"classifier": CLASSIFIER_MODEL, "detector": DETECTOR_MODEL, "reader": READER_MODEL, "judge": JUDGE_MODEL},
        "calls": {"classifier": len(turns), "detector": detector_calls, "reader": reader_calls, "judge": judge_calls},
        "bounds_usd": {"classifier": classifier, "detector": detector, "reader": reader, "judge": judge, "incremental_reserved": incremental, "prior_spend": prior_spend_usd, "cumulative_reserved": cumulative},
        "reserve_factor": RESERVE_FACTOR,
        "operational_stop_usd": OPERATIONAL_STOP_USD,
        "authorization_ceiling_usd": AUTHORIZATION_CEILING_USD,
    }


def enforce_corrected_budget(estimate: dict[str, Any]) -> None:
    """Reject before provider/DB work when the cumulative bound is unsafe."""
    cumulative = estimate["bounds_usd"]["cumulative_reserved"]
    if cumulative > AUTHORIZATION_CEILING_USD:
        raise RuntimeError(f"corrected BELIEF preflight cumulative bound ${cumulative:.2f} exceeds authorization ceiling ${AUTHORIZATION_CEILING_USD:.2f}")
    if cumulative > OPERATIONAL_STOP_USD:
        raise RuntimeError(f"corrected BELIEF preflight cumulative bound ${cumulative:.2f} exceeds operational stop ${OPERATIONAL_STOP_USD:.2f}")


def estimate_cost(
    questions: int,
    *,
    repetitions: int = REPETITIONS,
    reader_input_tokens: Decimal = READER_INPUT_TOKENS_PER_QUESTION,
    reader_output_tokens: Decimal = READER_OUTPUT_TOKENS_PER_QUESTION,
    judge_input_tokens: Decimal = JUDGE_INPUT_TOKENS_PER_QUESTION,
    judge_output_tokens: Decimal = JUDGE_OUTPUT_TOKENS_PER_QUESTION,
) -> CostEstimate:
    if questions < 1 or repetitions < 1:
        raise ValueError("questions and repetitions must be positive")
    calls = questions * len(ARMS) * repetitions
    reader_per_call = (
        reader_input_tokens * LUNA_INPUT_USD_PER_MILLION
        + reader_output_tokens * LUNA_OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)
    judge_per_call = (
        judge_input_tokens * GPT4O_INPUT_USD_PER_MILLION
        + judge_output_tokens * GPT4O_OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)
    reader_cost = reader_per_call * calls
    judge_cost = judge_per_call * calls
    total = (reader_cost + judge_cost) * RESERVE_FACTOR
    return CostEstimate(
        questions=questions,
        arms=len(ARMS),
        repetitions=repetitions,
        reader_calls=calls,
        judge_calls=calls,
        reader_cost_usd=reader_cost,
        judge_cost_usd=judge_cost,
        total_cost_usd=total,
        reserve_factor=RESERVE_FACTOR,
        operational_stop_usd=OPERATIONAL_STOP_USD,
        authorization_ceiling_usd=AUTHORIZATION_CEILING_USD,
    )


def _manifest_body(
    root: Path,
    dataset_path: Path,
    selected: list[Instance],
    *,
    per_type: int,
    seed: int,
    normalization_report_path: Path,
    normalization_source_path: Path,
) -> dict[str, Any]:
    dataset_hash = sha256_file(dataset_path)
    ordered_ids = [instance.question_id for instance in selected]
    selection = {
        "strategy": "equal_by_question_type_source_order",
        "question_types": list(QUESTION_TYPES),
        "per_type": per_type,
        "seed": seed,
        "ordered_question_ids": ordered_ids,
        "ordered_question_ids_sha256": sha256_bytes(_canonical(ordered_ids)),
    }
    return {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_AUTHORIZED",
        "dataset": {
            "name": dataset_path.name,
            "sha256": dataset_hash,
            "population_count": len(load_split(dataset_path)),
        },
        "normalization": _normalization_metadata(
            dataset_path,
            normalization_report_path,
            source_path=normalization_source_path,
        ),
        "selection": selection,
        "arms": list(ARMS),
        "repetitions": REPETITIONS,
        "ingest": {"mode": "dual", "representations": ["raw_memory", "episode_turns"]},
        "retrieval": {"top_k": 10, "recall_k": RECALL_K, "label_blind": True},
        "reader": {
            "provider": "openai",
            "model": READER_MODEL,
            "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
        },
        "judge": {"provider": "openai", "model": JUDGE_MODEL},
        "cost": estimate_cost(len(selected)).to_json(),
        "source_hashes": source_hashes(root),
        "execution_boundary": {
            "preparation_only": True,
            "provider_calls": False,
            "database_writes": False,
            "judge_calls": False,
            "authorization_required_before_execution": True,
        },
    }


def prepare_manifest(
    root: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    per_type: int = 6,
    seed: int = 0,
    normalization_report_path: Path,
    normalization_source_path: Path,
) -> dict[str, Any]:
    """Write the fixed six-per-type, provider-free pilot manifest and cost plan."""
    if per_type != 6:
        raise ValueError("the three-tier pilot requires exactly six questions per type")
    selected = select_questions(dataset_path, per_type=per_type, seed=seed)
    body = _manifest_body(
        root,
        dataset_path,
        selected,
        per_type=per_type,
        seed=seed,
        normalization_report_path=normalization_report_path,
        normalization_source_path=normalization_source_path,
    )
    body["manifest_sha256"] = _manifest_hash(body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return body


def load_manifest(
    root: Path,
    dataset_path: Path,
    path: Path,
    *,
    normalization_source_path: Path,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") != "PREPARED_NOT_AUTHORIZED":
        raise ValueError("pilot manifest is not a prepared v1 manifest")
    recorded_hash = value.pop("manifest_sha256", None)
    if recorded_hash != _manifest_hash(value):
        raise ValueError("pilot manifest hash mismatch")
    value["manifest_sha256"] = recorded_hash
    if value["dataset"]["sha256"] != sha256_file(dataset_path):
        raise ValueError("pilot dataset checksum mismatch")
    normalization = value.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("pilot manifest is missing mandatory normalization binding")
    if normalization.get("derived_sha256") != value["dataset"]["sha256"]:
        raise ValueError("pilot normalization derived checksum mismatch")
    if normalization.get("source_sha256") != sha256_file(normalization_source_path):
        raise ValueError("pilot normalization source checksum mismatch")
    report_filename = normalization.get("report_filename")
    if not isinstance(report_filename, str) or Path(report_filename).name != report_filename:
        raise ValueError("pilot normalization report filename is invalid")
    report_path = path.parent / report_filename
    if not report_path.is_file() or sha256_file(report_path) != normalization.get("report_sha256"):
        raise ValueError("pilot normalization report checksum mismatch")
    if normalization.get("exclusion_count") != len(normalization.get("excluded_question_ids", [])):
        raise ValueError("pilot normalization exclusion count mismatch")
    current_sources = source_hashes(root)
    if current_sources != value["source_hashes"]:
        raise ValueError("pilot source hash mismatch; regenerate the manifest")
    selection = value["selection"]
    ordered_ids = selection.get("ordered_question_ids", [])
    if selection.get("per_type") != 6 or len(ordered_ids) != 6 * len(QUESTION_TYPES):
        raise ValueError("pilot manifest must contain exactly six questions per type")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("pilot manifest contains duplicate question IDs")
    instances = {instance.question_id: instance for instance in load_split(dataset_path)}
    if set(ordered_ids) - set(instances):
        raise ValueError("pilot manifest contains unknown question IDs")
    counts = {question_type: 0 for question_type in QUESTION_TYPES}
    for question_id in ordered_ids:
        question_type = instances[question_id].question_type
        if question_type not in counts:
            raise ValueError("pilot manifest contains unsupported question type")
        counts[question_type] += 1
    if set(counts.values()) != {6}:
        raise ValueError("pilot manifest question types are not balanced")
    if tuple(value["arms"]) != ARMS:
        raise ValueError("pilot arms are not turns/belief/auto")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_hash_for(value: dict[str, Any]) -> str:
    return sha256_bytes(_canonical(value))


def corrected_manifest_body(
    root: Path,
    dataset_path: Path,
    selected: list[Instance],
    *,
    normalization_report_path: Path,
    normalization_source_path: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Describe the corrected, belief-only production-faithful pilot.

    This manifest is intentionally separate from the historical three-arm
    manifest.  Its ingest and retrieval fields are executable contract, not
    descriptive labels, and the source hashes bind the production seams that
    affect the result.
    """
    ids = [item.question_id for item in selected]
    return {
        "schema": CORRECTED_SCHEMA,
        "status": "PREPARED_NOT_AUTHORIZED",
        "dataset": {
            "name": dataset_path.name,
            "sha256": sha256_file(dataset_path),
            "population_count": len(load_split(dataset_path)),
        },
        "normalization": _normalization_metadata(
            dataset_path,
            normalization_report_path,
            source_path=normalization_source_path,
        ),
        "selection": {
            "strategy": "equal_by_question_type_source_order",
            "question_types": list(QUESTION_TYPES),
            "per_type": len(ids) // len(QUESTION_TYPES),
            "seed": seed,
            "ordered_question_ids": ids,
            "ordered_question_ids_sha256": sha256_bytes(_canonical(ids)),
        },
        "arm": CORRECTED_ARM,
        "ingest": {
            "mode": CORRECTED_INGEST_MODE,
            "item_granularity": "one_IngestItem_per_source_turn",
            "episode_turn_granularity": "one_episode_turn_per_source_turn",
            "source": "longmemeval",
            "metadata": ["benchmark", "benchmark_source", "session_id", "question_id", "turn_role"],
            "whole_session_shortcut": False,
        },
        "materialization": {
            "detector": "weft.views.belief_detector.detect_belief_updates",
            "writer": "weft.views.materializer.materialize_turn",
            "scope": "question_project_sandbox",
            "cursor": "none",
        },
        "retrieval": {
            "entrypoint": "weft.mcp.tools.weft_recall",
            "modes": ["auto", "belief", "turns"],
            "tier": CORRECTED_TIER,
            "claim_lookup": "public_belief_claims_first_when_project_id_null",
            "memory_fallback": "public_weft_recall_belief_path",
            "user_id": "unique_owner_per_run_and_question",
            "scope": CORRECTED_SCOPE,
            "label_blind": True,
            "answer_gold_metadata": False,
        },
        "reader": {
            "provider": "openai",
            "model": READER_MODEL,
            "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
            "context_capture": "exact_system_and_user_content_at_reader_boundary",
            "enumeration_renderer": "benchmarks.longmemeval.reader.format_recall_context",
        },
        "judge": {"provider": "openai", "model": JUDGE_MODEL},
        "cost": {
            "stages": {
                "ingest_classifier": {"status": "unknown", "provider_usage": "unknown"},
                "materialization_detector": {"status": "unknown", "provider_usage": "unknown"},
                "reader": {"status": "planned", "provider_usage": "captured_per_row"},
                "judge": {"status": "planned", "provider_usage": "captured_when_run"},
            },
            "operational_stop_usd": str(OPERATIONAL_STOP_USD),
            "authorization_ceiling_usd": str(AUTHORIZATION_CEILING_USD),
            "unknowns_are_not_zero": True,
        },
        "source_hashes": source_hashes(root),
        "execution_boundary": {
            "preparation_only": True,
            "provider_calls": False,
            "database_writes": False,
            "judge_calls": False,
            "authorization_required_before_execution": True,
        },
    }


def prepare_corrected_manifest(
    root: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    normalization_report_path: Path,
    normalization_source_path: Path,
    per_type: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a provider/database-free manifest for the corrected belief pilot."""
    if per_type != 6:
        raise ValueError("the corrected pilot requires exactly six questions per type")
    historical_manifest_path = root / "benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json"
    historical = json.loads(historical_manifest_path.read_text(encoding="utf-8"))
    if historical.get("schema") != SCHEMA:
        raise ValueError("corrected pilot requires the existing S36 pilot manifest")
    historical_ids = historical.get("selection", {}).get("ordered_question_ids", [])
    instances = {item.question_id: item for item in load_split(dataset_path)}
    if len(historical_ids) != 36 or set(historical_ids) - set(instances):
        raise ValueError("S36 selection is unavailable in the supplied dataset")
    selected = [instances[question_id] for question_id in historical_ids]
    body = corrected_manifest_body(
        root,
        dataset_path,
        selected,
        normalization_report_path=normalization_report_path,
        normalization_source_path=normalization_source_path,
        seed=seed,
    )
    body["manifest_sha256"] = _manifest_hash_for(body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return body


def load_corrected_manifest(
    root: Path,
    dataset_path: Path,
    path: Path,
    *,
    normalization_source_path: Path,
) -> dict[str, Any]:
    """Validate a corrected manifest and reject historical-arm manifests."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != CORRECTED_SCHEMA or value.get("status") != "PREPARED_NOT_AUTHORIZED":
        raise ValueError("pilot manifest is not a prepared corrected-belief manifest")
    recorded_hash = value.pop("manifest_sha256", None)
    if recorded_hash != _manifest_hash_for(value):
        raise ValueError("corrected pilot manifest hash mismatch")
    value["manifest_sha256"] = recorded_hash
    if value.get("dataset", {}).get("sha256") != sha256_file(dataset_path):
        raise ValueError("corrected pilot dataset checksum mismatch")
    normalization = value.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("corrected manifest is missing normalization binding")
    if normalization.get("source_sha256") != sha256_file(normalization_source_path):
        raise ValueError("corrected normalization source checksum mismatch")
    if tuple(value.get("selection", {}).get("question_types", ())) != QUESTION_TYPES:
        raise ValueError("corrected manifest question types are not balanced")
    ids = value["selection"].get("ordered_question_ids", [])
    if len(ids) != 36 or len(set(ids)) != len(ids):
        raise ValueError("corrected manifest must contain 36 unique question IDs")
    instances = {item.question_id: item for item in load_split(dataset_path)}
    if set(ids) - set(instances):
        raise ValueError("corrected manifest contains unknown question IDs")
    counts = {question_type: 0 for question_type in QUESTION_TYPES}
    for question_id in ids:
        question_type = instances[question_id].question_type
        if question_type not in counts:
            raise ValueError("corrected manifest contains unsupported question type")
        counts[question_type] += 1
    if set(counts.values()) != {6}:
        raise ValueError("corrected manifest question types are not balanced")
    if value.get("arm") != CORRECTED_ARM:
        raise ValueError("corrected manifest is not belief-only")
    if value.get("ingest", {}).get("whole_session_shortcut") is not False:
        raise ValueError("corrected manifest permits a whole-session ingestion shortcut")
    return value


async def _public_recall(
    pool: Any,
    embedder: Any,
    *,
    question: str,
    user_id: str,
    tier: str,
    limit: int,
) -> dict[str, Any]:
    """Call the same public MCP recall entrypoint used by normal queries."""
    from weft.auth import current_user_id
    from weft.cache import NullCache
    from weft.config import WeftConfig
    from weft.mcp.server import AppContext
    from weft.mcp.tools import weft_recall

    app = AppContext(pool=pool, cache=NullCache(), embedding=embedder, config=WeftConfig())

    async def _list_roots() -> list:
        return []

    ctx = types.SimpleNamespace(
        request_context=types.SimpleNamespace(lifespan_context=app),
        list_roots=_list_roots,
    )
    token: Token = current_user_id.set(user_id)
    try:
        return await weft_recall(
            ctx, query=question, project_id=None, user_id=user_id,
            tier=tier, limit=limit, mode="hybrid", retrieval_mode="face",
        )
    finally:
        current_user_id.reset(token)


def _corrected_ledger_path(output_dir: Path) -> Path:
    return output_dir / "corrected-belief-attempt-ledger.json"


def _load_corrected_ledger(path: Path, manifest: dict[str, Any], ordered_ids: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("corrected attempt ledger is unreadable; refusing restart") from exc
    if not isinstance(ledger, dict) or ledger.get("schema") != "weft.longmemeval.attempt-ledger.v1":
        raise RuntimeError("corrected attempt ledger schema is invalid; refusing restart")
    if ledger.get("manifest_sha256") != manifest["manifest_sha256"] or ledger.get("question_ids") != ordered_ids:
        raise RuntimeError("corrected attempt ledger is not bound to this manifest; refusing restart")
    completed = ledger.get("completed_ids", [])
    if not isinstance(completed, list) or len(completed) != len(set(completed)) or set(completed) - set(ordered_ids):
        raise RuntimeError("corrected attempt ledger completed IDs are inconsistent")
    failed = ledger.get("failed_ids", [])
    if not isinstance(failed, list) or len(failed) != len(set(failed)) or set(failed) - set(ordered_ids):
        raise RuntimeError("corrected attempt ledger failed IDs are inconsistent")
    try:
        Decimal(str(ledger["consumed_usd"]))
        Decimal(str(ledger["reserved_usd"]))
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        raise RuntimeError("corrected attempt ledger cost fields are invalid") from exc
    if ledger.get("status") == "failed" and len(completed) > MAX_RESUME_COMPLETED:
        raise RuntimeError("corrected pilot will not restart after failure beyond 50% completion")
    return ledger


def _write_corrected_ledger(path: Path, value: dict[str, Any]) -> None:
    _write_json(path, value)


def _make_corrected_classifier_provider() -> AnthropicTextGenerationProvider:
    """Construct the benchmark classifier with its explicit retry cap."""
    from anthropic import AsyncAnthropic

    return AnthropicTextGenerationProvider(
        AsyncAnthropic(max_retries=CLASSIFIER_MAX_RETRIES), owns_client=True
    )


def _make_corrected_detector() -> tuple[Detector, Any]:
    """Construct the benchmark detector and return its owned SDK client."""
    from anthropic import AsyncAnthropic
    from weft.views.belief_detector import detect_belief_updates

    client = AsyncAnthropic(max_retries=DETECTOR_MAX_RETRIES)
    return (lambda turn: detect_belief_updates(turn, client=client)), client


async def run_corrected_pilot(
    *,
    root: Path,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    normalization_source_path: Path,
    execute: bool = False,
    detector: Detector | None = None,
    generation_provider: TextGenerationProvider | None = None,
    pool=None,
    embedder=None,
    reader: Reader | None = None,
) -> dict[str, Any]:
    """Run the corrected single-arm production-faithful belief pilot.

    The live path requires the same explicit local target and provider boundary
    as the historical pilot, but it never calls the legacy dual substrate.  A
    detector/provider/DB/Reader may be injected for provider-free tests.
    """
    manifest = load_corrected_manifest(
        root,
        dataset_path,
        manifest_path,
        normalization_source_path=normalization_source_path,
    )
    instances = {instance.question_id: instance for instance in load_split(dataset_path)}
    ordered_ids = manifest["selection"]["ordered_question_ids"]
    rows_path = output_dir / "corrected-belief-rows.jsonl"
    ledger_path = _corrected_ledger_path(output_dir)
    ledger = _load_corrected_ledger(ledger_path, manifest, ordered_ids)
    existing: set[str] = set()
    row_statuses: dict[str, set[str]] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("corrected rows contain malformed JSON; refusing restart") from exc
            qid = row.get("question_id")
            status = row.get("status")
            if row.get("manifest_sha256") != manifest["manifest_sha256"] or qid not in ordered_ids or status not in {"ok", "error"}:
                raise RuntimeError("corrected rows contain an invalid manifest question ID or status")
            statuses = row_statuses.setdefault(qid, set())
            if status == "ok" and statuses:
                raise RuntimeError("corrected rows contain a duplicate successful question ID")
            statuses.add(status)
            if status == "ok":
                existing.add(qid)
    if existing and ledger is None:
        raise RuntimeError("corrected attempt ledger is required when resuming rows")
    if ledger is not None and set(ledger["completed_ids"]) != existing:
        raise RuntimeError("corrected attempt ledger does not match successful rows")
    prior_text = os.environ.get("LONGMEMEVAL_PRIOR_SPEND_USD", "").strip()
    if ledger is not None:
        prior_spend = Decimal(str(ledger["reserved_usd"]))
    else:
        if not prior_text:
            prior_text = "0"
        try:
            prior_spend = Decimal(prior_text)
        except Exception as exc:
            raise RuntimeError("LONGMEMEVAL_PRIOR_SPEND_USD must be a non-negative decimal") from exc
        if prior_spend < 0:
            raise RuntimeError("LONGMEMEVAL_PRIOR_SPEND_USD must be non-negative")
    pending_ids = [question_id for question_id in ordered_ids if question_id not in existing]
    if len(existing) == len(ordered_ids) and ledger is not None and ledger.get("status") == "complete":
        pending_ids = []
    estimate = corrected_cost_estimate(list(instances.values()), pending_ids, judge_questions=len(ordered_ids), prior_spend_usd=prior_spend)
    if ledger is not None and ledger.get("reserved_usd") is not None:
        # The ledger's reservation already includes this attempt's remaining
        # work; never add it again after a crash/restart.
        estimate["bounds_usd"]["cumulative_reserved"] = Decimal(str(ledger["reserved_usd"]))
    enforce_corrected_budget(estimate)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ledger is None:
        ledger = {
            "schema": "weft.longmemeval.attempt-ledger.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "question_ids": ordered_ids,
            "attempt_id": sha256_bytes(_canonical({"manifest": manifest["manifest_sha256"], "rows": str(rows_path)}))[:16],
            "status": "started",
            "completed_ids": [],
            "failed_ids": [],
            "consumed_usd": "0",
            "reserved_usd": str(prior_spend),
        }
        _write_corrected_ledger(ledger_path, ledger)
    if not execute and any(value is None for value in (pool, embedder, reader)):
        return {
            "status": "prepared",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "cost": estimate,
        }
    if not execute and pool is None:
        raise RuntimeError("corrected pilot requires --execute for live resources")
    if execute:
        if not os.environ.get("OPENAI_API_KEY") and generation_provider is None:
            raise RuntimeError("OPENAI_API_KEY is required for --execute")
        if os.environ.get("WEFT_EMBEDDING_PROVIDER", "fastembed") != "fastembed" and embedder is None:
            raise RuntimeError("corrected pilot requires local FastEmbed")
        dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL", "").strip()
        if pool is None and not dsn:
            raise RuntimeError("LONGMEMEVAL_DATABASE_URL is required for --execute")
        if pool is None:
            parsed = urlsplit(dsn.replace("+psycopg2", ""))
            if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"} or (parsed.port or 5432) != 55432 or parsed.path.lstrip("/") != "weftbench":
                raise RuntimeError("corrected pilot requires the dedicated local target postgresql://...@127.0.0.1:55432/weftbench")

    from benchmarks.longmemeval.adapter import _make_embedder, _make_pool

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "corrected-belief-summary.json"

    owns_pool = pool is None
    owns_embedder = embedder is None
    owns_reader = reader is None
    owns_classifier_provider = generation_provider is None
    owns_detector_client = detector is None
    classifier_provider = generation_provider
    detector_client = None
    if owns_classifier_provider:
        classifier_provider = _make_corrected_classifier_provider()
    if owns_detector_client:
        detector, detector_client = _make_corrected_detector()
    pool = pool or await _make_pool()
    embedder = embedder or _make_embedder()
    reader = reader or Reader(
        provider=OpenAITextGenerationProvider(max_retries=READER_MAX_RETRIES),
        model=READER_MODEL,
    )
    reader_provider = getattr(reader, "_provider", None) if owns_reader else None
    completed_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        with rows_path.open("a", encoding="utf-8") as output:
            for question_id in ordered_ids:
                if question_id in existing:
                    continue
                instance = instances[question_id]
                project_id = project_id_for(question_id)
                question_user_id = f"{CORRECTED_USER_ID}-{manifest['manifest_sha256'][:12]}-{question_id}"
                turn_session_map: dict[str, str] = {}
                turn_content_map: dict[str, str] = {}
                question_started = time.perf_counter()
                try:
                    from weft.auth import current_user_id
                    owner_token = current_user_id.set(question_user_id)
                    try:
                        inserted = await load_haystack(
                            pool,
                            embedder,
                            instance,
                            CORRECTED_INGEST_MODE,
                            generation_provider=classifier_provider,
                            config=None,
                            turn_session_map=turn_session_map,
                            turn_content_map=turn_content_map,
                        )
                        mat_stats = await materialize_question(pool, project_id, detector=detector)
                        shape = derive_task_shape(instance.question, instance.sessions)
                        policy = RetrievalPolicy(top_k=max(10, shape.top_k), overfetch_multiplier=4)
                        recall_response = await _public_recall(
                            pool, embedder, question=instance.question,
                            user_id=question_user_id, tier=CORRECTED_TIER,
                            limit=policy.top_k,
                        )
                    finally:
                        current_user_id.reset(owner_token)
                    response = await reader.read_answer(
                        question=instance.question,
                        question_date=instance.question_date,
                        task_shape=shape.task_shape,
                        recall_response=recall_response,
                        top_k=policy.top_k,
                    )
                    row = {
                        "question_id": question_id,
                        "question_type": instance.question_type,
                        "arm": CORRECTED_ARM,
                        "tier": CORRECTED_TIER,
                        "status": "ok",
                        "manifest_sha256": manifest["manifest_sha256"],
                        "dataset_sha256": manifest["dataset"]["sha256"],
                        "task_shape": shape.task_shape,
                        "routing_class": shape.routing_class,
                        "ingest_mode": CORRECTED_INGEST_MODE,
                        "ingest_item_count": len(turn_session_map),
                        "episode_turn_count": len(turn_session_map),
                        "materialize": mat_stats.to_dict(),
                        "retrieval_requested_tier": CORRECTED_TIER,
                        "retrieval_selected_tier": recall_response.get("tier", CORRECTED_TIER),
                        "retrieval_fallback": recall_response.get("tier_fallback"),
                        "retrieval_response": recall_response,
                        "retrieval_count": recall_response.get("count", 0),
                        "retrieved_ids": [item.get("id") for item in (recall_response.get("results") or recall_response.get("turns") or []) if isinstance(item, dict)],
                        "reader_provider": "openai",
                        "reader_model": response.model,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "provider_usage": {
                            "reader": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
                            "ingest_classifier": "unknown",
                            "materialization_detector": "unknown",
                            "judge": "not_run",
                        },
                        "hypothesis": response.hypothesis,
                        "reader_context": {
                            "system_prompt": response.system_prompt,
                            "user_content": response.user_content,
                        },
                        "retrieval_diagnostics": recall_response.get(
                            "diagnostics",
                            {"available": False, "reason": "public recall response does not expose router diagnostics"},
                        ),
                        "elapsed_ms": (time.perf_counter() - question_started) * 1000.0,
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "question_id": question_id,
                        "question_type": instance.question_type,
                        "arm": CORRECTED_ARM,
                        "status": "error",
                        "manifest_sha256": manifest["manifest_sha256"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    await cleanup_haystack(pool, instance)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if row["status"] == "ok":
                    existing.add(question_id)
                    ledger["completed_ids"] = sorted(existing, key=ordered_ids.index)
                else:
                    ledger["failed_ids"] = sorted(set(ledger["failed_ids"]) | {question_id}, key=ordered_ids.index)
                    ledger["status"] = "failed"
                ledger["reserved_usd"] = str(estimate["bounds_usd"]["cumulative_reserved"])
                _write_corrected_ledger(ledger_path, ledger)
                completed_rows.append(row)
                if row["status"] != "ok":
                    raise RuntimeError(row["error"])
    finally:
        if reader_provider is not None and hasattr(reader_provider, "aclose"):
            await reader_provider.aclose()
        if owns_classifier_provider and classifier_provider is not None and hasattr(classifier_provider, "aclose"):
            await classifier_provider.aclose()
        if owns_detector_client and detector_client is not None:
            close = getattr(detector_client, "aclose", None) or getattr(detector_client, "close", None)
            if callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result
        if owns_pool:
            await pool.close()
        if owns_embedder and hasattr(embedder, "aclose"):
            await embedder.aclose()

    summary = {
        "schema": CORRECTED_SCHEMA,
        "status": "complete" if len(existing) == len(ordered_ids) else "incomplete",
        "questions": len(ordered_ids),
        "arm": CORRECTED_ARM,
        "new_questions": len(completed_rows),
        "previously_complete_questions": len(existing),
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "rows_path": str(rows_path),
        "elapsed_seconds": time.monotonic() - started,
        "cost_stages": {
            "ingest_classifier": "unknown",
            "materialization_detector": "unknown",
            "reader": "captured_per_row",
            "judge": "not_run",
        },
        "judge_projection_command": (
            "uv run python -m benchmarks.longmemeval.pilot judge-projection "
            f"--rows {rows_path} --manifest {manifest_path} --output-dir {output_dir} --tier belief"
        ),
    }
    if summary["status"] == "complete":
        ledger["status"] = "complete"
        ledger["completed_ids"] = ordered_ids
        _write_corrected_ledger(ledger_path, ledger)
    _write_json(summary_path, summary)
    return summary


async def run_pilot(
    *,
    root: Path,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    normalization_source_path: Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Run the one-repetition three-arm pilot after explicit ``--execute``."""
    manifest = load_manifest(
        root,
        dataset_path,
        manifest_path,
        normalization_source_path=normalization_source_path,
    )
    estimate = estimate_cost(len(manifest["selection"]["ordered_question_ids"]))
    if estimate.total_cost_usd > OPERATIONAL_STOP_USD:
        raise RuntimeError(
            f"pilot estimate ${estimate.total_cost_usd:.2f} exceeds operational stop ${OPERATIONAL_STOP_USD:.2f}"
        )
    if not execute:
        return {"status": "prepared", "manifest": str(manifest_path), "cost": estimate.to_json()}
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --execute")
    if os.environ.get("WEFT_EMBEDDING_PROVIDER", "fastembed") != "fastembed":
        raise RuntimeError("pilot requires local FastEmbed; hosted embedding spend is not in this cost plan")
    dsn = os.environ.get("LONGMEMEVAL_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("LONGMEMEVAL_DATABASE_URL is required for --execute")
    parsed = urlsplit(dsn.replace("+psycopg2", ""))
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"} or (parsed.port or 5432) != 55432 or parsed.path.lstrip("/") != "weftbench":
        raise RuntimeError("pilot requires the dedicated local target postgresql://...@127.0.0.1:55432/weftbench")

    from benchmarks.longmemeval.adapter import BENCHMARK_USER_ID, _make_embedder, _make_pool

    instances = {instance.question_id: instance for instance in load_split(dataset_path)}
    ordered_ids = manifest["selection"]["ordered_question_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "pilot-rows.jsonl"
    summary_path = output_dir / "pilot-summary.json"
    existing: set[tuple[str, str, int]] = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("manifest_sha256") == manifest["manifest_sha256"]:
                existing.add((row.get("question_id"), row.get("tier"), int(row.get("repetition", 0))))

    pool = await _make_pool()
    embedder = _make_embedder()
    provider = OpenAITextGenerationProvider()
    reader = Reader(provider=provider, model=READER_MODEL)
    completed_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        with rows_path.open("a", encoding="utf-8") as output:
            for question_id in ordered_ids:
                instance = instances[question_id]
                project_id = project_id_for(question_id)
                turn_session_map: dict[str, str] = {}
                turn_content_map: dict[str, str] = {}
                try:
                    await load_haystack(
                        pool, embedder, instance, "dual",
                        turn_session_map=turn_session_map,
                        turn_content_map=turn_content_map,
                    )
                    shape = derive_task_shape(instance.question, instance.sessions)
                    policy = RetrievalPolicy(top_k=max(10, shape.top_k), overfetch_multiplier=4)
                    for tier in ARMS:
                        key = (question_id, tier, 0)
                        if key in existing:
                            continue
                        diagnostics = RetrievalDiagnostics()
                        question_started = time.perf_counter()
                        memories = await retrieve(
                            pool,
                            embedder,
                            question=instance.question,
                            project_id=project_id,
                            task_shape=shape,
                            policy=policy,
                            tier=tier,
                            user_id=BENCHMARK_USER_ID,
                            expected_turn_count=len(turn_session_map),
                            diagnostics=diagnostics,
                            turn_session_map=turn_session_map,
                        )
                        response = await reader.read_answer(
                            question=instance.question,
                            question_date=instance.question_date,
                            task_shape=shape.task_shape,
                            memories=memories,
                            top_k=policy.top_k,
                        )
                        retrieved_ids = [item.memory.id for item in memories[:RECALL_K]]
                        retrieved_sessions = sorted({turn_session_map[item_id] for item_id in retrieved_ids if item_id in turn_session_map})
                        gold_sessions = sorted({session.session_id for session in instance.sessions if session.has_answer})
                        retrieval_hit = (
                            bool(set(retrieved_sessions) & set(gold_sessions))
                            if retrieved_sessions
                            else None
                        )
                        row = {
                            "question_id": question_id,
                            "question_type": instance.question_type,
                            "tier": tier,
                            "repetition": 0,
                            "status": "ok",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "dataset_sha256": manifest["dataset"]["sha256"],
                            "task_shape": shape.task_shape,
                            "routing_class": shape.routing_class,
                            "reader_provider": "openai",
                            "reader_model": response.model,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "hypothesis": response.hypothesis,
                            "retrieved_ids": retrieved_ids,
                            "retrieved_session_ids": retrieved_sessions,
                            "gold_session_ids": gold_sessions,
                            "recall_at_10_hit": retrieval_hit,
                            "retrieval_metric_available": retrieval_hit is not None,
                            "retrieval_count": len(memories),
                            "retrieval_diagnostics": diagnostics.to_dict(gold_turn_ids=[], top_k=policy.top_k),
                            "elapsed_ms": (time.perf_counter() - question_started) * 1000.0,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        completed_rows.append(row)
                finally:
                    await cleanup_haystack(pool, instance)
    finally:
        await provider.aclose()
        await pool.close()

    expected_cells = len(ordered_ids) * len(ARMS)
    summary = {
        "status": "complete" if len(completed_rows) + len(existing) == expected_cells else "incomplete",
        "questions": len(ordered_ids),
        "arms": list(ARMS),
        "expected_cells": expected_cells,
        "new_cells": len(completed_rows),
        "previously_complete_cells": len(existing),
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "rows_path": str(rows_path),
        "cost_plan": estimate.to_json(),
        "judge_projection_command": "uv run python -m benchmarks.longmemeval.pilot judge-projection --rows " + str(rows_path),
    }
    _write_json(summary_path, summary)
    return summary


def judge_projection(rows_path: Path, manifest_path: Path, output_dir: Path, tier: str) -> Path:
    """Project one tier's successful cells to upstream judge JSONL."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = manifest["selection"]["ordered_question_ids"]
    rows = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("tier") or row.get("arm")) == tier and row.get("status") == "ok":
            if row.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError("row manifest hash mismatch")
            rows.append(row)
    by_id = {row["question_id"]: row for row in rows}
    if set(by_id) != set(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError(f"tier {tier!r} does not have a complete unique denominator")
    output = output_dir / f"hypotheses-{tier}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps({"question_id": question_id, "hypothesis": by_id[question_id]["hypothesis"]}, ensure_ascii=False) + "\n" for question_id in expected_ids),
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("--source", type=Path, required=True)
    normalize.add_argument("--derived", type=Path, required=True)
    normalize.add_argument("--report", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--normalization-report",
        type=Path,
        required=True,
        help="pilot-only normalization report binding a derived dataset to its source",
    )
    prepare.add_argument(
        "--normalization-source",
        type=Path,
        required=True,
        help="immutable upstream source used to verify normalization provenance",
    )
    prepare.add_argument("--per-type", type=int, default=6)
    prepare.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--normalization-source", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    corrected_prepare = sub.add_parser("prepare-corrected")
    corrected_prepare.add_argument("--dataset", type=Path, required=True)
    corrected_prepare.add_argument("--output", type=Path, required=True)
    corrected_prepare.add_argument("--normalization-report", type=Path, required=True)
    corrected_prepare.add_argument("--normalization-source", type=Path, required=True)
    corrected_prepare.add_argument("--per-type", type=int, default=6)
    corrected_prepare.add_argument("--seed", type=int, default=0)
    corrected_run = sub.add_parser("run-corrected")
    corrected_run.add_argument("--dataset", type=Path, required=True)
    corrected_run.add_argument("--manifest", type=Path, required=True)
    corrected_run.add_argument("--normalization-source", type=Path, required=True)
    corrected_run.add_argument("--output-dir", type=Path, required=True)
    corrected_run.add_argument("--execute", action="store_true")
    projection = sub.add_parser("judge-projection")
    projection.add_argument("--rows", type=Path, required=True)
    projection.add_argument("--manifest", type=Path, required=True)
    projection.add_argument("--output-dir", type=Path, required=True)
    projection.add_argument("--tier", choices=ARMS, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "normalize":
        report = normalize_dataset(args.source, args.derived, args.report)
        print(json.dumps({
            "status": report["status"],
            "source_sha256": report["source"]["sha256"],
            "derived_sha256": report["derived"]["sha256"],
            "source_records": report["source"]["record_count"],
            "derived_records": report["derived"]["record_count"],
            "exclusion_count": report["exclusion_count"],
            "collapsed_group_count": report["collapsed_group_count"],
            "report": str(args.report),
        }, indent=2))
        return 0
    if args.command == "prepare":
        manifest = prepare_manifest(
            root,
            args.dataset,
            args.output,
            per_type=args.per_type,
            seed=args.seed,
            normalization_report_path=args.normalization_report,
            normalization_source_path=args.normalization_source,
        )
        print(json.dumps({"status": manifest["status"], "manifest": str(args.output), "cost": manifest["cost"]}, indent=2))
        return 0
    if args.command == "prepare-corrected":
        manifest = prepare_corrected_manifest(
            root,
            args.dataset,
            args.output,
            per_type=args.per_type,
            seed=args.seed,
            normalization_report_path=args.normalization_report,
            normalization_source_path=args.normalization_source,
        )
        print(json.dumps({"status": manifest["status"], "manifest": str(args.output)}, indent=2))
        return 0
    if args.command == "judge-projection":
        print(judge_projection(args.rows, args.manifest, args.output_dir, args.tier))
        return 0
    if args.command == "run-corrected":
        result = asyncio.run(
            run_corrected_pilot(
                root=root,
                dataset_path=args.dataset,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                normalization_source_path=args.normalization_source,
                execute=args.execute,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    result = asyncio.run(
        run_pilot(
            root=root,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            normalization_source_path=args.normalization_source,
            execute=args.execute,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
