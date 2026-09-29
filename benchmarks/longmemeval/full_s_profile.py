"""Prepare the deterministic full LongMemEval-S turns-only run manifest.

This module is offline-only: it reads the cleaned source JSON and writes a
first-occurrence-normalized dataset plus a hash-bound profile manifest. It does
not import database, embedding, provider, or judge clients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.longmemeval.agent_workload import sha256_file

FULL_S_PROFILE = "gpt6-luna-full-s-turns-v1"
FULL_S_CASE_COUNT = 500
FULL_S_MAX_BUDGET_USD = 150.0
FULL_S_OPERATIONAL_STOP_USD = FULL_S_MAX_BUDGET_USD
FULL_S_RESERVE_FACTOR = 1.25
FULL_S_WRITER_MODEL = "gpt-6-luna"
FULL_S_JUDGE_MODEL = "gpt-4o"
FULL_S_ARM = "turns"
DEFAULT_SOURCE = Path("benchmarks/longmemeval/data/longmemeval_s_cleaned.json")
DEFAULT_NORMALIZED = Path("benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json")
DEFAULT_MANIFEST = Path("benchmarks/longmemeval/manifests/longmemeval_s_full_turns_manifest.json")

# The full-S runtime import closure of the faithful runner. faithful_s36
# imports agent_workload, full_s_profile, dataset, faithful_agent, and
# faithful_budget; faithful_agent adds task_shape; and the full-S calibrate/
# resume branches lazily import ingest, judge, and faithful_gateway.
# pilot.py (untracked), router.py, reader.py, and weft/text_generation.py
# belong to the retired selected-35 pilot path and are not pinned here.
SOURCE_HASH_PATHS = (
    "benchmarks/longmemeval/agent_workload.py",
    "benchmarks/longmemeval/dataset.py",
    "benchmarks/longmemeval/faithful_agent.py",
    "benchmarks/longmemeval/faithful_budget.py",
    "benchmarks/longmemeval/faithful_gateway.py",
    "benchmarks/longmemeval/faithful_s36.py",
    "benchmarks/longmemeval/full_s_profile.py",
    "benchmarks/longmemeval/ingest.py",
    "benchmarks/longmemeval/judge.py",
    "benchmarks/longmemeval/task_shape.py",
)


class FullSProfileError(ValueError):
    """Raised for malformed or ambiguous full-S source data."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_first_occurrence(records: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize duplicate session IDs within each question, keeping first occurrences.

    Parallel session ID/date/turn arrays are filtered together. Metadata records
    original array positions for each repeated ID and the retained position.
    Question records and their order are preserved exactly.
    """
    if not isinstance(records, list) or len(records) != FULL_S_CASE_COUNT:
        raise FullSProfileError(f"cleaned S source must contain exactly {FULL_S_CASE_COUNT} records")
    seen_questions: set[str] = set()
    normalized: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for record_index, original in enumerate(records):
        if not isinstance(original, dict):
            raise FullSProfileError(f"source record {record_index} must be an object")
        question_id = original.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise FullSProfileError(f"source record {record_index} has no question_id")
        if question_id in seen_questions:
            raise FullSProfileError(f"duplicate question_id {question_id!r}")
        seen_questions.add(question_id)
        session_ids = original.get("haystack_session_ids")
        dates = original.get("haystack_dates")
        sessions = original.get("haystack_sessions")
        if not all(isinstance(value, list) for value in (session_ids, dates, sessions)):
            raise FullSProfileError(f"question {question_id} has malformed session arrays")
        if not (len(session_ids) == len(dates) == len(sessions)):
            raise FullSProfileError(f"question {question_id} session arrays disagree")
        kept: dict[str, int] = {}
        keep_indexes: list[int] = []
        repeated: dict[str, list[dict[str, int]]] = {}
        for index, session_id in enumerate(session_ids):
            if not isinstance(session_id, str) or not session_id.strip():
                raise FullSProfileError(f"question {question_id} has an invalid session ID")
            if session_id not in kept:
                kept[session_id] = index
                keep_indexes.append(index)
                continue
            repeated.setdefault(session_id, [{"source_index": kept[session_id]}]).append(
                {"source_index": index}
            )
        row = dict(original)
        row["haystack_session_ids"] = [session_ids[index] for index in keep_indexes]
        row["haystack_dates"] = [dates[index] for index in keep_indexes]
        row["haystack_sessions"] = [sessions[index] for index in keep_indexes]
        normalized.append(row)
        for session_id, occurrences in sorted(repeated.items()):
            duplicate_groups.append({
                "question_id": question_id,
                "session_id": session_id,
                "kept_source_index": kept[session_id],
                "occurrences": occurrences,
                "discarded_source_indexes": [item["source_index"] for item in occurrences[1:]],
            })
    metadata = {
        "policy": "first_occurrence_wins_within_question",
        "question_count": len(normalized),
        "duplicate_session_id_group_count": len(duplicate_groups),
        "discarded_session_occurrence_count": sum(len(group["discarded_source_indexes"]) for group in duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }
    return normalized, metadata


def build_full_s_manifest(
    source_path: Path,
    normalized_path: Path,
    *,
    source_hash_paths: Sequence[str] = SOURCE_HASH_PATHS,
) -> dict[str, Any]:
    """Write normalized source data and return the deterministic turns manifest."""
    source_path = Path(source_path)
    normalized_path = Path(normalized_path)
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    try:
        source_records = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise FullSProfileError("cleaned S source is not valid JSON") from exc
    records, deduplication = normalize_first_occurrence(source_records)
    _write_json(normalized_path, records)
    normalized_sha = sha256_file(normalized_path)
    question_ids = [record["question_id"] for record in records]
    code_hashes: dict[str, str] = {}
    for name in source_hash_paths:
        path = Path(name)
        if not path.is_file():
            raise FullSProfileError(f"source for manifest hash is missing: {name}")
        code_hashes[name] = sha256_file(path)
    return {
        "schema": "weft.longmemeval.pilot.v1",
        "status": "PREPARED_NOT_AUTHORIZED",
        "profile": FULL_S_PROFILE,
        "dataset": {
            "name": normalized_path.name,
            "sha256": normalized_sha,
            "population_count": len(records),
        },
        "source_dataset": {
            "name": source_path.name,
            "sha256": source_sha,
            "record_count": len(records),
        },
        "normalization": {
            "schema": "weft.longmemeval.full-s-first-occurrence.v1",
            "source_sha256": source_sha,
            "normalized_sha256": normalized_sha,
            **deduplication,
        },
        "selection": {
            "strategy": "all_source_records_in_source_order",
            "count": len(question_ids),
            "ordered_question_ids": question_ids,
            "ordered_question_ids_sha256": hashlib.sha256(_canonical_bytes(question_ids)).hexdigest(),
        },
        "arms": [FULL_S_ARM],
        "repetitions": 1,
        "ingest": {"mode": "dual", "representations": ["raw_memory", "episode_turns"]},
        "retrieval": {"tier": "turns", "top_k": 10, "recall_k": 10, "label_blind": True},
        "reader": {"provider": "openai", "model": FULL_S_WRITER_MODEL, "max_output_tokens": 2048},
        "judge": {"provider": "openai", "model": FULL_S_JUDGE_MODEL},
        "cost": {
            "questions": len(question_ids),
            "arms": 1,
            "repetitions": 1,
            "reader_calls": len(question_ids),
            "judge_calls": len(question_ids),
            "estimate_basis": "faithful cumulative conservative token-reservation ledger",
            "reference_estimate_usd": "8.840000 for 35 cases; planning reference only",
            "reserve_factor": f"{FULL_S_RESERVE_FACTOR:.6f}",
            "operational_stop_usd": f"{FULL_S_OPERATIONAL_STOP_USD:.6f}",
            "max_budget_usd": f"{FULL_S_MAX_BUDGET_USD:.6f}",
            "stop_semantics": "refuse new provider reservations before dispatch when cumulative estimate reaches the $150 hard cap; in-flight requests settle",
        },
        "source_hashes": code_hashes,
        "execution_boundary": {
            "preparation_only": True,
            "provider_calls": False,
            "database_writes": False,
            "judge_calls": False,
        },
    }


def prepare_full_s_manifest(
    source_path: Path = DEFAULT_SOURCE,
    normalized_path: Path = DEFAULT_NORMALIZED,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    source_hash_paths: Sequence[str] = SOURCE_HASH_PATHS,
) -> dict[str, Any]:
    """Prepare the normalized dataset and atomically publish its run manifest."""
    manifest = build_full_s_manifest(source_path, normalized_path, source_hash_paths=source_hash_paths)
    _write_json(Path(manifest_path), manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = prepare_full_s_manifest(args.source, args.normalized, args.manifest)
    print(json.dumps({
        "status": "prepared_not_authorized",
        "question_count": manifest["selection"]["count"],
        "source_sha256": manifest["source_dataset"]["sha256"],
        "normalized_sha256": manifest["dataset"]["sha256"],
        "manifest": str(args.manifest),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
