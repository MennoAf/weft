#!/usr/bin/env python3
"""
agent_workload.py — Offline LongMemEval S36 agent-workload audit.

This script validates the preserved 36-case selection and describes the workload
shape needed by an agent-selected public-memory pilot. It reads only local JSON,
verifies the recorded source hashes, and emits counts and SHA-256 digests. It
never includes question, answer, or conversation text in its report.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-09-21
Python:  >= 3.12

Dependencies:
    (stdlib only)

Usage:
    See bottom of file for run commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_DERIVATIVE_PATH = Path(
    "/tmp/longmemeval-pilot-normalization-20260921/"
    "longmemeval_s_pilot_derivative.json"
)
DEFAULT_ORIGINAL_PATH = Path(
    "/tmp/longmemeval-source-20260921/data/longmemeval_s_cleaned.json"
)
DEFAULT_MANIFEST_PATH = Path(
    "benchmarks/longmemeval/manifests/longmemeval_s_pilot_manifest.json"
)
DEFAULT_DERIVATIVE_SHA256 = (
    "8b825543dc35734d6f61e578abd73021b21d4db53128975a4c7997be9cff4135"
)
DEFAULT_ORIGINAL_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
EXPECTED_CASE_COUNT = 36


class WorkloadAuditError(ValueError):
    """Raised when the offline workload contract is not satisfied."""


def _canonical_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for stable, text-free digests."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without changing or copying it.

    Args:
        path: Existing file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    """Read and decode a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(
    path: Path,
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> dict[str, Any]:
    """Load a manifest and validate its ordered selection shape.

    ``expected_case_count`` is configurable only to keep unit fixtures small;
    the CLI and normal audit default to the approved 36-case selection.
    """
    value = _read_json(path)
    if not isinstance(value, dict):
        raise WorkloadAuditError("manifest must be a JSON object")
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise WorkloadAuditError("manifest.selection must be an object")
    ids = selection.get("ordered_question_ids")
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise WorkloadAuditError("manifest selection must contain string IDs")
    if len(ids) != expected_case_count or len(set(ids)) != len(ids):
        raise WorkloadAuditError(
            f"manifest must preserve exactly {expected_case_count} unique IDs"
        )
    return value


def _validate_hash(path: Path, expected: str, label: str) -> str:
    """Require an existing file's SHA-256 to equal its recorded value."""
    if not path.is_file():
        raise WorkloadAuditError(f"{label} dataset is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise WorkloadAuditError(f"{label} dataset SHA-256 mismatch")
    return actual


def _require_records(value: Any, label: str) -> list[dict[str, Any]]:
    """Validate that a decoded dataset is a list of object records."""
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise WorkloadAuditError(f"{label} dataset must be a JSON array of objects")
    return value


def _parse_date(value: Any) -> date | None:
    """Parse common LongMemEval date forms without external dependencies."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    for fmt in (
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d (%a) %H:%M",
    ):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    # ISO timestamps with a timezone are reduced to their calendar date.
    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _turns(record: Mapping[str, Any], index: int) -> tuple[list[str], Counter[str], Counter[str]]:
    """Return turn roles/content lengths, rejecting malformed turns."""
    sessions = record.get("haystack_sessions")
    if not isinstance(sessions, list):
        raise WorkloadAuditError(f"case {index} has malformed haystack_sessions")
    # This helper is intentionally unused for the case level; retained as a
    # narrow validation utility for callers that inspect one record.
    del sessions
    return [], Counter(), Counter()


def _history_digest(record: Mapping[str, Any]) -> str:
    """Digest only case history; gold question/answer fields are excluded."""
    history = {
        "haystack_dates": record.get("haystack_dates"),
        "haystack_session_ids": record.get("haystack_session_ids"),
        "haystack_sessions": record.get("haystack_sessions"),
    }
    return _sha256_bytes(_canonical_bytes(history))


def _content_digest(turns: Sequence[Mapping[str, Any]]) -> str:
    """Digest session role/content pairs, excluding identifiers and dates."""
    public_turns = [
        {"role": turn.get("role"), "content": turn.get("content")}
        for turn in turns
    ]
    return _sha256_bytes(_canonical_bytes(public_turns))


def _session_digest(session_id: str, session_date: Any, turns: Sequence[Mapping[str, Any]]) -> str:
    """Digest a session's structural history, including its ID and date."""
    return _sha256_bytes(
        _canonical_bytes(
            {"session_id": session_id, "date": session_date, "turns": list(turns)}
        )
    )


def _session_summary(session_id: str, session_date: Any, turns: Any, case_index: int) -> dict[str, Any]:
    """Summarize one session without returning turn text."""
    if not isinstance(session_id, str) or not session_id:
        raise WorkloadAuditError(f"case {case_index} has malformed session ID")
    if not isinstance(turns, list) or any(not isinstance(turn, dict) for turn in turns):
        raise WorkloadAuditError(f"case {case_index} has malformed session turns")
    role_counts: Counter[str] = Counter()
    chars_by_role: Counter[str] = Counter()
    clean_turns: list[dict[str, Any]] = []
    for turn in turns:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise WorkloadAuditError(f"case {case_index} has malformed turn fields")
        role_counts[role] += 1
        chars_by_role[role] += len(content)
        clean_turns.append({"role": role, "content": content})
    content_digest = _content_digest(clean_turns)
    return {
        "session_id": session_id,
        "date": session_date,
        "date_available": isinstance(session_date, str) and bool(session_date.strip()),
        "date_parseable": _parse_date(session_date) is not None,
        "turn_count": len(clean_turns),
        "characters": sum(chars_by_role.values()),
        "turns_by_role": dict(sorted(role_counts.items())),
        "characters_by_role": dict(sorted(chars_by_role.items())),
        "content_digest": content_digest,
        "session_digest": _session_digest(session_id, session_date, clean_turns),
    }


def _case_summary(record: Mapping[str, Any], case_index: int) -> dict[str, Any]:
    """Summarize one selected case using only non-gold structural metadata."""
    question_id = record.get("question_id")
    question_type = record.get("question_type")
    ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    sessions = record.get("haystack_sessions")
    if not isinstance(question_id, str) or not question_id:
        raise WorkloadAuditError(f"selected case {case_index} has no question_id")
    if not isinstance(question_type, str) or not question_type:
        raise WorkloadAuditError(f"case {question_id} has no question_type")
    if not isinstance(ids, list) or not isinstance(dates, list) or not isinstance(sessions, list):
        raise WorkloadAuditError(f"case {question_id} has malformed session arrays")
    if not (len(ids) == len(dates) == len(sessions)):
        raise WorkloadAuditError(f"case {question_id} session arrays disagree")

    session_rows = [
        _session_summary(session_id, session_date, turns, case_index)
        for session_id, session_date, turns in zip(ids, dates, sessions)
    ]
    role_counts: Counter[str] = Counter()
    chars_by_role: Counter[str] = Counter()
    for row in session_rows:
        role_counts.update(row["turns_by_role"])
        chars_by_role.update(row["characters_by_role"])
    parsed_dates = [_parse_date(row["date"]) for row in session_rows]
    available = [value for value in parsed_dates if value is not None]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "session_count": len(session_rows),
        "turn_count": sum(row["turn_count"] for row in session_rows),
        "characters": sum(row["characters"] for row in session_rows),
        "turns_by_role": dict(sorted(role_counts.items())),
        "characters_by_role": dict(sorted(chars_by_role.items())),
        "sessions": session_rows,
        "history_digest": _history_digest(record),
        "chronological_order": {
            "date_fields_present": all(row["date_available"] for row in session_rows),
            "parseable_date_count": len(available),
            "ordered_non_decreasing": len(available) == len(parsed_dates)
            and all(left <= right for left, right in zip(available, available[1:])),
        },
    }


def _select_records(records: Sequence[Mapping[str, Any]], ordered_ids: Sequence[str], label: str) -> list[Mapping[str, Any]]:
    """Select records in manifest order, never by reselection or sorting."""
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if isinstance(question_id, str):
            if question_id in by_id:
                raise WorkloadAuditError(f"{label} dataset has duplicate question_id")
            by_id[question_id] = record
    missing = [question_id for question_id in ordered_ids if question_id not in by_id]
    if missing:
        raise WorkloadAuditError(f"{label} dataset is missing selected IDs")
    return [by_id[question_id] for question_id in ordered_ids]


def _repetition_report(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe repeated IDs and digest overlap without exposing session text."""
    occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    digest_to_sessions: defaultdict[str, set[str]] = defaultdict(set)
    for case in cases:
        for session in case["sessions"]:
            occurrences[session["session_id"]].append(
                {
                    "question_id": case["question_id"],
                    "content_digest": session["content_digest"],
                    "session_digest": session["session_digest"],
                }
            )
            digest_to_sessions[session["content_digest"]].add(session["session_id"])
    repeated = {
        session_id: {
            "occurrence_count": len(rows),
            "content_digests": sorted({row["content_digest"] for row in rows}),
            "conflict": len({row["content_digest"] for row in rows}) > 1,
            "occurrences": rows,
        }
        for session_id, rows in sorted(occurrences.items())
        if len(rows) > 1
    }
    overlaps = {
        digest: sorted(session_ids)
        for digest, session_ids in sorted(digest_to_sessions.items())
        if len(session_ids) > 1
    }
    return {
        "unique_session_id_count": len(occurrences),
        "repeated_session_id_count": len(repeated),
        "conflicting_repeated_session_id_count": sum(
            item["conflict"] for item in repeated.values()
        ),
        "repeated_session_ids": repeated,
        "content_digest_overlap_count": len(overlaps),
        "content_digest_overlap": overlaps,
    }


def audit_workload(
    derivative_path: Path = DEFAULT_DERIVATIVE_PATH,
    original_path: Path = DEFAULT_ORIGINAL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    expected_derivative_sha256: str = DEFAULT_DERIVATIVE_SHA256,
    expected_original_sha256: str = DEFAULT_ORIGINAL_SHA256,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> dict[str, Any]:
    """Run the complete offline S36 workload audit.

    Args:
        derivative_path: Preserved normalized derivative JSON path.
        original_path: Preserved upstream cleaned JSON path.
        manifest_path: Existing manifest containing the ordered 36 IDs.
        expected_derivative_sha256: Recorded derivative digest.
        expected_original_sha256: Recorded source digest.

    Returns:
        A JSON-compatible, gold-blind report containing hashes, counts,
        per-case/session lengths, repetition diagnostics, ordering diagnostics,
        and history digests.

    Raises:
        WorkloadAuditError: If files, hashes, records, or selection invariants fail.
    """
    manifest = load_manifest(manifest_path, expected_case_count=expected_case_count)
    ordered_ids = manifest["selection"]["ordered_question_ids"]
    derivative_hash = _validate_hash(derivative_path, expected_derivative_sha256, "derivative")
    original_hash = _validate_hash(original_path, expected_original_sha256, "original")
    derivative_records = _require_records(_read_json(derivative_path), "derivative")
    original_records = _require_records(_read_json(original_path), "original")
    selected_derivative = _select_records(derivative_records, ordered_ids, "derivative")
    selected_original = _select_records(original_records, ordered_ids, "original")
    # Confirm both preserved materials contain the same selected ID sequence;
    # no ordering or re-selection is performed here.
    if [row.get("question_id") for row in selected_original] != list(ordered_ids):
        raise WorkloadAuditError("original selected IDs do not match manifest order")
    cases = [_case_summary(record, index) for index, record in enumerate(selected_derivative)]
    original_history_digests = [
        _history_digest(record) for record in selected_original
    ]
    derivative_history_digests = [case["history_digest"] for case in cases]
    total_roles: Counter[str] = Counter()
    total_chars_by_role: Counter[str] = Counter()
    for case in cases:
        total_roles.update(case["turns_by_role"])
        total_chars_by_role.update(case["characters_by_role"])
    chronology = {
        "case_count": len(cases),
        "all_cases_have_parseable_session_dates": all(
            case["chronological_order"]["parseable_date_count"] == case["session_count"]
            for case in cases
        ),
        "ordered_case_count": sum(
            case["chronological_order"]["ordered_non_decreasing"] for case in cases
        ),
        "non_chronological_question_ids": [
            case["question_id"]
            for case in cases
            if not case["chronological_order"]["ordered_non_decreasing"]
        ],
    }
    return {
        "schema": "weft.longmemeval.agent-workload.v1",
        "gold_blind": True,
        "selection": {
            "count": len(ordered_ids),
            "ordered_question_ids": list(ordered_ids),
            "ordered_question_ids_sha256": _sha256_bytes(_canonical_bytes(ordered_ids)),
            "manifest_path": str(manifest_path),
        },
        "datasets": {
            "derivative": {
                "path": str(derivative_path),
                "sha256": derivative_hash,
                "record_count": len(derivative_records),
                "selected_count": len(selected_derivative),
            },
            "original": {
                "path": str(original_path),
                "sha256": original_hash,
                "record_count": len(original_records),
                "selected_count": len(selected_original),
            },
        },
        "workload": {
            "case_count": len(cases),
            "session_count": sum(case["session_count"] for case in cases),
            "turn_count": sum(case["turn_count"] for case in cases),
            "characters": sum(case["characters"] for case in cases),
            "turns_by_role": dict(sorted(total_roles.items())),
            "characters_by_role": dict(sorted(total_chars_by_role.items())),
            "case_lengths": cases,
        },
        "repetition": _repetition_report(cases),
        "chronological_order": chronology,
        "history_digest_comparison": {
            "selected_case_count": len(cases),
            "derivative_history_digests": derivative_history_digests,
            "original_history_digests": original_history_digests,
            "matching_digest_count": sum(
                left == right
                for left, right in zip(derivative_history_digests, original_history_digests)
            ),
        },
        "agent_workload_contract": {
            "memory_write_frequency": "once_per_session_optional_agent_selected_public_memory",
            "recall_surface": "public_recall",
            "per_turn_classifier_or_detector_required": False,
            "paid_apis_called": False,
            "database_writes": False,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivative", type=Path, default=DEFAULT_DERIVATIVE_PATH)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, help="Write the gold-blind report JSON here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI and return a process exit code."""
    args = _parse_args(argv)
    try:
        report = audit_workload(args.derivative, args.original, args.manifest)
    except (OSError, json.JSONDecodeError, WorkloadAuditError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": "ok", "case_count": report["workload"]["case_count"], "session_count": report["workload"]["session_count"], "turn_count": report["workload"]["turn_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#    No additional dependencies; use the repository environment.
#
# 2. Basic usage:
#    uv run python benchmarks/longmemeval/agent_workload.py
#
# 3. With options:
#    uv run python benchmarks/longmemeval/agent_workload.py \
#      --output artifacts/belief-recall-0bk79s/agent-workload.json
#
# 4. Expected output:
#    A one-line count receipt on stdout and an optional gold-blind JSON report.
#
# ═══════════════════════════════════════════════════════════════
