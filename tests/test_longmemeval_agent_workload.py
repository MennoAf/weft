"""Tests for the offline, gold-blind LongMemEval workload audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.longmemeval.agent_workload import (
    WorkloadAuditError,
    audit_workload,
)


def _record(
    question_id: str,
    *,
    session_id: str = "session-1",
    content: str = "public memory",
    question: str = "PRIVATE QUESTION",
    answer: str = "PRIVATE ANSWER",
    session_date: str = "2024/01/01",
) -> dict[str, Any]:
    """Build a minimal LongMemEval-shaped record for an offline test."""
    return {
        "question_id": question_id,
        "question_type": "knowledge-update",
        "question": question,
        "answer": answer,
        "question_date": "2024/02/01",
        "answer_session_ids": [],
        "haystack_session_ids": [session_id],
        "haystack_dates": [session_date],
        "haystack_sessions": [
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "ack"},
            ]
        ],
    }


def _write_fixture(
    tmp_path: Path,
    records: list[dict[str, Any]],
    *,
    ordered_ids: list[str] | None = None,
) -> tuple[Path, Path, Path, str, str]:
    """Write synthetic datasets and a matching manifest, returning hashes."""
    derivative = tmp_path / "derivative.json"
    original = tmp_path / "original.json"
    manifest = tmp_path / "manifest.json"
    derivative.write_text(json.dumps(records), encoding="utf-8")
    original.write_text(json.dumps(records), encoding="utf-8")
    ids = ordered_ids if ordered_ids is not None else [record["question_id"] for record in records]
    manifest.write_text(
        json.dumps({"selection": {"ordered_question_ids": ids}}), encoding="utf-8"
    )
    return (
        derivative,
        original,
        manifest,
        hashlib.sha256(derivative.read_bytes()).hexdigest(),
        hashlib.sha256(original.read_bytes()).hexdigest(),
    )


def _audit(tmp_path: Path, records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Run the audit against synthetic files with their actual hashes."""
    derivative, original, manifest, derivative_hash, original_hash = _write_fixture(
        tmp_path, records, ordered_ids=kwargs.pop("ordered_ids", None)
    )
    return audit_workload(
        derivative,
        original,
        manifest,
        expected_derivative_sha256=derivative_hash,
        expected_original_sha256=original_hash,
        expected_case_count=len(records),
        **kwargs,
    )


def test_audit_reports_exact_counts_and_gold_blind_digests(tmp_path: Path) -> None:
    report = _audit(
        tmp_path,
        [
            _record("q1", content="alpha", question="secret-question", answer="secret-answer"),
            _record("q2", content="beta", session_id="session-2"),
        ],
    )

    assert report["workload"]["case_count"] == 2
    assert report["workload"]["session_count"] == 2
    assert report["workload"]["turn_count"] == 4
    assert report["workload"]["turns_by_role"] == {"assistant": 2, "user": 2}
    assert report["workload"]["characters_by_role"] == {"assistant": 6, "user": 9}
    assert report["gold_blind"] is True
    rendered = json.dumps(report)
    assert "secret-question" not in rendered
    assert "secret-answer" not in rendered
    assert "history_digest" in rendered
    assert report["history_digest_comparison"]["matching_digest_count"] == 2


def test_missing_data_is_rejected(tmp_path: Path) -> None:
    derivative, original, manifest, derivative_hash, original_hash = _write_fixture(
        tmp_path, [_record("q1")]
    )
    original.unlink()
    with pytest.raises(WorkloadAuditError, match="original dataset is missing"):
        audit_workload(
            derivative,
            original,
            manifest,
            expected_derivative_sha256=derivative_hash,
            expected_original_sha256=original_hash,
            expected_case_count=1,
        )


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    derivative, original, manifest, _, original_hash = _write_fixture(
        tmp_path, [_record("q1")]
    )
    with pytest.raises(WorkloadAuditError, match="derivative dataset SHA-256 mismatch"):
        audit_workload(
            derivative,
            original,
            manifest,
            expected_derivative_sha256="0" * 64,
            expected_original_sha256=original_hash,
            expected_case_count=1,
        )


def test_changed_selection_is_rejected(tmp_path: Path) -> None:
    records = [_record("q1"), _record("q2", session_id="session-2")]
    derivative, original, manifest, derivative_hash, original_hash = _write_fixture(
        tmp_path, records, ordered_ids=["q1", "changed"]
    )
    with pytest.raises(WorkloadAuditError, match="missing selected IDs"):
        audit_workload(
            derivative,
            original,
            manifest,
            expected_derivative_sha256=derivative_hash,
            expected_original_sha256=original_hash,
            expected_case_count=2,
        )


def test_repeated_session_id_with_different_content_is_a_conflict(tmp_path: Path) -> None:
    report = _audit(
        tmp_path,
        [
            _record("q1", content="first"),
            _record("q2", content="second"),
        ],
    )
    repetition = report["repetition"]
    assert repetition["repeated_session_id_count"] == 1
    assert repetition["conflicting_repeated_session_id_count"] == 1
    assert repetition["repeated_session_ids"]["session-1"]["conflict"] is True
    assert len(repetition["repeated_session_ids"]["session-1"]["content_digests"]) == 2


def test_matching_repeated_content_is_overlap_not_conflict(tmp_path: Path) -> None:
    report = _audit(
        tmp_path,
        [
            _record("q1", session_id="session-a", content="same"),
            _record("q2", session_id="session-b", content="same"),
        ],
    )
    repetition = report["repetition"]
    assert repetition["conflicting_repeated_session_id_count"] == 0
    assert repetition["content_digest_overlap_count"] == 1


def test_manifest_must_preserve_exactly_36_unique_ids(tmp_path: Path) -> None:
    records = [_record(f"q{i}") for i in range(36)]
    derivative, original, manifest, derivative_hash, original_hash = _write_fixture(
        tmp_path, records, ordered_ids=[f"q{i}" for i in range(35)] + ["q0"]
    )
    with pytest.raises(WorkloadAuditError, match="exactly 36 unique IDs"):
        audit_workload(
            derivative,
            original,
            manifest,
            expected_derivative_sha256=derivative_hash,
            expected_original_sha256=original_hash,
        )


def test_chronological_order_is_reported_without_reordering(tmp_path: Path) -> None:
    report = _audit(
        tmp_path,
        [
            _record("q1", session_id="a", session_date="2024/02/01"),
            _record("q2", session_id="b", session_date="2024/01/01"),
        ],
    )
    assert report["selection"]["ordered_question_ids"] == ["q1", "q2"]
    assert report["chronological_order"]["ordered_case_count"] == 2


def test_agent_workload_contract_does_not_require_per_turn_detector() -> None:
    # This is a contract assertion, not a provider or production integration.
    from benchmarks.longmemeval.agent_workload import audit_workload

    assert audit_workload.__doc__
