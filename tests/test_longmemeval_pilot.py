from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.longmemeval import pilot
from benchmarks.longmemeval.dataset import Instance, Session, Turn


def _dataset(path: Path) -> None:
    rows = []
    for index, question_type in enumerate(pilot.QUESTION_TYPES):
        for offset in range(6):
            question_id = f"{question_type}-{index}-{offset}"
            rows.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "question": f"Question {question_id}",
                    "answer": "answer",
                    "question_date": "2024-01-01",
                    "haystack_session_ids": [f"session-{question_id}"],
                    "haystack_dates": ["2023/01/01"],
                    "haystack_sessions": [[{"role": "user", "content": "answer"}]],
                    "answer_session_ids": [f"session-{question_id}"],
                }
            )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _row(question_id: str, question_type: str, session_ids: list[str], dates: list[str], sessions: list[list[dict[str, str]]]) -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"Question {question_id}",
        "answer": "answer",
        "question_date": "2024-01-01",
        "haystack_session_ids": session_ids,
        "haystack_dates": dates,
        "haystack_sessions": sessions,
        "answer_session_ids": [session_ids[0]],
    }


def test_normalize_dataset_excludes_conflicts_and_collapses_identicals(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    derived = tmp_path / "derived.json"
    report_path = tmp_path / "normalization.json"
    identical = [{"role": "user", "content": "same"}]
    conflicting_a = [{"role": "user", "content": "first"}]
    conflicting_b = [{"role": "user", "content": "second"}]
    rows = [
        _row("keep", pilot.QUESTION_TYPES[0], ["same", "same"], ["2023/01/01", "2023/01/01"], [identical, identical]),
        _row("drop", pilot.QUESTION_TYPES[1], ["conflict", "conflict"], ["2023/01/01", "2023/01/02"], [conflicting_a, conflicting_b]),
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")

    report = pilot.normalize_dataset(source, derived, report_path)

    assert report["source"]["record_count"] == 2
    assert report["derived"]["record_count"] == 1
    assert report["exclusion_count"] == 1
    assert report["collapsed_group_count"] == 1
    normalized = json.loads(derived.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in normalized] == ["keep"]
    assert normalized[0]["haystack_session_ids"] == ["same"]
    assert report["exclusions"][0]["question_id"] == "drop"
    group = report["exclusions"][0]["duplicate_groups"][0]
    assert group["session_id"] == "conflict"
    assert len({item["payload_sha256"] for item in group["occurrences"]}) == 2


def test_normalize_dataset_rejects_source_alias_and_validates_retained_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    derived = tmp_path / "derived.json"
    report_path = tmp_path / "normalization.json"
    row = _row(
        "keep",
        pilot.QUESTION_TYPES[0],
        ["same", "same"],
        ["2023/01/01", "2023/01/01"],
        [[{"role": "user", "content": "same"}], [{"role": "user", "content": "same"}]],
    )
    source.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable source"):
        pilot.normalize_dataset(source, source, report_path)
    pilot.normalize_dataset(source, derived, report_path)
    loaded_rows = json.loads(derived.read_text(encoding="utf-8"))
    assert len(loaded_rows) == 1
    assert Instance.from_dict(loaded_rows[0]).question_id == "keep"


def test_normalize_dataset_preserves_source_and_typed_loader_rejects_bad_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    derived = tmp_path / "derived.json"
    report_path = tmp_path / "normalization.json"
    rows = [_row("bad", pilot.QUESTION_TYPES[0], ["one"], ["2023/01/01"], [[{"role": "user", "content": "x"}]])]
    source.write_text(json.dumps(rows), encoding="utf-8")
    source_before = source.read_bytes()
    pilot.normalize_dataset(source, derived, report_path)
    assert source.read_bytes() == source_before


def test_select_questions_is_equal_and_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    _dataset(dataset)
    first = pilot.select_questions(dataset, per_type=2, seed=7)
    second = pilot.select_questions(dataset, per_type=2, seed=7)
    assert [item.question_id for item in first] == [item.question_id for item in second]
    counts = {question_type: 0 for question_type in pilot.QUESTION_TYPES}
    for item in first:
        counts[item.question_type] += 1
    assert set(counts.values()) == {2}


def test_prepare_manifest_requires_normalized_six_per_type_packet(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    dataset = tmp_path / "dataset.json"
    report_path = tmp_path / "normalization.json"
    output = tmp_path / "pilot-manifest.json"
    _dataset(source)
    report = pilot.normalize_dataset(source, dataset, report_path)
    manifest = pilot.prepare_manifest(
        Path.cwd(),
        dataset,
        output,
        normalization_report_path=report_path,
        normalization_source_path=source,
        seed=0,
    )
    assert manifest["status"] == "PREPARED_NOT_AUTHORIZED"
    assert manifest["arms"] == ["turns", "belief", "auto"]
    assert manifest["selection"]["per_type"] == 6
    assert manifest["cost"]["reader_calls"] == 108
    assert manifest["normalization"]["source_sha256"] == report["source"]["sha256"]
    loaded = pilot.load_manifest(
        Path.cwd(), dataset, output, normalization_source_path=source
    )
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]


def test_load_manifest_rejects_unbound_or_unbalanced_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    dataset = tmp_path / "dataset.json"
    report_path = tmp_path / "normalization.json"
    output = tmp_path / "pilot-manifest.json"
    _dataset(source)
    pilot.normalize_dataset(source, dataset, report_path)
    manifest = pilot.prepare_manifest(
        Path.cwd(),
        dataset,
        output,
        normalization_report_path=report_path,
        normalization_source_path=source,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["normalization"] = None
    value["manifest_sha256"] = pilot._manifest_hash({k: v for k, v in value.items() if k != "manifest_sha256"})
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization binding"):
        pilot.load_manifest(Path.cwd(), dataset, output, normalization_source_path=source)
    value["normalization"] = manifest["normalization"]
    value["manifest_sha256"] = pilot._manifest_hash({k: v for k, v in value.items() if k != "manifest_sha256"})
    output.write_text(json.dumps(value), encoding="utf-8")
    report_path.write_text(report_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="report checksum"):
        pilot.load_manifest(Path.cwd(), dataset, output, normalization_source_path=source)
    assert manifest["selection"]["per_type"] == 6


def test_cost_guard_is_below_fifty_for_six_question_smoke() -> None:
    estimate = pilot.estimate_cost(6)
    assert estimate.total_cost_usd < pilot.OPERATIONAL_STOP_USD
    assert estimate.total_cost_usd < pilot.AUTHORIZATION_CEILING_USD


def test_cost_guard_rejects_invalid_population() -> None:
    with pytest.raises(ValueError):
        pilot.estimate_cost(0)


def test_reader_model_and_judge_are_explicit() -> None:
    assert pilot.READER_MODEL == "gpt-5.6-luna"
    assert pilot.JUDGE_MODEL == "gpt-4o"
    assert pilot.ARMS == ("turns", "belief", "auto")
