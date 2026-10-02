"""RC-FL-17 provider-free benchmark summary acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.longmemeval.judge import summarize_pipeline


def _write_reference(path: Path, rows: list[object]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _artifacts(
    tmp_path: Path,
    *,
    references: list[object],
    hypotheses: list[object],
    results: list[object],
) -> tuple[Path, Path, Path]:
    ref = tmp_path / "reference.json"
    hyp = tmp_path / "hypotheses.jsonl"
    result = tmp_path / "results.jsonl"
    _write_reference(ref, references)
    _write_jsonl(hyp, hypotheses)
    _write_jsonl(result, results)
    return ref, hyp, result


def _reference(qid: str, question_type: str = "single-session-user") -> dict:
    return {"question_id": qid, "question_type": question_type}


def _hypothesis(qid: str) -> dict:
    return {"question_id": qid, "hypothesis": f"answer-{qid}"}


def _result(qid: str, label: object = True) -> dict:
    return {"question_id": qid, "autoeval_label": {"label": label}}


def test_missing_hypotheses_and_results_stay_in_reference_denominator(tmp_path):
    ref, hyp, result = _artifacts(
        tmp_path,
        references=[
            _reference("q1", "known"),
            _reference("q2", "known"),
            _reference("q3", "missing-hypothesis"),
            _reference("q4", "missing-result"),
        ],
        hypotheses=[_hypothesis("q1"), _hypothesis("q2"), _hypothesis("q4")],
        results=[_result("q1", True)],
    )

    metrics = summarize_pipeline(ref_path=ref, hyp_path=hyp, result_path=result)

    assert metrics["n_total"] == 4
    assert metrics["expected_questions"] == 4
    assert metrics["n_correct_total"] == 1
    assert metrics["overall_accuracy"] == pytest.approx(0.25)
    assert metrics["hypotheses_produced"] == 3
    assert metrics["judge_results_produced"] == 1
    assert metrics["missing_hypotheses"] == 1
    assert metrics["hypotheses_without_judge_results"] == 2
    assert metrics["complete"] is False

    by_type = {row["question_type"]: row for row in metrics["by_type"]}
    assert by_type["missing-hypothesis"]["n"] == 1
    assert by_type["missing-hypothesis"]["missing_hypotheses"] == 1
    assert by_type["missing-result"]["hypotheses_without_judge_results"] == 1


@pytest.mark.parametrize(
    ("artifact", "rows", "message"),
    [
        (
            "reference",
            [_reference("q1"), _reference("q1")],
            "duplicate reference question_id: q1",
        ),
        (
            "hypothesis",
            [_hypothesis("q1"), _hypothesis("q1")],
            "duplicate hypothesis question_id: q1",
        ),
        (
            "result",
            [_result("q1"), _result("q1")],
            "duplicate result question_id: q1",
        ),
    ],
)
def test_duplicate_reference_hypothesis_and_result_ids_fail_closed(
    tmp_path: Path, artifact: str, rows: list[object], message: str
):
    references = [_reference("q1")]
    hypotheses = [_hypothesis("q1")]
    results = [_result("q1")]
    if artifact == "reference":
        references = rows
    elif artifact == "hypothesis":
        hypotheses = rows
    else:
        results = rows
    paths = _artifacts(
        tmp_path,
        references=references,
        hypotheses=hypotheses,
        results=results,
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        summarize_pipeline(ref_path=paths[0], hyp_path=paths[1], result_path=paths[2])


@pytest.mark.parametrize(
    ("artifact", "rows", "message"),
    [
        (
            "reference",
            [{"question_type": "missing-id"}],
            "reference row 1 has missing/invalid question_id",
        ),
        (
            "reference",
            [{"question_id": 1, "question_type": "malformed-id"}],
            "reference row 1 has missing/invalid question_id",
        ),
        (
            "hypothesis",
            [{"hypothesis": "missing-id"}],
            "hypothesis row 1 has missing/invalid question_id",
        ),
        (
            "hypothesis",
            [{"question_id": 1, "hypothesis": "malformed-id"}],
            "hypothesis row 1 has missing/invalid question_id",
        ),
        (
            "result",
            [{"autoeval_label": {"label": True}}],
            "result row 1 has missing/invalid question_id",
        ),
        (
            "result",
            [{"question_id": 1, "autoeval_label": {"label": True}}],
            "result row 1 has missing/invalid question_id",
        ),
    ],
)
def test_malformed_reference_hypothesis_and_result_ids_fail_closed(
    tmp_path: Path, artifact: str, rows: list[object], message: str
):
    references = [_reference("q1")]
    hypotheses = [_hypothesis("q1")]
    results = [_result("q1")]
    if artifact == "reference":
        references = rows
    elif artifact == "hypothesis":
        hypotheses = rows
    else:
        results = rows
    paths = _artifacts(
        tmp_path,
        references=references,
        hypotheses=hypotheses,
        results=results,
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        summarize_pipeline(ref_path=paths[0], hyp_path=paths[1], result_path=paths[2])


@pytest.mark.parametrize(
    ("artifact", "rows", "message"),
    [
        (
            "hypothesis",
            [_hypothesis("unknown")],
            "unknown hypothesis question_id: unknown",
        ),
        (
            "result",
            [_result("unknown")],
            "unknown result question_id: unknown",
        ),
    ],
)
def test_unknown_hypothesis_and_result_ids_fail_closed(
    tmp_path: Path, artifact: str, rows: list[object], message: str
):
    references = [_reference("q1")]
    hypotheses = [_hypothesis("q1")]
    results = [_result("q1")]
    if artifact == "hypothesis":
        hypotheses = rows
    else:
        results = rows
    paths = _artifacts(
        tmp_path,
        references=references,
        hypotheses=hypotheses,
        results=results,
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        summarize_pipeline(ref_path=paths[0], hyp_path=paths[1], result_path=paths[2])


@pytest.mark.parametrize(
    "label",
    [None, {"model": "judge"}, {"label": "true"}, {"label": 1}],
)
def test_malformed_judge_labels_fail_closed(tmp_path: Path, label: object):
    result_row = {"question_id": "q1"}
    if label is not None:
        result_row["autoeval_label"] = label
    ref, hyp, result = _artifacts(
        tmp_path,
        references=[_reference("q1")],
        hypotheses=[_hypothesis("q1")],
        results=[result_row],
    )

    with pytest.raises(
        ValueError,
        match=r"^missing/malformed autoeval label for question_id: q1$",
    ):
        summarize_pipeline(ref_path=ref, hyp_path=hyp, result_path=result)
