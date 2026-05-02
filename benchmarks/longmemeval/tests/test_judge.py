"""Unit tests for the LongMemEval judge wrapper.

These tests do NOT call OpenAI. They exercise the wrapper's path resolution,
command construction, and result-summarization logic. The actual subprocess
invocation is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.longmemeval.judge import (
    _build_command,
    _default_ref_for,
    _resolve_longmemeval_root,
    _result_path_for,
    run_judge,
    summarize_results,
)


# ----------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------


def _make_fake_lme(root: Path) -> Path:
    """Build a minimal LongMemEval-shaped directory for path tests."""
    (root / "src" / "evaluation").mkdir(parents=True)
    (root / "src" / "evaluation" / "evaluate_qa.py").write_text(
        "# stub evaluator", encoding="utf-8",
    )
    (root / "data").mkdir()
    return root


def test_resolve_via_env(tmp_path, monkeypatch):
    fake = _make_fake_lme(tmp_path / "lme")
    monkeypatch.setenv("LONGMEMEVAL_PATH", str(fake))
    assert _resolve_longmemeval_root() == fake.resolve()


def test_resolve_env_missing_evaluator_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGMEMEVAL_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="evaluate_qa.py"):
        _resolve_longmemeval_root()


def test_default_ref_for_oracle(tmp_path):
    fake = _make_fake_lme(tmp_path / "lme")
    ref = fake / "data" / "longmemeval_oracle.json"
    ref.write_text("[]", encoding="utf-8")
    hyp = tmp_path / "longmemeval_oracle_extracted_20260430T204018Z.jsonl"
    hyp.write_text("", encoding="utf-8")
    assert _default_ref_for(hyp, fake) == ref


def test_default_ref_filename_too_short_raises(tmp_path):
    fake = _make_fake_lme(tmp_path / "lme")
    hyp = tmp_path / "weird.jsonl"
    with pytest.raises(ValueError, match="convention"):
        _default_ref_for(hyp, fake)


def test_default_ref_missing_split_raises(tmp_path):
    fake = _make_fake_lme(tmp_path / "lme")
    hyp = tmp_path / "longmemeval_madeup_extracted_20260430T204018Z.jsonl"
    hyp.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="reference dataset"):
        _default_ref_for(hyp, fake)


# ----------------------------------------------------------------------
# Command construction
# ----------------------------------------------------------------------


def test_build_command_uses_uv_with_deps(tmp_path):
    fake = _make_fake_lme(tmp_path / "lme")
    hyp = tmp_path / "h.jsonl"
    ref = tmp_path / "r.json"
    hyp.write_text("", encoding="utf-8")
    ref.write_text("[]", encoding="utf-8")
    cmd = _build_command(
        longmemeval_root=fake, hyp_path=hyp, ref_path=ref, model="gpt-4o",
    )
    # The wrapper must layer openai/backoff into an ephemeral env so the
    # upstream evaluator can import them without us pinning them in Weft.
    assert cmd[0:2] == ["uv", "run"]
    flat = " ".join(cmd)
    assert "--with openai" in flat
    assert "--with backoff" in flat
    assert "evaluate_qa.py" in flat
    assert "gpt-4o" in cmd


def test_result_path_naming():
    p = Path("/tmp/run.jsonl")
    assert _result_path_for(p, "gpt-4o").name == "run.jsonl.eval-results-gpt-4o"


# ----------------------------------------------------------------------
# Result summarization
# ----------------------------------------------------------------------


def _write_result(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_ref(path: Path, instances: list[dict]) -> None:
    path.write_text(json.dumps(instances), encoding="utf-8")


def test_summarize_per_type_accuracy(tmp_path):
    ref = tmp_path / "ref.json"
    _write_ref(
        ref,
        [
            {"question_id": "q1", "question_type": "multi-session"},
            {"question_id": "q2", "question_type": "multi-session"},
            {"question_id": "q3", "question_type": "temporal-reasoning"},
        ],
    )
    result = tmp_path / "out.jsonl.eval-results-gpt-4o"
    _write_result(
        result,
        [
            {"question_id": "q1", "autoeval_label": {"label": True}},
            {"question_id": "q2", "autoeval_label": {"label": False}},
            {"question_id": "q3", "autoeval_label": {"label": True}},
        ],
    )
    metrics = summarize_results(result_path=result, ref_path=ref)
    assert metrics["overall_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    by = {row["question_type"]: row for row in metrics["by_type"]}
    assert by["multi-session"]["accuracy"] == pytest.approx(0.5)
    assert by["temporal-reasoning"]["accuracy"] == pytest.approx(1.0)
    assert by["multi-session"]["n"] == 2


def test_summarize_task_averaged_treats_types_equally(tmp_path):
    """Task-averaged accuracy weights each question type the same — so a
    type with 3 questions doesn't drown out a type with 30. This is the
    headline metric LongMemEval reports."""
    ref = tmp_path / "ref.json"
    _write_ref(
        ref,
        # multi-session: 1/1 = 100%, single-session-user: 0/3 = 0%
        [
            {"question_id": "ms1", "question_type": "multi-session"},
            {"question_id": "ssu1", "question_type": "single-session-user"},
            {"question_id": "ssu2", "question_type": "single-session-user"},
            {"question_id": "ssu3", "question_type": "single-session-user"},
        ],
    )
    result = tmp_path / "out.jsonl.eval-results-gpt-4o"
    _write_result(
        result,
        [
            {"question_id": "ms1", "autoeval_label": {"label": True}},
            {"question_id": "ssu1", "autoeval_label": {"label": False}},
            {"question_id": "ssu2", "autoeval_label": {"label": False}},
            {"question_id": "ssu3", "autoeval_label": {"label": False}},
        ],
    )
    metrics = summarize_results(result_path=result, ref_path=ref)
    # overall is 1/4 = 0.25, but task-avg is (1.0 + 0.0)/2 = 0.5
    assert metrics["overall_accuracy"] == pytest.approx(0.25)
    assert metrics["task_averaged_accuracy"] == pytest.approx(0.5)


def test_summarize_skips_blank_lines(tmp_path):
    ref = tmp_path / "ref.json"
    _write_ref(ref, [{"question_id": "q1", "question_type": "multi-session"}])
    result = tmp_path / "out.jsonl.eval-results-gpt-4o"
    result.write_text(
        '{"question_id": "q1", "autoeval_label": {"label": true}}\n\n\n',
        encoding="utf-8",
    )
    metrics = summarize_results(result_path=result, ref_path=ref)
    assert metrics["n_total"] == 1


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def test_run_judge_skip_if_exists_does_not_subprocess(tmp_path):
    """When labelled output already exists and skip_if_exists=True, the
    wrapper must NOT call OpenAI — that's the whole point of the flag."""
    fake = _make_fake_lme(tmp_path / "lme")
    ref = fake / "data" / "longmemeval_oracle.json"
    _write_ref(ref, [{"question_id": "q1", "question_type": "multi-session"}])
    hyp = tmp_path / "longmemeval_oracle_raw_20260430T194312Z.jsonl"
    hyp.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "blue"}) + "\n",
        encoding="utf-8",
    )
    result = _result_path_for(hyp, "gpt-4o")
    _write_result(
        result, [{"question_id": "q1", "autoeval_label": {"label": True}}],
    )

    with patch("benchmarks.longmemeval.judge.subprocess.run") as fake_run:
        metrics = run_judge(
            hyp_path=hyp,
            ref_path=ref,
            model="gpt-4o",
            longmemeval_root=fake,
            skip_if_exists=True,
        )
    fake_run.assert_not_called()
    assert metrics["n_total"] == 1
    metrics_path = hyp.with_suffix(hyp.suffix + ".metrics.json")
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text())["overall_accuracy"] == 1.0


def test_run_judge_unsupported_model_raises(tmp_path):
    fake = _make_fake_lme(tmp_path / "lme")
    hyp = tmp_path / "longmemeval_oracle_raw_x.jsonl"
    hyp.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported judge model"):
        run_judge(
            hyp_path=hyp, ref_path=None, model="claude-opus",
            longmemeval_root=fake,
        )


def test_run_judge_missing_openai_key_when_running(tmp_path, monkeypatch):
    """Without OPENAI_API_KEY, the wrapper must refuse to invoke before
    the subprocess starts — failing fast saves us from a confusing error
    deep inside the upstream evaluator."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = _make_fake_lme(tmp_path / "lme")
    ref = fake / "data" / "longmemeval_oracle.json"
    _write_ref(ref, [])
    hyp = tmp_path / "longmemeval_oracle_raw_20260430T194312Z.jsonl"
    hyp.write_text("", encoding="utf-8")

    # No pre-existing result file → subprocess path is selected.
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_judge(
            hyp_path=hyp, ref_path=ref, model="gpt-4o",
            longmemeval_root=fake, skip_if_exists=True,
        )
