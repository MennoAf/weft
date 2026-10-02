"""Offline CLI boundary regressions for the faithful pilot."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval import faithful_s36 as runner


@pytest.mark.parametrize("command", ["prepare", "calibrate", "resume"])
def test_cli_parses_without_duplicate_options(command):
    """Every execution phase binds the official judge source explicitly."""
    argv = [command, "--dataset", "dataset.json", "--manifest", "manifest.json",
            "--judge-root", "/tmp/official"]
    if command != "resume":
        argv += ["--owner-id", "benchmark-owner"]
    args = runner._parse_args(argv)
    assert args.judge_root == Path("/tmp/official")
    assert args.command == command


def test_resume_cli_still_requires_judge_root():
    with pytest.raises(SystemExit):
        runner._parse_args([
            "resume", "--dataset", "dataset.json", "--manifest", "manifest.json",
        ])


def test_fresh_profile_cli_requires_explicit_model_and_cap(tmp_path):
    args = runner._parse_args([
        "prepare", "--dataset", "dataset.json", "--manifest", "manifest.json",
        "--judge-root", "/tmp/official", "--owner-id", "benchmark-owner",
        "--profile", runner.FRESH_RUN_PROFILE,
    ])
    with pytest.raises(runner.ExecutionGateError, match="requires --writer-model"):
        asyncio.run(runner._main_async(args))


def test_fresh_profile_cli_requires_distinct_explicit_artifact_binding():
    args = runner._parse_args([
        "prepare", "--dataset", "dataset.json", "--manifest", "manifest.json",
        "--judge-root", "/tmp/official", "--owner-id", "benchmark-owner",
        "--profile", runner.FRESH_RUN_PROFILE,
        "--writer-model", "gpt-6-luna", "--max-budget-usd", "50",
    ])
    assert args.profile == runner.FRESH_RUN_PROFILE
    assert args.writer_model == "gpt-6-luna"
    assert args.max_budget_usd == 50.0
    assert runner.GPT6_SELECTED35_ARTIFACT_NAMESPACE != runner.ARTIFACT_NAMESPACE


def test_prepare_cli_forwards_judge_binding(monkeypatch, tmp_path):
    """Preparation must not drop the parsed official-source path."""
    captured = {}

    def prepare(dataset, manifest, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ordered_question_ids=tuple(range(36)))

    monkeypatch.setattr(runner, "prepare_run", prepare)
    args = runner._parse_args([
        "prepare", "--dataset", "dataset.json", "--manifest", "manifest.json",
        "--judge-root", "/tmp/official", "--owner-id", "benchmark-owner",
        "--artifact-root", str(tmp_path),
    ])
    assert asyncio.run(runner._main_async(args)) == 0
    assert captured["judge_root"] == Path("/tmp/official")
    assert captured["owner_id"] == "benchmark-owner"
