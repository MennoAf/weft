"""RC-FL-04 tests for the public/private artifact boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_rc_artifact_boundary import (
    CATEGORY_NAMES,
    classify_path,
    render_receipt,
    scan_repository,
)


ROOT = Path(__file__).resolve().parents[1]


def _touch(root: Path, relative: str, content: str = "fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_required_boundary_categories_classify_without_deleting(tmp_path: Path) -> None:
    files = {
        "benchmarks/longmemeval/runs/run/results.json": "{}\n",
        "nested/.worktrees/feature/weft.py": "print('fixture')\n",
        "evidence/raw/receipt.json": '{"status":"historical"}\n',
        "private/operational/deploy-notes.md": "operator-only note\n",
        "docs/public-guide.md": "public documentation\n",
    }
    for relative, content in files.items():
        _touch(tmp_path, relative, content)

    report = scan_repository(tmp_path)

    assert set(report["category_counts"]) >= set(CATEGORY_NAMES)
    assert report["category_counts"]["generated-benchmark"] == 1
    assert report["category_counts"]["nested-worktree"] == 1
    assert report["category_counts"]["raw-receipt"] == 1
    assert report["category_counts"]["private-operational-doc"] == 1
    assert report["credential-shaped"]["match_count"] == 0
    assert (tmp_path / "nested/.worktrees/feature/weft.py").exists()


def test_credential_shaped_matches_are_flagged_and_values_are_never_reported(
    tmp_path: Path,
) -> None:
    secret = "synthetic-test-value-do-not-publish"
    _touch(tmp_path, "docs/example.txt", f"WEFT_API_KEY={secret}\n")
    _touch(tmp_path, "private/operational/config.env", f"TOKEN='{secret}'\n")

    report = scan_repository(tmp_path)
    encoded = json.dumps(report, sort_keys=True)

    assert report["credential-shaped"]["match_count"] == 2
    assert report["credential-shaped"]["disposition"] == "flagged-and-excluded"
    assert secret not in encoded
    assert "WEFT_API_KEY" not in encoded
    assert "TOKEN" not in encoded


def test_path_classifier_preserves_historical_overlay_and_copied_identity() -> None:
    assert classify_path("docs/release-candidate-plan.md") == "historical"
    assert classify_path("artifacts/linux-amd64-acceptance-20260912/README.md") == "copied"
    assert classify_path(".ci-rc-worktree/report.md") == "overlay"
    assert classify_path(".rc-candidate-worktree/README.md") == "overlay"
    assert classify_path("tls-release/README.md") == "overlay"
    assert classify_path("weft/cli.py") == "current-source"


def test_receipt_is_redacted_and_states_scan_limits(tmp_path: Path) -> None:
    _touch(tmp_path, "benchmarks/longmemeval/runs/run.json", "{}\n")
    report = scan_repository(tmp_path)
    receipt = render_receipt(report, receipt_path="evidence/rc-finish-line/artifact-boundary.md")

    assert "generated benchmark" in receipt.lower()
    assert "nested worktree" in receipt.lower()
    assert "raw receipt" in receipt.lower()
    assert "private operational" in receipt.lower()
    assert "current-source" in receipt
    assert "historical" in receipt
    assert "overlay" in receipt
    assert "copied" in receipt
    assert "limitations" in receipt.lower()
    assert "history rewrite" in receipt.lower()
    assert "secret clearance" in receipt.lower()
    assert str(tmp_path) not in receipt
    assert "file://" not in receipt
    assert "https://" not in receipt


def test_existing_ignore_rules_cover_generated_benchmark_and_nested_worktree_paths() -> None:
    for relative in (
        "benchmarks/longmemeval/runs/synthetic.json",
        "benchmarks/longmemeval/data/example_cleaned.json",
        "benchmarks/longmemeval/snapshots/v1/example.csv.gz",
        "feature/.worktrees/nested/file.txt",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--", relative],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, relative


def test_cli_writes_only_requested_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "boundary.md"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_rc_artifact_boundary.py"),
            "--root",
            str(tmp_path),
            "--receipt",
            str(receipt),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert receipt.exists()
    assert "credential" in receipt.read_text(encoding="utf-8").lower()
    assert not list(tmp_path.glob("**/*secret*"))
