"""Focused contract tests for the release-candidate provenance receipt."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.capture_rc_provenance import _status_records, collect_provenance


ROOT = Path(__file__).resolve().parents[1]


def test_receipt_captures_exact_candidate_boundary() -> None:
    receipt = collect_provenance(ROOT)
    source_control = receipt["source_control"]

    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert source_control["head_sha"] == expected_head
    assert re.fullmatch(r"[0-9a-f]{40}", source_control["head_sha"])
    assert source_control["branch"] == expected_branch
    # This checkout is intentionally dirty before this leaf; a clean claim would
    # hide the release boundary and invalidate the receipt.
    assert source_control["dirty"] is True
    assert source_control["untracked"] is True
    assert receipt["release_boundary"]["candidate_status"] == "dirty-worktree"


def test_receipt_hashes_dirty_and_untracked_content_without_reading_it(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    untracked = tmp_path / "untracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    untracked.write_text("untracked\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "RC Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    tracked.write_text("dirty\n", encoding="utf-8")

    dirty, untracked_state, hashes = _status_records(tmp_path, None)

    assert dirty is True
    assert untracked_state is True
    assert hashes["tracked.txt"] == {
        "status": " M",
        "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
    }
    assert hashes["untracked.txt"] == {
        "status": "untracked",
        "sha256": hashlib.sha256(untracked.read_bytes()).hexdigest(),
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        for entry in hashes.values()
        if entry["sha256"] is not None
    )


def test_receipt_has_package_lock_and_evidence_class_identity() -> None:
    receipt = collect_provenance(ROOT)

    assert receipt["package"]["name"] == "weft-memory"
    assert receipt["package"]["version"] == "1.0.0rc1"
    lock = receipt["dependency_lock"]
    assert lock["path"] == "uv.lock"
    assert re.fullmatch(r"[0-9a-f]{64}", lock["sha256"])
    assert lock["format"]["version"] == 1
    assert lock["format"]["revision"] == 2
    assert set(receipt["evidence_classes"]) == {
        "current-source",
        "historical",
        "overlay",
        "copied",
    }


def test_receipt_is_json_safe_and_does_not_emit_urls_or_output_hash() -> None:
    receipt = collect_provenance(ROOT, output_path=ROOT / "evidence/rc-finish-line/provenance.json")
    encoded = json.dumps(receipt, sort_keys=True)

    assert "://" not in encoded
    assert "Authorization:" not in encoded
    assert "Bearer " not in encoded
    assert "evidence/rc-finish-line/provenance.json" not in receipt["source_control"]["file_hashes"]
    assert receipt["release_boundary"]["source_control_mutated"] is False
