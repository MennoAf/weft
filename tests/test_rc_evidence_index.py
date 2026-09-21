"""RC-FL-22 deterministic final evidence index contract tests."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import verify_rc_evidence as verifier

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "evidence/rc-finish-line/final-index.md"
SIDECAR = ROOT / "evidence/rc-finish-line/final-index.sha256"
DISPOSITION = ROOT / "evidence/rc-finish-line/operator-disposition.md"


def _index() -> dict:
    return verifier._json_block(INDEX)


def test_final_index_is_complete_unique_and_hash_bound() -> None:
    report = verifier.verify(INDEX, SIDECAR, DISPOSITION, ROOT)
    assert report["entry_count"] == 22
    assert report["overall_disposition"] == "HOLD"
    assert sum(report["disposition_counts"].values()) == 22


def test_historical_and_pending_provenance_are_explicit() -> None:
    index = _index()
    entries = {entry["acceptance_id"]: entry for entry in index["entries"]}
    assert entries["RC-FL-15"]["source_commit"] == "115f46444b45f33b82143d49c7d698c87f78f5c0"
    assert entries["RC-FL-15"]["source_branch"] == "finch/rc-fl15-inrepo-0b97sp"
    assert entries["RC-FL-19"]["source_commit"] == "e8a10818500f745aa1cea59eb362505e19920633"
    assert entries["RC-FL-19"]["source_branch"] == "finch/rc-fl19-inrepo-0b97sp"
    assert entries["RC-FL-22"]["source_commit"] == verifier.SELF_PROVENANCE_COMMIT
    assert entries["RC-FL-22"]["source_branch"] == verifier.SELF_PROVENANCE_BRANCH


def test_tamper_and_stale_hashes_are_rejected() -> None:
    index = _index()
    mutated = copy.deepcopy(index)
    mutated["entries"][0]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.validate_index(mutated, ROOT)


def test_missing_and_duplicate_acceptances_are_rejected() -> None:
    index = _index()
    missing = copy.deepcopy(index)
    missing["entries"].pop()
    with pytest.raises(ValueError, match="exactly 22"):
        verifier.validate_index(missing, ROOT)
    duplicate = copy.deepcopy(index)
    duplicate["entries"][1]["acceptance_id"] = duplicate["entries"][0]["acceptance_id"]
    with pytest.raises(ValueError, match="duplicate"):
        verifier.validate_index(duplicate, ROOT)


def test_invalid_status_and_disposition_inconsistency_are_rejected() -> None:
    index = _index()
    invalid = copy.deepcopy(index)
    invalid["entries"][0]["disposition"] = "PASS"
    with pytest.raises(ValueError, match="invalid disposition"):
        verifier.validate_index(invalid, ROOT)
    disposition = verifier._json_block(DISPOSITION)
    disposition["overall_disposition"] = "BLOCKED"
    with pytest.raises(ValueError, match="inconsistent"):
        verifier.validate_disposition(disposition, index)


def test_sidecar_tamper_is_rejected(tmp_path: Path) -> None:
    sidecar = tmp_path / "index.sha256"
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        verifier.verify(INDEX, sidecar, DISPOSITION, ROOT)


def test_path_traversal_and_missing_evidence_are_rejected() -> None:
    index = _index()
    traversal = copy.deepcopy(index)
    traversal["entries"][0]["evidence"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="repository-relative|escapes"):
        verifier.validate_index(traversal, ROOT)
    missing = copy.deepcopy(index)
    missing["entries"][0]["evidence"][0]["path"] = "evidence/rc-finish-line/no-such-file.md"
    with pytest.raises(FileNotFoundError, match="missing"):
        verifier.validate_index(missing, ROOT)


def test_metadata_adversarial_strings_are_rejected() -> None:
    index = _index()
    cases = (("title", "TOKEN=actual-secret-value", "credential"), ("result", "/Users/person/worktree", "private"), ("limitations", "../outside", "traversal"))
    for field, value, message in cases:
        mutated = copy.deepcopy(index)
        mutated["entries"][0][field] = value
        with pytest.raises(ValueError, match=message):
            verifier.validate_index(mutated, ROOT)


def test_secret_and_private_absolute_path_evidence_are_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("WEFT_API_KEY=actual-secret-value\n", encoding="utf-8")
    private = tmp_path / "private.txt"
    private.write_text("source /Users/person/private/worktree\n", encoding="utf-8")
    index = _index()
    for path, message, name in ((secret, "credential", "_rc_secret_fixture.txt"), (private, "private", "_rc_private_fixture.txt")):
        target = ROOT / "tests" / name
        target.write_bytes(path.read_bytes())
        try:
            mutated = copy.deepcopy(index)
            mutated["entries"][0]["evidence"][0]["path"] = target.relative_to(ROOT).as_posix()
            mutated["entries"][0]["evidence"][0]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            with pytest.raises(ValueError, match=message):
                verifier.validate_index(mutated, ROOT)
        finally:
            target.unlink()


def test_provenance_commit_path_and_branch_are_verified(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, timeout=5)
    (repo / "receipt.txt").write_text("receipt\n", encoding="utf-8")
    subprocess.run(["git", "add", "receipt.txt"], cwd=repo, check=True, timeout=5)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "receipt"], cwd=repo, check=True, timeout=5)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    index = _index()
    entry = copy.deepcopy(index["entries"][0])
    entry["source_commit"] = commit
    entry["source_branch"] = "main"
    (repo / "uncommitted.txt").write_text("uncommitted receipt\n", encoding="utf-8")
    entry["evidence"][0]["path"] = "uncommitted.txt"
    entry["evidence"][0]["sha256"] = verifier.sha256_bytes((repo / "uncommitted.txt").read_bytes())
    with pytest.raises(ValueError, match="commit does not contain evidence"):
        verifier._validate_entry(entry, repo, set())


def test_existing_branch_must_contain_normal_provenance_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, timeout=5)
    (repo / "receipt.txt").write_text("receipt\n", encoding="utf-8")
    subprocess.run(["git", "add", "receipt.txt"], cwd=repo, check=True, timeout=5)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "receipt"], cwd=repo, check=True, timeout=5)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    index = _index()
    entry = copy.deepcopy(index["entries"][0])
    entry["source_commit"] = commit
    entry["source_branch"] = "main"
    entry["evidence"][0]["path"] = "receipt.txt"
    entry["evidence"][0]["sha256"] = verifier.sha256_bytes((repo / "receipt.txt").read_bytes())
    (repo / "second.txt").write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "add", "second.txt"], cwd=repo, check=True, timeout=5)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "second"], cwd=repo, check=True, timeout=5)
    entry["source_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    subprocess.run(["git", "branch", "other", "HEAD~1"], cwd=repo, check=True, timeout=5)
    with pytest.raises(ValueError, match="not contained"):
        entry["source_branch"] = "other"
        verifier._validate_entry(entry, repo, set())


def test_pending_provenance_is_restricted_to_rc_fl22() -> None:
    index = _index()
    entry = copy.deepcopy(index["entries"][0])
    entry["source_commit"] = verifier.SELF_PROVENANCE_COMMIT
    with pytest.raises(ValueError, match="only for RC-FL-22"):
        verifier.validate_index({**index, "entries": [entry if e is index["entries"][0] else e for e in index["entries"]]}, ROOT)


def test_ready_is_rejected_when_qualification_or_hosted_boundary_is_not_clear() -> None:
    index = _index()
    ready = copy.deepcopy(index)
    ready["overall_disposition"] = "READY"
    ready["entries"] = [{**entry, "disposition": "READY"} if entry["acceptance_id"] == "RC-FL-20" else entry for entry in ready["entries"]]
    with pytest.raises(ValueError, match="cannot be READY"):
        verifier.validate_index(ready, ROOT)
