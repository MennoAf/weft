"""Verify the deterministic RC finish-line evidence index and disposition."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
INDEX_SCHEMA = "weft.rc-finish-line.index.v1"
DISPOSITION_SCHEMA = "weft.rc-finish-line.disposition.v1"
STATUSES = {"READY", "HOLD", "BLOCKED", "NOT RUN"}
EXPECTED_IDS = tuple(f"RC-FL-{i:02d}" for i in range(1, 23))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s:=(\[\]\"'])/(?:Users|home|private|srv|var/lib|tmp)/[^\s\"'`]+"
    r"|(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"'`]+"
)
SECRET_RE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]{16,}|-----BEGIN .*PRIVATE KEY-----|"
    r"(?:API_KEY|TOKEN|PASSWORD|SECRET)\s*[:=]\s*[^<>{}$\s]+)"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
SELF_PROVENANCE_COMMIT = "CURRENT_WORKTREE_PENDING_COMMIT"
SELF_PROVENANCE_BRANCH = "finch/rc-fl22-inrepo-0b97sp"
SELF_PROVENANCE_ID = "RC-FL-22"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_block(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    blocks = JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise ValueError(f"{path}: expected exactly one JSON block")
    value = json.loads(blocks[0])
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON block must be an object")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"evidence path is not repository-relative: {relative!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes repository root: {relative}") from exc
    if (root / relative).is_symlink():
        raise ValueError(f"evidence path is symlinked: {relative}")
    return path


def _expect_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{label} has unexpected keys: expected {sorted(expected)}, got {actual}"
        )


def _validate_string_safety(value: Any, label: str, *, allow_command: bool = False) -> None:
    """Reject secrets and filesystem/control-string injection in metadata."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} is required")
    if PRIVATE_PATH_RE.search(value):
        raise ValueError(f"private absolute path in metadata: {label}")
    if SECRET_RE.search(value):
        raise ValueError(f"credential-shaped material in metadata: {label}")
    if CONTROL_RE.search(value) or "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe control or backslash in metadata: {label}")
    if ".." in value:
        raise ValueError(f"path traversal marker in metadata: {label}")
    if allow_command:
        # Commands are descriptive receipts, not executable input.  They may use
        # relative paths and flags, but may not carry private paths or traversal.
        return


def _run_git(root: Path, args: Iterable[str]) -> subprocess.CompletedProcess[str]:
    """Run one fixed-argv, local-only Git query with a bounded timeout."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env={},
    )


def _validate_branch(branch: str, label: str) -> None:
    _validate_string_safety(branch, label)
    if (
        not BRANCH_RE.fullmatch(branch)
        or branch in {"HEAD", "FETCH_HEAD", "ORIG_HEAD"}
        or branch.startswith("refs/")
        or branch.startswith("/")
        or branch.startswith(".")
        or branch.endswith("/")
        or branch.endswith(".")
        or "/." in branch
        or ".." in branch
        or "@{" in branch
    ):
        raise ValueError(f"{label} is not a canonical branch name")


def _validate_provenance(
    entry: dict[str, Any], root: Path, evidence_paths: list[str]
) -> None:
    acceptance = entry["acceptance_id"]
    commit = entry["source_commit"]
    branch = entry["source_branch"]
    _validate_string_safety(commit, f"{acceptance} source_commit")
    _validate_string_safety(branch, f"{acceptance} source_branch")
    _validate_branch(branch, f"{acceptance} source_branch")

    if commit == SELF_PROVENANCE_COMMIT:
        if acceptance != SELF_PROVENANCE_ID:
            raise ValueError("pending worktree provenance is allowed only for RC-FL-22")
        if branch != SELF_PROVENANCE_BRANCH:
            raise ValueError("RC-FL-22 pending provenance has unexpected source branch")
        for relative in evidence_paths:
            if not _safe_path(root, relative).is_file():
                raise FileNotFoundError(
                    f"RC-FL-22 pending evidence path is missing: {relative}"
                )
        return

    if not COMMIT_RE.fullmatch(commit):
        raise ValueError(f"{acceptance} source_commit must be lowercase 40-hex or pending marker")
    object_type = _run_git(root, ["cat-file", "-t", commit])
    if object_type.returncode != 0 or object_type.stdout.strip() != "commit":
        raise ValueError(f"{acceptance} source commit object does not exist: {commit}")
    for relative in evidence_paths:
        probe = f"{commit}:{relative}"
        if _run_git(root, ["cat-file", "-e", probe]).returncode != 0:
            raise ValueError(f"{acceptance} source commit does not contain evidence: {probe}")

    # Local branches are authoritative when present.  Historical/ephemeral
    # branch labels may no longer exist, so absence is informational rather than
    # a failure; an existing branch must contain the claimed commit.
    branch_ref = f"refs/heads/{branch}"
    branch_result = _run_git(root, ["show-ref", "--verify", "--quiet", branch_ref])
    if branch_result.returncode == 0:
        contained = _run_git(root, ["merge-base", "--is-ancestor", commit, branch])
        if contained.returncode != 0:
            raise ValueError(
                f"{acceptance} source commit is not contained by local source branch: "
                f"{commit} !<= {branch}"
            )


def _validate_entry(entry: Any, root: Path, seen_paths: set[str]) -> None:
    keys = {
        "acceptance_id",
        "title",
        "disposition",
        "evidence",
        "source_commit",
        "source_branch",
        "source_lineage",
        "command",
        "result",
        "limitations",
    }
    _expect_keys(entry, keys, "entry")
    acceptance = entry["acceptance_id"]
    if acceptance not in EXPECTED_IDS:
        raise ValueError(f"unknown acceptance ID: {acceptance}")
    _validate_string_safety(entry["title"], f"{acceptance} title")
    if entry["disposition"] not in STATUSES:
        raise ValueError(f"{acceptance} has invalid disposition")
    for field in (
        "source_commit",
        "source_branch",
        "source_lineage",
        "command",
        "result",
        "limitations",
    ):
        _validate_string_safety(
            entry[field], f"{acceptance} {field}", allow_command=field == "command"
        )
    if not isinstance(entry["evidence"], list) or not entry["evidence"]:
        raise ValueError(f"{acceptance} requires evidence")
    evidence_paths: list[str] = []
    for item in entry["evidence"]:
        _expect_keys(item, {"path", "sha256"}, f"{acceptance} evidence")
        relative = item["path"]
        if relative in {
            "evidence/rc-finish-line/final-index.md",
            "evidence/rc-finish-line/final-index.sha256",
            "evidence/rc-finish-line/operator-disposition.md",
        }:
            raise ValueError(f"{acceptance} cannot hash final index/disposition artifacts")
        if not SHA256_RE.fullmatch(item["sha256"]):
            raise ValueError(f"{acceptance} evidence hash must be lowercase sha256")
        if relative in seen_paths:
            raise ValueError(f"duplicate evidence path: {relative}")
        seen_paths.add(relative)
        path = _safe_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"evidence path is missing: {relative}")
        data = path.read_bytes()
        if sha256_bytes(data) != item["sha256"]:
            raise ValueError(f"evidence hash mismatch: {relative}")
        text = data.decode("utf-8", errors="replace")
        if PRIVATE_PATH_RE.search(text):
            raise ValueError(f"private absolute path in evidence: {relative}")
        if SECRET_RE.search(text):
            raise ValueError(f"credential-shaped material in evidence: {relative}")
        evidence_paths.append(relative)
    _validate_provenance(entry, root, evidence_paths)


def validate_index(index: dict[str, Any], root: Path = ROOT) -> dict[str, int]:
    _expect_keys(index, {"schema", "hash_algorithm", "overall_disposition", "entries"}, "index")
    _validate_string_safety(index["schema"], "index schema")
    _validate_string_safety(index["hash_algorithm"], "index hash_algorithm")
    if index["schema"] != INDEX_SCHEMA or index["hash_algorithm"] != "sha256":
        raise ValueError("index schema or hash algorithm mismatch")
    if index["overall_disposition"] not in STATUSES:
        raise ValueError("index overall disposition is invalid")
    entries = index["entries"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_IDS):
        raise ValueError("index must contain exactly 22 entries")
    ids = [entry.get("acceptance_id") if isinstance(entry, dict) else None for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate acceptance entries")
    if set(ids) != set(EXPECTED_IDS):
        missing = sorted(set(EXPECTED_IDS) - set(ids))
        extra = sorted(set(ids) - set(EXPECTED_IDS))
        raise ValueError(f"acceptance completeness mismatch: missing={missing} extra={extra}")
    seen_paths: set[str] = set()
    for entry in entries:
        _validate_entry(entry, root.resolve(), seen_paths)
    counts = Counter(entry["disposition"] for entry in entries)
    by_id = {entry["acceptance_id"]: entry for entry in entries}
    if index["overall_disposition"] == "READY":
        if by_id["RC-FL-20"]["disposition"] != "READY" or by_id["RC-FL-21"]["disposition"] == "BLOCKED":
            raise ValueError("index cannot be READY while qualification is unauthorized or hosted boundary is blocked")
    if by_id["RC-FL-21"]["disposition"] == "BLOCKED" and "external status not yet supplied" not in by_id["RC-FL-21"]["limitations"].lower():
        raise ValueError("RC-FL-21 BLOCKED entry requires exact external-status limitation")
    return dict(counts)


def validate_disposition(disposition: dict[str, Any], index: dict[str, Any]) -> None:
    _expect_keys(disposition, {"schema", "decision_date_utc", "overall_disposition", "promotion_authorized", "decision", "operator_action_required"}, "disposition")
    for field in ("schema", "decision_date_utc", "overall_disposition", "decision", "operator_action_required"):
        _validate_string_safety(disposition[field], f"disposition {field}")
    if disposition["schema"] != DISPOSITION_SCHEMA:
        raise ValueError("disposition schema mismatch")
    if not re.fullmatch(r"2026-09-17", disposition["decision_date_utc"]):
        raise ValueError("decision date must be the deliberate 2026-09-17 UTC date")
    if disposition["overall_disposition"] != index["overall_disposition"]:
        raise ValueError("disposition is inconsistent with index")
    if disposition["overall_disposition"] != "HOLD" or disposition["promotion_authorized"] is not False:
        raise ValueError("operator disposition must be HOLD with promotion_authorized false")


def verify(index_path: Path, index_sha256_path: Path, disposition_path: Path, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    index_path = index_path if index_path.is_absolute() else root / index_path
    index_sha256_path = index_sha256_path if index_sha256_path.is_absolute() else root / index_sha256_path
    disposition_path = disposition_path if disposition_path.is_absolute() else root / disposition_path
    index_bytes = index_path.read_bytes()
    sidecar = index_sha256_path.read_text(encoding="utf-8")
    if not re.fullmatch(r"[0-9a-f]{64}\n", sidecar):
        raise ValueError("index sidecar must be one lowercase sha256 line")
    if sidecar.strip() != sha256_bytes(index_bytes):
        raise ValueError("index sidecar mismatch")
    index = _json_block(index_path)
    counts = validate_index(index, root)
    disposition = _json_block(disposition_path)
    validate_disposition(disposition, index)
    return {"overall_disposition": index["overall_disposition"], "entry_count": len(index["entries"]), "disposition_counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", type=Path, required=True)
    parser.add_argument("--disposition", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify(args.index, args.index_sha256, args.disposition)
    print(f"RC evidence verified: entries={report['entry_count']} disposition={report['overall_disposition']} counts={report['disposition_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
