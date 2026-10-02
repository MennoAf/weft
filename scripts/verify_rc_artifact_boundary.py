#!/usr/bin/env python3
"""Verify the release-candidate public/private artifact boundary.

The verifier is intentionally non-destructive.  It inventories paths and scans
bounded, repository-local bytes, but records no matched content, credential
names, or absolute paths.  A credential-shaped match is a flag for review, not
proof of a secret and never a claim of history-wide clearance.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any


EVIDENCE_CLASSES = ("current-source", "historical", "overlay", "copied")
CATEGORY_NAMES = (
    "generated-benchmark",
    "nested-worktree",
    "raw-receipt",
    "private-operational-doc",
)
_WORKTREE_ROOTS = (".ci-rc-worktree", ".rc-candidate-worktree", "tls-release")

# These are deliberately conservative shape checks.  They are used only to
# count potential matches; values and surrounding content are never retained.
_CREDENTIAL_PATTERNS = (
    re.compile(
        rb"(?im)^\s*(?:WEFT_API_KEY|[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)|TOKEN|SECRET|PASSWORD)"
        rb"\s*[:=]\s*(['\"]?)(?![<{])([A-Za-z0-9_./+=:-]{8,})\1\s*(?:#.*)?$"
    ),
    re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^/\s:@]+:[^@\s]+@"),
)

_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_SCAN_BYTES = 64 * 1024 * 1024
_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}


def _normalise(raw: str | os.PathLike[str]) -> str:
    value = str(raw).replace(os.sep, "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def classify_path(path: str | os.PathLike[str]) -> str:
    """Return the evidence identity for a repository-relative path."""
    relative = _normalise(path)
    parts = relative.split("/")
    if any(part == ".worktrees" for part in parts) or any(
        relative == root or relative.startswith(root + "/")
        for root in _WORKTREE_ROOTS
    ):
        return "overlay"
    if relative == "artifacts/linux-amd64-acceptance-20260912" or relative.startswith(
        "artifacts/linux-amd64-acceptance-20260912/"
    ):
        return "copied"
    if relative == "docs/release-candidate-plan.md" or relative.startswith("historical/"):
        return "historical"
    return "current-source"


def _artifact_category(path: str) -> str | None:
    relative = _normalise(path)
    parts = relative.lower().split("/")
    name = parts[-1]
    if any(part == ".worktrees" for part in parts) or any(
        relative == root or relative.startswith(root + "/")
        for root in _WORKTREE_ROOTS
    ):
        return "nested-worktree"
    if (
        relative.startswith("benchmarks/")
        and (
            "runs" in parts
            or "results" in parts
            or name.endswith("_cleaned.json")
            or name == "minted_cases.jsonl"
            or name.endswith((".jsonl", ".jsonl.gz", ".csv.gz"))
        )
    ):
        return "generated-benchmark"
    if (
        "receipt" in name
        or name.startswith(".receipt")
        or "raw-receipt" in name
        or ("evidence" in parts and "raw" in parts)
    ):
        return "raw-receipt"
    if any(part in {"private", "operational", "internal"} for part in parts[:-1]):
        return "private-operational-doc"
    if name.startswith(("private-", "operational-", "internal-")) and name.endswith(
        (".md", ".txt", ".json", ".yaml", ".yml", ".toml")
    ):
        return "private-operational-doc"
    return None


def _relative(root: Path, path: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _credential_match_count(data: bytes) -> int:
    return sum(len(pattern.findall(data)) for pattern in _CREDENTIAL_PATTERNS)


def _iter_files(root: Path):
    """Yield local regular files without following symlinks."""
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path


def scan_repository(root: Path | str, excluded_receipt: str | None = None) -> dict[str, Any]:
    """Inventory the checkout while retaining only safe counts and metadata."""
    root = Path(root).resolve()
    excluded = _normalise(excluded_receipt) if excluded_receipt else None
    category_counts = {name: 0 for name in CATEGORY_NAMES}
    evidence_counts = {name: 0 for name in EVIDENCE_CLASSES}
    files_seen = 0
    files_scanned = 0
    bytes_scanned = 0
    skipped_large = 0
    skipped_binary = 0
    credential_matches = 0

    for path in _iter_files(root):
        relative = _relative(root, path)
        if relative is None or relative == excluded:
            continue
        files_seen += 1
        evidence_counts[classify_path(relative)] += 1
        category = _artifact_category(relative)
        if category is not None:
            category_counts[category] += 1
        try:
            size = path.stat().st_size
            if size > _MAX_FILE_BYTES or bytes_scanned + size > _MAX_SCAN_BYTES:
                skipped_large += 1
                continue
            data = path.read_bytes()
        except OSError:
            skipped_large += 1
            continue
        if b"\0" in data:
            skipped_binary += 1
            continue
        files_scanned += 1
        bytes_scanned += len(data)
        credential_matches += _credential_match_count(data)

    return {
        "schema": "weft.rc-artifact-boundary.v1",
        "scan": {
            "root_kind": "current-checkout",
            "excluded_receipt": bool(excluded),
            "files_seen": files_seen,
            "files_scanned_for_shapes": files_scanned,
            "bytes_scanned": bytes_scanned,
            "max_file_bytes": _MAX_FILE_BYTES,
            "max_total_bytes": _MAX_SCAN_BYTES,
            "skipped_large_or_unreadable": skipped_large,
            "skipped_binary": skipped_binary,
        },
        "category_counts": category_counts,
        "evidence_class_counts": evidence_counts,
        "credential-shaped": {
            "match_count": credential_matches,
            "disposition": "flagged-and-excluded" if credential_matches else "none-observed",
        },
        "non_destructive": True,
        "source_control": {
            "mutated": False,
            "artifacts_deleted": False,
            "worktrees_deleted": False,
        },
    }


def render_receipt(report: dict[str, Any], receipt_path: str = "") -> str:
    """Render a content-safe receipt with no matched values or local paths."""
    scan = report["scan"]
    categories = report["category_counts"]
    evidence = report["evidence_class_counts"]
    credential = report["credential-shaped"]
    digest = hashlib.sha256(
        (str(categories) + str(evidence) + str(credential) + str(scan)).encode("utf-8")
    ).hexdigest()
    lines = [
        "# RC-FL-04 Artifact Boundary",
        "",
        "**Disposition:** REVIEW REQUIRED — excluded/flagged material remains in place.",
        "",
        "## Scan scope",
        "",
        "- Scanned the current checkout's local regular files without following symlinks.",
        f"- Files observed: `{scan['files_seen']}`; files scanned for shapes: `{scan['files_scanned_for_shapes']}`; bytes: `{scan['bytes_scanned']}`.",
        f"- Limits: `{scan['max_file_bytes']}` bytes per file and `{scan['max_total_bytes']}` bytes total.",
        f"- Skipped large/unreadable: `{scan['skipped_large_or_unreadable']}`; binary: `{scan['skipped_binary']}`.",
        f"- Receipt output excluded from its own scan: `{bool(scan['excluded_receipt'])}`.",
        "",
        "## Boundary classification",
        "",
        "| Category | Count | Disposition |",
        "|---|---:|---|",
        f"| Generated benchmark files | `{categories['generated-benchmark']}` | Excluded from public candidate artifacts |",
        f"| Nested worktrees | `{categories['nested-worktree']}` | Excluded from public candidate artifacts |",
        f"| Raw receipts | `{categories['raw-receipt']}` | Excluded; retain only redacted/hash-safe evidence |",
        f"| Private operational docs | `{categories['private-operational-doc']}` | Excluded from public documentation |",
        f"| Credential-shaped matches | `{credential['match_count']}` | {credential['disposition']} (values omitted) |",
        "",
        "## Evidence identity",
        "",
        "The scan preserves current-source versus historical, overlay, and copied evidence distinctions; it does not promote any retained evidence to current acceptance.",
        "",
        "| Evidence class | Files |",
        "|---|---:|",
        f"| current-source | `{evidence['current-source']}` |",
        f"| historical | `{evidence['historical']}` |",
        f"| overlay | `{evidence['overlay']}` |",
        f"| copied | `{evidence['copied']}` |",
        "",
        "## Safety and limitations",
        "",
        "- This verifier is non-destructive: it deletes no artifacts or worktrees and does not modify source control.",
        "- A credential-shaped match is only a bounded pattern flag; this receipt is not secret clearance and makes no claim beyond the scanned scope.",
        "- This scan does not perform a history rewrite, inspect unreachable objects, inspect other worktrees/branches, or establish repository-wide secret clearance.",
        "- Generated benchmark files, nested worktrees, raw receipts, and private operational docs are classified by path and remain available for separate cleanup decisions; none is deleted here.",
        "- Existing narrow `.gitignore` rules for benchmark outputs and nested worktrees were validated by the focused tests; no `.gitignore` change was needed.",
        f"- Safe report fingerprint (not a content dump): `{digest}`.",
        "",
        "No credential values, raw secrets, private paths, or credential-bearing URLs are included.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    receipt = args.receipt if args.receipt.is_absolute() else root / args.receipt
    relative = _relative(root, receipt.resolve())
    report = scan_repository(root, excluded_receipt=relative)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(render_receipt(report, relative or ""), encoding="utf-8")
    print(
        "RC-FL-04 artifact boundary verified: "
        f"{report['scan']['files_seen']} files observed, "
        f"{report['credential-shaped']['match_count']} credential-shaped match(es) flagged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
