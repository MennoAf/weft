#!/usr/bin/env python3
"""Capture a redacted, hash-bound release-candidate provenance receipt.

This script is deliberately read-only with respect to source control.  It only
writes the JSON path explicitly supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


EVIDENCE_CLASSES = ("current-source", "historical", "overlay", "copied")


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    )
    return result.stdout


def _sha256(path: Path, root: Path | None = None) -> str | None:
    """Return a file's content hash, without following links outside root."""
    if not path.is_file():
        return None
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
        except OSError:
            return None
        # A repository symlink that points outside the checkout is not content
        # we are authorized to read for this receipt.
        if root is not None and not target.is_relative_to(root):
            return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _relative_path(raw: bytes) -> str:
    return os.fsdecode(raw).replace("\\", "/")


def _status_records(root: Path, excluded: str | None) -> tuple[bool, bool, dict[str, dict[str, Any]]]:
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    records = raw.split(b"\0")
    dirty = False
    untracked = False
    file_hashes: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 3 or record[2:3] != b" ":
            continue
        status = record[:2].decode("ascii", errors="replace")
        path = _relative_path(record[3:])
        # With porcelain -z, rename/copy records carry the old path as the
        # next NUL-delimited field; hash the destination, which is the current
        # worktree content, and do not expose the old path unnecessarily.
        if status[0] in "RC" or status[1] in "RC":
            if index < len(records):
                index += 1
        if excluded and path == excluded:
            continue
        dirty = True
        is_untracked = status == "??"
        untracked = untracked or is_untracked
        path_obj = root / path
        file_hashes[path] = {
            "status": "untracked" if is_untracked else status,
            "sha256": _sha256(path_obj, root),
        }
    return dirty, untracked, dict(sorted(file_hashes.items()))


def _package_metadata(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    project = data.get("project", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "metadata_path": "pyproject.toml",
        "metadata_sha256": _sha256(path, root),
    }


def _lock_metadata(root: Path) -> dict[str, Any]:
    path = root / "uv.lock"
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    return {
        "path": "uv.lock",
        "sha256": _sha256(path),
        "format": {
            "version": lock.get("version"),
            "revision": lock.get("revision"),
            "requires_python": lock.get("requires-python"),
        },
        "package_count": len(lock.get("package", [])),
    }


def _evidence_class_details(root: Path, file_hashes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracked = set(os.fsdecode(item) for item in _git(root, "ls-files", "-z").split(b"\0") if item)
    overlay_paths = sorted(file_hashes)
    historical_paths = [
        path for path in ("docs/release-candidate-plan.md",) if (root / path).exists()
    ]
    copied_roots = [
        path for path in ("artifacts/linux-amd64-acceptance-20260912",) if (root / path).exists()
    ]
    return {
        "current-source": {
            "meaning": "Source and package metadata in this checkout; this receipt binds only their current bytes and revision.",
            "tracked_file_count": len(tracked),
            "representative_paths": ["weft/", "pyproject.toml", "uv.lock"],
        },
        "historical": {
            "meaning": "Prior plans or reports retained for lineage; not current-candidate acceptance evidence.",
            "paths_present": historical_paths,
        },
        "overlay": {
            "meaning": "Worktree changes layered over HEAD; explicitly outside a clean candidate boundary.",
            "paths_hashed": overlay_paths,
        },
        "copied": {
            "meaning": "Evidence copied from another run or environment; it is not reclassified as a current run.",
            "known_roots_present": copied_roots,
        },
    }


def collect_provenance(root: Path | str, output_path: Path | str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output_relative = None
    if output_path is not None:
        output = Path(output_path).resolve()
        try:
            output_relative = output.relative_to(root).as_posix()
        except ValueError:
            output_relative = None

    dirty, untracked, file_hashes = _status_records(root, output_relative)
    head_sha = _git(root, "rev-parse", "HEAD").decode().strip()
    branch = _git(root, "branch", "--show-current").decode().strip()
    details = _evidence_class_details(root, file_hashes)
    receipt: dict[str, Any] = {
        "schema": "weft.rc-provenance.v1",
        "source_control": {
            "head_sha": head_sha,
            "branch": branch or None,
            "dirty": dirty,
            "untracked": untracked,
            "file_hashes": file_hashes,
        },
        "package": _package_metadata(root),
        "dependency_lock": _lock_metadata(root),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "evidence_class_details": details,
        "release_boundary": {
            "candidate_status": "dirty-worktree" if dirty or untracked else "clean-worktree",
            "dirty_state_explicit": True,
            "untracked_state_explicit": True,
            "source_control_mutated": False,
            "excluded_output_path": output_relative,
            "limitations": [
                "Historical, overlay, and copied classes are labels, not current acceptance claims.",
                "A dirty or untracked worktree is not a clean release candidate.",
            ],
        },
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    receipt = collect_provenance(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
