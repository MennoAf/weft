#!/usr/bin/env python3
"""Refuse unsafe production deployments from this repository.

Production deploys must come from a clean, synchronized ``main`` checkout.
Internal benchmark source is allowed in the repository, but generated datasets,
snapshots, runs, and oversized Git blobs are not allowed on the deploy ref.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


MAX_BLOB_BYTES = 100 * 1024 * 1024
REQUIRED_FLY_SETTINGS = {
    "app": 'app = "weft-mcp"',
    "migration_mode": "WEFT_MIGRATION_MODE = \"verify\"",
    "transport": "WEFT_TRANSPORT = \"streamable-http\"",
    "health_path": "path = \"/healthz\"",
}
FORBIDDEN_TRACKED_PATTERNS = (
    "/data/",
    "/runs/",
    "/results/",
    "/snapshots/",
    "/fixtures/",
)


class GuardError(RuntimeError):
    """A production-deployment precondition failed."""


def run(*args: str) -> str:
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_paths() -> list[str]:
    return run("git", "ls-files").splitlines()


def check_branch_and_sync() -> None:
    branch = run("git", "branch", "--show-current")
    if branch != "main":
        raise GuardError(f"production deploy requires branch 'main' (found {branch!r})")

    try:
        remote_head = run("git", "rev-parse", "--verify", "origin/main")
    except subprocess.CalledProcessError as exc:
        raise GuardError("origin/main is unavailable; run 'git fetch origin main' first") from exc

    local_head = run("git", "rev-parse", "HEAD")
    if local_head != remote_head:
        raise GuardError(
            "local main is not synchronized with origin/main; "
            "run 'git fetch origin && git reset --hard origin/main' only after preserving work"
        )


def check_worktree() -> None:
    dirty = run("git", "status", "--porcelain", "--untracked-files=all")
    relevant = [
        line
        for line in dirty.splitlines()
        if not line.endswith((".ci-rc-worktree/", ".rc-candidate-worktree/", "tls-release/"))
    ]
    if relevant:
        raise GuardError(
            "working tree is dirty; commit or stash all deploy-ref changes before deploying"
        )


def check_fly_config(repo_root: Path) -> None:
    config = repo_root / "fly.toml"
    if not config.is_file():
        raise GuardError(
            "private production fly.toml is missing from this clone; restore it "
            "from the deployment secret store or operator backup, and do not deploy "
            "the sanitized deploy/examples/fly/fly.example.toml"
        )
    text = config.read_text(encoding="utf-8")
    for label, setting in REQUIRED_FLY_SETTINGS.items():
        if setting not in text:
            raise GuardError(f"fly.toml is missing required production setting: {label}")


def check_tracked_artifacts(paths: list[str]) -> None:
    forbidden = [
        path
        for path in paths
        if path.startswith("benchmarks/")
        and any(marker in f"/{path}" for marker in FORBIDDEN_TRACKED_PATTERNS)
    ]
    if forbidden:
        joined = ", ".join(forbidden[:5])
        suffix = "..." if len(forbidden) > 5 else ""
        raise GuardError(f"generated/internal benchmark artifacts are tracked: {joined}{suffix}")


def check_blob_sizes() -> None:
    objects = subprocess.Popen(
        ["git", "rev-list", "--objects", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert objects.stdout is not None
    sizes = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)",
        ],
        stdin=objects.stdout,
        check=True,
        capture_output=True,
        text=True,
    )
    objects.wait()
    oversized: list[str] = []
    for line in sizes.stdout.splitlines():
        kind, _object_id, size, *path = line.split(" ", 3)
        if kind == "blob" and int(size) > MAX_BLOB_BYTES:
            oversized.append(path[0] if path else _object_id)
    if oversized:
        raise GuardError(
            "production history contains Git blobs over 100 MB: "
            + ", ".join(oversized[:5])
        )


def check_docker_boundary(repo_root: Path) -> None:
    dockerfile = repo_root / "Dockerfile"
    if not dockerfile.is_file():
        raise GuardError("Dockerfile is missing")
    text = dockerfile.read_text(encoding="utf-8")
    if re.search(r"^\s*COPY\s+benchmarks(?:/|\s)", text, re.MULTILINE):
        raise GuardError("Dockerfile copies internal benchmark content into the production image")


def main() -> int:
    try:
        repo_root = Path(run("git", "rev-parse", "--show-toplevel"))
        check_branch_and_sync()
        check_worktree()
        check_fly_config(repo_root)
        paths = tracked_paths()
        check_tracked_artifacts(paths)
        check_blob_sizes()
        check_docker_boundary(repo_root)
    except (GuardError, subprocess.CalledProcessError) as exc:
        print(f"PRODUCTION DEPLOY BLOCKED: {exc}", file=sys.stderr)
        return 1

    print("Production deploy guard passed: clean synchronized main, safe history, and production config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
