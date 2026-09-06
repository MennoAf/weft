#!/usr/bin/env python3
"""Refuse unsafe production deployments from this repository.

Production deploys must come from a clean checkout at an approved remote
production tag. Internal benchmark source is allowed in the repository, but
generated datasets, snapshots, runs, and oversized Git blobs are not allowed
on the deploy ref.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


APPROVED_PRODUCTION_TAG = re.compile(r"production-\d{4}-\d{2}-\d{2}(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?\Z")
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


def check_approved_production_ref() -> None:
    """Require an approved production tag at HEAD and on the origin remote.

    Tags are the explicit operator approval boundary. The remote check prevents
    a locally-created or moved tag from authorizing a deployment.
    """
    head = run("git", "rev-parse", "HEAD")
    local_tags = run("git", "tag", "--points-at", "HEAD").splitlines()
    approved_tags = [tag for tag in local_tags if APPROVED_PRODUCTION_TAG.fullmatch(tag)]
    if not approved_tags:
        raise GuardError(
            "production deploy requires an approved tag matching "
            "production-YYYY-MM-DD[-suffix] at HEAD"
        )
    if len(approved_tags) > 1:
        raise GuardError(
            "production deploy requires exactly one approved tag at HEAD; "
            f"found {', '.join(sorted(approved_tags))}"
        )

    tag = approved_tags[0]
    try:
        advertised = run(
            "git",
            "ls-remote",
            "--exit-code",
            "origin",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        )
    except subprocess.CalledProcessError as exc:
        raise GuardError(
            f"approved production tag {tag!r} is not available on origin; "
            f"run 'git fetch origin refs/tags/{tag}:refs/tags/{tag}' first"
        ) from exc
    remote_hashes = {
        fields[0]
        for line in advertised.splitlines()
        if len(fields := line.split()) == 2
        and fields[1] in {f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"}
    }
    if not remote_hashes:
        raise GuardError(f"origin returned no usable hash for production tag {tag!r}")

    if head not in remote_hashes:
        raise GuardError(
            f"approved production tag {tag!r} does not resolve to HEAD on origin"
        )


def check_checkout_ref() -> None:
    """Allow main or a detached checkout, with approval enforced by the tag check."""
    branch = run("git", "branch", "--show-current")
    if branch not in ("main", ""):
        raise GuardError(
            "production deploy requires main or a detached approved-tag checkout "
            f"(found {branch!r})"
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
        check_checkout_ref()
        check_approved_production_ref()
        check_worktree()
        check_fly_config(repo_root)
        paths = tracked_paths()
        check_tracked_artifacts(paths)
        check_blob_sizes()
        check_docker_boundary(repo_root)
    except (GuardError, subprocess.CalledProcessError) as exc:
        print(f"PRODUCTION DEPLOY BLOCKED: {exc}", file=sys.stderr)
        return 1

    print("Production deploy guard passed: approved production tag, clean ref, safe history, and production config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
