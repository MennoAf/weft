"""Tests for the production deployment reference guard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_GUARD_PATH = Path(__file__).parents[1] / "scripts" / "deploy_production_guard.py"
_SPEC = importlib.util.spec_from_file_location("deploy_production_guard", _GUARD_PATH)
assert _SPEC and _SPEC.loader
_GUARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GUARD)


HEAD = "a" * 40
OTHER = "b" * 40


def _mock_git(monkeypatch: pytest.MonkeyPatch, *, branch: str, tags: str, remote: str) -> None:
    responses = {
        ("git", "rev-parse", "HEAD"): HEAD,
        ("git", "branch", "--show-current"): branch,
        ("git", "tag", "--points-at", "HEAD"): tags,
        (
            "git",
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/tags/production-2026-09-03",
            "refs/tags/production-2026-09-03^{}",
        ): remote,
    }

    def fake_run(*args: str) -> str:
        try:
            return responses[args]
        except KeyError as exc:
            raise AssertionError(f"unexpected git command: {args!r}") from exc

    monkeypatch.setattr(_GUARD, "run", fake_run)


def test_approved_remote_production_tag_at_head_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git(
        monkeypatch,
        branch="",
        tags="production-2026-09-03",
        remote=f"{HEAD}\trefs/tags/production-2026-09-03^{{}}",
    )

    _GUARD.check_checkout_ref()
    _GUARD.check_approved_production_ref()


def test_main_with_approved_tag_at_head_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git(
        monkeypatch,
        branch="main",
        tags="production-2026-09-03",
        remote=f"{HEAD}\trefs/tags/production-2026-09-03",
    )

    _GUARD.check_checkout_ref()
    _GUARD.check_approved_production_ref()


@pytest.mark.parametrize(
    "tags",
    ["", "v1.0.0", "production-latest", "production-2026-9-3"],
)
def test_unapproved_tag_at_head_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tags: str
) -> None:
    _mock_git(
        monkeypatch,
        branch="",
        tags=tags,
        remote="",
    )

    with pytest.raises(_GUARD.GuardError, match="approved tag"):
        _GUARD.check_approved_production_ref()


def test_multiple_approved_tags_at_head_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git(
        monkeypatch,
        branch="",
        tags="production-2026-09-03\nproduction-2026-09-03-hotfix",
        remote="",
    )

    with pytest.raises(_GUARD.GuardError, match="exactly one approved tag"):
        _GUARD.check_approved_production_ref()


def test_local_only_tag_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git(
        monkeypatch,
        branch="",
        tags="production-2026-09-03",
        remote="",
    )

    monkeypatch.setattr(
        _GUARD,
        "run",
        lambda *args: (
            HEAD
            if args == ("git", "rev-parse", "HEAD")
            else "production-2026-09-03"
            if args == ("git", "tag", "--points-at", "HEAD")
            else (_ for _ in ()).throw(_GUARD.subprocess.CalledProcessError(2, args))
        ),
    )

    with pytest.raises(_GUARD.GuardError, match="not available on origin"):
        _GUARD.check_approved_production_ref()


def test_tag_pointing_at_different_remote_commit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_git(
        monkeypatch,
        branch="",
        tags="production-2026-09-03",
        remote=f"{OTHER}\trefs/tags/production-2026-09-03^{{}}",
    )

    with pytest.raises(_GUARD.GuardError, match="does not resolve to HEAD"):
        _GUARD.check_approved_production_ref()


def test_non_main_named_branch_is_rejected_even_with_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git(
        monkeypatch,
        branch="recovery/v72-production-clean",
        tags="production-2026-09-03",
        remote=f"{HEAD}\trefs/tags/production-2026-09-03^{{}}",
    )

    with pytest.raises(_GUARD.GuardError, match="main or a detached"):
        _GUARD.check_checkout_ref()


def test_dirty_worktree_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GUARD, "run", lambda *args: " M docs/configuration.md")

    with pytest.raises(_GUARD.GuardError, match="working tree is dirty"):
        _GUARD.check_worktree()
