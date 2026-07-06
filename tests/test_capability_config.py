"""Tests for Capability Registry configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capability_registry.config import RegistryConfig, RepoConfig, load_config


def test_load_toml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "capability.toml"
    config_path.write_text(
        """
dry_run = false
weft_project_id = "weft-public"

[[repos]]
slug = "muttr"
root = "../muttr"
include_globs = ["**/*.py", "scripts/*.py"]
exclude_dirs = [".git", "__pycache__", ".venv"]

[repos.manual_capability_slugs]
"crawl/escalation.py" = ["bot-block-hardening", "crawler-escalation"]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == RegistryConfig(
        repos=[
            RepoConfig(
                slug="muttr",
                root="../muttr",
                include_globs=["**/*.py", "scripts/*.py"],
                exclude_dirs=[".git", "__pycache__", ".venv"],
                manual_capability_slugs={
                    "crawl/escalation.py": [
                        "bot-block-hardening",
                        "crawler-escalation",
                    ]
                },
            )
        ],
        dry_run=False,
        weft_project_id="weft-public",
    )


def test_load_json_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "capability.json"
    config_path.write_text(
        json.dumps({"repos": [{"slug": "weft", "root": "/missing/weft"}]}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dry_run is True
    assert config.weft_project_id == ""
    assert config.repos[0].include_globs == ["**/*.py"]
    assert config.repos[0].exclude_dirs == [
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
    ]
    assert config.repos[0].manual_capability_slugs == {}


def test_repo_config_defaults_are_not_shared() -> None:
    first = RepoConfig(slug="first", root="/first")
    second = RepoConfig(slug="second", root="/second")

    first.include_globs.append("scripts/*.py")
    first.exclude_dirs.append("dist")
    first.manual_capability_slugs["a.py"] = ["crawler"]

    assert second.include_globs == ["**/*.py"]
    assert second.exclude_dirs == [
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
    ]
    assert second.manual_capability_slugs == {}


def test_load_config_requires_repos_list(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"dry_run": True}), encoding="utf-8")

    with pytest.raises(ValueError, match='Config must contain "repos" list'):
        load_config(config_path)


def test_load_config_wraps_invalid_repo_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps({"repos": [{"slug": "missing-root"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid repo config"):
        load_config(config_path)


def test_load_config_allows_empty_repos(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.json"
    config_path.write_text(json.dumps({"repos": []}), encoding="utf-8")

    assert load_config(config_path).repos == []


def test_example_config_is_parseable() -> None:
    config = load_config(Path("capability_registry/example_config.toml"))

    assert config.dry_run is True
    assert config.repos[0].slug == "muttr"
    assert "crawl/escalation.py" in config.repos[0].manual_capability_slugs
