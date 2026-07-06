"""Configuration models and loader for the Capability Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


DEFAULT_INCLUDE_GLOBS = ["**/*.py"]
DEFAULT_EXCLUDE_DIRS = [".git", "__pycache__", ".venv", "venv", "node_modules"]


@dataclass
class RepoConfig:
    """Configuration for scanning one source repository."""

    slug: str
    root: str
    include_globs: list[str] = field(default_factory=lambda: DEFAULT_INCLUDE_GLOBS.copy())
    exclude_dirs: list[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_DIRS.copy())
    manual_capability_slugs: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class RegistryConfig:
    """Top-level Capability Registry configuration."""

    repos: list[RepoConfig]
    dry_run: bool = True
    weft_project_id: str = ""


def load_config(path: Path) -> RegistryConfig:
    """Load TOML or JSON registry configuration from ``path``."""
    try:
        raw = _load_raw_config(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    except ValueError:
        raise
    except Exception as exc:
        if _is_toml_decode_error(exc):
            raise ValueError(f"Invalid TOML config {path}: {exc}") from exc
        raise

    return _build_registry_config(raw)


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".toml":
        if tomllib is None:
            raise ImportError(
                "Loading .toml config requires Python 3.11+ or the tomli package "
                f"(current Python: {sys.version_info.major}.{sys.version_info.minor})."
            )
        return tomllib.loads(text)
    return json.loads(text)


def _build_registry_config(raw: dict[str, Any]) -> RegistryConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON/TOML object")

    repos_raw = raw.get("repos")
    if not isinstance(repos_raw, list):
        raise ValueError('Config must contain "repos" list')

    repos = [_build_repo_config(index, repo_raw) for index, repo_raw in enumerate(repos_raw)]

    dry_run = raw.get("dry_run", True)
    if not isinstance(dry_run, bool):
        raise ValueError("Config field 'dry_run' must be a bool")

    weft_project_id = raw.get("weft_project_id", "")
    if not isinstance(weft_project_id, str):
        raise ValueError("Config field 'weft_project_id' must be a string")

    return RegistryConfig(
        repos=repos,
        dry_run=dry_run,
        weft_project_id=weft_project_id,
    )


def _build_repo_config(index: int, raw: Any) -> RepoConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid repo config at index {index}: expected object")

    try:
        slug = raw["slug"]
        root = raw["root"]
    except KeyError as exc:
        raise ValueError(f"Invalid repo config at index {index}: missing {exc}") from exc

    if not isinstance(slug, str):
        raise ValueError(f"Invalid repo config at index {index}: slug must be a string")
    if not isinstance(root, str):
        raise ValueError(f"Invalid repo config at index {index}: root must be a string")

    include_globs = _string_list(
        raw.get("include_globs", DEFAULT_INCLUDE_GLOBS),
        f"repos[{index}].include_globs",
    )
    exclude_dirs = _string_list(
        raw.get("exclude_dirs", DEFAULT_EXCLUDE_DIRS),
        f"repos[{index}].exclude_dirs",
    )
    manual_capability_slugs = _manual_slug_map(
        raw.get("manual_capability_slugs", {}),
        f"repos[{index}].manual_capability_slugs",
    )

    return RepoConfig(
        slug=slug,
        root=root,
        include_globs=include_globs,
        exclude_dirs=exclude_dirs,
        manual_capability_slugs=manual_capability_slugs,
    )


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Config field '{field_name}' must be a list of strings")
    return list(value)


def _manual_slug_map(value: Any, field_name: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{field_name}' must be a string-to-list map")

    result: dict[str, list[str]] = {}
    for file_path, slugs in value.items():
        if not isinstance(file_path, str):
            raise ValueError(f"Config field '{field_name}' keys must be strings")
        result[file_path] = _string_list(slugs, f"{field_name}.{file_path}")
    return result


def _is_toml_decode_error(exc: Exception) -> bool:
    if tomllib is None:
        return False
    toml_decode_error: Optional[type[Exception]] = getattr(
        tomllib,
        "TOMLDecodeError",
        None,
    )
    return toml_decode_error is not None and isinstance(exc, toml_decode_error)


__all__ = ["RepoConfig", "RegistryConfig", "load_config"]
