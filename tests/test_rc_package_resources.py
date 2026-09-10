"""Focused foundation tests for the package Compose resource accessor."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from weft.resources import ComposeResourceError, compose_file_path


_RESOURCE = "docker-compose.weft.yml"


class _TraversableResource:
    def __init__(self, path: Path, present: bool) -> None:
        self.path = path
        self.present = present

    def is_file(self) -> bool:
        return self.present


class _TraversablePackage:
    def __init__(self, resource: _TraversableResource) -> None:
        self.resource = resource

    def joinpath(self, name: str) -> _TraversableResource:
        assert name == _RESOURCE
        return self.resource


def _write_source_layout(root: Path, *, project_name: str = "weft-memory") -> Path:
    (root / "weft" / "data").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n', encoding="utf-8"
    )
    compose = root / _RESOURCE
    compose.write_text("services:\n  test:\n    image: example\n", encoding="utf-8")
    return root / "weft" / "data"


def test_package_data_wins_over_source_fallback(tmp_path: Path) -> None:
    """A present package resource is resolved before any checkout fallback."""
    package_path = tmp_path / _RESOURCE
    package_path.write_text("package-data", encoding="utf-8")
    package = _TraversablePackage(
        _TraversableResource(package_path, present=True)
    )

    @contextmanager
    def fake_as_file(resource: _TraversableResource):
        assert resource is package.resource
        yield resource.path

    with (
        patch("weft.resources.importlib.resources.files", return_value=package),
        patch("weft.resources.importlib.resources.as_file", fake_as_file),
        patch("weft.resources._source_fallback", side_effect=AssertionError),
    ):
        with compose_file_path() as resolved:
            assert resolved == package_path
            assert resolved.read_text(encoding="utf-8") == "package-data"


def test_package_materialization_path_expires_at_context_exit(tmp_path: Path) -> None:
    """A materialized path is usable only while the resource context is open."""
    materialized = tmp_path / _RESOURCE
    package = _TraversablePackage(
        _TraversableResource(tmp_path / "package-resource", present=True)
    )
    alive = False

    @contextmanager
    def fake_as_file(_resource: _TraversableResource):
        nonlocal alive
        materialized.write_text("materialized", encoding="utf-8")
        alive = True
        try:
            yield materialized
        finally:
            alive = False
            materialized.unlink()

    with (
        patch("weft.resources.importlib.resources.files", return_value=package),
        patch("weft.resources.importlib.resources.as_file", fake_as_file),
    ):
        with compose_file_path() as resolved:
            assert alive
            assert resolved.exists()
            assert resolved.read_text(encoding="utf-8") == "materialized"
        assert not alive
        assert not resolved.exists()


@pytest.mark.parametrize(
    "caller_error",
    [OSError("caller os"), FileNotFoundError("caller fnf"), RuntimeError("caller runtime")],
)
def test_package_caller_exception_is_identical_and_cleanup_runs(
    tmp_path: Path, caller_error: Exception
) -> None:
    """Caller failures pass through unchanged while materialization still closes."""
    package = _TraversablePackage(
        _TraversableResource(tmp_path / "package-resource", present=True)
    )
    cleanup = False

    @contextmanager
    def fake_as_file(_resource: _TraversableResource):
        nonlocal cleanup
        try:
            yield tmp_path / _RESOURCE
        finally:
            cleanup = True

    with (
        patch("weft.resources.importlib.resources.files", return_value=package),
        patch("weft.resources.importlib.resources.as_file", fake_as_file),
    ):
        with pytest.raises(type(caller_error)) as caught:
            with compose_file_path():
                raise caller_error

    assert caught.value is caller_error
    assert str(caught.value) == str(caller_error)
    assert cleanup


def test_package_presence_probe_error_is_actionable_and_stops_setup(
    tmp_path: Path,
) -> None:
    """A failing presence probe preserves its cause and stops resolution setup."""
    package = _TraversablePackage(
        _TraversableResource(tmp_path / "package-resource", present=False)
    )
    probe_error = OSError("stat denied")
    body_entered = False

    with (
        patch("weft.resources.importlib.resources.files", return_value=package),
        patch(
            "weft.resources.importlib.resources.as_file",
            side_effect=AssertionError("materialization must not run"),
        ) as as_file,
        patch(
            "weft.resources._source_fallback",
            side_effect=AssertionError("source fallback must not run"),
        ) as source_fallback,
        patch.object(package.resource, "is_file", side_effect=probe_error),
    ):
        with pytest.raises(
            ComposeResourceError, match="Unable to inspect packaged"
        ) as caught:
            with compose_file_path():
                body_entered = True

    assert caught.value.__cause__ is probe_error
    assert not body_entered
    as_file.assert_not_called()
    source_fallback.assert_not_called()


def test_package_materialization_setup_error_is_actionable(tmp_path: Path) -> None:
    """Only an as_file setup failure is translated to ComposeResourceError."""
    package = _TraversablePackage(
        _TraversableResource(tmp_path / "package-resource", present=True)
    )
    setup_error = OSError("materialization failed")

    @contextmanager
    def failing_as_file(_resource: _TraversableResource):
        raise setup_error
        yield  # pragma: no cover

    with (
        patch("weft.resources.importlib.resources.files", return_value=package),
        patch("weft.resources.importlib.resources.as_file", failing_as_file),
    ):
        with pytest.raises(ComposeResourceError, match="Unable to materialize packaged") as caught:
            with compose_file_path():
                pass

    assert caught.value.__cause__ is setup_error


def test_real_anchored_source_fallback(tmp_path: Path) -> None:
    """A source checkout is accepted from the filesystem-backed package origin."""
    package_origin = _write_source_layout(tmp_path)
    package = _TraversablePackage(
        _TraversableResource(package_origin / _RESOURCE, present=False)
    )

    with patch("weft.resources.importlib.resources.files", return_value=package_origin):
        with compose_file_path() as resolved:
            assert resolved == tmp_path / _RESOURCE
            assert resolved.read_text(encoding="utf-8").startswith("services:")


def test_source_fallback_rejects_mismatched_project_metadata(tmp_path: Path) -> None:
    """A nearby but unrelated project cannot authorize source fallback."""
    package_origin = _write_source_layout(tmp_path, project_name="other-project")

    with patch("weft.resources.importlib.resources.files", return_value=package_origin):
        with pytest.raises(ComposeResourceError, match="Unable to resolve"):
            with compose_file_path():
                pass


def test_source_fallback_rejects_complete_noncanonical_origin(tmp_path: Path) -> None:
    """A complete vendor/package/data layout is not the canonical source origin."""
    source_root = tmp_path / "vendor"
    package_origin = source_root / "package" / "data"
    package_origin.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "weft-memory"\n', encoding="utf-8"
    )
    (source_root / "weft").mkdir()
    (source_root / _RESOURCE).write_text("decoy", encoding="utf-8")

    with patch("weft.resources.importlib.resources.files", return_value=package_origin):
        with pytest.raises(ComposeResourceError, match="Unable to resolve"):
            with compose_file_path():
                pass


@pytest.mark.parametrize(
    "layout_kind",
    ["installed", "missing-package", "missing-compose"],
)
def test_source_fallback_rejects_unanchored_or_incomplete_layout(
    tmp_path: Path, layout_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed-looking and incomplete layouts never trigger arbitrary lookup."""
    isolated_cwd = tmp_path / "cwd"
    isolated_cwd.mkdir()
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()
    monkeypatch.chdir(isolated_cwd)
    monkeypatch.setenv("HOME", str(isolated_home))

    if layout_kind == "installed":
        package_origin = (
            tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "weft" / "data"
        )
        package_origin.mkdir(parents=True)
        # A tempting parent Compose file must not be discovered by searching.
        (tmp_path / _RESOURCE).write_text("wrong-parent", encoding="utf-8")
    elif layout_kind == "missing-package":
        package_origin = tmp_path / "unrelated-package" / "data"
        package_origin.mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "weft-memory"\n', encoding="utf-8"
        )
        (tmp_path / _RESOURCE).write_text("wrong-layout", encoding="utf-8")
        # The adjacent ``weft`` source package is deliberately absent.
    else:
        package_origin = _write_source_layout(tmp_path)
        (tmp_path / _RESOURCE).unlink()

    with patch("weft.resources.importlib.resources.files", return_value=package_origin):
        with pytest.raises(ComposeResourceError, match="Unable to resolve"):
            with compose_file_path():
                pass


def test_missing_non_filesystem_resource_has_actionable_error(tmp_path: Path) -> None:
    """A missing installed/non-filesystem resource does not use checkout paths."""
    package = _TraversablePackage(
        _TraversableResource(tmp_path / _RESOURCE, present=False)
    )
    with patch("weft.resources.importlib.resources.files", return_value=package):
        with pytest.raises(
            ComposeResourceError,
            match="install package data or run from an anchored weft-memory checkout",
        ):
            with compose_file_path():
                pass


def test_missing_package_is_reported_as_resource_error() -> None:
    """An unavailable package produces the same actionable typed failure."""
    with patch(
        "weft.resources.importlib.resources.files",
        side_effect=ModuleNotFoundError("weft.data"),
    ):
        with pytest.raises(ComposeResourceError, match="weft.data is unavailable"):
            with compose_file_path():
                pass
