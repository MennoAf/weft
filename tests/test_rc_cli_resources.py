"""Release-candidate proof for packaged Compose CLI wiring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from weft import cli


_RESOURCE_MEMBER = "weft/data/docker-compose.weft.yml"
_ROOT_MEMBER = "docker-compose.weft.yml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _build_wheel(output_dir: Path) -> Path:
    output_dir.mkdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels!r}"
    return wheels[0]


def _install_wheel_isolated(wheel: Path, root: Path) -> tuple[Path, Path]:
    """Install the new wheel and only needed CLI deps, entirely from uv cache."""
    venv = root / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    python = _venv_python(venv)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            "--no-deps",
            str(wheel),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    # These are the only direct imports needed to load weft.cli.  The wheel is
    # deliberately installed separately with --no-deps so proof distinguishes
    # the new artifact from its dependency environment.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            "click",
            "pydantic",
            "pyyaml",
            "python-dotenv",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return venv, python


def test_real_wheel_contains_only_package_resource_and_installs_isolated(tmp_path: Path) -> None:
    """The actual wheel works from a fresh cwd/HOME without checkout imports."""
    dist = tmp_path / "dist"
    wheel = _build_wheel(dist)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert _RESOURCE_MEMBER in names
        assert _ROOT_MEMBER not in names
        packaged = archive.read(_RESOURCE_MEMBER)
    assert packaged.startswith(b"services:")

    install_root = tmp_path / "install"
    install_root.mkdir()
    _venv, python = _install_wheel_isolated(wheel, install_root)
    isolated_cwd = tmp_path / "cwd"
    isolated_home = tmp_path / "home"
    isolated_cwd.mkdir()
    isolated_home.mkdir()
    script = textwrap.dedent(
        """
        import asyncio
        import os
        from contextlib import contextmanager
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch

        from click.testing import CliRunner
        import weft
        from weft import cli
        from weft.resources import compose_file_path

        repo = Path(os.environ["REPO_ROOT"]).resolve()
        assert repo not in Path(weft.__file__).resolve().parents
        with compose_file_path() as resource:
            assert resource.is_file()
            assert resource.read_bytes().startswith(b"services:")
            assert repo not in resource.resolve().parents

        def fake_run(argv, **kwargs):
            assert argv[:4] == ["docker", "compose", "-f", argv[3]]
            compose = Path(argv[3])
            assert compose.is_file()
            assert compose.read_bytes().startswith(b"services:")
            return MagicMock(returncode=0, stderr="")

        async def migrations(_config):
            return ["v1"]

        with (
            patch.object(cli.subprocess, "run", side_effect=fake_run),
            patch.object(cli, "load_config", return_value=object()),
            patch.object(cli, "_run_owner_migrations", side_effect=migrations),
            patch.object(cli, "_register_mcp"),
        ):
            runner = CliRunner()
            up_result = runner.invoke(cli.cli, ["up"])
            assert up_result.exit_code == 0, up_result.output
            down_result = runner.invoke(cli.cli, ["down"])
            assert down_result.exit_code == 0, down_result.output
        """
    )
    env = {"HOME": str(isolated_home), "REPO_ROOT": str(_repo_root())}
    installed = subprocess.run(
        [str(python), "-c", script],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr


@contextmanager
def _tracked_compose_context(path: Path, events: list[tuple[str, object]]):
    events.append(("enter", path))
    try:
        yield path
    finally:
        events.append(("exit", path))


def _result(returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stderr=stderr)


def test_up_uses_compose_context_and_preserves_order(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text("services:\n", encoding="utf-8")
    events: list[tuple[str, object]] = []

    def run(argv, **_kwargs):
        events.append(("docker", tuple(argv)))
        assert events[0] == ("enter", compose)
        assert Path(argv[3]) == compose
        return _result()

    async def migrations(_config):
        events.append(("migrations", None))
        assert events[-2][0] == "docker"
        return ["v1"]

    def register(_project_dir):
        events.append(("register", None))
        assert events[-2][0] == "migrations"

    with (
        patch.object(cli, "compose_file_path", lambda: _tracked_compose_context(compose, events)),
        patch.object(cli.subprocess, "run", side_effect=run),
        patch.object(cli, "load_config", return_value=object()),
        patch.object(cli, "_run_owner_migrations", side_effect=migrations),
        patch.object(cli, "_register_mcp", side_effect=register),
    ):
        result = CliRunner().invoke(cli.cli, ["up"])

    assert result.exit_code == 0, result.output
    assert [event[0] for event in events] == [
        "enter",
        "docker",
        "migrations",
        "register",
        "exit",
    ]


def test_up_preserves_error_and_closes_compose_context(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text("services:\n", encoding="utf-8")
    events: list[tuple[str, object]] = []

    with (
        patch.object(cli, "compose_file_path", lambda: _tracked_compose_context(compose, events)),
        patch.object(cli.subprocess, "run", return_value=_result(1, "docker failed")),
        patch.object(cli, "_run_owner_migrations", new_callable=AsyncMock) as migrations,
        patch.object(cli, "_register_mcp") as register,
    ):
        result = CliRunner().invoke(cli.cli, ["up"])

    assert result.exit_code == 1
    assert "Error: docker failed" in result.output
    assert [event[0] for event in events] == ["enter", "exit"]
    migrations.assert_not_called()
    register.assert_not_called()


def test_down_uses_compose_context_and_preserves_error_behavior(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text("services:\n", encoding="utf-8")
    events: list[tuple[str, object]] = []

    def run(argv, **_kwargs):
        events.append(("docker", tuple(argv)))
        assert events == [("enter", compose), ("docker", tuple(argv))]
        assert Path(argv[3]) == compose
        return _result()

    with (
        patch.object(cli, "compose_file_path", lambda: _tracked_compose_context(compose, events)),
        patch.object(cli.subprocess, "run", side_effect=run),
    ):
        result = CliRunner().invoke(cli.cli, ["down"])

    assert result.exit_code == 0, result.output
    assert [event[0] for event in events] == ["enter", "docker", "exit"]

    events.clear()
    with (
        patch.object(cli, "compose_file_path", lambda: _tracked_compose_context(compose, events)),
        patch.object(cli.subprocess, "run", return_value=_result(1, "cannot stop")),
    ):
        result = CliRunner().invoke(cli.cli, ["down"])

    assert result.exit_code == 1
    assert "Error: cannot stop" in result.output
    assert [event[0] for event in events] == ["enter", "exit"]


def test_missing_resource_error_is_actionable_for_both_commands() -> None:
    from weft.resources import ComposeResourceError

    error = ComposeResourceError(
        "Unable to resolve 'docker-compose.weft.yml': package data is absent"
    )
    with patch.object(cli, "compose_file_path", MagicMock(side_effect=error)):
        for command in ("up", "down"):
            result = CliRunner().invoke(cli.cli, [command])
            assert result.exit_code != 0
            assert result.exception is not None
            assert "Unable to resolve 'docker-compose.weft.yml'" in str(result.exception)
