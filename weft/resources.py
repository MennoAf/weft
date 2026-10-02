"""Access packaged runtime resources with a narrowly scoped source fallback."""

from __future__ import annotations

import importlib.resources
import tomllib
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator


_RESOURCE_NAME = "docker-compose.weft.yml"
_PROJECT_NAME = "weft-memory"


class ComposeResourceError(FileNotFoundError):
    """Raised when the Compose resource cannot be resolved safely."""


def _source_fallback(package_root: object) -> Path | None:
    """Return the authored checkout resource for an anchored source package.

    The package origin is the only source of the candidate root.  In
    particular, this function never consults the current directory, HOME, or
    arbitrary parent directories.
    """
    if not isinstance(package_root, Path):
        return None

    source_root = package_root.parent.parent
    expected_origin = source_root / "weft" / "data"
    if package_root != expected_origin:
        return None

    source_package = source_root / "weft"
    metadata_path = source_root / "pyproject.toml"
    compose_path = source_root / _RESOURCE_NAME

    if not source_package.is_dir() or not metadata_path.is_file():
        return None

    try:
        with metadata_path.open("rb") as metadata_file:
            metadata = tomllib.load(metadata_file)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    if metadata.get("project", {}).get("name") != _PROJECT_NAME:
        return None
    if not compose_path.is_file():
        return None
    return compose_path


def _missing_resource_error(package_root: object) -> ComposeResourceError:
    """Build an actionable error without exposing arbitrary filesystem data."""
    if isinstance(package_root, Path):
        detail = (
            "package data is absent and its filesystem origin is not an "
            "anchored weft-memory source checkout"
        )
    else:
        detail = "package data is absent from the installed/non-filesystem package"
    return ComposeResourceError(
        f"Unable to resolve {_RESOURCE_NAME!r}: {detail}; "
        "install package data or run from an anchored weft-memory checkout"
    )


@contextmanager
def compose_file_path() -> Iterator[Path]:
    """Yield the Compose path while any package materialization is alive.

    Package data is authoritative.  A source checkout is accepted only when
    the ``weft.data`` filesystem origin is adjacent to the expected project
    metadata, source package, and one authored root Compose file.  A yielded
    package path must not be retained after this context exits because
    ``importlib.resources.as_file`` may remove its temporary materialization.
    """
    try:
        package_root = importlib.resources.files("weft.data")
        resource = package_root.joinpath(_RESOURCE_NAME)
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        raise ComposeResourceError(
            f"Unable to resolve {_RESOURCE_NAME!r}: weft.data is unavailable; "
            "install package data or run from an anchored weft-memory checkout"
        ) from exc

    try:
        resource_present = resource.is_file()
    except OSError as exc:
        raise ComposeResourceError(
            f"Unable to inspect packaged {_RESOURCE_NAME!r}; "
            "verify package data is readable or run from an anchored "
            "weft-memory checkout"
        ) from exc

    if resource_present:
        stack = ExitStack()
        try:
            try:
                materialized = stack.enter_context(
                    importlib.resources.as_file(resource)
                )
            except (OSError, FileNotFoundError) as exc:
                raise ComposeResourceError(
                    f"Unable to materialize packaged {_RESOURCE_NAME!r}; "
                    "the package resource is unavailable"
                ) from exc
            yield Path(materialized)
        finally:
            stack.close()
        return

    source_path = _source_fallback(package_root)
    if source_path is not None:
        yield source_path
        return

    raise _missing_resource_error(package_root)
