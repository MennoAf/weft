"""Safe allocation and atomic publication for benchmark outputs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class OutputAllocationError(ValueError):
    """Raised when an output would overwrite or contaminate its source."""


def allocate_new_output(source: Path | str, destination: Path | str) -> Path:
    """Validate and reserve a new destination path.

    ``source`` and ``destination`` are resolved before comparison.  A
    destination may not exist, equal the source, or be nested below the source
    directory.  The parent is created, but the destination itself is never
    created or overwritten; callers publish it atomically afterwards.
    """
    source_path = Path(source).expanduser().resolve(strict=False)
    destination_path = Path(destination).expanduser().resolve(strict=False)
    if destination_path == source_path:
        raise OutputAllocationError("destination equals source")
    if source_path in destination_path.parents:
        raise OutputAllocationError("destination is nested under source")
    if destination_path.exists():
        raise FileExistsError(f"output already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    return destination_path


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def publish_json_atomic(destination: Path | str, payload: Any) -> Path:
    """Publish JSON atomically without replacing an existing destination."""
    target = Path(destination).expanduser().resolve(strict=False)
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        # link() is atomic and, unlike replace(), refuses a concurrent winner.
        os.link(temporary, target)
        os.unlink(temporary)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
