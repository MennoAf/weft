"""Tests for Capability Registry Python source scanning."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap

from capability_registry.models import CapabilityEntry
from capability_registry.scanner import (
    compute_file_hash,
    extract_entries_from_file,
    scan_repo,
)
from capability_registry.slug_classifier import classify_entries


def make_temp_py_file(content: str) -> tuple[Path, Path]:
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / "sample.py"
    file_path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return temp_dir, file_path


def test_extract_module_docstring() -> None:
    temp_dir, file_path = make_temp_py_file(
        '''
        """A bot-block hardening module."""

        def helper():
            """Function docstring."""
            return None
        '''
    )
    try:
        entries = extract_entries_from_file("test-repo", temp_dir, file_path)
    finally:
        shutil.rmtree(temp_dir)

    module_entry = next(entry for entry in entries if entry.symbol_kind == "module")
    assert module_entry.docstring is not None
    assert "bot-block" in module_entry.docstring


def test_extract_function_entries() -> None:
    temp_dir, file_path = make_temp_py_file(
        """
        def foo():
            pass

        def bar():
            pass
        """
    )
    try:
        entries = extract_entries_from_file("test-repo", temp_dir, file_path)
    finally:
        shutil.rmtree(temp_dir)

    assert len(entries) == 3
    assert [entry.symbol_name for entry in entries if entry.symbol_kind == "function"] == [
        "foo",
        "bar",
    ]


def test_extract_class_entries() -> None:
    temp_dir, file_path = make_temp_py_file(
        '''
        class LazyEscalationPolicy:
            """Escalates only after cheap retries fail."""
        '''
    )
    try:
        entries = extract_entries_from_file("test-repo", temp_dir, file_path)
    finally:
        shutil.rmtree(temp_dir)

    assert any(
        entry.symbol_name == "LazyEscalationPolicy" and entry.symbol_kind == "class"
        for entry in entries
    )


def test_extract_imports() -> None:
    temp_dir, file_path = make_temp_py_file(
        """
        import requests
        from time import sleep

        def helper():
            pass
        """
    )
    try:
        entries = extract_entries_from_file("test-repo", temp_dir, file_path)
    finally:
        shutil.rmtree(temp_dir)

    module_entry = next(entry for entry in entries if entry.symbol_kind == "module")
    assert module_entry.imports == ["requests", "time"]


def test_file_hash_is_sha256() -> None:
    temp_dir, file_path = make_temp_py_file("VALUE = 1\n")
    try:
        file_hash = compute_file_hash(file_path)
    finally:
        shutil.rmtree(temp_dir)

    assert len(file_hash) == 64
    assert all(char in "0123456789abcdef" for char in file_hash)


def test_syntax_error_file() -> None:
    temp_dir, file_path = make_temp_py_file("def foo(: pass")
    try:
        entries = extract_entries_from_file("test-repo", temp_dir, file_path)
    finally:
        shutil.rmtree(temp_dir)

    assert len(entries) == 1
    assert entries[0].symbol_kind == "unparseable"
    assert entries[0].docstring is not None


def test_slug_classifier_bot_block() -> None:
    entry = CapabilityEntry(
        repo_slug="test-repo",
        file_path="sample.py",
        docstring="bot block detection",
    )

    classify_entries([entry])

    assert "bot-block-hardening" in entry.capability_slugs


def test_scan_repo_finds_py_files() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / "one.py").write_text("ONE = 1\n", encoding="utf-8")
        (temp_dir / "two.py").write_text("TWO = 2\n", encoding="utf-8")

        entries = scan_repo("test-repo", temp_dir)
    finally:
        shutil.rmtree(temp_dir)

    module_paths = {entry.file_path for entry in entries if entry.symbol_kind == "module"}
    assert {"one.py", "two.py"}.issubset(module_paths)


def test_scan_repo_excludes_venv() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / "root.py").write_text("ROOT = 1\n", encoding="utf-8")
        venv_dir = temp_dir / ".venv"
        venv_dir.mkdir()
        (venv_dir / "ignored.py").write_text("IGNORED = 1\n", encoding="utf-8")

        entries = scan_repo("test-repo", temp_dir)
    finally:
        shutil.rmtree(temp_dir)

    assert all(".venv" not in entry.file_path for entry in entries)
    assert any(entry.file_path == "root.py" for entry in entries)


def test_scan_repo_is_deterministic() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / "b.py").write_text("B = 1\n", encoding="utf-8")
        (temp_dir / "a.py").write_text("A = 1\n", encoding="utf-8")

        first = scan_repo("test-repo", temp_dir)
        second = scan_repo("test-repo", temp_dir)
    finally:
        shutil.rmtree(temp_dir)

    assert first == second


def _run_main() -> int:
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                test_fn()
            except Exception as exc:  # pragma: no cover - manual runner
                failures += 1
                print(f"FAIL {name}: {exc}")
            else:
                print(f"PASS {name}")
    return failures


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(Path.cwd()))
    sys.exit(_run_main())
