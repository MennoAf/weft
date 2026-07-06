"""AST-based Python source scanner for the Capability Registry."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from capability_registry.models import CapabilityEntry


DEFAULT_INCLUDE_GLOBS = ["**/*.py"]
DEFAULT_EXCLUDE_DIRS = [
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
]


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest for ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_imports(tree: ast.Module) -> List[str]:
    """Extract sorted top-level import package names from an AST module."""
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return sorted(imports)


def extract_entries_from_file(
    repo_slug: str,
    repo_root: Path,
    file_path: Path,
) -> List[CapabilityEntry]:
    """Parse one Python file into module, function, and class entries."""
    file_hash = compute_file_hash(file_path)
    relative_path = str(file_path.relative_to(repo_root))
    text = file_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [
            CapabilityEntry(
                repo_slug=repo_slug,
                file_path=relative_path,
                symbol_kind="unparseable",
                docstring=str(exc),
                file_hash=file_hash,
            )
        ]

    imports = extract_imports(tree)
    entries: list[CapabilityEntry] = [
        CapabilityEntry(
            repo_slug=repo_slug,
            file_path=relative_path,
            symbol_kind="module",
            docstring=ast.get_docstring(tree),
            imports=imports,
            file_hash=file_hash,
        )
    ]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(
                CapabilityEntry(
                    repo_slug=repo_slug,
                    file_path=relative_path,
                    symbol_name=node.name,
                    symbol_kind="function",
                    docstring=ast.get_docstring(node),
                    imports=imports,
                    file_hash=file_hash,
                )
            )
        elif isinstance(node, ast.ClassDef):
            entries.append(
                CapabilityEntry(
                    repo_slug=repo_slug,
                    file_path=relative_path,
                    symbol_name=node.name,
                    symbol_kind="class",
                    docstring=ast.get_docstring(node),
                    imports=imports,
                    file_hash=file_hash,
                )
            )

    return entries


def scan_repo(
    repo_slug: str,
    repo_root: Path,
    include_globs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> List[CapabilityEntry]:
    """Scan Python files under ``repo_root`` and return capability entries."""
    include_globs = include_globs or DEFAULT_INCLUDE_GLOBS
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS

    file_paths: set[Path] = set()
    for pattern in include_globs:
        file_paths.update(path for path in repo_root.rglob(pattern) if path.is_file())

    entries: list[CapabilityEntry] = []
    for file_path in sorted(file_paths):
        if any(parent.name in exclude_dirs for parent in file_path.parents):
            continue
        entries.extend(extract_entries_from_file(repo_slug, repo_root, file_path))
    return entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a Python repo for capability entries.")
    parser.add_argument("--repo-slug", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entries = scan_repo(args.repo_slug, args.repo_root)
    if args.json:
        print(json.dumps([dataclasses.asdict(entry) for entry in entries], indent=2))
        return

    file_count = len({entry.file_path for entry in entries})
    print(f"Scanned {file_count} files, extracted {len(entries)} entries.")


if __name__ == "__main__":
    main()
