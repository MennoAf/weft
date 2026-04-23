"""Additive-only signature guard for single-user code paths.

Prevents silent regressions on public function signatures in ``weft.store``
and ``weft.mcp.tools``. Phase 1 froze the signatures that existing
single-user callers rely on; from here, changes must be strictly additive:

  - Every baseline parameter must still exist with the same name and kind.
  - A baseline parameter that had a default must still have one (value free
    to change, but it cannot become required).
  - New parameters are allowed anywhere, as long as they are keyword-only
    or carry a default (so existing callers don't break).
  - Removing a parameter, renaming it, changing its kind, or removing a
    default is a violation.

If a signature change is intentional (e.g. deprecation complete), regenerate
the baseline:

    uv run python tests/_record_baseline.py

and commit ``tests/baseline_signatures.json`` alongside the change.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

_BASELINE_PATH = Path(__file__).parent / "baseline_signatures.json"


def _load_baseline() -> dict[str, dict[str, dict]]:
    if not _BASELINE_PATH.exists():
        pytest.skip(
            "tests/baseline_signatures.json not found — regenerate via "
            "`uv run python tests/_record_baseline.py`"
        )
    with _BASELINE_PATH.open() as f:
        return json.load(f)


def _current_params(module_name: str, func_name: str) -> list[dict]:
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name, None)
    if func is None:
        return []
    sig = inspect.signature(func)
    return [
        {
            "name": p.name,
            "kind": p.kind.name,
            "has_default": p.default is not inspect.Parameter.empty,
        }
        for p in sig.parameters.values()
    ]


def _check_additive(module_name: str, func_name: str, baseline_params: list[dict]) -> None:
    """Assert current signature is an additive-only extension of baseline."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, func_name), (
        f"{module_name}.{func_name} was removed — guard forbids deletions"
    )

    current = _current_params(module_name, func_name)
    current_by_name = {p["name"]: p for p in current}

    for bp in baseline_params:
        name = bp["name"]
        assert name in current_by_name, (
            f"{module_name}.{func_name}: parameter '{name}' was removed or "
            f"renamed (baseline kind={bp['kind']})"
        )
        cp = current_by_name[name]
        assert cp["kind"] == bp["kind"], (
            f"{module_name}.{func_name}: parameter '{name}' changed kind "
            f"{bp['kind']} → {cp['kind']} (breaks call sites)"
        )
        if bp["has_default"]:
            assert cp["has_default"], (
                f"{module_name}.{func_name}: parameter '{name}' lost its "
                f"default — became required, breaks existing callers"
            )

    # New parameters must not appear as positional-only required params, or
    # they'll shift positions for existing callers.
    baseline_names = {p["name"] for p in baseline_params}
    for cp in current:
        if cp["name"] in baseline_names:
            continue
        # New parameter — must be keyword-only, *args/**kwargs, or have default.
        if cp["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD", "KEYWORD_ONLY"):
            continue
        assert cp["has_default"], (
            f"{module_name}.{func_name}: new parameter '{cp['name']}' is "
            f"required and positional — breaks existing callers. Make it "
            f"keyword-only or give it a default."
        )


def _collect_cases() -> list[tuple[str, str, list[dict]]]:
    baseline = _load_baseline()
    cases: list[tuple[str, str, list[dict]]] = []
    for module_name, funcs in baseline.items():
        for func_name, meta in funcs.items():
            cases.append((module_name, func_name, meta["parameters"]))
    return cases


_CASES = _collect_cases() if _BASELINE_PATH.exists() else []


@pytest.mark.parametrize(
    ("module_name", "func_name", "baseline_params"),
    _CASES,
    ids=[f"{m}.{f}" for m, f, _ in _CASES],
)
def test_signature_is_additive(module_name: str, func_name: str, baseline_params: list[dict]) -> None:
    """Each baseline function's current signature is additive-only vs baseline."""
    _check_additive(module_name, func_name, baseline_params)


def test_baseline_file_exists() -> None:
    """Baseline file must be present for the guard to have meaning."""
    assert _BASELINE_PATH.exists(), (
        f"baseline_signatures.json missing at {_BASELINE_PATH} — regenerate "
        f"via `uv run python tests/_record_baseline.py`"
    )


def test_no_functions_removed_from_store() -> None:
    """Explicit no-deletion check for weft.store (guard against the whole module changing shape)."""
    baseline = _load_baseline()
    mod = importlib.import_module("weft.store")
    for name in baseline.get("weft.store", {}):
        assert hasattr(mod, name), f"weft.store.{name} was removed"


def test_no_functions_removed_from_mcp_tools() -> None:
    """Explicit no-deletion check for weft.mcp.tools."""
    baseline = _load_baseline()
    mod = importlib.import_module("weft.mcp.tools")
    for name in baseline.get("weft.mcp.tools", {}):
        assert hasattr(mod, name), f"weft.mcp.tools.{name} was removed"
