"""Regenerate tests/baseline_signatures.json from current module state.

Run after an intentional signature change on weft.store or weft.mcp.tools:

    uv run python tests/_record_baseline.py

Commit the updated JSON alongside the signature change so the additive-only
guard (tests/test_additive_guard.py) stays in sync.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

MODULES = ("weft.store", "weft.mcp.tools")
OUT_PATH = Path(__file__).parent / "baseline_signatures.json"

# When this file is executed directly, Python puts ``tests/`` ahead of the
# repository root on sys.path.  That makes the local ``tests/mcp`` directory
# shadow the installed MCP package while importing ``weft.mcp.tools``.
REPO_ROOT = OUT_PATH.parent.parent.resolve()
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
TESTS_DIR = OUT_PATH.parent.resolve()
while str(TESTS_DIR) in sys.path:
    sys.path.remove(str(TESTS_DIR))


def _public_funcs(module_name: str) -> dict[str, dict]:
    mod = importlib.import_module(module_name)
    funcs: dict[str, dict] = {}
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module_name:
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        funcs[name] = {
            "parameters": [
                {
                    "name": p.name,
                    "kind": p.kind.name,
                    "has_default": p.default is not inspect.Parameter.empty,
                }
                for p in sig.parameters.values()
            ]
        }
    return funcs


def main() -> None:
    baseline = {mod: _public_funcs(mod) for mod in MODULES}
    with OUT_PATH.open("w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")
    total = sum(len(v) for v in baseline.values())
    print(f"Wrote {OUT_PATH} ({total} functions across {len(MODULES)} modules)")


if __name__ == "__main__":
    main()
