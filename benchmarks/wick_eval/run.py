"""Anticipated Wick recall fixture loader + per-shape coverage report.

Usage: uv run python benchmarks/wick_eval/run.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "dataset.json"


def load_dataset() -> list[dict]:
    """Load the Wick recall fixture dataset."""
    with DATASET.open() as f:
        return json.load(f)


def coverage_report(rows: list[dict]) -> dict:
    """Compute per-shape and per-tier coverage statistics."""
    by_shape = Counter(r["shape"] for r in rows)
    by_tier = Counter(r["current_weft_tier"] for r in rows)
    belief_view_hits = sum(1 for r in rows if r["belief_view_addresses_this"])
    return {
        "n": len(rows),
        "by_shape": dict(by_shape),
        "by_tier": dict(by_tier),
        "belief_view_addresses": belief_view_hits,
        "belief_view_pct": round(belief_view_hits / len(rows), 3),
    }


def main() -> None:
    """Load fixture and report coverage."""
    rows = load_dataset()
    report = coverage_report(rows)
    print(json.dumps(report, indent=2))
    assert report["n"] >= 11, f"fixture below n=11 floor: {report['n']}"


if __name__ == "__main__":
    main()
