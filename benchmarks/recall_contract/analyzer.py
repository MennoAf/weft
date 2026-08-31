"""Read-only artifact analysis with complete-denominator and input hashing gates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifact_io import sha256_file
from .funnel import TerminalFunnel, aggregate_metrics


def analyze_artifacts(funnel: TerminalFunnel, inputs: Mapping[str, Path | str]) -> dict[str, Any]:
    """Return metrics plus hashes of every declared input; never writes files.

    Hashes are computed both before and after metric aggregation.  A changed
    input means the analyzer observed a moving source and fails closed.
    """
    before = {name: sha256_file(path) for name, path in sorted(inputs.items())}
    metrics = aggregate_metrics(funnel)
    after = {name: sha256_file(path) for name, path in sorted(inputs.items())}
    if before != after:
        raise RuntimeError("input mutated during read-only analysis")
    return {
        "analyzer_version": "recall-artifact-analyzer-v1",
        "complete_denominator": funnel.complete_denominator,
        "input_hashes": before,
        "metrics": metrics,
        "read_only": True,
    }


def diagnose_gap(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Give a stage-aware, honest diagnosis without inferring semantic lift."""
    metrics = dict(analysis.get("metrics", {}))
    if not analysis.get("complete_denominator", False):
        return {"status": "blocked", "primary_gap": "incomplete_denominator", "claimable": False}
    coverage = float(metrics.get("oracle_coverage", 0.0))
    losses = metrics.get("primary_loss", {})
    diagnosis = {
        "status": "measured_gap",
        "primary_gap": None,
        "loss_counts": dict(losses),
        "claimable": coverage >= 1.0,
        "observed_explanation": (
            "The 25-case deterministic fixture demonstrates contract behavior on explicit stable identities; "
            "natural-prose replay evidence must remain structurally unevaluated unless identity and completeness "
            "oracles exist. Neither source alone proves general semantic lift."
        ),
    }
    non_none = {key: value for key, value in losses.items() if key != "none"}
    if non_none:
        diagnosis["primary_gap"] = max(non_none, key=non_none.get)
    else:
        diagnosis["status"] = "no_gap_observed"
    return diagnosis
