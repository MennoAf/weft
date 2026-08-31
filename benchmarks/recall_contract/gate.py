"""Fail-closed promotion decisions for benchmark artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .funnel import TerminalFunnel, aggregate_metrics


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    allowed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def evaluate_promotion(
    funnel: TerminalFunnel,
    *,
    min_oracle_coverage: float = 1.0,
    max_primary_loss: float = 0.0,
    require_provider_free: bool = True,
) -> PromotionDecision:
    """Evaluate whether a terminal funnel can be promoted to a claim-bearing run."""
    reasons: list[str] = []
    try:
        metrics = aggregate_metrics(funnel)
    except ValueError as exc:
        return PromotionDecision(False, (f"invalid_funnel: {exc}",), {})
    if not funnel.complete_denominator:
        reasons.append("incomplete_denominator")
    if funnel.oracle_coverage < min_oracle_coverage:
        reasons.append("oracle_coverage_below_gate")
    loss_count = sum(value for key, value in metrics["primary_loss"].items() if key != "none")
    denominator = max(1, metrics["denominator"])
    if loss_count / denominator > max_primary_loss:
        reasons.append("primary_loss_above_gate")
    if require_provider_free and any(
        int(row.budgets.get("provider_calls", row.scores.get("provider_calls", 0)) or 0) > 0
        for row in funnel.rows
    ):
        reasons.append("provider_calls_present")
    return PromotionDecision(not reasons, tuple(reasons), metrics)
