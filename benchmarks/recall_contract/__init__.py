"""Fail-closed, provider-free contracts for recall benchmark artifacts."""

from .analyzer import analyze_artifacts, diagnose_gap
from .artifact_io import allocate_new_output, publish_json_atomic, sha256_file
from .funnel import (
    FUNNEL_VERSION,
    REQUIRED_STAGES,
    StageRecord,
    TerminalFunnel,
    TerminalRow,
    aggregate_metrics,
)
from .gate import evaluate_promotion
from .safety import classify_command, run_autonomous

__all__ = [
    "FUNNEL_VERSION",
    "REQUIRED_STAGES",
    "StageRecord",
    "TerminalFunnel",
    "TerminalRow",
    "aggregate_metrics",
    "allocate_new_output",
    "publish_json_atomic",
    "sha256_file",
    "analyze_artifacts",
    "diagnose_gap",
    "evaluate_promotion",
    "classify_command",
    "run_autonomous",
]
