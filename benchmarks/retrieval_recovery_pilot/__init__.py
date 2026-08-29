"""Provider-free paired retrieval-recovery pilot evaluator."""

from .adapter import FrozenSnapshot, LivePilotAdapter, PilotEmbedding, SnapshotNamespace
from .evaluator import (
    PILOT_VERSION,
    Arm,
    ArmMetrics,
    FrozenCase,
    PilotReport,
    RecallResult,
    RecoveryPilot,
    Scope,
    evaluate_cases,
    load_cases,
    write_report,
)

__all__ = [
    "PILOT_VERSION",
    "Arm",
    "ArmMetrics",
    "FrozenCase",
    "PilotReport",
    "RecallResult",
    "RecoveryPilot",
    "Scope",
    "evaluate_cases",
    "load_cases",
    "write_report",
    "FrozenSnapshot",
    "LivePilotAdapter",
    "PilotEmbedding",
    "SnapshotNamespace",
]
