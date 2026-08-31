"""Autonomous benchmark command safety: offline, bounded, and fail closed."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

MAX_SECONDS = 300
_PROVIDER_ENV_MARKERS = (
    "API_KEY", "API_TOKEN", "ANTHROPIC", "OPENAI", "GEMINI", "COHERE",
    "DATABASE_URL", "DATABASE_PASSWORD", "LONGMEMEVAL_DATABASE",
)
_DENIED_TERMS = (
    "provider", "judge", "evaluate_qa", "detached", "nohup", "screen",
    "tmux", "&",
)
_DENIED_LIVE_MODULES = (
    "benchmarks.longmemeval.opt_n",
    "benchmarks.longmemeval.adapter",
    "benchmarks.longmemeval.judge",
    "benchmarks.opt_n.live",
    "run_detached",
)
_ALLOWED_PREFIXES = (
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("uv", "run", "pytest"),
)
_ARTIFACT_REPLAY_PREFIXES = (
    ("python", "-m", "benchmarks.compositional_recall.replay_longmemeval"),
    ("python3", "-m", "benchmarks.compositional_recall.replay_longmemeval"),
    ("uv", "run", "python", "-m", "benchmarks.compositional_recall.replay_longmemeval"),
)
_ARTIFACT_REPORT_PREFIXES = (
    ("python", "-m", "benchmarks.recall_contract.report"),
    ("python3", "-m", "benchmarks.recall_contract.report"),
    ("uv", "run", "python", "-m", "benchmarks.recall_contract.report"),
)


@dataclass(frozen=True, slots=True)
class CommandDecision:
    allowed: bool
    reason: str
    argv: tuple[str, ...]


def _has_provider_env(env: Mapping[str, str]) -> bool:
    return any(any(marker in key.upper() for marker in _PROVIDER_ENV_MARKERS) for key in env)


def classify_command(command: Sequence[str] | str, *, env: Mapping[str, str] | None = None) -> CommandDecision:
    argv = tuple(command.split()) if isinstance(command, str) else tuple(str(item) for item in command)
    if not argv:
        return CommandDecision(False, "unclassified command", argv)
    lowered = " ".join(argv).casefold()
    if any(term in lowered for term in _DENIED_TERMS) or any(module in lowered for module in _DENIED_LIVE_MODULES):
        return CommandDecision(False, "provider/judge/live benchmark/detached command denied", argv)
    if env is not None and _has_provider_env(env):
        return CommandDecision(False, "provider credential environment denied", argv)
    executable = argv[0].rsplit("/", 1)[-1]
    normalized = (executable,) + argv[1:]
    if any(normalized[: len(prefix)] == prefix for prefix in _ARTIFACT_REPLAY_PREFIXES):
        if "--dataset" not in argv or "--output" not in argv:
            return CommandDecision(False, "artifact replay requires explicit dataset and output", argv)
        return CommandDecision(True, "offline artifact-only replay command", argv)
    if any(normalized[: len(prefix)] == prefix for prefix in _ARTIFACT_REPORT_PREFIXES):
        required = {"--compositional-run", "--replay", "--replay-manifest", "--dataset", "--snapshot-manifest", "--complete-marker", "--failing-report", "--output"}
        if not required.issubset(argv):
            return CommandDecision(False, "artifact report requires explicit frozen inputs and output", argv)
        return CommandDecision(True, "offline artifact-only diagnosis command", argv)
    if not any(normalized[: len(prefix)] == prefix for prefix in _ALLOWED_PREFIXES):
        return CommandDecision(False, "unclassified command", argv)
    return CommandDecision(True, "offline provider-free command", argv)


def run_autonomous(command: Sequence[str] | str, *, env: Mapping[str, str] | None = None, timeout: int = MAX_SECONDS, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run an allowed command, capped at 300 seconds and never detached."""
    decision = classify_command(command, env=env)
    if not decision.allowed:
        raise PermissionError(decision.reason)
    if timeout < 1 or timeout > MAX_SECONDS:
        raise ValueError(f"timeout must be between 1 and {MAX_SECONDS} seconds")
    clean_env = {key: value for key, value in dict(env or os.environ).items() if not any(marker in key.upper() for marker in _PROVIDER_ENV_MARKERS)}
    return subprocess.run(decision.argv, cwd=cwd, env=clean_env, text=True, capture_output=True, timeout=min(timeout, MAX_SECONDS), check=False)
