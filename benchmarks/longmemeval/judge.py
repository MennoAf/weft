"""Wrapper around LongMemEval's upstream judge (``evaluate_qa.py``).

The upstream evaluator lives in the cloned LongMemEval repo, calls OpenAI
gpt-4o once per question to produce a yes/no autoeval label, and writes
results to ``<hyp_file>.eval-results-<model>``. This module:

  * Locates the LongMemEval clone (``LONGMEMEVAL_PATH`` env, or the default
    sibling-of-Weft layout used in the README).
  * Runs the evaluator in an ephemeral ``uv run --with`` env so the
    upstream deps (``openai``, ``backoff``, ``tqdm``) don't have to be
    pinned into Weft's pyproject.
  * After scoring, parses the result JSONL and emits a per-question-type
    breakdown summary as ``<hyp_file>.metrics.json`` next to the upstream
    output. This is the file we actually compare across runs.

Cost: roughly $5–15 in OpenAI judge calls for a 500-question oracle run
(gpt-4o-2024-08-06). The wrapper does NOT call OpenAI itself; it just
builds the subprocess invocation. Spend lives in the user's OpenAI account.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o"
SUPPORTED_MODELS = ("gpt-4o", "gpt-4o-mini")


# ----------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------


def _resolve_longmemeval_root() -> Path:
    """Find the LongMemEval clone.

    Resolution order:
      1. ``LONGMEMEVAL_PATH`` env var (absolute or relative to cwd).
      2. ``../LongMemEval`` relative to the Weft repo root (README default).
      3. The path used by the existing benchmark stats files
         (``Personal/langchain/LongMemEval``).
    """
    env = os.environ.get("LONGMEMEVAL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "src" / "evaluation" / "evaluate_qa.py").exists():
            return p
        raise FileNotFoundError(
            f"LONGMEMEVAL_PATH={env!r} does not contain "
            f"src/evaluation/evaluate_qa.py"
        )

    # Weft repo root is two levels up from this file (benchmarks/longmemeval).
    weft_root = Path(__file__).resolve().parents[2]
    candidates = [
        weft_root.parent / "LongMemEval",
        Path.home() / "Documents/code_projects/Personal/langchain/LongMemEval",
    ]
    for c in candidates:
        if (c / "src" / "evaluation" / "evaluate_qa.py").exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate the LongMemEval clone. Set LONGMEMEVAL_PATH or "
        "clone it as a sibling of the Weft repo:\n"
        "  git clone https://github.com/xiaowu0162/LongMemEval ../LongMemEval"
    )


def _default_ref_for(hyp_path: Path, longmemeval_root: Path) -> Path:
    """Infer the reference dataset from the hypothesis filename.

    Hypothesis files follow ``<split>_<mode>[_<filter-slug>]_<timestamp>.jsonl``
    (see adapter.py). The split name is everything before the mode marker
    (``raw`` / ``extracted``); whatever sits between the mode and the
    trailing timestamp (a question-type slug, ``filteredN``, etc.) is
    metadata about the run, not part of the split name.
    """
    # Layouts handled:
    #   longmemeval_oracle_extracted_20260430T204018Z.jsonl
    #   longmemeval_oracle_extracted_multi-session_20260502T233202Z.jsonl
    #   longmemeval_oracle_turns_tier-turns_filtered4_20260504T015042Z.jsonl
    stem = hyp_path.stem  # drops .jsonl
    parts = stem.split("_")
    mode_idx = next(
        (i for i, p in enumerate(parts) if p in ("raw", "extracted", "turns")), -1,
    )
    if mode_idx <= 0:
        raise ValueError(
            f"hypothesis filename {hyp_path.name!r} does not match the "
            f"<split>_<mode>[_<filter>]_<timestamp> convention; "
            f"pass --ref explicitly"
        )
    split = "_".join(parts[:mode_idx])
    ref = longmemeval_root / "data" / f"{split}.json"
    if not ref.exists():
        raise FileNotFoundError(
            f"Could not infer reference dataset from {hyp_path.name!r}. "
            f"Tried {ref}. Pass --ref explicitly."
        )
    return ref


# ----------------------------------------------------------------------
# Subprocess invocation
# ----------------------------------------------------------------------


def _build_command(
    *,
    longmemeval_root: Path,
    hyp_path: Path,
    ref_path: Path,
    model: str,
) -> list[str]:
    """Construct the argv for the upstream judge.

    We use ``uv run --with`` to layer ``openai``, ``backoff``, ``tqdm``,
    and ``numpy`` on top of whatever interpreter ``uv`` selects. Working
    directory for the subprocess is the LongMemEval root so its relative
    imports resolve. Paths in argv are absolute.
    """
    if not shutil.which("uv"):
        raise RuntimeError(
            "uv is required to run the judge wrapper "
            "(installs openai/backoff into an ephemeral env)."
        )
    evaluator = longmemeval_root / "src" / "evaluation" / "evaluate_qa.py"
    return [
        "uv", "run",
        "--with", "openai",
        "--with", "backoff",
        "--with", "tqdm",
        "--with", "numpy",
        "python", str(evaluator),
        model,
        str(hyp_path.resolve()),
        str(ref_path.resolve()),
    ]


def _result_path_for(hyp_path: Path, model: str) -> Path:
    """Path the upstream evaluator writes its labelled JSONL to."""
    return hyp_path.with_suffix(hyp_path.suffix + f".eval-results-{model}")


# ----------------------------------------------------------------------
# Per-type metrics
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeMetrics:
    """Pass-rate for one question type."""

    question_type: str
    n: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0


def summarize_results(
    *,
    result_path: Path,
    ref_path: Path,
) -> dict:
    """Build a per-question-type metrics summary from a labelled result file.

    The upstream ``print_qa_metrics.py`` prints to stdout and is hardcoded
    to gpt-4o-2024-08-06. We re-implement a tiny version here that:
      * works for any judge model
      * returns structured JSON instead of stdout-only
      * preserves the per-type accuracy breakdown that we actually care
        about for tracking improvements over time
    """
    qid2type: dict[str, str] = {}
    with ref_path.open(encoding="utf-8") as f:
        for entry in json.load(f):
            qid2type[entry["question_id"]] = entry["question_type"]

    by_type: dict[str, list[bool]] = defaultdict(list)
    with result_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry["question_id"]
            qtype = qid2type.get(qid, "unknown")
            label = entry.get("autoeval_label", {}).get("label")
            by_type[qtype].append(bool(label))

    metrics = []
    total_n = 0
    total_correct = 0
    for qtype in sorted(by_type):
        labels = by_type[qtype]
        m = TypeMetrics(
            question_type=qtype,
            n=len(labels),
            n_correct=sum(1 for x in labels if x),
        )
        metrics.append(
            {
                "question_type": m.question_type,
                "n": m.n,
                "n_correct": m.n_correct,
                "accuracy": round(m.accuracy, 4),
            }
        )
        total_n += m.n
        total_correct += m.n_correct

    overall_acc = total_correct / total_n if total_n else 0.0
    # Task-averaged accuracy treats each type equally — the headline metric
    # the LongMemEval paper reports.
    task_avg = (
        sum(m["accuracy"] for m in metrics) / len(metrics) if metrics else 0.0
    )
    return {
        "result_file": str(result_path),
        "ref_file": str(ref_path),
        "overall_accuracy": round(overall_acc, 4),
        "task_averaged_accuracy": round(task_avg, 4),
        "n_total": total_n,
        "n_correct_total": total_correct,
        "by_type": metrics,
    }


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def run_judge(
    *,
    hyp_path: Path,
    ref_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    longmemeval_root: Path | None = None,
    skip_if_exists: bool = True,
) -> dict:
    """Run the upstream judge against a hypothesis file and summarize.

    Args:
        hyp_path: Path to the ``{question_id, hypothesis}`` JSONL produced
            by ``adapter.py``.
        ref_path: LongMemEval reference dataset. Inferred from hyp filename
            if omitted.
        model: Judge model. ``gpt-4o`` matches the LongMemEval paper.
        longmemeval_root: Override the LongMemEval clone location.
        skip_if_exists: If True and the result file already exists with a
            non-zero size, skip the OpenAI call and just summarize.

    Returns:
        The metrics dict written to ``<hyp_path>.metrics.json``.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported judge model {model!r}; pick one of {SUPPORTED_MODELS}"
        )
    if not hyp_path.exists():
        raise FileNotFoundError(f"hypothesis file not found: {hyp_path}")

    root = longmemeval_root or _resolve_longmemeval_root()
    ref = ref_path or _default_ref_for(hyp_path, root)
    if not ref.exists():
        raise FileNotFoundError(f"reference dataset not found: {ref}")

    result_path = _result_path_for(hyp_path, model)
    if skip_if_exists and result_path.exists() and result_path.stat().st_size > 0:
        logger.info("reusing existing labelled output: %s", result_path)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set — required for the OpenAI judge."
            )
        cmd = _build_command(
            longmemeval_root=root, hyp_path=hyp_path, ref_path=ref, model=model,
        )
        logger.info("running upstream judge: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)

    metrics = summarize_results(result_path=result_path, ref_path=ref)
    metrics_path = hyp_path.with_suffix(hyp_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("wrote metrics summary: %s", metrics_path)
    return metrics


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


@click.command()
@click.option(
    "--hyp",
    "hyp_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Hypothesis JSONL file from adapter.py.",
)
@click.option(
    "--ref",
    "ref_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="LongMemEval reference dataset. Inferred from --hyp filename if omitted.",
)
@click.option(
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Judge model.",
)
@click.option(
    "--longmemeval-path",
    "longmemeval_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override the LongMemEval clone location (else $LONGMEMEVAL_PATH or sibling-of-Weft).",
)
@click.option(
    "--rerun",
    is_flag=True,
    default=False,
    help="Re-call the OpenAI judge even if a labelled result file already exists.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def cli(
    hyp_path: Path,
    ref_path: Path | None,
    model: str,
    longmemeval_root: Path | None,
    rerun: bool,
    log_level: str,
) -> None:
    """Score a Weft hypothesis file with the LongMemEval upstream judge."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Mirror adapter.py: load standard Weft .env locations so OPENAI_API_KEY
    # in ~/.weft/.env "just works" without per-invocation prefixing.
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")
    try:
        metrics = run_judge(
            hyp_path=hyp_path,
            ref_path=ref_path,
            model=model,
            longmemeval_root=longmemeval_root,
            skip_if_exists=not rerun,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    click.echo("\nLongMemEval results:")
    click.echo(f"  overall accuracy:        {metrics['overall_accuracy']:.4f}")
    click.echo(f"  task-averaged accuracy:  {metrics['task_averaged_accuracy']:.4f}")
    click.echo(f"  n={metrics['n_total']} correct={metrics['n_correct_total']}")
    click.echo("")
    click.echo("  by question type:")
    for row in metrics["by_type"]:
        click.echo(
            f"    {row['question_type']:<28} "
            f"{row['n_correct']:>4}/{row['n']:<4}  "
            f"{row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    cli()
