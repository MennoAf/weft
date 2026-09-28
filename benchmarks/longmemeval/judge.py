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
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o"
SUPPORTED_MODELS = ("gpt-4o", "gpt-4o-mini")
# Preserved evaluator model aliases (source/evaluation/evaluate_qa.py).
_MODEL_ZOO = {
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
}


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


def _load_jsonl(path: Path, *, artifact: str) -> list[dict]:
    """Load a JSONL artifact, rejecting non-object rows."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{artifact} row {line_number} must be a JSON object"
                )
            rows.append(row)
    return rows


def _index_unique(
    rows: list[dict],
    *,
    artifact: str,
    allowed_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Index rows by question_id and enforce artifact-boundary integrity."""
    indexed: dict[str, dict] = {}
    for row_number, row in enumerate(rows, start=1):
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(
                f"{artifact} row {row_number} has missing/invalid question_id"
            )
        if qid in indexed:
            raise ValueError(f"duplicate {artifact} question_id: {qid}")
        if allowed_ids is not None and qid not in allowed_ids:
            raise ValueError(f"unknown {artifact} question_id: {qid}")
        indexed[qid] = row
    return indexed


def _result_label(row: dict, *, qid: str) -> bool:
    """Read current nested or legacy boolean judge labels strictly."""
    raw_label = row.get("autoeval_label")
    if isinstance(raw_label, dict):
        raw_label = raw_label.get("label")
    if not isinstance(raw_label, bool):
        raise ValueError(f"missing/malformed autoeval label for question_id: {qid}")
    return raw_label


def summarize_pipeline(
    *,
    ref_path: Path,
    hyp_path: Path,
    result_path: Path,
) -> dict:
    """Summarize the complete reference → hypothesis → judge pipeline.

    The reference population is always the denominator. Missing hypotheses and
    missing judge results therefore count as incorrect, while remaining visible
    as distinct pipeline failures. Duplicate IDs and IDs absent from the
    preceding artifact boundary fail loudly rather than corrupting metrics.
    """
    with ref_path.open(encoding="utf-8") as f:
        ref_rows = json.load(f)
    if not isinstance(ref_rows, list):
        raise ValueError("reference artifact must be a JSON array")

    refs = _index_unique(ref_rows, artifact="reference")
    hypotheses = _index_unique(
        _load_jsonl(hyp_path, artifact="hypothesis"),
        artifact="hypothesis",
        allowed_ids=set(refs),
    )
    results = _index_unique(
        _load_jsonl(result_path, artifact="result"),
        artifact="result",
        allowed_ids=set(hypotheses),
    )
    labels = {qid: _result_label(row, qid=qid) for qid, row in results.items()}

    expected_by_type: dict[str, list[str]] = defaultdict(list)
    for qid, row in refs.items():
        qtype = row.get("question_type")
        if not isinstance(qtype, str) or not qtype:
            raise ValueError(f"reference question {qid} has invalid question_type")
        expected_by_type[qtype].append(qid)

    metrics: list[dict] = []
    raw_type_accuracies: list[float] = []
    total_correct = 0
    for qtype in sorted(expected_by_type):
        qids = expected_by_type[qtype]
        correct = sum(1 for qid in qids if labels.get(qid) is True)
        produced_hypotheses = sum(1 for qid in qids if qid in hypotheses)
        produced_results = sum(1 for qid in qids if qid in results)
        m = TypeMetrics(qtype, len(qids), correct)
        raw_type_accuracies.append(m.accuracy)
        total_correct += correct
        metrics.append(
            {
                "question_type": qtype,
                "n": m.n,
                "n_correct": correct,
                "accuracy": round(m.accuracy, 4),
                "hypotheses_produced": produced_hypotheses,
                "judge_results_produced": produced_results,
                "missing_hypotheses": m.n - produced_hypotheses,
                "hypotheses_without_judge_results": (
                    produced_hypotheses - produced_results
                ),
            }
        )

    expected = len(refs)
    missing_hypotheses = expected - len(hypotheses)
    missing_results = len(hypotheses) - len(results)
    return {
        # Metrics may be published. Keep host usernames/mount layouts out of
        # the artifact while preserving stable sidecar identity.
        "result_file": result_path.name,
        "hypothesis_file": hyp_path.name,
        "ref_file": ref_path.name,
        "paths_redacted": True,
        "overall_accuracy": round(total_correct / expected, 4) if expected else 0.0,
        "task_averaged_accuracy": (
            round(sum(raw_type_accuracies) / len(raw_type_accuracies), 4)
            if raw_type_accuracies else 0.0
        ),
        "n_total": expected,
        "n_correct_total": total_correct,
        "expected_questions": expected,
        "hypotheses_produced": len(hypotheses),
        "judge_results_produced": len(results),
        "missing_hypotheses": missing_hypotheses,
        "hypotheses_without_judge_results": missing_results,
        "complete": missing_hypotheses == 0 and missing_results == 0,
        "boundary_validation": {
            "duplicate_reference_ids": 0,
            "duplicate_hypothesis_ids": 0,
            "duplicate_result_ids": 0,
            "unknown_hypothesis_ids": 0,
            "unknown_result_ids": 0,
        },
        "by_type": metrics,
    }


def summarize_results(*, result_path: Path, ref_path: Path) -> dict:
    """Compatibility wrapper for legacy two-artifact callers.

    This cannot distinguish a missing hypothesis from a missing judge result,
    so it treats every labelled result as a produced hypothesis. New callers
    must use :func:`summarize_pipeline`.
    """
    result_rows = _load_jsonl(result_path, artifact="result")
    compatibility_hyp_path = result_path.with_suffix(result_path.suffix + ".hyp.tmp")
    try:
        with compatibility_hyp_path.open("w", encoding="utf-8") as f:
            for row in result_rows:
                f.write(json.dumps({"question_id": row.get("question_id")}) + "\n")
        return summarize_pipeline(
            ref_path=ref_path,
            hyp_path=compatibility_hyp_path,
            result_path=result_path,
        )
    finally:
        compatibility_hyp_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def _official_prompt_loader(source_root: Path):
    """Load the preserved evaluator's prompt function without changing it."""
    source = source_root / "src" / "evaluation" / "evaluate_qa.py"
    if not source.is_file():
        raise FileNotFoundError(f"preserved evaluator not found: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_anscheck_prompt"),
        None,
    )
    if function is None:
        raise RuntimeError(f"preserved evaluator lacks get_anscheck_prompt: {source}")
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["get_anscheck_prompt"]


def run_bounded_judge(
    *,
    hyp_path: Path,
    ref_path: Path,
    source_root: Path,
    model: str = DEFAULT_MODEL,
    client=None,
    max_retries: int = 4,
) -> dict:
    """Run official grading semantics with finite SDK and outer retries."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported judge model {model!r}; pick one of {SUPPORTED_MODELS}")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    prompt_fn = _official_prompt_loader(source_root)
    with ref_path.open(encoding="utf-8") as f:
        references = json.load(f)
    hypotheses = _load_jsonl(hyp_path, artifact="hypothesis")
    reference_by_id = {row["question_id"]: row for row in references}
    owned = client is None
    if owned:
        from openai import OpenAI
        client = OpenAI(max_retries=0)
    result_path = _result_path_for(hyp_path, model)
    try:
        with result_path.open("w", encoding="utf-8") as out:
            for entry in hypotheses:
                reference = reference_by_id[entry["question_id"]]
                prompt = prompt_fn(
                    reference["question_type"], reference["question"],
                    reference["answer"], entry["hypothesis"],
                    abstention="_abs" in entry["question_id"],
                )
                kwargs = {
                    "model": _MODEL_ZOO[model][0],
                    "messages": [{"role": "user", "content": prompt}],
                    "n": 1, "temperature": 0, "max_tokens": 10,
                }
                last_exc = None
                for attempt in range(max_retries + 1):
                    try:
                        completion = client.chat.completions.create(**kwargs)
                        last_exc = None
                        break
                    except Exception as exc:  # bounded retry wrapper
                        last_exc = exc
                if last_exc is not None:
                    raise last_exc
                response = completion.choices[0].message.content.strip()
                entry = dict(entry)
                entry["autoeval_label"] = {
                    "model": _MODEL_ZOO[model][0], "label": "yes" in response.lower()
                }
                out.write(json.dumps(entry) + "\n")
    finally:
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return summarize_pipeline(ref_path=ref_path, hyp_path=hyp_path, result_path=result_path)


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

    metrics = summarize_pipeline(
        ref_path=ref,
        hyp_path=hyp_path,
        result_path=result_path,
    )
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
    click.echo(f"  taREDACTED accuracy:  {metrics['task_averaged_accuracy']:.4f}")
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
