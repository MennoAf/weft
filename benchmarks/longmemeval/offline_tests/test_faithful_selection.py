"""Hermetic offline regressions for faithful LongMemEval cohort selection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks.longmemeval import faithful_s36
from benchmarks.longmemeval.faithful_budget import LedgerBindingError
from benchmarks.longmemeval.faithful_s36 import (
    FaithfulRunError,
    RunPaths,
    _effective_selection_policy,
    _normalize_selection_policy,
    _select_instances,
    prepare_run,
)


# The canonical S36 order from longmemeval_s_pilot_manifest.json. The test
# constructs its own manifest and synthetic dataset; it never loads the hosted
# dataset or touches a durable benchmark run directory.
CANONICAL_QUESTION_IDS = (
    "118b2229",
    "7527f7e2",
    "c960da58",
    "ec81a493",
    "4fd1909e",
    "c14c00dd",
    "gpt4_15e38248",
    "80ec1f4f",
    "eeda8a6d",
    "0edc2aef",
    "35a27287",
    "54026fce",
    "95228167",
    "a89d7624",
    "1c0ddc50",
    "4bc144e2",
    "8979f9ec",
    "a346bb18",
    "gpt4_e061b84f",
    "5e1b23de",
    "gpt4_7bc6cf22",
    "71017277",
    "gpt4_d6585ce9",
    "gpt4_9a159967",
    "a1eacc2a",
    "6a27ffc2",
    "50635ada",
    "dfde3500",
    "42ec0761",
    "f685340e_abs",
    "7e00a6cb",
    "8464fc84",
    "8752c811",
    "6222b6eb",
    "352ab8bd",
    "16c90bf4",
)
EXCLUDED_QUESTION_ID = "7527f7e2"


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write a local S36 manifest and dataset with deliberately reversed rows."""
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(
        json.dumps({"selection": {"ordered_question_ids": list(CANONICAL_QUESTION_IDS)}}),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for question_id in reversed(CANONICAL_QUESTION_IDS):
        records.append({
            "question_id": question_id,
            "question_type": "single-session-user",
            "question": f"synthetic question for {question_id}",
            "answer": "synthetic answer",
            "question_date": "2024/01/01",
            "haystack_session_ids": [f"session-{question_id}"],
            "haystack_dates": ["2024/01/01"],
            "haystack_sessions": [[{"role": "user", "content": f"history-{question_id}"}]],
        })
    dataset_path = tmp_path / "synthetic-dataset.json"
    dataset_path.write_text(json.dumps(records), encoding="utf-8")
    return dataset_path, manifest_path


def test_default_selection_uses_canonical_manifest_order(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_synthetic_inputs(tmp_path)

    selected, manifest = _select_instances(dataset_path, manifest_path)

    assert tuple(instance.question_id for instance in selected) == CANONICAL_QUESTION_IDS
    assert manifest["selection"]["ordered_question_ids"] == list(CANONICAL_QUESTION_IDS)
    assert selected[0].question_id == "118b2229"
    assert selected[-1].question_id == "16c90bf4"


def test_explicit_cohort_omits_7527f7e2_and_keeps_canonical_order(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_synthetic_inputs(tmp_path)
    cohort = [
        question_id
        for question_id in reversed(CANONICAL_QUESTION_IDS)
        if question_id != EXCLUDED_QUESTION_ID
    ]

    selected, _ = _select_instances(
        dataset_path,
        manifest_path,
        include_question_ids=cohort,
    )
    selected_ids = [instance.question_id for instance in selected]

    assert EXCLUDED_QUESTION_ID not in selected_ids
    assert selected_ids == [
        question_id
        for question_id in CANONICAL_QUESTION_IDS
        if question_id != EXCLUDED_QUESTION_ID
    ]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"include_question_ids": "118b2229"}, "must be a sequence"),
        ({"include_question_ids": [""]}, "non-empty question IDs"),
        (
            {"include_question_ids": ["118b2229", "118b2229"]},
            "duplicate question IDs",
        ),
        ({"include_question_ids": ["not-in-the-manifest"]}, "outside the canonical manifest"),
        (
            {
                "include_question_ids": ["118b2229"],
                "exclude_question_ids": ["118b2229"],
            },
            "both includes and excludes",
        ),
        ({"include_question_ids": []}, "selected no manifest cases"),
    ],
)
def test_selection_policy_rejects_invalid_or_ambiguous_ids(
    policy: dict[str, Any], message: str,
) -> None:
    with pytest.raises(FaithfulRunError, match=message):
        _normalize_selection_policy(CANONICAL_QUESTION_IDS, **policy)


def test_exclusion_only_bound_policy_reselects_without_excluded_question(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_synthetic_inputs(tmp_path)
    prepared = prepare_run(
        dataset_path,
        manifest_path,
        paths=RunPaths.from_root(tmp_path / "temporary-exclusion-run"),
        owner_id="offline-owner",
        exclude_question_ids=[EXCLUDED_QUESTION_ID],
    )

    # Calibration/resume resolve the omitted CLI options from the prepared
    # binding, then pass these normalized values back to _select_instances.
    # An exclusion-only policy persists include_question_ids=None (JSON null)
    # with excluded IDs retained; re-selection must preserve that policy.
    policy = _effective_selection_policy(
        prepared.binding,
        CANONICAL_QUESTION_IDS,
        include_question_ids=None,
        exclude_question_ids=None,
    )
    selected, _ = _select_instances(
        dataset_path,
        manifest_path,
        include_question_ids=policy["include_question_ids"],
        exclude_question_ids=policy["exclude_question_ids"],
    )

    selected_ids = [instance.question_id for instance in selected]
    assert len(selected_ids) == len(CANONICAL_QUESTION_IDS) - 1
    assert EXCLUDED_QUESTION_ID not in selected_ids
    assert selected_ids == [
        question_id
        for question_id in CANONICAL_QUESTION_IDS
        if question_id != EXCLUDED_QUESTION_ID
    ]


def test_prepared_selection_policy_rejects_mismatch_and_tampering(tmp_path: Path) -> None:
    dataset_path, manifest_path = _write_synthetic_inputs(tmp_path)
    cohort = [CANONICAL_QUESTION_IDS[2], CANONICAL_QUESTION_IDS[0]]
    prepared = prepare_run(
        dataset_path,
        manifest_path,
        paths=RunPaths.from_root(tmp_path / "temporary-run"),
        owner_id="offline-owner",
        include_question_ids=cohort,
    )
    expected_cohort = [CANONICAL_QUESTION_IDS[0], CANONICAL_QUESTION_IDS[2]]

    assert list(prepared.ordered_question_ids) == expected_cohort
    assert _effective_selection_policy(
        prepared.binding,
        CANONICAL_QUESTION_IDS,
        include_question_ids=None,
        exclude_question_ids=None,
    ) == {
        "include_question_ids": expected_cohort,
        "exclude_question_ids": None,
    }

    with pytest.raises(LedgerBindingError, match="selection policy differs from prepared binding"):
        _effective_selection_policy(
            prepared.binding,
            CANONICAL_QUESTION_IDS,
            include_question_ids=[CANONICAL_QUESTION_IDS[0]],
            exclude_question_ids=None,
        )

    tampered_binding = dict(prepared.binding)
    tampered_binding["selection_policy_include_question_ids"] = json.dumps(
        [CANONICAL_QUESTION_IDS[1]], separators=(",", ":")
    )
    with pytest.raises(LedgerBindingError, match="selection policy differs from prepared binding"):
        _effective_selection_policy(
            tampered_binding,
            CANONICAL_QUESTION_IDS,
            include_question_ids=None,
            exclude_question_ids=None,
        )


@pytest.mark.parametrize(
    ("process_value", "cwd_value", "home_value", "expected_dsn"),
    [
        (
            "synthetic-process-dsn",
            "synthetic-cwd-dsn",
            "synthetic-home-dsn",
            "synthetic-process-dsn",
        ),
        (None, "synthetic-cwd-dsn", "synthetic-home-dsn", "synthetic-cwd-dsn"),
        (None, None, "synthetic-home-dsn", "synthetic-home-dsn"),
    ],
    ids=["process-environment-wins", "cwd-env-wins-over-home-env", "home-env-fallback"],
)
def test_cli_loads_dotenv_before_parsing_environment_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_value: str | None,
    cwd_value: str | None,
    home_value: str | None,
    expected_dsn: str,
) -> None:
    """The CLI honors process > cwd .env > ~/.weft/.env without real credentials."""
    cwd = tmp_path / "working-directory"
    home = tmp_path / "synthetic-home"
    cwd.mkdir()
    (home / ".weft").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", lambda: home)

    env_key = "LONGMEMEVAL_DATABASE_URL"
    monkeypatch.delenv(env_key, raising=False)
    if process_value is not None:
        monkeypatch.setenv(env_key, process_value)

    env_values = {
        cwd / ".env": cwd_value,
        home / ".weft" / ".env": home_value,
    }
    events: list[Path | str] = []

    def fake_load_dotenv(path: Path) -> bool:
        events.append(path)
        value = env_values[path]
        if value is not None:
            os.environ.setdefault(env_key, value)
        return value is not None

    # Replace python-dotenv itself so no real .env file can be read.
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=fake_load_dotenv),
    )

    parse_args = faithful_s36._parse_args

    def recording_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        events.append("parse")
        return parse_args(argv)

    captured_args: list[argparse.Namespace] = []

    async def capture_execution(args: argparse.Namespace) -> int:
        events.append("execute")
        captured_args.append(args)
        return 0

    monkeypatch.setattr(faithful_s36, "_parse_args", recording_parse_args)
    monkeypatch.setattr(faithful_s36, "_main_async", capture_execution)

    assert faithful_s36.main([
        "resume",
        "--dataset", "unused-dataset.json",
        "--manifest", "unused-manifest.json",
        "--judge-root", "unused-judge-root",
    ]) == 0

    assert events == [cwd / ".env", home / ".weft" / ".env", "parse", "execute"]
    assert captured_args[0].dsn == expected_dsn


@pytest.mark.parametrize(
    ("owner_id", "agent_id"),
    [
        ("different-owner", "faithful-s36"),
        ("offline-owner", "different-agent"),
    ],
)
def test_prepare_refuses_prepared_run_identity_mismatch(
    tmp_path: Path, owner_id: str, agent_id: str,
) -> None:
    dataset_path, manifest_path = _write_synthetic_inputs(tmp_path)
    paths = RunPaths.from_root(tmp_path / "temporary-run")
    cohort = [CANONICAL_QUESTION_IDS[0], CANONICAL_QUESTION_IDS[2]]
    prepare_run(
        dataset_path,
        manifest_path,
        paths=paths,
        owner_id="offline-owner",
        agent_id="faithful-s36",
        include_question_ids=cohort,
    )

    with pytest.raises(LedgerBindingError, match="prepared manifest binding differs"):
        prepare_run(
            dataset_path,
            manifest_path,
            paths=paths,
            owner_id=owner_id,
            agent_id=agent_id,
            include_question_ids=cohort,
        )
