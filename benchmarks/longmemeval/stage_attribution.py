"""Offline, source-annotated stage attribution for faithful LongMemEval.

This benchmark-local module is pure: it has no provider, database, filesystem,
or network effects. It requires a manually source-reviewed target and its
concrete source turn; it never infers a target from the benchmark reference
answer. Reports contain coverage states and indexes, not raw target, source,
memory, retrieval, or answer text.

Input contract (all keys required; unknown keys rejected)::

    {
      "schema": "weft.longmemeval.stage-attribution-input.v2",
      "target": {
        "fact": "45 minutes each way",
        "source_turn": {
          "session_id": "synthetic-session-1",
          "turn_index": 0,
          "role": "user",
          "content": "My commute is 45 minutes each way."
        },
        "source_basis": "manual_source_review"
      },
      "stages": {
        "writer_selection": {
          "captured": true,
          "contents": ["Commute: 45 minutes each way."]
        },
        "persisted_memory_readback": {
          "captured": true,
          "verified_after_save": true,
          "contents": ["Commute: 45 minutes each way."]
        },
        "retrieved_memory_evidence": {"captured": true, "contents": ["..."]},
        "final_answer": {"captured": true, "text": "45 minutes each way"},
        "judge": {"captured": true, "label": true}
      }
    }

``writer_selection.contents`` records the exact content candidates supplied to
``weft_remember`` before persistence. ``captured: true`` with an empty list is
an observed negative; ``captured: false`` with ``contents: null`` means the
observation is unavailable. A persistence gap is reported only when a captured
writer candidate contains the exact target and a separately captured,
independently verified post-save readback does not. Raw evidence content is
never copied into the report.

The commute example is synthetic and appears only in this contract/test fixture.
The evaluator validates that the supplied user turn contains the exact fact;
human review is still responsible for verifying source authenticity. Exact
matching requires the complete annotated value, unit, and qualifier/period; it
does not attempt semantic equivalence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Mapping

INPUT_SCHEMA = "weft.longmemeval.stage-attribution-input.v2"
REPORT_SCHEMA = "weft.longmemeval.stage-attribution-report.v2"
MAX_FACT_CHARS = 256
MAX_SESSION_ID_CHARS = 200
MAX_SOURCE_CHARS = 4_000
MAX_EVIDENCE_CHARS = 4_000
MAX_EVIDENCE_ITEMS = 32


@dataclass(frozen=True, slots=True)
class TargetAnnotation:
    """Manually reviewed fact and concrete source-turn citation."""

    fact: str
    session_id: str
    turn_index: int
    source_content: str
    source_basis: str


@dataclass(frozen=True, slots=True)
class MemoryCapture:
    """Explicit captured memory evidence; empty contents is a negative."""

    captured: bool
    contents: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class PersistedMemoryCapture:
    """Independent post-save readback with explicit capture and verification."""

    captured: bool
    verified_after_save: bool
    contents: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class TextCapture:
    """Captured text stage, where an empty string is a negative observation."""

    captured: bool
    text: str | None


@dataclass(frozen=True, slots=True)
class JudgeCapture:
    """Observed binary judge label, separate from exact-fact coverage."""

    captured: bool
    label: bool | None


@dataclass(frozen=True, slots=True)
class StageObservation:
    """Bounded, content-free observation for a stage."""

    captured: bool
    status: str
    item_count: int | None
    matching_item_indexes: tuple[int, ...]
    verified_after_save: bool | None = None


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """Strict JSON-ready report; attribution names an observed first gap only."""

    schema: str
    source_annotation: Mapping[str, Any]
    stages: Mapping[str, StageObservation]
    attribution: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Return the report contract with JSON array values for indexes."""
        stages: dict[str, dict[str, Any]] = {}
        for name, observation in self.stages.items():
            stage = asdict(observation)
            stage["matching_item_indexes"] = list(observation.matching_item_indexes)
            stages[name] = stage
        return {
            "schema": self.schema,
            "source_annotation": dict(self.source_annotation),
            "stages": stages,
            "attribution": self.attribution,
            "interpretation": self.interpretation,
        }


def evaluate_stage_attribution(payload: Mapping[str, Any]) -> AttributionReport:
    """Evaluate exact annotated-fact coverage across captured stages.

    This reports observed literal coverage, not semantic equivalence or causal
    responsibility. Source authenticity remains a human review obligation.
    """
    root = _mapping(payload, "input", {"schema", "target", "stages"})
    if root["schema"] != INPUT_SCHEMA:
        raise ValueError(f"input.schema must equal {INPUT_SCHEMA!r}")

    annotation = _parse_annotation(root["target"])
    stage_data = _mapping(
        root["stages"],
        "stages",
        {
            "writer_selection",
            "persisted_memory_readback",
            "retrieved_memory_evidence",
            "final_answer",
            "judge",
        },
    )
    writer = _parse_memory_capture(stage_data["writer_selection"], "stages.writer_selection")
    persisted_capture = _parse_persisted_memory_capture(
        stage_data["persisted_memory_readback"], "stages.persisted_memory_readback"
    )
    retrieved_capture = _parse_memory_capture(
        stage_data["retrieved_memory_evidence"], "stages.retrieved_memory_evidence"
    )
    answer = _parse_text_capture(stage_data["final_answer"], "stages.final_answer")
    judge = _parse_judge_capture(stage_data["judge"], "stages.judge")

    writer_observation = _memory_observation(
        annotation.fact, writer.captured, writer.contents
    )
    persisted = _memory_observation(
        annotation.fact,
        persisted_capture.captured,
        persisted_capture.contents,
        verified_after_save=persisted_capture.verified_after_save,
    )
    retrieved = _memory_observation(annotation.fact, retrieved_capture.captured, retrieved_capture.contents)
    answer_observation = _text_observation(annotation.fact, answer)
    judge_observation = StageObservation(
        captured=judge.captured,
        status=("not_captured" if not judge.captured else "accepted" if judge.label else "rejected"),
        item_count=1 if judge.captured else None,
        matching_item_indexes=(),
    )
    stages = {
        "source_target": StageObservation(True, "present", 1, (0,)),
        "writer_selection": writer_observation,
        "persisted_memory_contents": persisted,
        "retrieved_memory_evidence": retrieved,
        "final_answer": answer_observation,
        "judge": judge_observation,
    }

    return AttributionReport(
        schema=REPORT_SCHEMA,
        source_annotation={
            "source_basis": annotation.source_basis,
            "source_kind": "haystack_session_user_turn",
            "source_location_sha256": _sha256(f"{annotation.session_id}#turn:{annotation.turn_index}"),
            "validation": "source_turn_contains_exact_target",
        },
        stages=stages,
        attribution=_first_observed_gap(
            writer_observation, persisted, retrieved, answer_observation, judge
        ),
        interpretation="observed_exact_phrase_coverage_only; not causal attribution or benchmark scoring",
    )


def _parse_annotation(value: object) -> TargetAnnotation:
    raw = _mapping(value, "target", {"fact", "source_turn", "source_basis"})
    fact = _string(raw["fact"], "target.fact", MAX_FACT_CHARS)
    source = _mapping(
        raw["source_turn"],
        "target.source_turn",
        {"session_id", "turn_index", "role", "content"},
    )
    session_id = _string(source["session_id"], "target.source_turn.session_id", MAX_SESSION_ID_CHARS)
    turn_index = source["turn_index"]
    if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
        raise ValueError("target.source_turn.turn_index must be a non-negative integer")
    role = _string(source["role"], "target.source_turn.role", 16)
    if role != "user":
        raise ValueError("target.source_turn.role must be 'user'")
    content = _string(source["content"], "target.source_turn.content", MAX_SOURCE_CHARS)
    basis = _string(raw["source_basis"], "target.source_basis", 64)
    if basis != "manual_source_review":
        raise ValueError("target.source_basis must be 'manual_source_review'")
    if not _contains_exact_target(content, fact):
        raise ValueError("target.source_turn.content must contain target.fact as an exact phrase")
    return TargetAnnotation(fact, session_id, turn_index, content, basis)


def _parse_memory_capture(value: object, path: str) -> MemoryCapture:
    raw = _mapping(value, path, {"captured", "contents"})
    captured = _boolean(raw["captured"], f"{path}.captured")
    contents = _parse_contents(raw["contents"], path, captured)
    return MemoryCapture(captured, contents)


def _parse_persisted_memory_capture(value: object, path: str) -> PersistedMemoryCapture:
    raw = _mapping(value, path, {"captured", "verified_after_save", "contents"})
    captured = _boolean(raw["captured"], f"{path}.captured")
    verified = _boolean(raw["verified_after_save"], f"{path}.verified_after_save")
    if not captured and verified:
        raise ValueError(f"{path}.verified_after_save cannot be true when captured is false")
    contents = _parse_contents(raw["contents"], path, captured)
    return PersistedMemoryCapture(captured, verified, contents)


def _parse_contents(value: object, path: str, captured: bool) -> tuple[str, ...] | None:
    if not captured:
        if value is not None:
            raise ValueError(f"{path}.contents must be null when captured is false")
        return None
    if not isinstance(value, list):
        raise ValueError(f"{path}.contents must be a list when captured is true")
    if len(value) > MAX_EVIDENCE_ITEMS:
        raise ValueError(f"{path}.contents exceeds {MAX_EVIDENCE_ITEMS} items")
    return tuple(
        _string(item, f"{path}.contents[{index}]", MAX_EVIDENCE_CHARS, allow_empty=True)
        for index, item in enumerate(value)
    )


def _parse_text_capture(value: object, path: str) -> TextCapture:
    raw = _mapping(value, path, {"captured", "text"})
    captured = _boolean(raw["captured"], f"{path}.captured")
    text = raw["text"]
    if not captured:
        if text is not None:
            raise ValueError(f"{path}.text must be null when captured is false")
        return TextCapture(False, None)
    return TextCapture(True, _string(text, f"{path}.text", MAX_EVIDENCE_CHARS, allow_empty=True))


def _parse_judge_capture(value: object, path: str) -> JudgeCapture:
    raw = _mapping(value, path, {"captured", "label"})
    captured = _boolean(raw["captured"], f"{path}.captured")
    label = raw["label"]
    if not captured:
        if label is not None:
            raise ValueError(f"{path}.label must be null when captured is false")
        return JudgeCapture(False, None)
    if not isinstance(label, bool):
        raise ValueError(f"{path}.label must be a boolean when captured is true")
    return JudgeCapture(True, label)


def _mapping(value: object, path: str, expected_keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    keys = set(value.keys())
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{path} keys must be strings")
    missing = expected_keys - keys
    unknown = keys - expected_keys
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unknown:
            details.append(f"unknown keys: {sorted(unknown)}")
        raise ValueError(f"{path} has " + "; ".join(details))
    return value


def _string(value: object, path: str, max_chars: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if len(value) > max_chars:
        raise ValueError(f"{path} exceeds {max_chars} characters")
    if not allow_empty and not value.strip():
        raise ValueError(f"{path} must not be empty")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _memory_observation(
    fact: str,
    captured: bool,
    contents: tuple[str, ...] | None,
    *,
    verified_after_save: bool | None = None,
) -> StageObservation:
    if not captured:
        return StageObservation(
            captured=False,
            status="not_captured",
            item_count=None,
            matching_item_indexes=(),
            verified_after_save=verified_after_save,
        )
    assert contents is not None  # Guaranteed by strict boundary validation.
    matches = tuple(index for index, text in enumerate(contents) if _contains_exact_target(text, fact))
    if verified_after_save is False:
        status = "unverified"
    else:
        status = "present" if matches else "missing"
    return StageObservation(
        captured=True,
        status=status,
        item_count=len(contents),
        matching_item_indexes=matches,
        verified_after_save=verified_after_save,
    )


def _text_observation(fact: str, capture: TextCapture) -> StageObservation:
    if not capture.captured:
        return StageObservation(False, "not_captured", None, ())
    assert capture.text is not None  # Guaranteed by strict boundary validation.
    matches = (0,) if _contains_exact_target(capture.text, fact) else ()
    return StageObservation(True, "present" if matches else "missing", 1, matches)


def _first_observed_gap(
    writer: StageObservation,
    persisted: StageObservation,
    retrieved: StageObservation,
    answer: StageObservation,
    judge: JudgeCapture,
) -> str:
    if not writer.captured:
        return "missing_writer_selection_capture"
    if writer.status == "missing":
        return "writer_selection_miss"
    if not persisted.captured:
        return "missing_persisted_readback"
    if persisted.verified_after_save is not True:
        return "missing_persisted_readback_verification"
    if persisted.status == "missing":
        return "persistence_gap"
    if retrieved.status == "not_captured":
        return "missing_retrieval_evidence_capture"
    if retrieved.status == "missing":
        return "retrieval_miss"
    if answer.status == "not_captured":
        return "missing_final_answer_capture"
    if answer.status == "missing":
        return "reader_miss"
    if not judge.captured:
        return "missing_judge_label"
    if judge.label is False:
        return "judge_rejected_exactly_covered_answer"
    return "exact_evidence_chain_judge_accepted"


def _contains_exact_target(text: str, fact: str) -> bool:
    normalized_text = _normalize(text)
    normalized_fact = _normalize(fact)
    phrase = r"\s+".join(re.escape(part) for part in normalized_fact.split())
    return re.search(rf"(?<!\w){phrase}(?!\w)", normalized_text) is not None


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
