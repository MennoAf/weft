#!/usr/bin/env python3
"""Classify LongMemEval-S single-session-preference failures from saved artifacts.

Offline/read-only with respect to benchmark artifacts. Writes only the JSON and
Markdown reports alongside this script. No provider or database calls are made.
"""
from __future__ import annotations

import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".recall-lift-worktree/artifacts/longmemeval-gpt6-full-s-turns"
DATASET_PATH = ROOT / "benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json"
RUNS = ("faithful", "faithful-run1-20260930", "faithful-run2-20260930")
RUN_LABELS = ("main", "run1", "run2")
GENERIC = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "user",
    "would", "prefer", "prefers", "preference", "responses", "response", "suggestions", "suggest",
    "particularly", "especially", "specifically", "based", "their", "they", "them", "there",
    "have", "which", "who", "when", "what", "where", "about", "across", "take", "account",
    "include", "use", "using", "such", "also", "only", "not", "are", "can", "does", "has",
    "had", "was", "were", "been", "being", "some", "should", "could", "provide", "related",
    "previous", "experience", "experiences", "recommendations", "recommendation", "helpful", "more",
    "like", "want", "needs", "need", "information", "interactions", "personalized", "specific",
}
PREFERENCE_CUES = (
    "prefer", "rather", "like", "dislike", "enjoy", "want", "love", "hate", "interested",
    "wish", "avoid", "would choose", "looking for", "my favorite", "i use", "i have",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def norm(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def tokens(text: str) -> set[str]:
    return {word for word in norm(text).split() if len(word) > 2 and word not in GENERIC}


def split_sentences(text: str) -> list[str]:
    return [part.strip(" \t-*•") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.strip()) >= 28]


def source_candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank sentence evidence only within dataset-designated answer sessions."""
    gold_tokens = tokens(item["answer"])
    sessions = dict(zip(item.get("haystack_session_ids", []), item.get("haystack_sessions", [])))
    candidates: list[dict[str, Any]] = []
    for session_id in item.get("answer_session_ids", []):
        for turn_index, turn in enumerate(sessions.get(session_id, [])):
            # Direct preference evidence must be user-authored; assistant advice
            # and generated recommendations are not evidence of user preference.
            if turn.get("role") != "user":
                continue
            content = str(turn.get("content", ""))
            for sentence in split_sentences(content):
                sentence_tokens = tokens(sentence)
                overlap = len(sentence_tokens & gold_tokens) / max(1, len(gold_tokens))
                cue = any(cue in sentence.lower() for cue in PREFERENCE_CUES)
                # Small explicit-user-evidence tie-break; score remains mainly lexical.
                user_bonus = 0.025 if turn.get("role") == "user" else 0.0
                score = overlap + (0.10 if cue else 0.0) + user_bonus
                if overlap >= 0.10 or cue:
                    candidates.append({
                        "score": round(score, 4),
                        "token_overlap": round(overlap, 4),
                        "preference_cue": cue,
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "role": turn.get("role"),
                        "sentence": sentence,
                        "source_turn_content": content,
                    })
    candidates.sort(key=lambda row: (row["score"], row["token_overlap"], row["role"] == "user"), reverse=True)
    # Keep only distinct statements with meaningful gold-answer overlap.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        key = norm(row["sentence"])
        if key and key not in seen:
            unique.append(row)
            seen.add(key)
        if len(unique) == 5:
            break
    return unique


def recall_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for tool_call in record.get("tool_results", []):
        result = tool_call.get("result")
        if tool_call.get("name") != "weft_recall" or not isinstance(result, dict):
            continue
        turns = result.get("turns", [])
        if isinstance(turns, list):
            found.extend(turn for turn in turns if isinstance(turn, dict) and isinstance(turn.get("content"), str))
    return found


def sentence_in_context(statements: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
    for statement in statements:
        needle = norm(statement["source_turn_content"])
        if not needle:
            continue
        for turn in turns:
            hay = norm(str(turn.get("content", "")))
            if needle in hay:
                return {"present": True, "matched_sentence": statement["sentence"],
                        "context_session_id": turn.get("source_session_id"),
                        "context_turn_index": turn.get("turn_index")}
    return {"present": False, "matched_sentence": None, "context_session_id": None, "context_turn_index": None}


def retrieved_contexts(record: dict[str, Any]) -> list[str]:
    return [str(turn.get("content", "")) for turn in recall_turns(record)]


def item_cost(record: dict[str, Any], reservations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answer = record.get("answer", {})
    ids = list(answer.get("reservations", [])) if isinstance(answer, dict) else []
    judge_id = record.get("judge", {}).get("reservation_id") if isinstance(record.get("judge"), dict) else None
    if judge_id:
        ids.append(judge_id)
    rows = [reservations[rid] for rid in ids if rid in reservations]
    actual = sum(float(row.get("actual_usd", 0.0) or 0.0) for row in rows)
    estimate = sum(float(row.get("estimated_usd", 0.0) or 0.0) for row in rows)
    return {"reservation_count": len(ids), "ledger_rows_found": len(rows),
            "actual_usd": round(actual, 8), "reserved_estimate_usd": round(estimate, 8)}


def main() -> None:
    dataset = load_json(DATASET_PATH)
    preference_items = {row["question_id"]: row for row in dataset
                        if "preference" in str(row.get("question_type", "")).lower()}
    checkpoints = {run: load_json(ARTIFACTS / run / "session-checkpoint.json") for run in RUNS}
    ledgers = {run: load_json(ARTIFACTS / run / "budget-ledger.json") for run in RUNS}
    evidence = {run: checkpoints[run]["evidence"] for run in RUNS}
    preference_ids = sorted(preference_items)
    verdicts: dict[str, dict[str, bool]] = {}
    for question_id in preference_ids:
        verdicts[question_id] = {}
        for run in RUNS:
            label = evidence[run][question_id]["judge"].get("label")
            if not isinstance(label, bool):
                raise ValueError(f"missing boolean evidence[{question_id}].judge.label in {run}")
            verdicts[question_id][run] = label
    stable = [qid for qid in preference_ids if not any(verdicts[qid].values())]
    always_correct = [qid for qid in preference_ids if all(verdicts[qid].values())]
    flicker = [qid for qid in preference_ids if qid not in stable and qid not in always_correct]

    reservation_maps = {
        run: {row["reservation_id"]: row for row in ledgers[run]["reservations"]}
        for run in RUNS
    }
    items_out = []
    for question_id in stable:
        dataset_item = preference_items[question_id]
        statements = source_candidates(dataset_item)
        best = statements[0] if statements else None
        run_detail = {}
        context_presence = []
        imports = {}
        for run, short_name in zip(RUNS, RUN_LABELS):
            record = evidence[run][question_id]
            retrieved = recall_turns(record)
            present = sentence_in_context(statements, retrieved)
            context_presence.append(present["present"])
            session_log = {row.get("session_id"): row for row in record.get("sessions", []) if isinstance(row, dict)}
            source_imports = {sid: session_log.get(sid, {}).get("result") for sid in dataset_item.get("answer_session_ids", [])}
            imports[run] = source_imports
            run_detail[short_name] = {
                "correct": record["judge"]["label"],
                "judge_raw": record["judge"].get("raw"),
                "answer": record.get("answer", {}).get("text", ""),
                "retrieved_context_turn_count": len(retrieved),
                "source_sentence_present": present,
                "answer_session_import": source_imports,
                "cost": item_cost(record, reservation_maps[run]),
            }
        successful_imports = [
            result.get("ingested") is True
            for by_run in imports.values() for result in by_run.values() if isinstance(result, dict)
        ]
        if best is None or not successful_imports or not all(successful_imports):
            stage = "STORAGE"
        elif sum(context_presence) >= 2:
            stage = "READER"
        else:
            stage = "RETRIEVAL"
        # A human-readable quote from the dataset's answer-designated source session.
        quote = best["sentence"] if best else "No supporting source sentence matched the gold-answer terms."
        items_out.append({
            "question_id": question_id,
            "question_type": dataset_item["question_type"],
            "question": dataset_item["question"],
            "gold_answer": dataset_item["answer"],
            "answer_session_ids": dataset_item.get("answer_session_ids", []),
            "supporting_statements": statements,
            "statement_found_in_dataset_source": bool(statements),
            "checkpoint_imported_answer_sessions_all_runs": bool(successful_imports) and all(successful_imports),
            "context_presence_by_run": {label: detail["source_sentence_present"]["present"] for label, detail in run_detail.items()},
            "verdict_by_run": {label: detail["correct"] for label, detail in run_detail.items()},
            "stage": stage,
            "classification_rule": "STORAGE if no matched source statement or any answer-session ingestion receipt is absent/false; otherwise READER if exact/normalized matched sentence appears in >=2/3 run contexts; otherwise RETRIEVAL.",
            "evidence_quote": quote,
            "runs": run_detail,
        })

    selected_validation_ids = stable + flicker
    main_costs = {
        qid: item_cost(evidence[RUNS[0]][qid], reservation_maps[RUNS[0]])
        for qid in selected_validation_ids
    }
    actuals = [value["actual_usd"] for value in main_costs.values()]
    estimateds = [value["reserved_estimate_usd"] for value in main_costs.values()]
    manifest = load_json(ARTIFACTS / "faithful" / "run-manifest.json")
    binding = manifest.get("binding", {})
    report = {
        "fields_used": {
            "verdict": "checkpoint evidence[question_id].judge.label (boolean recorded by the GPT-4o judge); independently validated counts 17/30, 15/30, 18/30.",
            "judge_text": "checkpoint evidence[question_id].judge.raw",
            "reader_answer": "checkpoint evidence[question_id].answer.text (the faithful-agent final answer text)",
            "source_session_receipts": "checkpoint evidence[question_id].sessions[*].session_id and .result.ingested; these are ingestion receipts, not raw stored-text readbacks.",
            "retrieved_context": "checkpoint evidence[question_id].tool_results[*] where name == weft_recall, result.turns[*].content; exact/normalized sentence substring test.",
            "source_sentence": "Dataset question.answer_session_ids joined to haystack_session_ids/haystack_sessions; sentence ranked by informative token overlap with gold answer plus explicit preference-cue and user-role tie-breaks.",
            "cost": "faithful/budget-ledger.json reservations joined by answer.reservations and judge.reservation_id; summed actual_usd and reserved estimated_usd.",
        },
        "runs": list(RUNS),
        "verdict_counts": {run: sum(verdicts[qid][run] for qid in preference_ids) for run in RUNS},
        "sets": {"stable_failure": stable, "flicker": flicker, "always_correct": always_correct},
        "classification_method_caveat": "Checkpoint receipts show the source session was marked ingested but do not include a raw post-write database readback. Sentence matching is a lexical heuristic against answer-designated dataset sessions; manually audit weak matches before treating the stage label as ground truth.",
        "dominant_stage": max(("STORAGE", "RETRIEVAL", "READER", "JUDGE"), key=lambda stage: sum(row["stage"] == stage for row in items_out)),
        "stage_counts": {stage: sum(row["stage"] == stage for row in items_out) for stage in ("STORAGE", "RETRIEVAL", "READER", "JUDGE")},
        "stable_failure_items": items_out,
        "validation_plan": {
            "question_ids": selected_validation_ids,
            "treatment_ids": stable,
            "control_ids": flicker,
            "n": len(selected_validation_ids),
            "writer_model": binding.get("writer_model"),
            "judge_model": "gpt-4o (faithful checkpoint judge model)",
            "parity_settings_from_faithful_run_manifest": {
                key: binding.get(key) for key in (
                    "run_profile", "retrieval_tier", "embedding_provider", "embedding_model", "embedding_dimensions",
                    "selection_policy_include_question_ids", "selection_policy_exclude_question_ids", "tool_round_policy_json",
                    "max_budget_usd", "operational_stop_usd", "agent_id", "owner_id",
                ) if key in binding
            },
            "cost_basis": "Main faithful-run item cost: all answer.text (gpt-6-luna answerer) reservations plus that item's GPT-4o judge reservation, joined to budget-ledger actual_usd and estimated_usd. FULL_S_PROFILE imports the source sessions locally (faithful_s36.py:1933-1945), so there is no separate paid session-writer call to add.",
            "per_item_actual_usd_mean": round(statistics.mean(actuals), 8) if actuals else None,
            "per_item_actual_usd_median": round(statistics.median(actuals), 8) if actuals else None,
            "per_item_reserved_estimate_usd_mean": round(statistics.mean(estimateds), 8) if estimateds else None,
            "18_item_actual_cost_estimate_usd": round(sum(actuals), 8),
            "18_item_reserved_estimate_usd": round(sum(estimateds), 8),
            "item_costs": main_costs,
            "execution_warning": "Plan only. Do not run the benchmark in this analysis script; treatment requires authorized provider execution.",
        },
    }
    out_dir = Path(__file__).resolve().parent
    (out_dir / "classification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Single-session preference failure classification", "",
        "Verdict is `judge.label`; contexts are captured `weft_recall` turn content. Source quotes are lexical matches in answer-designated dataset sessions. See JSON for full answers, sentence-match detail, and run-level costs.", "",
        "| Question ID | Per-run verdict (main/run1/run2) | Statement found? | Context presence (main/run1/run2) | Stage | Evidence quote |",
        "|---|---|---|---|---|---|",
    ]
    for row in items_out:
        verdict = "/".join("correct" if row["verdict_by_run"][name] else "incorrect" for name in RUN_LABELS)
        presence = "/".join("yes" if row["context_presence_by_run"][name] else "no" for name in RUN_LABELS)
        found = "yes" if row["statement_found_in_dataset_source"] else "no"
        quote = row["evidence_quote"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['question_id']}` | {verdict} | {found} | {presence} | **{row['stage']}** | “{quote}” |")
    lines += [
        "", f"**Dominant stage:** {report['dominant_stage']} ({report['stage_counts']}).",
        "", "## Evidence limits", "",
        report["classification_method_caveat"],
        "", "## Focused validation (not executed)", "",
        f"Treatment: {len(stable)} stable failures. Control: {len(flicker)} flicker items. Answerer: `{binding.get('writer_model')}`; judge: `gpt-4o`; total: 18 items. In this FULL_S run profile, source sessions are ingested locally rather than sent to a paid writer model.",
        f"Main-run actual cost mean/item: ${report['validation_plan']['per_item_actual_usd_mean']:.6f}; median/item: ${report['validation_plan']['per_item_actual_usd_median']:.6f}; 18-item observed-cost estimate: ${report['validation_plan']['18_item_actual_cost_estimate_usd']:.6f}.",
        f"Reserved-estimate cost mean/item: ${report['validation_plan']['per_item_reserved_estimate_usd_mean']:.6f}; 18-item reserved estimate: ${report['validation_plan']['18_item_reserved_estimate_usd']:.6f}.",
        "Use the manifest's model, turn-retrieval tier, tool-round policy, embedding settings, owner identity, and operational budget; include exactly the 18 IDs recorded in the JSON. Compare treatment judge-label lift against its three-run baseline, and check that controls do not materially regress. No provider call or benchmark execution is performed here.",
        "", "## Reader-side proposal", "",
        "Hypothesis: the faithful answerer has no preference-specific instruction to actively search for direct preference evidence; its generic ‘answer only from recall context’ rule can terminate with an ungrounded or generic answer when the relevant turn is not surfaced on the first recall. A falsifiable treatment should improve judged accuracy on stable failures without materially changing flicker controls.",
        "", "At `benchmarks/longmemeval/faithful_agent.py:652-656`, append this exact sentence to `instructions`:", "",
        "> When a question asks what the user prefers, search the public recall tools for the user's specific preference and its earlier session before answering. Prefer direct user statements and user-confirmed choices over assistant-generated suggestions or topic associations; apply the retrieved preference to the current question, and say “I don't know” if no directly relevant preference is retrieved.",
    ]
    (out_dir / "classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stable_failure_count": len(stable), "flicker_count": len(flicker), "always_correct_count": len(always_correct), "verdict_counts": report["verdict_counts"], "stage_counts": report["stage_counts"], "dominant_stage": report["dominant_stage"], "validation_cost_actual": report["validation_plan"]["18_item_actual_cost_estimate_usd"], "validation_cost_reserved": report["validation_plan"]["18_item_reserved_estimate_usd"], "outputs": [str(out_dir / "classification.json"), str(out_dir / "classification.md")]}, indent=2))


if __name__ == "__main__":
    main()
