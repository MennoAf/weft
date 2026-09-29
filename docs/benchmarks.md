# Benchmarks

Weft is benchmarked against [LongMemEval](https://github.com/xiaowu0162/LongMemEval), a multi-session memory evaluation dataset for chat assistants. 

This document explains how we created the benchmark harness, what was measured, and how you can run the numbers yourself. 

You can review the harness yourself in `benchmarks/longmemeval/`.

## A note on what Weft optimizes for

Weft is built to save what you tell it to save. 

When you say "remember this," Weft does. It adds provenance and context, but at your direction. It does not curate your conversations into benchmark-shaped facts.

This is a deliberate design choice. Weft prioritizes predictability and trust over maximum recall, even if that means leaving points on the table. 

I don't want to aggressively summarize everything just to make a score higher. When Weft gains an autonomous classifier, it will be because it makes the core product better. 

But that isn't today. 

The numbers below measure the system as it actually ships, not a benchmark-tuned configuration.

## Results — LongMemEval-S, full split (500 questions)

Run date: 2026-09-29. One authorized paid run; no repetitions yet (see the publication protocol below).

| Metric | Value |
|--------|-------|
| Overall accuracy (judged) | **335 / 496 = 67.54%** (67.0% against the full selected 500) |
| Task-averaged accuracy | **70.06%** |
| Completed / selected | 496 / 500 |
| Failed cases | 4 (3 exceeded the harness's conservative context bound; 1 operational casualty, journaled and reported) |
| Writer model | `gpt-6-luna` (agent workload with tool rounds) |
| Judge | `gpt-4o`, official LongMemEval answer-check prompt, abstention-aware |
| Retrieval | turn tier, `top_k=10`, dual ingest (raw memories + episode turns) |
| Embeddings | local FastEmbed `BAAI/bge-small-en-v1.5` (768-dim storage contract) |
| Actual provider spend | **$0.65** (conservative in-run ledger estimate: $4.79) |
| Wall time | ~5.3 hours |

> **Note:** The benchmark drove the models through direct paid API calls rather than a subscription-based harness, so no platform-injected system prompt influenced the responses.

**Per question type:**

| Question type | Accuracy |
|---------------|----------|
| single-session-assistant | 53/56 = 94.6% |
| single-session-user | 64/68 = 94.1% |
| temporal-reasoning | 88/131 = 67.2% |
| single-session-preference | 17/30 = 56.7% |
| knowledge-update | 43/78 = 55.1% |
| multi-session | 70/133 = 52.6% |

The current weakest question types are where memory systems earn their keep: multi-session recall, synthesis, knowledge-update (where old facts are replaced), and single-session preference (17/30 — small sample).

This is also where the current Weft model of "remember only what the user asks" struggles the most. 

These are active improvement targets, and the first update I'm making to Weft after launch. 

## How this run was built

- **Agent workload, not retrieval-only.** For each of the 500 questions, a fresh agent ingested the question's full haystack history through Weft's normal write path — raw memories *and* per-turn episode records (dual representation) — then answered the question using Weft's recall tools under a bounded tool-round policy. The model in the benchmark never sees the question_type, answer, or questionID, which might allow it to "[cheat](https://mediumroast.dev/blog/we-were-not-beating-longmemeval/)"
- **Turn-tier retrieval.** Answers were produced from turn-level hybrid recall over the ingested corpus (`--tier turns`), the representation LongMemEval's multi-session questions stress.
- **Official scoring.** Every hypothesis was judged by `gpt-4o` with LongMemEval's official answer-check prompt. The judge reviewed the answer from the agent against what it expected. Only answers that passed the judge were marked correct.
- **Isolation.** The run wrote only to a disposable local Docker Postgres (`lme_bench` database). Database identity was verified before execution (Postgres `system_identifier` matched host-vs-container; loopback bind confirmed). Nothing touched any hosted database.
- **Budget discipline.** A conservative reservation ledger priced every provider call before dispatch. Before the full run, a 4-question paid calibration measured the real per-case cost and produced the spend projection; the run was explicitly approved against that projection. Thinking tokens were not calculated, so the system used a "best guess" estimate for the conservative run ledger.
- **Integrity.** The exact source files, dataset checksum, manifest, selection order, pricing, and tool-round policy are pinned by hash in the run manifest; the runner refuses to execute if any pinned file changes. Per-question checkpoints were written before each provider call; the run is crash-resumable, and every recovery is journaled.

## Validating it yourself

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, a LongMemEval checkout with the cleaned S dataset, and provider credentials for the writer and judge models.

```bash
# 1. Local infrastructure (disposable Postgres + Redis)
docker compose -f docker-compose.weft.yml up -d

# 2. Create a benchmark database (the runner refuses databases named "weft")
docker compose -f docker-compose.weft.yml exec -T postgres \
  psql -U weft -d weft -c 'CREATE DATABASE lme_bench'

# 3. Point every Weft DB variable at the local benchmark database
export DATABASE_URL="postgresql://weft:weft_local@127.0.0.1:5433/lme_bench"
export WEFT_DATABASE_URL="$DATABASE_URL"
export LONGMEMEVAL_DATABASE_URL="$DATABASE_URL"

# 4. Prepare the run (offline: normalizes the dataset, pins source hashes,
#    writes the manifest — 500 questions, first-occurrence dedupe)
uv run python -m benchmarks.longmemeval.faithful_s36 prepare \
  --dataset benchmarks/longmemeval/data/longmemeval_s_full_first_occurrence.json \
  --manifest benchmarks/longmemeval/manifests/longmemeval_s_full_turns_manifest.json \
  --profile gpt6-luna-full-s-turns-v1 --writer-model gpt-6-luna \
  --max-budget-usd 150 --owner-id <your-run-id> \
  --judge-root /path/to/LongMemEval

# 5. Calibrate (paid, small): measures real per-case cost, ends HOLD_FOR_APPROVAL
uv run python -m benchmarks.longmemeval.faithful_s36 calibrate ... \
  --case-limit 4 --execute --dsn "$DATABASE_URL"

# 6. Approve against the measured projection (human gate, recorded in the receipt)
uv run python -m benchmarks.longmemeval.faithful_s36 approve ... \
  --projected-total-usd <measured> --approved-by <you>

# 7. Run the full split (resumable; checkpoints survive interruption)
uv run python -m benchmarks.longmemeval.faithful_s36 resume ... \
  --execute --dsn "$DATABASE_URL"
```

Each step's full flag set is printed by `--help`. Artifacts land under the run's artifact root: `execution-receipt.json` (status, denominators, budget), `budget-ledger.json` (every provider reservation, estimate vs actual), and `session-checkpoint.json` (per-question evidence including each judge verdict). The score is recomputable from the checkpoint: overall accuracy = judged rows with `judge.label: true` over all judged rows.

## Limitations

- **Single repetition.** One complete run is evidence, not a publication gate — the protocol below calls for 3–5 independent repetitions before headline claims.
- **Model and system are confounded.** These numbers measure Weft-as-shipped with `gpt-6-luna` as the writer. The numbers might change based on the model you use for your system.
- **Estimate vs invoice.** The ledger's $4.79 is a conservative reservation estimate, not a billing guarantee. Actuals are reported separately and were 14% of estimate on this run.
- **User-directed memory policy.** See the note at the top: the current write path saves what the user designates. Systems that auto-curate conversations trade that predictability for recall, and will score differently on this benchmark.
- **Provider-free contract tests prove correctness, not quality.** They verify task shape, artifacts, and fail-closed accounting without any provider call; only paid runs measure answer quality.

## Repeatability and publication protocol

A single aggregate run is not a keep/scrap or publication gate:

1. Pin the code commit, dataset checksum, model identifiers, prompts, question IDs and order, and seeds. (The run manifest does this mechanically; the runner refuses drift.)
2. Run 3–5 independent repetitions per arm, or hold stochastic substrate constant with a versioned fixed-materialization snapshot and repeat only the intended variable.
3. Preserve raw hypotheses, judge results, ledgers, receipts, and failure logs for every repetition.
4. Report stages separately: ingestion completion, retrieval recall, answer correctness, abstention behavior, and infrastructure failures. Failed cases stay in the denominator and are counted by cause.
5. Record estimated cost before execution and actual cost afterward; paid runs require explicit operator approval against the measured projection.

## Session continuity A/B/C benchmark

The deterministic continuity suite under `benchmarks/personal_agent/` tests a
separate outcome from LongMemEval: whether a later session can recover omitted
episodic evidence while keeping concise handoff as the primary continuity
layer.

- **Arm A:** handoff plus durable memories only.
- **Arm B:** Arm A plus bounded targeted turn recall.
- **Arm C:** Arm B plus opt-in materialized beliefs with `evidence_turn_ids`.

Deterministic tests cover activation precision, exact turn IDs, chronology,
exact wording, rejected rationale, supersession, structural containment of
instruction-shaped text inside a quoted-evidence envelope, incomplete evidence,
and project/user isolation. They do **not** prove reader-model instruction
non-compliance or answer quality. `continuity_eval.py` writes deterministic
per-arm mechanics and keeps paid metrics explicitly `PENDING-PAID-EVALUATION`.

```bash
uv run pytest benchmarks/personal_agent/tests/test_paah_continuity.py -q
```

## Cost expectations (LongMemEval-S, agent workload, 2026-09-29 run)

| Stage | Measured |
|-------|----------|
| 4-question calibration smoke | $0.0045 actual / $0.037 reserved |
| Full 500-question run | **$0.65 actual** / $4.79 reserved (~5.3 h) |

## Roadmap gates

Falsifiable claims for the retrieval roadmap, recorded against commits:

- **Multi-session and knowledge-update lift** — the two weakest types (52.6%, 55.1%) are the active targets; a retrieval change ships only if these move without regressing single-session types.
- **Turn-tier recall@10** — turn recall@10 must lift ≥3 points over baseline; Oracle must lift ≥1.
