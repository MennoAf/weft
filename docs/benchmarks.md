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

## Results — LongMemEval-S recall lift (three repeated executions, September 30, 2026)

The lift was measured by repeating the same selected 500-question set three times with the same first-occurrence dataset, writer (`gpt-6-luna`), judge (`gpt-4o`, official abstention-aware answer-check prompt), and turn-tier retrieval profile. These are repeated executions of the same selected question set, not statistically independent samples. Of the 1,500 selected question slots, 1,491 were judged across the three runs — 9 selected cases went unjudged, and the per-run denominators below disclose them.

| Run | Judged / selected | Correct / judged | Accuracy |
|-----|-------------------|------------------|----------|
| 1 | 495 / 500 | 395 / 495 | 79.8% |
| 2 | 499 / 500 | 404 / 499 | 81.0% |
| 3 | 497 / 500 | 393 / 497 | 79.1% |
| **All three** | **1,491 / 1,500** | **1,192 / 1,491 (pooled)** | **80.0% pooled** |

The unweighted mean of the three run accuracies is **~80.0%**, versus the **67.54%** historical baseline — **+12.4 percentage points**. The pooled judged-row rate is 1,192/1,491 = 80.0%. Denominators stay visible because the nine unjudged selected cases mean the per-run scores are computed over slightly different judged sets.

Actual combined provider spend across the three runs: **$3.34** (reservation ledgers were conservative estimates, not invoices).

### Per question type (historical baseline → recall-lift mean)

| Question type | Baseline (single run, 2026-09-29) | Recall-lift mean (3 runs) | Change | Lifted-run range |
|---------------|-----------------------------------|---------------------------|--------|------------------|
| knowledge-update | 43/78 = 55.1% | 78.6% | **+23.5 pp** | 75.6–82.1% |
| multi-session | 70/133 = 52.6% | 71.3% | **+18.7 pp** | 67.4–74.2% |
| temporal-reasoning | 88/131 = 67.2% | 77.9% | **+10.7 pp** | run-level details in receipts |
| single-session-assistant | 53/56 = 94.6% | 98.2% | **+3.6 pp** | identical across all 3 runs |
| single-session-user | 64/68 = 94.1% | 97.6% | **+3.5 pp** | run-level details in receipts |
| single-session-preference | 17/30 = 56.7% | 55.6% | **−1.1 pp** | 50.0–60.0% (30 judged each run) |

The overall score is weighted by judged-question counts. The unweighted mean of the six category deltas is ~+9.8 pp — don't present the category-average delta as the overall gain.

The lift landed mostly where the baseline was weakest: knowledge-update, multi-session, and temporal reasoning moved substantially, and single-session assistant/user gained a few points from already-strong levels. Single-session preference is the one category that did not improve — it was variable across runs (50.0–60.0%) and slightly below its historical baseline on the mean.

### Single-session preference: cautious first-pass read

The same 30 single-session-preference question IDs were judged in every run:

| Run | Correct | Judged | Accuracy |
|-----|---------|--------|----------|
| 1 | 15 | 30 | 50.0% |
| 2 | 18 | 30 | 60.0% |
| 3 | 17 | 30 | 56.7% |
| Mean | 50 | 90 run-question judgments | 55.6% |

Across those 30 shared IDs, 12 were judged correct in all three runs, 9 were incorrect in all three, and 9 changed label across runs. That is a stable hard-case subset alongside real run-to-run writer/judge variability. The 90 repeated judgments are not 90 independent questions.

A preliminary join of each run's retrieved turn source-session IDs to each question's annotated answer session suggests a mixture of retrieval misses (some stable misses returned no turns from the gold answer session) and downstream answer/judge variability (some misses returned gold-session turns and still flipped label between runs). This diagnosis is **preliminary and unconfirmed**: a gold-session hit is only a coarse proxy — it does not prove the exact preference-bearing fact was returned or that the agent attended to it — and the planned independent preference review did not complete. Exact-fact annotation and a completed independent review are still needed before publishing any mechanism claim.

### Historical baseline — pre-lift single run (2026-09-29)

Kept for the before/after comparison above; these were the numbers before the recall lift.

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

> **Note:** The benchmark drove the models through direct paid API calls rather than a subscription-based harness, so no platform-injected system prompt influenced the responses. This applies to the baseline run and all three repeated runs.

## How this run was built

The 2026-09-29 baseline run and all three repeated executions used the same construction:

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
export DATABASE_URL="postgresql://weft:weft_local@localhost:5433/lme_bench"
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

- **Repeated executions, not independent samples.** The current headline comes from three repeated executions of the same selected 500-question set — 1,491 of 1,500 selected slots judged, with 9 unjudged. The protocol below still calls for 3–5 independent repetitions per arm (or a versioned fixed-materialization design) before publication-grade claims; that independent-sample standard has not been met.
- **Model and system are confounded.** These numbers measure Weft-as-shipped with `gpt-6-luna` as the writer. The numbers might change based on the model you use for your system.
- **Estimate vs invoice.** Reservation ledgers are conservative estimates, not billing guarantees. On the baseline run the $4.79 reservation came in at $0.65 actual (14% of estimate); the three recall-lift runs totaled $3.34 actual combined.
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

## Cost expectations (LongMemEval-S, agent workload)

| Stage | Measured |
|-------|----------|
| 4-question calibration smoke (baseline run) | $0.0045 actual / $0.037 reserved |
| Baseline full 500-question run (2026-09-29, historical) | **$0.65 actual** / $4.79 reserved (~5.3 h) |
| Three recall-lift repetitions (2026-09-30) | **$3.34 actual combined** |

## Roadmap gates

Falsifiable claims for the retrieval roadmap, recorded against commits:

- **Multi-session and knowledge-update lift** — met in the three-run recall lift: 52.6% → 71.3% and 55.1% → 78.6% means (historical baseline → lift), with single-session assistant/user means up at 98.2%/97.6%. Further retrieval changes must preserve these gains.
- **Single-session preference** — still flat (56.7% historical baseline → 55.6% mean, −1.1 pp, 50.0–60.0% across runs) and now the active improvement target; a change ships only if preference moves without regressing the other types.
- **Turn-tier recall@10** — turn recall@10 must lift ≥3 points over baseline; Oracle must lift ≥1.
