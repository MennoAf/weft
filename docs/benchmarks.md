# Benchmarks

Weft is benchmarked against [LongMemEval](https://github.com/xiaowu0162/LongMemEval), a multi-session memory evaluation dataset for chat assistants. Numbers below are reproducible from the harness in `benchmarks/longmemeval/`.

## Qualification evidence classes

The public benchmark contract leads with the shipped local FastEmbed profile:
`BAAI/bge-small-en-v1.5` (provider `fastembed`, model
`BAAI/bge-small-en-v1.5`). Its native/signal width is **384**; Weft stores and
emits **768** dimensions for the local pgvector contract, with zero padding that
adds no semantic information. Dimensions describe the vector interface, not
parameter count.

Every future arm manifest and report MUST identify provider, model,
native/signal dimensions, storage/output dimensions, profile/snapshot identity,
and fixed controls. Controls include dataset split/checksum, question IDs and
order, ingest representation, routing, retrieval tier and `top_k`, Reader
model and prompt, judge/scoring, and retry behavior. This is a requirement for
new qualification evidence, not a claim that every historical artifact already
contains all fields.

Evidence classes are deliberately separate:

1. **Provider-free contract tests** use deterministic fake/spy providers. They
   prove label-blind task shape and runtime traces, artifact/output contracts,
   and fail-closed coverage accounting. They make no provider calls and do not
   prove answer quality.
2. **Labeled live-provider smoke tests** are an operator-authorized, separately
   labeled small-fixture check against a live provider. They are not
   provider-free evidence and do not establish a qualification lift.
3. **Paid qualification** is a separately authorized repeated run with frozen
   inputs/configuration, raw artifacts, judge results, costs, and failure
   categories. It remains **HOLD** here; no paid execution is performed by
   this repository's contract tests.

Historical runs remain historical evidence. Their summaries must not be
promoted into a current lift claim or used to hide missing hypotheses/judge
results. Hosted OpenAI arms are optional later comparisons only, after explicit
authorization; they are not executed here.

## What we measure

LongMemEval ships two haystack tiers:

- **LongMemEval-S** (~40 sessions/question, ~100K tokens) — the published leaderboard tier
- **LongMemEval-M** (~500 sessions/question, ~1M tokens) — stress-tests retrieval at scale

We score on two metrics:

- **Answer correctness** — judged by GPT-4o per LongMemEval's evaluator
- **recall@10** — does the gold-evidence turn appear in the top-10 retrieved candidates? Independent of the answering model

The `recall@10` metric isolates retrieval quality from generation quality, so improvements to the storage/retrieval layer surface immediately rather than getting lost in answering noise.

## Current numbers

> *Results from in-progress baselining — table below will be updated as runs land.*

### LongMemEval-S

| Configuration | Overall | Single-session-assistant | Single-session-preference | Source |
|---------------|---------|--------------------------|---------------------------|--------|
| Honest-extracted baseline (n=500) | 43.6% | — | 16% | 2026-05-03 |
| Raw-fallback (n=500) | 42.0% | — | 27% | 2026-05-03 |

Honest extraction (LLM-derived beliefs) and raw fallback (full-turn ingest) have different failure modes — see `weft_search_all` for the analysis memories tagged `longmemeval`.

### LongMemEval-M

**Publication status: unresolved.** Historical M runs exist, but their summaries were generated from labelled-result rows rather than the complete reference population. Missing hypotheses and missing judge results could disappear from the denominator. Those artifacts must be rescored through the reference → hypothesis → result contract before any M headline is published.

| Configuration | recall@10 | Answer correctness | Notes |
|---------------|-----------|--------------------|----|
| `--mode turns --tier turns` historical runs | retained in raw artifacts | **not publication-ready** | Results vary materially across May runs; ingestion completion, retrieval, reader accuracy, and infrastructure failures must be reported separately. |

The repaired scorer uses the reference question set as the denominator. A question with no produced hypothesis or no judge label remains incorrect and is reported under its distinct pipeline-failure category. Duplicate or unknown IDs and malformed labels fail the summary rather than being silently counted or omitted.

## How to reproduce

### Prerequisites

```bash
# Clone LongMemEval
git clone https://github.com/xiaowu0162/LongMemEval ~/code/LongMemEval
# Download the cleaned datasets per the LongMemEval README
```

### Run the harness

```bash
WEFT_DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
  uv run python -m benchmarks.longmemeval.adapter \
    --dataset ~/code/LongMemEval/data/longmemeval_m_cleaned.json \
    --mode turns \
    --tier turns \
    --stratified-frac 0.5 \
    --sample-seed 0
```

Outputs land in `benchmarks/longmemeval/results/` — both an answers JSONL (for the LongMemEval evaluator) and a `recall_at_10` summary.

### Score the answers

Use Weft's wrapper so the metrics file receives all three pipeline artifacts and applies the complete reference denominator:

```bash
OPENAI_API_KEY=... uv run python -m benchmarks.longmemeval.judge \
  --hyp benchmarks/longmemeval/results/<your-run>.jsonl \
  --ref ~/code/LongMemEval/data/longmemeval_m_cleaned.json \
  --model gpt-4o
```

The upstream evaluator writes `<hyp>.eval-results-<model>`; Weft then writes `<hyp>.metrics.json` with expected, hypothesis, judge-result, missing-stage, correctness, and per-type counts. Scoring an existing labelled sidecar is free; generating missing labels is paid.

### Repeatability and publication protocol

A single aggregate run is not a keep/scrap or publication gate. Before publishing M or comparing retrieval arms:

1. Pin the code commit, dataset checksum, model identifiers, prompt/configuration, sample IDs, and random seeds.
2. Either run **3–5 independent repetitions per arm**, or hold stochastic substrate generation constant with a versioned fixed-materialization snapshot and repeat only the intended variable.
3. Preserve raw hypotheses, labelled results, metrics, stats, and failure logs for every repetition.
4. Report distributions and these stages separately: ingestion completion, retrieval recall, reader accuracy, preference compliance, enumeration, unsupported answers, and infrastructure failures.
5. Treat missing hypotheses/results as incorrect. Do not compress elapsed time into evidence of completed coverage.
6. Record estimated cost before execution and actual cost afterward. Paid runs require explicit operator approval.

### Session continuity A/B/C benchmark

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
per-arm mechanics and keeps those paid metrics explicitly pending. Repeated
paid reader evaluation remains
`PENDING-PAID-EVALUATION`, and neither Arm B automation nor Arm C materializer
wiring is enabled by the benchmark.

Run deterministic mechanics without an LLM:

```bash
uv run pytest benchmarks/personal_agent/tests/test_paah_continuity.py -q
```

A future paid run must repeat each enabled arm 3–5 times, preserve raw
artifacts, and report correctness, unsupported claims, evidence citation,
stale/superseded answers, instruction non-compliance, activation precision,
latency, token cost, and infrastructure failures separately.

### Cost expectations

| Tier | Wall time | Embedding cost (OpenAI) | Judge cost (GPT-4o) |
|------|-----------|-------------------------|--------------------|
| S, n=3 smoke | ~5 min | <$0.05 | <$0.05 |
| M, strat50% (n≈251) | ~5–6 hours | ~$5–8 | ~$2–3 |
| M, full (n=500) | ~10–12 hours | ~$10–15 | ~$4–6 |

## RC-FL-20 qualification preparation

`uv run python scripts/prepare_rc_qualification.py --output evidence/qualification.md --budget evidence/benchmark-budget.json` emits the deterministic qualification run card and budget. The generated status is always **`PREPARED — NOT AUTHORIZED`**. Preparation does not execute a provider, database, hosted arm, Reader, judge, or paid call; the budget is a non-authorizing spend guardrail, not permission to execute.

The run card validates an immutable A/B/C matrix: required local FastEmbed `BAAI/bge-small-en-v1.5` primary (native/signal 384, stored/output 768), optional hosted OpenAI `text-embedding-3-small` controlled comparison (1536/768), and optional hosted `text-embedding-3-large` ceiling comparison (3072/3072). Every arm's role, provider/model, local/hosted state, execution state/text, credential requirement, comparison semantics, profile ID, snapshot ID, and dimension fields are canonical. Hosted profiles remain disabled until a separate dated operator authorization supplies scope, frozen prices, and a non-authorizing spend guardrail.

Controls include exact execution-boundary keys (`authorization_required_before_execution=true`, credentials prohibited from manifest/CLI, preparation-only, no provider/paid/database calls, hosted disabled, and no authorization placeholder), label/gold-blind TaskShape derivation, and source-byte provenance. `recall_k=10` is the retrieval metric cutoff; Reader candidate width is the per-question TaskShape `top_k` (10 or 30), widened only by explicit `max(requested_top_k, derived_top_k)`. The validator re-reads every declared repository-relative source under an explicit root, rejects traversal/symlinks and path-set changes, and recomputes the source-boundary digest from current bytes. It never trusts HEAD metadata or its own stored hashes.

Both planned tiers are explicit: S smoke (5) plus M stratified (251) equals **256 questions per repetition**. Reader and judge each have 256 question-stage calls per repetition and 768 over three repetitions. Embedding item count remains dataset-dependent until a frozen ingest manifest exists. Local embedding provider spend is separately known as $0, but local all-stage and all-arm totals remain unknown until Reader/judge prices and item counts are known; no known total `$0` is emitted. Provider-free contract, metamorphic, summary, and local Docker receipts remain contract evidence only, never lift claims; historical results remain historical.

The external dataset checksum, exact sampled question-ID digest, ordered ID manifest/hash, dated authorization, frozen prices, and cleared blockers are deliberately absent in this checkout. A separate execution-readiness validator accepts only lowercase 64-hex identities, exactly 251 unique nonempty ordered IDs, and canonical bindings. `ordered_question_ids_sha256` is SHA-256 of the canonical ordered-ID JSON list; `question_ids_and_order_digest` is SHA-256 of canonical JSON `{"dataset_sha256": <dataset checksum>, "ordered_question_ids": <list>}`, so an arbitrary digest or reordered/duplicated list cannot pass. Authorization is a canonical record over operator/decision IDs, the exact qualification purpose and hosted arms, max spend/currency, timezone-aware `authorized_at`, and canonical model/profile IDs plus `prices_sha256`; its record hash is recomputed, never trusted. Each embedding/Reader/judge price is an amount/unit/currency/source/timezone-aware `as_of` record, is bound by `prices_sha256`, and participates in readiness/budget arithmetic; the maximum-spend guardrail remains a hard stop, not approval. Preparation reads no environment values and does not load `.env`; readiness accepts only unique sanitized key names (never values), rejecting forbidden or secret/path-shaped names. Canonical serialization is UTF-8 JSON with sorted keys and compact separators, hashed with SHA-256.

## Falsification gates

The retrieval roadmap commits to falsifiable claims at each phase, not pure improvements:

- **Phase 1 (turn-tier compounding + RRF dispatch)** — turn recall@10 must lift ≥3 points over baseline; Oracle must lift ≥1
- **Phase 2 (hierarchical retrieval)** — M-tier overall ≥ 0.75 with `WEFT_HIERARCHICAL=1`

Numbers are recorded against the corresponding commit in `tests/baselines.md` and the `longmemeval` topic in Weft itself. If a phase fails its gate, the design is wrong, not the test — post-mortem before proceeding.
