# Benchmarks

Weft is benchmarked against [LongMemEval](https://github.com/xiaowu0162/LongMemEval), a multi-session memory evaluation dataset for chat assistants. Numbers below are reproducible from the harness in `benchmarks/longmemeval/`.

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

| Configuration | recall@10 | Answer correctness | Notes |
|---------------|-----------|--------------------|----|
| `--mode turns --tier turns` baseline | *running* | *running* | P0.2 baseline, captured 2026-05-07 |

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

```bash
python ~/code/LongMemEval/src/evaluation/evaluate_qa.py \
  gpt-4o \
  benchmarks/longmemeval/results/<your-run>.jsonl \
  ~/code/LongMemEval/data/longmemeval_m_cleaned.json
```

### Cost expectations

| Tier | Wall time | Embedding cost (OpenAI) | Judge cost (GPT-4o) |
|------|-----------|-------------------------|--------------------|
| S, n=3 smoke | ~5 min | <$0.05 | <$0.05 |
| M, strat50% (n≈251) | ~5–6 hours | ~$5–8 | ~$2–3 |
| M, full (n=500) | ~10–12 hours | ~$10–15 | ~$4–6 |

## Falsification gates

The retrieval roadmap commits to falsifiable claims at each phase, not pure improvements:

- **Phase 1 (turn-tier compounding + RRF dispatch)** — turn recall@10 must lift ≥3 points over baseline; Oracle must lift ≥1
- **Phase 2 (hierarchical retrieval)** — M-tier overall ≥ 0.75 with `WEFT_HIERARCHICAL=1`

Numbers are recorded against the corresponding commit in `tests/baselines.md` and the `longmemeval` topic in Weft itself. If a phase fails its gate, the design is wrong, not the test — post-mortem before proceeding.
