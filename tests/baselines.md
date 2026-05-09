# Weft Benchmark Baselines

Locked numbers for the 10× retrieval tackle path. Every Phase 1/2 leaf
measures itself against these — if a leaf can't show movement here,
its falsification gate fails and the plan revises rather than ships.

## Headline numbers

| Number | Value | Captured | Commit |
| --- | --- | --- | --- |
| **P0.1 — LongMemEval-M strat50, turns/turns (QA accuracy)** | overall **0.7689** / taREDACTED **0.7785** (n=251) | 2026-05-07 | `be4d352` |
| **P0.2 — turn-tier recall@10 (M strat50)** | **0.9482** (238/251) | 2026-05-07 | `34f1063` |

### Falsification gates (targets — these are revisited per leaf)

| Gate | Target | Absolute | Source |
| --- | --- | --- | --- |
| **P1.A4** turn-tier recall@10 lift vs P0.2 | ≥ 3 points | **≥ 0.978** | EPIC `loom-6e86575c` |
| **P1.B3** Tier 1.5 RRF Oracle lift | ≥ 1 point, stable | (S baseline + 1pt) | EPIC `loom-6e86575c` |
| **Phase 2** M-tier under `WEFT_HIERARCHICAL=on` | overall > 0.75 | **> 0.75 QA** | EPIC `loom-531d1c44` |

> Phase 2 gate sits below the P0.1 baseline (0.7689). The original verdict
> assumed a lower starting point. Treat overall > 0.75 as a floor — the
> informative signal is per-question-type lift on `knowledge-update` and
> `temporal-reasoning` (the haystack-noise classes). Revisit the gate
> wording before claiming Phase 2 success on overall accuracy alone.

## P0.1 — LongMemEval-M (50% stratified) baseline

**Numbers (n=251, judge gpt-4o):**

| metric | value |
| --- | --- |
| overall accuracy | 0.7689 (193/251) |
| taREDACTED accuracy | 0.7785 |
| single-session-assistant | 28/28 = 1.0000 |
| single-session-user | 34/35 = 0.9714 |
| multi-session | 50/67 = 0.7463 |
| temporal-reasoning | 46/67 = 0.6866 |
| knowledge-update | 26/39 = 0.6667 |
| single-session-preference | 9/15 = 0.6000 |

**Reproduce:**

```bash
# 1. Boot local Postgres + run migrations
docker compose -f docker-compose.weft.yml up -d
WEFT_DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
  uv run python -c "import asyncio, asyncpg; from weft.db.migrations._runner import run_migrations; \
  asyncio.run((lambda: (lambda p: run_migrations(p))(asyncpg.create_pool('postgresql://weft:weft_local@localhost:5433/weft')))())"

# 2. Get the M dataset (one-time, ~2.5 GB)
curl -L -o ../langchain/LongMemEval/data/longmemeval_m_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json

# 3. Run the adapter (~5.5 hr wall-clock on M-pro, n=251)
WEFT_DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../langchain/LongMemEval/data/longmemeval_m_cleaned.json \
    --mode turns --tier turns \
    --stratified-frac 0.5 --sample-seed 0

# 4. Judge (~3 min)
uv run python -m benchmarks.longmemeval.judge \
  --hyp benchmarks/longmemeval/results/longmemeval_m_cleaned_turns_tier-turns_strat50s0_<TS>.jsonl
```

**Cost:** ~$3.30 Anthropic (Sonnet, 1.04M tokens in / 11k out) + ~$3–5 OpenAI judge + minor embedding cost. Under the $15–30 budget the verdict's tackle path called out.

**Key observations vs the comparable 2026-05-04 S strat50 turns run (overall 0.8367, task-avg 0.8569):**

- Scaling from S (~40 sessions/q) to M (~500 sessions/q) costs ~7 points overall, ~8 points taREDACTED. That's the price of haystack noise the descent layer is meant to cut.
- `single-session-assistant` held perfect (28/28). Within-session retrieval is unaffected by haystack scale.
- `knowledge-update` dropped hardest (0.8462 → 0.6667, −18 pts). 12.5× more distractor sessions, more chances to retrieve the wrong update. This is the cleanest signal for hierarchical retrieval.
- `temporal-reasoning` (0.7910 → 0.6866) and `multi-session` (0.7612 → 0.7463) are the synthesis classes Phase 1's turn-tier compounding targets.

**Note on the Phase 2 gate.** Phase 2's done_when says "M-tier under flag-on exceeds 0.75." The pre-Phase-2 baseline already sits at **0.7689 overall / 0.7785 task-avg** because turn-tier (Branch A) is already shipped. The gate may have been written assuming a lower baseline; revisit before claiming Phase 2 success on overall accuracy alone. Per-question-type lifts (especially knowledge-update and temporal-reasoning) are likely the more informative signals.

**Run artifacts:**

- Hypothesis JSONL: `benchmarks/longmemeval/results/longmemeval_m_cleaned_turns_tier-turns_strat50s0_20260507T001033Z.jsonl`
- Eval-results: `<jsonl>.eval-results-gpt-4o`
- Metrics summary: `<jsonl>.metrics.json`
- Run log: `benchmarks/longmemeval/results/m_strat50s0_turns_run_20260507T001029Z.log`

## P0.2 — Turn-tier recall@10 (M strat50)

**Numbers (n=251, k=10, gold = `gold_session_ids`):**

| metric | value |
| --- | --- |
| recall@10 overall | **0.9482** (238/251) |
| single-session-assistant | 28/28 = 1.0000 |
| single-session-user | 35/35 = 1.0000 |
| knowledge-update | 38/39 = 0.9744 |
| multi-session | 64/67 = 0.9552 |
| single-session-preference | 14/15 = 0.9333 |
| temporal-reasoning | 59/67 = 0.8806 |

**Reproduce:** Same command as P0.1. Recall@k is auto-computed alongside
the QA hypothesis when `--mode turns --tier turns` is set (instrumentation
landed in commit `8ed3d49`). The summary file lands next to the hypothesis
JSONL as `<stem>_recall_at_10_summary.json`; per-question hits are in
`<stem>_recall_at_10.jsonl`.

```bash
WEFT_DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../langchain/LongMemEval/data/longmemeval_m_cleaned.json \
    --mode turns --tier turns \
    --stratified-frac 0.5 --sample-seed 0
```

**Run cost:** 5h34m wall, 1.04M input / 10.7k output tokens, 119 210 sessions
ingested, 0 failed questions. ~17 turns lost their vector to the
`text-embedding-3-small` 8192-token input cap (issue `weft-b1bd07c2`); only
1/13 misses correlated with that ceiling, so the structural recall ceiling
on this run is ~1 question, not 17.

**QA-vs-retrieval gap.** The headline number that matters for Phase 1 / 2 is
recall@10. The gap between recall@10 and QA accuracy is the answering/
extraction layer, not retrieval:

| Type | QA acc (P0.1) | recall@10 (P0.2) | gap |
| --- | --- | --- | --- |
| single-session-assistant | 1.0000 | 1.0000 | 0.000 |
| single-session-user | 0.9714 | 1.0000 | 0.029 |
| multi-session | 0.7463 | 0.9552 | 0.209 |
| knowledge-update | 0.6667 | 0.9744 | 0.308 |
| temporal-reasoning | 0.6866 | 0.8806 | 0.194 |
| single-session-preference | 0.6000 | 0.9333 | 0.333 |

`single-session-preference` and `knowledge-update` show the largest gaps:
retrieval finds the gold session, the answer comes back wrong. Phase 1 / 2
gates measure recall lift; QA lift is downstream of that.

**Miss cluster (13 misses out of 251):**

- temporal-reasoning: 8 (62%)
- multi-session: 3
- knowledge-update: 1
- single-session-preference: 1

Temporal-reasoning is the load-bearing class for Phase 2 hierarchical
descent. If hierarchical doesn't lift it specifically, the verdict's
falsifiable claim weakens.

**Run artifacts:**

- Hypothesis JSONL: `benchmarks/longmemeval/results/longmemeval_m_cleaned_turns_tier-turns_strat50s0_20260507T150321Z.jsonl`
- Per-question recall: `<jsonl-stem>_recall_at_10.jsonl`
- Summary: `<jsonl-stem>_recall_at_10_summary.json`
- Stats: `<jsonl>.stats.json`
