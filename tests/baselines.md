# Weft Benchmark Baselines

Locked numbers for the 10× retrieval tackle path. Every Phase 1/2 leaf
measures itself against these — if a leaf can't show movement here,
its falsification gate fails and the plan revises rather than ships.

## Headline numbers

| Number | Value | Captured | Commit |
| --- | --- | --- | --- |
| **P0.1 — LongMemEval-M strat50, turns/turns** | overall **0.7689** / taREDACTED **0.7785** (n=251) | 2026-05-07 | `be4d352` |
| P0.2 — turn-tier recall@10 | _pending — `loom-22bf6b24`_ | — | — |

### Falsification gates (targets — these are revisited per leaf)

| Gate | Target | Source |
| --- | --- | --- |
| **P1.A4** turn-tier recall@10 lift vs P0.2 | ≥ 3 points | EPIC `loom-6e86575c` |
| **P1.B3** Tier 1.5 RRF Oracle lift | ≥ 1 point, stable | EPIC `loom-6e86575c` |
| **Phase 2** M-tier under `WEFT_HIERARCHICAL=on` | overall > 0.75 | EPIC `loom-531d1c44` |

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

## P0.2 — Turn-tier recall@10

_Not yet captured. Tracked as `loom-22bf6b24`. Adds a small extension to the
adapter (or a sibling script) to compute recall@10 against the gold evidence
sessions in M. Lock the number here when it lands._
