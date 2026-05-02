# LongMemEval × Weft

Adapter that runs Weft against the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) memory benchmark.

## Why this exists

LongMemEval tests memory systems on five abilities — single-session recall, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. The benchmark's plug-in contract is trivially file-based (emit JSONL of `{question_id, hypothesis}`), so the cost of running it is small. The interesting question is empirical: **does Weft's belief-shaped extraction preserve enough fidelity for question types that need temporal precision and multi-session synthesis, vs. raw dialogue-trace storage?** Both ingest modes are supported here so the benchmark can answer that.

## Pipeline

Per question:

1. Sandbox to `project_id="lme_<question_id>"`
2. Ingest the haystack via the chosen mode
3. Hybrid recall (vector + BM25 + RRF) for the question
4. Claude Reader produces a hypothesis (system prompt cached per question type)
5. Append `{question_id, hypothesis}` JSONL line
6. Hard-delete the question's memories (unless `--no-cleanup`)

## Ingest modes

- **`raw`** — each session is written as one `MemoryType.fact` memory, full dialogue preserved. Embedding computed locally. No LLM extraction. This is the high-fidelity baseline.
- **`extracted`** — each session is fed through Weft's existing `ingest_pipeline.process()`, which classifies intent, resolves entities, and writes derived memories. This is what production Weft does for Slack/Obsidian ingest.

## Setup

```bash
# 1. Get the dataset (sibling of Weft repo)
git clone https://github.com/xiaowu0162/LongMemEval ../LongMemEval

# 2. Make sure Weft Postgres is running
docker compose -f docker-compose.weft.yml up -d

# 3. Make sure ANTHROPIC_API_KEY is in env (for the Reader)
export ANTHROPIC_API_KEY=sk-...
```

## Usage

```bash
# Smoke test — 5 questions on Oracle (cheapest split, evidence sessions only)
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../LongMemEval/data/longmemeval_oracle.json \
    --mode raw --limit 5

# Full Oracle run, raw mode
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../LongMemEval/data/longmemeval_oracle.json \
    --mode raw

# Same, extracted mode (the architectural comparison)
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../LongMemEval/data/longmemeval_oracle.json \
    --mode extracted

# Headline LongMemEval_S run (~$5–15 in judge cost when scored)
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../LongMemEval/data/longmemeval_s.json \
    --mode raw
```

Output:

- `results/<split>_<mode>_<UTC-timestamp>.jsonl` — hypotheses (the eval contract)
- `results/<split>_<mode>_<UTC-timestamp>.jsonl.stats.json` — token + timing stats

## Scoring with the upstream evaluator

The hypotheses file is the exact format LongMemEval's judge expects. From the LongMemEval clone:

```bash
cd ../LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o \
    ../../../Weft/benchmarks/longmemeval/results/<file>.jsonl \
    ../../data/longmemeval_oracle.json
python print_qa_metrics.py <labeled_file>
```

Judge cost is roughly $5–15 per full split run.

## Tests

```bash
uv run pytest benchmarks/longmemeval/tests/ -v
```

Smoke tests use a stubbed Reader (no Anthropic calls) but exercise real Postgres + pgvector + FastEmbed via testcontainers. They verify:

- Dataset loader parses LongMemEval JSON
- Raw-mode ingest produces one JSONL line per question with the correct contract
- `--no-cleanup` leaves memories in the database
- Hybrid recall actually surfaces the evidence session for a softball question (the structural readiness check)

## Files

- `dataset.py` — typed loader (`Instance`, `Session`, `Turn`)
- `ingest.py` — raw vs extracted mode handlers, project sandboxing, cleanup
- `reader.py` — Claude Reader with question-type-specific system prompts and prompt caching
- `adapter.py` — orchestration + CLI (entry point: `python -m benchmarks.longmemeval.adapter`)
- `tests/test_smoke.py` — integration smoke tests against real Postgres
