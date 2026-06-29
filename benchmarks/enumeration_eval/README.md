# Enumeration Eval Harness

Measures **recall@membership** for collection retrieval under two paths:

1. **Oracle Path** (deterministic): `gather_topic_memories` — must return ALL M members
   - Single run per fixture
   - Assert recall@membership == 1.0
   - Proves V1 completeness (unbounded gather)

2. **Candidate Path** (stochastic): `weft_recall` — natural-language semantic recall
   - k≥5 runs per fixture
   - Report min/median/max spread
   - Diagnostic signal for retrieval consistency

## Metric

```
recall@membership = |returned ∩ members(C)| / |members(C)|
```

For a fixture collection C with known membership set M:
- Calculate intersection of returned memories with M
- Divide by expected cardinality |M|

## Fixtures

All fixtures use the **Jim Boblaw** synthetic persona (house rule: never real names).

### Collections

- **Plants** (12 members): "Jim grows tomatoes...", "Jim has a pothos...", etc.
  - Topic tag: `plants`
- **Medications** (8 members): "Jim takes aspirin...", "Jim uses metformin...", etc.
  - Topic tag: `medications`
- **Books** (5 members): "Jim read 'The Great Gatsby'...", etc.
  - Topic tag: `books`

## Running

### Via entrypoint (standalone)

```bash
# Requires Docker (testcontainers spins up Postgres pgvector + Redis)
uv run python -m benchmarks.enumeration_eval
```

Output: 
- Detailed logs to stdout
- `benchmarks/enumeration_eval/results.json` (JSON report)

### Via pytest (optional smoke test)

```bash
uv run pytest benchmarks/enumeration_eval/tests/ -v
```

## Architecture

- **fixtures.py**: Jim Boblaw persona + known-membership collections
- **harness.py**: `enum_gather_oracle()` and `enum_recall_candidate()` implementations
- **__main__.py**: Runnable entrypoint
- **conftest.py**: Reuses testcontainers fixtures from `tests/conftest.py`

## Expected Results

- **Oracle path**: 100% (all collections return 1.0 recall@membership)
- **Candidate path**: Typically lower; spread indicates consistency (lower spread = more reliable)
