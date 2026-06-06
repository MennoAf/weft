# Weft Benchmark Baselines

Locked numbers for the 10× retrieval tackle path. Every Phase 1/2 leaf
measures itself against these — if a leaf can't show movement here,
its falsification gate fails and the plan revises rather than ships.

## Headline numbers

| Number | Value | Captured | Commit |
| --- | --- | --- | --- |
| **P0.1 — LongMemEval-M strat50, turns/turns (QA accuracy)** | overall **0.7689** / task-averaged **0.7785** (n=251) | 2026-05-07 | `be4d352` |
| **P0.2 — turn-tier recall@10 (M strat50)** | **0.9482** (238/251) | 2026-05-07 | `34f1063` |
| **P1.A5 Run 1 — warm rerank-ON recall@10** | **0.9482** (238/251 norm., 238/249 succ.) | 2026-05-09 | `a33ad6d` |
| **P1.A5 Run 1 — warm rerank-ON QA accuracy** | **0.7430** (185/249), task-avg **0.7432** | 2026-05-09 | _this commit_ |
| ~~P1.A5 Run 2 — warm rerank-OFF recall@10~~ | ~~pending~~ | — | **skipped — decision `weft-b8efcc24`** |

### Falsification gates (targets — these are revisited per leaf)

| Gate | Target | Absolute | Source |
| --- | --- | --- | --- |
| ~~**P1.A4** turn-tier recall@10 lift vs P0.2~~ | ~~≥ 3 points~~ | **REFRAMED** | see "Track A measurement caveat" below |
| **P1.B3** Tier 1.5 RRF Oracle lift | ≥ 1 point, stable | (S baseline + 1pt) | EPIC `loom-6e86575c` |
| **Phase 2** M-tier under `WEFT_HIERARCHICAL=on` | overall > 0.75 | **> 0.75 QA** | EPIC `loom-531d1c44` |

> Phase 2 gate sits below the P0.1 baseline (0.7689). The original verdict
> assumed a lower starting point. Treat overall > 0.75 as a floor — the
> informative signal is per-question-type lift on `knowledge-update` and
> `temporal-reasoning` (the haystack-noise classes). Revisit the gate
> wording before claiming Phase 2 success on overall accuracy alone.

### Track A measurement caveat (P1.A4 reframe)

The Phase 1 Track A falsification gate ("turn-tier compounding lifts
recall@10 by ≥ 3 points") is **structurally untestable on the cold-DB
LongMemEval harness.** Two compounding reasons:

1. **Baseline contamination.** P0.2 (`recall@10 = 0.9482` at commit
   `34f1063`) was captured AFTER all three Track A items merged
   (`b12857d` A1, `6c95374` A2, `35669b1` A3). The locked baseline IS
   the post-Track-A number. There's no on-file pre-Track-A counterfactual.
2. **Cold-DB harness.** Each benchmark question runs against a fresh
   database. New turns ingest at the default `usefulness_score = 0.7`,
   `last_boosted_at = None`. That makes `usefulness_factor ≈ 0.85` for
   every turn (constant — cancels in ranking) and disables the decay
   path. In benchmark conditions, the Track A rerank reduces to
   `RRF × recency_by_occurred_at`. The compounding boost signal the
   verdict's falsifiable claim was about is not exercised.

**Decision (2026-05-08).** Track A stays on main as a structural piece
— the loop is wired, awaiting accumulation in real personal-recall use.
The cold-DB falsification gate is retired. P1.A4 (`loom-d32a8353`)
cancelled-as-reframed. A warm-boost harness variant is filed as
`loom-3363e387` (P1.A5): pre-warm phase that simulates N rounds of
recall + boost, plus a `WEFT_TURN_RERANK_DISABLE` flag for clean A/B
on the same warmed dataset.

Diagnosis logged at `weft-dea4cce7`.

### P1.A5 Run 1 — warm rerank-ON (M strat50, warm-boost rounds=3)

**Headline:** recall@10 = **0.9482** (238/251 normalized, 238/249 on
succeeded subset). **Numerically identical to the cold P0.2 baseline.**

**Per-question-type recall@10 vs P0.2 cold:**

| type | cold P0.2 | warm Run 1 | hit Δ |
| --- | --- | --- | --- |
| single-session-assistant | 28/28 = 1.000 | 27/27 = 1.000 (1 failed) | -1 (lost to failure) |
| single-session-user | 35/35 = 1.000 | 35/35 = 1.000 | 0 |
| multi-session | 64/67 = 0.9552 | 64/67 = 0.9552 | 0 |
| **knowledge-update** | 38/39 = 0.9744 | **39/39 = 1.0000** | **+1 (rerank-attributable)** |
| temporal-reasoning | 59/67 = 0.8806 | 59/66 = 0.8939 (1 failed) | 0 |
| single-session-preference | 14/15 = 0.9333 | 14/15 = 0.9333 | 0 |

**Net: +1 hit on knowledge-update, −1 hit lost to a tsquery failure on
single-session-assistant.** Apples-to-apples on the 249-question subset:
Track A's compounding rerank moves at most **+1 question** (≈ +0.4pt),
well below the ≥ 3pt falsification threshold.

**Run details:**

- elapsed: 28 229s (7h50m wall)
- 1.02M input tokens, 11.1k output, 0 cached
- 2 questions failed (`tsquery stack too small` on warmup queries with
  pathological content prefixes — fixed in commit `a33ad6d` for future
  runs, but Run 1 was on unpatched code)
- warm-boost telemetry: 747 rounds ran, 7 470 queries, 74 570 accesses,
  73 832 boosts (i.e. the loop was firmly exercised)
- file: `benchmarks/longmemeval/results/longmemeval_m_cleaned_turns_tier-turns_strat50s0_warm3_20260509T012034Z*`

**Verdict (final — Run 2 skipped):**

The Lodestar verdict's claim — _"Track A turn-tier compounding lifts
recall@10 by ≥ 3 points"_ — is **falsified.** Even with 74k boost
events accumulated across 3 rounds × 10 queries × ~250 questions, the
rerank surfaces exactly **one** previously-missed gold session.

**Run 2 (rerank-OFF) was the planned rerank-isolation control.** It is
skipped as decision `weft-b8efcc24` (2026-05-09): Run 1 alone falsifies
the +3pt claim; Run 2 only answers the architectural curiosity "does
the rerank do anything beyond RRF order?" — not worth ~8h + ~$10 right
now. If we ever want that signal, do it cheap on a 50-Q subset.

**Worse than no-op on QA.** The 2026-05-09 warm3 hypotheses were judged
with `gpt-4o` (commit see git log) and the warm-boost rerank actively
**regressed** total accuracy on the 249 questions both runs answered:

| metric | P0.1 baseline (no warm-boost) | P1.A5 Run 1 (warm3) | Δ |
| --- | ---: | ---: | ---: |
| overall accuracy | 0.7689 (193/251) | 0.7430 (185/249) | **−2.59 pt** |
| task-averaged accuracy | 0.7785 | 0.7432 | **−3.53 pt** |
| knowledge-update | 26/39 = 0.6667 | 26/39 = 0.6667 | 0 |
| multi-session | 50/67 = 0.7463 | 48/67 = 0.7164 | −2.99 pt (lost 2) |
| single-session-assistant | 28/28 = 1.0000 | 27/27 = 1.0000 | −1 hit (q failed in ingest) |
| single-session-preference | 9/15 = 0.6000 | 7/15 = 0.4667 | **−13.33 pt (lost 2)** |
| single-session-user | 34/35 = 0.9714 | 33/35 = 0.9429 | −2.85 pt (lost 1) |
| temporal-reasoning | 46/67 = 0.6866 | 44/66 = 0.6667 | −1.99 pt (lost 2 + 1 ingest fail) |

The boost loop did exactly what it was designed to do — re-rank turns
by accumulated usefulness — but that re-ranking nudged out turns the
Reader was actually using to synthesize correct answers. Net: the
rerank is not just unhelpful, it's mildly harmful at the QA layer.

Files:
- `benchmarks/longmemeval/results/longmemeval_m_cleaned_turns_tier-turns_strat50s0_warm3_20260509T012034Z.jsonl.eval-results-gpt-4o`
- `<same>.metrics.json`
- Local-only validation summary: `benchmarks/longmemeval/results/turn_tier_validation_2026_05_09.md`

**Reproduce Run 1:**

```bash
WEFT_DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
DATABASE_URL="postgresql://weft:weft_local@localhost:5433/weft" \
uv run python -m benchmarks.longmemeval.adapter \
    --dataset ../langchain/LongMemEval/data/longmemeval_m_cleaned.json \
    --mode turns --tier turns \
    --stratified-frac 0.5 --sample-seed 0 \
    --warm-boost-rounds 3
```

## P0.1 — LongMemEval-M (50% stratified) baseline

**Numbers (n=251, judge gpt-4o):**

| metric | value |
| --- | --- |
| overall accuracy | 0.7689 (193/251) |
| task-averaged accuracy | 0.7785 |
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

- Scaling from S (~40 sessions/q) to M (~500 sessions/q) costs ~7 points overall, ~8 points task-averaged. That's the price of haystack noise the descent layer is meant to cut.
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

## Turn-tier validation — temporal-reasoning empty-rate (loom-ee973301)

**Question:** does turn-tier ingest + retrieval drop the temporal-reasoning
empty-rate below 15% (the gap diagnosed in `project_recall_completeness_diagnosis.md`)?

**Verdict:** ✅ **Yes — 3.0% on M-tier turn-tier baseline (vs <15% target).**

Naive empty-detection on the JSONL hypotheses (empty string, "I don't
know", "no information", "cannot determine"). Same row across the same
strat50 sample for both M-tier rows.

| Question type | Oracle raw (n=500) | Oracle extracted post-fence (n=500) | M turn-tier baseline (n=251) | M turn-tier warm3 (n=249) |
| --- | ---: | ---: | ---: | ---: |
| knowledge-update | 3.8% | 5.1% | **0.0%** | 5.1% |
| multi-session | 3.0% | 6.8% | 6.0% | 3.0% |
| single-session-assistant | 0.0% | 0.0% | 0.0% | 0.0% |
| single-session-preference | 0.0% | 0.0% | 0.0% | 0.0% |
| single-session-user | 0.0% | 0.0% | 0.0% | 0.0% |
| **temporal-reasoning** | 5.3% | 14.3% | **3.0%** | **10.6%** |

The diagnosis-era memory referenced "raw 37% empty, extracted 31%
empty" on temporal-reasoning. Naive empty-detection on the current
(cleaned) dataset shows lower numbers across the board — the original
diagnosis number was likely measured under a stricter "did not produce a
useful answer" definition. Either way, the M-tier turn-tier baseline at
**3.0%** clears the original target by a wide margin, and the
post-fence Oracle extracted at 14.3% is right at the threshold —
showing the gain from belief→turn ingest is real on the load-bearing
question class.

**Total accuracy floor (no worse than the better of raw/extracted):**

The M-tier turn-tier baseline at **0.7689 overall / 0.7785 task-avg**
beats every Oracle raw/extracted run on every shared question type
(comparison split, but Oracle is the easier benchmark — turn-tier wins
on the harder M split anyway).

`loom-ee973301` is satisfied. Local-only summary doc with the full
comparison table: `benchmarks/longmemeval/results/turn_tier_validation_2026_05_09.md`.

## Recall-vs-Reader bucketing spike (2026-05-09 evening)

**Question:** of the 58 M-tier turn-tier baseline failures, are they
retrieval failures (gold context not in top-10) or Reader failures
(gold context retrieved, answer still wrong)? The answer decides
whether Phase 2 hierarchical retrieval (loom-531d1c44) ships or pivots.

**Inputs (no fresh run — both files already on disk):**

- QA eval: `longmemeval_m_cleaned_turns_tier-turns_strat50s0_20260507T001033Z.jsonl.eval-results-gpt-4o`
- Recall@10: `longmemeval_m_cleaned_turns_tier-turns_strat50s0_20260507T150321Z_recall_at_10.jsonl`

Joined by `question_id` — perfect overlap (251/251).

**Verdict:** ✅ **Reader is the bottleneck. 81% of failures had gold context retrieved.**

Cross-tab (n=251):

| | QA OK | QA FAIL | Total |
|---|---:|---:|---:|
| Recall@10 HIT | 191 | **47** (Reader fails) | 238 |
| Recall@10 MISS | 2 | **11** (Retrieval fails) | 13 |
| **Total** | 193 | 58 | 251 |

- P(QA correct \| HIT) = 191/238 = **80.3%** — the Reader's ceiling on
  the retrieved context.
- P(QA correct \| MISS) = 2/13 = **15.4%** — lucky-guess floor when
  no gold context is in the prompt.

**Headroom (assuming current P(correct\|HIT) on the new context):**

| Layer fix | New accuracy | Δ |
| --- | ---: | ---: |
| Current overall | 76.89% | — |
| Reader fixes every HIT_FAIL | **95.62%** | +18.73pt |
| Retrieval fixes every MISS_FAIL | 81.27% | +4.38pt |

Reader-to-retrieval headroom ratio: **4.3x**.

**By question type — Reader-share of failures:**

| type | n | pass% | H_OK | H_FL | M_OK | M_FL | Reader-share of fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| knowledge-update | 39 | 66.7% | 25 | 13 | 1 | 0 | **13/13 (100%)** |
| multi-session | 67 | 74.6% | 49 | 15 | 1 | 2 | **15/17 (88%)** |
| single-session-assistant | 28 | 100.0% | 28 | 0 | 0 | 0 | n/a |
| single-session-preference | 15 | 60.0% | 9 | 5 | 0 | 1 | **5/6 (83%)** |
| single-session-user | 35 | 97.1% | 34 | 1 | 0 | 0 | 1/1 (100%) |
| temporal-reasoning | 67 | 68.7% | 46 | 13 | 0 | 8 | **13/21 (62%)** |

knowledge-update is the cleanest signal: **every** failing question
had gold context retrieved. There is nothing for hierarchical
retrieval to fix on knowledge-update — the entire gap is the Reader
not reasoning correctly over an updated fact.

Temporal-reasoning is the only type where retrieval-side losses
matter (8 of 21 failures). Even there, Reader-fix headroom (+19pt
on the type) is larger than retrieval-fix headroom (+12pt on the
type).

**Implication for Phase 2 (loom-531d1c44):** hierarchical retrieval
caps at +4.38pt overall, almost all of it concentrated in
temporal-reasoning. The overall accuracy ceiling for any
retrieval-only intervention on this baseline is **81.27%**.
The 95.62% ceiling lives downstream — in whatever is reading the
context.

**Run artifacts:**

- JSON cross-tab: `benchmarks/longmemeval/results/recall_vs_reader_bucketing_2026_05_09.json`
- Inputs: the two files listed above

### Correction: the 81% headline overstated Reader-share

The 4-bucket cross-tab above uses `recall_at_k_hit` as-recorded in the
recall@10 file, which is **session-level**: a question is HIT if at
least one gold session is represented in the 10 retrieved turns. For
multi-session questions where there are 3, 5, or 6 gold sessions and
only 1-2 are retrieved, the metric still reports HIT — even though
the answer-bearing turn(s) are likely in the missed gold sessions.

A 12-question stratified eyeball on HIT_FAIL examples surfaced this
quickly (e.g. qid `0977f2af` "kitchen gadget before Air Fryer" — gold
session for *Instant Pot* was not retrieved, only the *Air Fryer* gold
session; recall@10 reports HIT but the Reader cannot answer).

**Re-bucketing the 47 HIT_FAIL questions by gold-session coverage:**

| Bucket | n_OK | n_FAIL | Notes |
| --- | ---: | ---: | --- |
| HIT_FULL (all gold sessions retrieved) | 156 | 20 | cleanest "Reader-only failure" |
| HIT_PARTIAL (some gold sessions missed) | 35 | 27 | retrieval covered partially |
| MISS (no gold sessions retrieved) | 2 | 11 | clean retrieval miss |

**Failure share corrected:** Reader-only **34.5%**, retrieval-side
total **65.5%** (HIT_PARTIAL + MISS). This **inverts** the
session-level headline.

**Corrected ceilings:**

| Layer fix | New accuracy | Δ |
| --- | ---: | ---: |
| Reader fixes every HIT_FULL fail | 84.86% | +7.97pt |
| Retrieval fixes every PARTIAL+MISS fail | **92.03%** | **+15.14pt** |

Retrieval has ~1.9× the headroom of the Reader.

**By type — the failure regimes are now distinct:**

| type | fails | HIT_FULL | HIT_PART | MISS | Reader-only % |
| --- | ---: | ---: | ---: | ---: | ---: |
| knowledge-update | 13 | 9 | 4 | 0 | **69.2%** |
| single-session-preference | 6 | 5 | 0 | 1 | **83.3%** |
| single-session-user | 1 | 1 | 0 | 0 | 100% |
| multi-session | 17 | 3 | 12 | 2 | **17.6%** |
| temporal-reasoning | 21 | 2 | 11 | 8 | **9.5%** |

knowledge-update + preference + user (n=20 fails) is overwhelmingly
**Reader-driven** — gold context is fully retrieved and the Reader
fails to reason over it (recency-priority, preference application,
abstention). multi-session + temporal-reasoning (n=38 fails) is
overwhelmingly **retrieval-driven** — partial gold-session coverage
is the dominant failure shape.

**Sample-12 failure-mode taxonomy (HIT_FAIL inspection):**

1. **Recall-hit-but-turn-miss** — gold session in retrieved set, but
   answer-bearing turn isn't (multi-session, temporal-reasoning).
2. **Reader recency/update failure** — multiple values in context,
   Reader picks the older or more-discussed one (e.g. Hawaii vs Paris
   for "most recent family trip").
3. **Reader hallucination on `_abs` (should-abstain) questions** — gold
   context is silent on the asked detail, Reader fabricates from world
   knowledge (e.g. bus-cost estimate, vintage *films* answered with
   vintage-*camera* duration).
4. **Reader preference-blindness** — preference signal in context,
   Reader gives a generic answer (e.g. cultural-events question whose
   gold session is full of language-learning context).
5. Likely judge noise — 1 of 12 borderline.

**Implication for Phase 2 (loom-531d1c44):** hierarchical retrieval is
the right intervention for the multi-session + temporal-reasoning
cluster (38 fails, mostly retrieval-side; ceiling +15.14pt). A small
Reader-prompt pass — abstention rule for `_abs`-style questions,
recency-priority rule for updates, explicit preference-application
instruction — also has standalone value for the knowledge-update +
preference + user cluster (20 fails, mostly Reader-side; ceiling
+7.97pt). The two interventions don't compete: they target disjoint
question-type clusters.

Cheapest sequence: Reader prompt pass first (a few hours, no infra
work), measure delta, then commit to Phase 2 retrieval for the
remaining retrieval-side fails.

**Run artifact (corrected):** `benchmarks/longmemeval/results/recall_vs_reader_bucketing_2026_05_09_extended.json`
(6-bucket cross-tab + per-question gold-coverage detail).

## Turn-level recall@k instrumentation (loom-f658cd55)

New telemetry fields added to the per-question recall JSONL (turn-mode + turn-tier only):

- `turn_level_recall_at_k_hit` — `true` if the gold answer text appears (via substring or
  word-boundary match) in any of the top-k retrieved turns' content.
- `n_retrieved_turns_with_content` — count of retrieved turn IDs that resolved to content
  in the in-memory `turn_content_map`. Values below `k` indicate coverage gaps in
  instrumentation (e.g. turn IDs that were not found in the map).

Heuristic: gold answer is lowercased and stripped, then matched against each turn's
content (also lowercased + stripped). Answers longer than 3 characters use substring
containment; answers of 1-3 characters use a `\b<gold>\b` word-boundary regex to avoid
false positives like "no" matching "north". See `_answer_text_match` in `adapter.py` for
documented false-positive and false-negative cases.

**The 2026-05-07 M-tier jsonl cannot be retrofitted.** Turn IDs are minted as
`et-{_short_id()}` at ingest time (random per run); the existing JSONL's
`retrieved_turn_ids` are dead references with no content recoverable from disk. A fresh
M-tier run with `capture_recall=True` (automatic when `--mode turns --tier turns`) will
produce the first turn-level baseline — that is a separate task. This closes the
session-level-only gap flagged in Ghost Audit finding `weft-496166ed`.

## Wick recall fixture (loom-efffb521)

Hand-crafted fixture of 14 anticipated Wick recall use cases (n=14),
sourced from Jason's personal-agent expectations and Finch council
2026-05-09. Lives at `benchmarks/wick_eval/dataset.json`. Each entry
specifies question text, shape-tag, expected-answer pattern,
current-Weft-tier that should serve it, and belief-view scope flag.
Used as design checklist for loom-540df5f7 (belief-view spec) and
proxy eval for loom-1fe75d00 (M-tier eval gate). Real-world Wick
eval (n≥50) is a separate task (loom-b2c02183) blocked on Wick shipping.

## Belief-view detector canary — harness (locked 2026-05-26)

The belief-detector canary measures the Haiku turn-to-claim detector's **over-extraction**
discipline (spec §6.2): does it emit belief claims on turns that should yield none? The
fixture (`benchmarks/wick_eval/canary_belief_detector.json`) is **32 turns** spanning the
full role + claim-count distribution from loom-72bd0d59:

| Bucket | Count | Notes |
| --- | --- | --- |
| user-role | 20 | claims + no-claim + third-party + adversarial |
| assistant-role | 6 | agent-suggested claims + acknowledgements |
| tool-role | 3 | expected_claims = [] |
| system-role | 2 | expected_claims = [] |
| single-claim | 9 | one expected attribute |
| multi-claim | 3 | ≥2 expected attributes |
| no-claim (expected []) | 21 | greetings, questions, ack, third-party, tool, system, adversarial |
| adversarial injection | 3 | "ignore prior facts…" — must abstain (§6.1) |

> **Source caveat — SYNTHETIC, not real turns.** loom-72bd0d59 specified ≥30 *real* turns from
> `episode_turns` history. That premise is **not satisfiable yet**: real input (Discord) routes
> to `memories` via `ingest_pipeline`, and the only writer of `episode_turns` is `weft_turn_append`
> (the MCP tool Wick calls) — and Wick is not live (blocked, loom-b2c02183). `episode_turns`
> contains only benchmark residue. So this fixture is synthetic (Jim-Boblaw-style personas, no
> real PII). The real-turn canary is blocked on Wick shipping; re-curate from real traffic then.

**Reproduce:**

```bash
uv run python benchmarks/wick_eval/run_belief_canary.py \
    --fixture benchmarks/wick_eval/canary_belief_detector.json
```

**Locked numbers (32 turns, real Haiku detector, ~$0.06/run):**

| Metric | Value | Gate | Verdict |
| --- | --- | --- | --- |
| **abstention_violation_rate** | **0.0** (0 / 21 no-claim turns) | < 0.15 | **PASS** |
| overall_over_extraction_rate | 0.6 (9 / 15 emitted) | — | naming drift, see below |
| total_emitted_claims | 15 | | |
| abstention_violations | 0 | | |

**Interpretation.** The detector emitted **zero claims on all 21 no-claim turns** — including the
3 adversarial-injection turns (§6.1), all tool/system output, and third-party narration. That is
the over-extraction signal that matters, and it is perfect (0%). The headline
`overall_over_extraction_rate` of 0.6 is **entirely attribute-naming drift**, not over-extraction:
all 7 flagged `false_positive_turns` (004, 005, 006, 007, 008, 009, 011) are *legitimate* claims
where the detector identified the right fact but named the attribute differently than the gold
label (e.g. `reading.current_book` vs `reading.current-book`; `health.allergies` vs
`health.allergy`; on 008 it split "new job as PM" into role + job-status). No hallucinated claims.

Because the detector is non-deterministic and `overall_over_extraction_rate` does exact-string set
difference, that metric drifts run-to-run and is **not a usable gate**. The
`abstention_violation_rate` (added to the harness in this pass) is naming-independent and is the
gate metric going forward. Follow-up (Weft issue): the exact-string metric should be downgraded to
diagnostic-only, or paired with a synonym/normalization layer, before it's trusted as a threshold.

## Reader-prompt v2 — anti-frequency / anti-fabrication / extract-preference (2026-05-16)

Targeted patches to `benchmarks/longmemeval/reader.py` system prompts addressing
the Reader-driven failure clusters identified by the Recall-vs-Reader bucketing
spike (see corrected analysis above). All three patches gate to specific question
types — no risk of collateral on already-passing classes.

| Patch | Question type | Failure mode addressed | Source taxonomy item |
| --- | --- | --- | --- |
| Anti-frequency recency rule | `knowledge-update` | Reader picks older fact when discussed more (Hawaii vs Paris) | Failure mode #2 |
| Anti-fabrication abstention rule | all `_abs` types | Reader fills gaps from world knowledge (bus cost, vintage films vs cameras) | Failure mode #3 |
| Extract-and-apply preference rule | `single-session-preference` | Reader gives generic answer when preference signal exists | Failure mode #4 |

**Falsifiable claim:** the recall-vs-Reader analysis identified +7.97pt overall
accuracy headroom across the Reader-driven clusters (HIT_FULL failures on
knowledge-update + preference + single-session-user). Reader-prompt v2 should
move that ceiling — not fully close it, but materially lift overall accuracy
above the P0.1 baseline of 0.7689 on a re-run.

**Measurement plan:** the in-flight Phase 2 benchmark (`loom-87e4d8bf`, started
2026-05-16 10:47 UTC) loaded the OLD Reader at startup and will measure
hierarchical retrieval against the v1 Reader. A subsequent M-tier re-run
post-Phase-2-completion measures v2-Reader + (whichever WEFT_HIERARCHICAL state
was decided). Tracked by a follow-up Loom task — see the Loom queue for
the M-tier Reader-v2 re-run.

**Why ship before measurement:** the failure taxonomy is published; the patches
are tight; the cost of a wrong patch is small (each is type-gated). The cheaper
sequence per the Recall-vs-Reader analysis (line 460-462) is Reader prompt pass
first, then commit to Phase 2 retrieval for the remaining retrieval-side fails.
This commit ships the Reader pass; Phase 2 measurement decides the rest.

## Phase 2 hierarchical — RETRACTED: the falsification was a confound (corrected 2026-05-26)

> **⚠️ The 2026-05-16 "hierarchical falsification" recorded below is INVALID — hierarchical
> descent was never actually measured.** `WEFT_HIERARCHICAL` is **inert** on the LongMemEval
> `--tier turns` path: `retrieve(tier="turns")` → `_retrieve_turns` (router.py:152) only calls
> `temporal_anchor` / flat `recall_turns`; the flag is read solely in `recall_both`
> (turn_recall.py:343) and the MCP `weft_recall` tool (tools.py:388), neither on the benchmark
> path, and `recall_turns_hierarchical` was never invoked. `_retrieve_turns` is byte-identical
> since commit 5c11e14 (May 4), so the May 16 "hier" run and the May 7 "flat" baseline executed
> **identical flat retrieval**. The −5pt overall delta is explained by 15 question crashes in the
> May 16 run (cause unrecoverable — log lost to a failed nohup) plus reader/judge nondeterminism
> across two runs 9 days apart. The "−6% ingest" was a **counting artifact**: per-question ingest
> was identical (~474.9 sessions/q); the 7,137-session gap = the 15 failed questions' haystacks
> never tallied (`adapter.py:515-523` `continue`s before `sessions_ingested += n_sessions`). The
> H1–H5 diagnosis (incl. H5 "descent ignores question_date") therefore describes FLAT turn-tier
> retrieval, misattributed to descent.
>
> **Outcome:** hierarchical's real LongMemEval performance is UNMEASURED. The track (EPIC
> `loom-531d1c44` + children `loom-31a8db2c`/`loom-c6989702`/`loom-23999517`) was shelved
> 2026-05-26 and Phase 2 retrieval reallocated to belief-view (`loom-44565df2`). Inert-flag
> cleanup tracked as `loom-c2dc5637`. Full record + proofs: pinned Weft anti-pattern
> `weft-be0e9374`. Local diagnosis docs under `benchmarks/longmemeval/results/` (gitignored)
> carry the same correction banner.

### Original (now-retracted) entry — kept for history

The Phase 2 hierarchical retrieval M-tier run on 2026-05-16 failed its gate
(overall 0.7161 vs required ≥ 0.75, and regressed against the flat 0.7689
baseline). `WEFT_HIERARCHICAL` is off-by-default and Phase 2 was NOT merged.
Post-mortem: `benchmarks/longmemeval/results/phase2_hierarchical_postmortem_2026_05_16.md`.
Diagnosis epic (`loom-f1fdf679`) ran four leaves; synthesis with recommendation:
`benchmarks/longmemeval/results/phase2_hierarchical_diagnosis_2026_05_22.md`.
Headline: H5 (descent ignores question_date temporal anchors) is the
highest-leverage actionable finding; recommendation is a smaller H5-only
experiment on temporal-reasoning subset before any broader revise-or-shelve
decision.

