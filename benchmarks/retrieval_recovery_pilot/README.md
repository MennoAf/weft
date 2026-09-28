# Retrieval-recovery pilot

This is a small, provider-free A/B harness for measuring **candidate/evidence
retrieval lift**, not answer quality. It is intentionally separate from the
frozen `terminal-funnel-v1` benchmark schema and does not call a Reader, judge,
or final-answer generator.

## Frozen cases and adapter

`fixtures.json` contains eight structural cases (procedural, exact identifier,
turn-only, comparison, chronology, unknown, conflict, and scope control).
Fixture IDs are stable labels, not database IDs. A real run supplies a frozen
snapshot adapter that maps those labels to actual stable `memory:`, `turn:`, or
`claim:` IDs before returning results.

The adapter is an injected async callable:

```python
from benchmarks.retrieval_recovery_pilot import Arm, FrozenCase, RecallResult, load_cases, evaluate_cases

cases = load_cases("benchmarks/retrieval_recovery_pilot/fixtures.json")

async def recall(case: FrozenCase, arm: Arm) -> RecallResult:
    # Run the same frozen snapshot query in both arms.  Only the additive
    # recovery block may differ in the deterministic arm.
    response = await snapshot.lookup(case.query, case.scope.to_dict(), recovery_mode=arm.value)
    return RecallResult.from_mapping(response)

report = await evaluate_cases(cases, recall, provenance={"snapshot": "pilot-snapshot-v1"})
```

`RecallResult` accepts a legacy response plus the additive `recovery` block,
including the public `RecoveryOutcome.to_public_dict()` shape. Recovery
candidates are always unioned with baseline IDs; they never replace, reorder,
or authorize legacy results.

## Materiality gate

A report is material only when its denominator is complete (every case × both
arms), treatment has zero legacy drift and scope violations, candidate lift is
reported separately from authoritative evidence, and all unknown/incomplete /
conflict cases remain fail-closed. Comparison and chronology recovery requires
all mandatory branches/anchors; one-sided evidence is not a rescue. Cap/error
counters and non-gold candidate rate must be reviewed before promotion. The
pilot never proves final-answer correctness and should not trigger a full or
paid benchmark without an explicit run card.

## Paired answer-quality pilot

`quality_run.py` is a separate provider-free pilot for the explicit `weft_answer`
seam. It calls the same frozen snapshot once with `recovery_mode="off"` and once
with `recovery_mode="deterministic"`, then scores only the public answer contract:
status, completeness, evidence status, typed/rendered answer, authoritative
citations, and mandatory branch/anchor coverage. It never changes production
answer semantics or the retrieval-only report schema.

The checked-in `quality_fixtures.json` contains 12 bounded cases: the eight
structural recovery controls plus four selected LongMemEval questions from
`benchmarks/longmemeval/data/longmemeval_s_cleaned.json`. LongMemEval rows retain
the exact question ID, gold answer, and `answer_session_ids` for audit provenance.
The live quality runner never sends these cases to the synthetic pilot namespace.
It first requires a complete snapshot manifest and `.complete` marker whose
checksum and selected-question mapping match the cleaned S dataset. If that
canonical mapping is absent or mismatched, the default preflight fails closed
with a no-write blocked result. The explicit live runner may instead materialize
the bounded selected source sessions/turns as canonical scoped memory rows,
retaining deterministic source-session/source-turn tags and re-reading every ID
through the normal memory authority path before scoring. Only those verified
bounded memory IDs become authoritative citations; synthetic controls remain
separate.

Readiness is write-free and requires no provider or database:

```bash
uv run python -m benchmarks.retrieval_recovery_pilot.quality_run
```

A live run requires an explicit isolated test DSN and writes only the named
versioned artifact; it does not run the full benchmark:

```bash
uv run python -m benchmarks.retrieval_recovery_pilot.quality_run \
  --seed-and-run --dsn "$RETRIEVAL_RECOVERY_QUALITY_DSN" \
  --output benchmarks/retrieval_recovery_pilot/runs/quality-latest.json
```

The report has a complete case×arm denominator, paired deltas, baseline accuracy,
treatment accuracy, positive rescues, baseline-control regressions, false
sufficiency, citation/authority correctness, incomplete/conflict preservation,
scope/legacy drift, provider-call count, and fail-closed materiality gates. Raw
candidate content is never copied into quality artifacts.

To publish a report without overwriting an input, use `write_report(report,
source=..., destination=...)`; it reuses the recall-contract allocation and
atomic publication helpers.
