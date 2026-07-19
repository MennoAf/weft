# Evidence and Continuity Validation Ledger — July 2026

**Baseline commit:** `d7a1f3575868172add934402e4889c8bfb129eec`
**Validation started:** 2026-07-18
**External provenance:** `/Users/jasonbauman/Documents/code_projects/Personal/weft_review/` (evidence only; not an executable specification)

## Governing continuity contract

```text
prime -> concise handoff and durable state
          |
          +-- ordinary continuity question: answer without raw turns
          |
          +-- question needs omitted episodic evidence:
                targeted turn recall -> return only relevant evidence
```

Handoff remains the primary continuity layer. Raw turns are targeted evidence and recovery only. Prime must not include raw turns automatically, and materialization remains benchmark/manual-only until repeated comparative evidence supports production wiring.

## Findings status

| Claim | Status | Current evidence / disposition |
|---|---|---|
| LongMemEval summary omits failed questions from its denominator | **confirmed-current at baseline; repaired in this program** | Baseline `benchmarks/longmemeval/judge.py::summarize_results` counted labelled result rows only. `summarize_pipeline(ref, hyp, result)` now uses the unique reference population as denominator and separates missing hypotheses from hypotheses lacking judge results. |
| Ingest classifier fabricates a confidence-zero `general_note` containing normalized raw input after malformed/truncated/API failure | **confirmed at baseline; repaired** | `weft/ingest_pipeline.py::classify` now validates content/stop reason, salvages only complete structured items, and otherwise abstains with `[]`. Transport, empty, malformed, truncated, fenced, and unknown-type cases assert that no raw fallback blob is stored. |
| Automatic decay provides meaningful lifecycle archival for ordinary memories | **rejected/overstated; review-only branch implemented** | `run_decay` reports candidates by default and mutates only through explicit `apply=True`; scheduled consolidation always passes `apply=False`. The score matrix and regression tests protect pinned/immortal records and prove ordinary operation cannot make a destructive decay transition. |
| Omitted-project `weft_prime` can hang indefinitely on silent `roots/list` reverse RPC | **fixed and transport-verified locally** | `_detect_project_id` bounds `ctx.list_roots()` to two seconds. `tests/test_prime_streamable_http.py` crosses the registered FastMCP Streamable HTTP boundary for absent, empty, and silent roots; explicit-project calls remain scoped and budgeted. A deployed Fly smoke remains an external release check. |
| Production RLS is sufficient because policy SQL exists | **staging-equivalent verified; production deployment-dependent** | Non-superuser/non-`BYPASSRLS` application-role tests cover two-user CRUD, unset/invalid identity, and required role/policy invariants. The actual Fly/Supabase effective role remains a public-release gate documented in `docs/production-release-gates.md`. |
| Prime should automatically consume raw turns/materialized beliefs because turn-tier LongMemEval is strong | **deterministic mechanics complete; paid outcome evaluation pending** | The continuity suite proves prime remains handoff-first, handoff-sufficient queries skip turns, and targeted rationale/chronology/exact-wording queries return bounded scoped evidence. Arm C remains opt-in. Status remains `PENDING-PAID-EVALUATION`; no production wiring was enabled. |
| Zero-call tools are safe to delete | **rejected/overstated; telemetry hardened** | Recorder tasks are retained/drained; heartbeats, failures, successful writes, gaps, and shutdown state are recorded. Classification requires 30 valid coverage days. `inventory/preliminary-tool-inventory.json` is evidence only; manifest removals require an approved deprecation record. |
| Materializer should be registered automatically | **requires experiment** | `materialize_pending_turns` exists but is not a default worker. It remains opt-in until Arm C reproducibly beats Arm B without safety regressions. |
| Weft should be rewritten around SQLite or peripheral systems bulk-deleted | **rejected/out of scope** | The configured datastore remains Postgres. No rewrite, bulk deletion, or new product arms are part of this validation program. |

## Baseline checks

Run from repository root with synthetic/local fixtures only:

```bash
uv run pytest benchmarks/longmemeval/tests/test_judge.py -q
uv run pytest tests/test_ingest_pipeline.py tests/test_consolidation.py tests/test_mcp_tools.py -q
```

Baseline results at the commit above:

- LongMemEval judge tests: **13 passed**.
- Ingest/consolidation/MCP tests: **131 passed**.
- A combined invocation was invalid because benchmark and repository `conftest.py` plugin registration collided; suites are intentionally invoked separately.

## Experimental artifact contract

Every benchmark or deployment-validation artifact must record:

- code commit;
- UTC timestamp;
- configuration and model/version pins;
- dataset or deterministic fixture identity;
- expected, produced, failed, and malformed counts by pipeline stage;
- raw artifact path and redaction status;
- infrastructure failures separately from retrieval/reader correctness;
- paid-run estimated and actual cost where applicable.

Paid LongMemEval and continuity-reader runs are operator-triggered only. Until repeated Arm A/B evaluation is approved and run, continuity status is `PENDING-PAID-EVALUATION`. No single M run is a publication or keep/scrap gate.

## Deterministic continuity acceptance status

The benchmark-only A/B/C harness lives under `benchmarks/personal_agent/`:

- Arm A: concise handoff and durable memories only;
- Arm B: Arm A plus bounded, targeted turn recall through real `weft_recall(tier="auto")`;
- Arm C: Arm B plus opt-in materialized beliefs carrying `evidence_turn_ids`.

Synthetic fixtures include final state, next action, rejected rationale, three-stage chronology, exact wording, omitted detail, superseded dialogue, quoted instruction-shaped text, and cross-project/cross-user distractors. The database-backed tests use a restricted non-superuser/non-`BYPASSRLS` role and grant only the tables needed by unified recall. Raw turns remain labelled quoted evidence and cannot override authoritative handoff/durable state.

Verified commands:

```bash
uv run pytest tests/test_turn_recall.py -q
uv run pytest benchmarks/personal_agent/tests/test_paah_continuity.py -q
```

Results after final remediation on 2026-07-18: the turn-router suite and continuity package pass independently; the continuity package includes DB-backed isolation plus deterministic run-artifact coverage. Repository and benchmark suites are run separately because their `conftest.py` plugin registration collides when collected together.

## Tool-usage evidence status

Migration 69 adds `weft_tool_usage_coverage`. Recorder tasks are strongly retained and drained before pool shutdown. Daily coverage records recorder version, heartbeat range, successful writes, failures, and shutdown-drain state. `get_tool_usage_summary` reports expected/valid/gap days and refuses to call zero usage observed when coverage is incomplete. Deprecation eligibility requires **30 valid days**, not 30 elapsed calendar days.

Artifacts:

- `inventory/public-tool-manifest.json` — checked-in public MCP baseline;
- `inventory/approved-deprecations.json` — explicit approvals only;
- `inventory/preliminary-tool-inventory.json` — dependency evidence with unknown owner/replacement facts marked `UNCONFIRMED`.

The manifest test rejects a removed public tool unless an approved deprecation record exists. The preliminary inventory is not a deprecation recommendation. The real 30-valid-day observation window cannot be compressed into tests and remains open.

## LLM completion/failure policy audit

Durable-state call sites were reviewed for truncation and fabrication risk:

| Call site | Output role | Policy |
|---|---|---|
| `weft/ingest_pipeline.py::classify` | Routes text into memories/alerts/entities | **salvage or abstain** — inspect content/stop reason, recover only complete array items, otherwise return `[]`; never synthesize raw input into an intent. |
| `weft/views/belief_detector.py` | Materializes turn evidence into claims | **salvage or abstain** — conservative partial-array recovery, field validation, confidence/provenance gates. |
| `weft/views/aggregate_detector.py` and replay batch parser | Materializes aggregate claims | **abstain** — API/parse failures produce no claims; confidence and evidence IDs are validated. A dedicated stop-reason check remains advisable but no raw fallback is fabricated. |
| `weft/views/topic_synthesis.py` | Optional narrative digest with provenance | **typed error/abstain** — parse or API failures return non-success status and do not write beliefs. |
| `weft/ingest.py` source-file summaries | Operator-invoked descriptive prose | **raise** — response errors propagate and abort the ingest operation; output is not interpreted as a durable fact classifier. |
| `weft/quarantine_review.py` | Instruction-shape safety classification | **raise/retain unchecked** within an explicit timeout — per-row failure leaves the row unapproved rather than promoting it. |

This phase adds tests only to the classifier path that previously fabricated durable state. Broader client-wrapper standardization is out of scope.

## Documentation reconciliation

- `docs/findings-2026-07-16-prime-hang.md` is preserved as the original incident report; its “not yet fixed” status is historical and must be annotated with the current timeout fix and pending transport verification.
- `docs/connection-recovery.md` remains the operational runbook and should gain a scenario for one hanging tool while other calls work, plus the user-scope configuration shadowing evidence recorded by the tribunal handoff.
- External tribunal prose stays in the external workspace. Only reproducible claims and implementation-relevant provenance are summarized here.
