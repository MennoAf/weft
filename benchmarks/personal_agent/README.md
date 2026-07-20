# PAAH — Personal-Agent Acceptance Harness

Proves Weft **answers** real personal-agent query shapes correctly, with
**structural** assertions over seeded synthetic data — not LLM-judged, not
LongMemEval. It is the repeatable scoreboard the project has lacked since the
2026-05-03 LongMemEval baseline. Design spec: Weft memory `weft-b9582b8f`.

## What it does (enumeration shape — first slice)

1. **Seed** a known-cardinality manifest (`manifest.py`) through the **real
   `weft_remember` write path** — embedding, topic tagging, validation, and the
   pre-insert dedup check all run. The seeder verifies actual stored cardinality
   against the manifest, so any dedup collapse surfaces instead of silently
   corrupting the ground truth.
2. **Ask** `weft_recall` enumeration questions ("how many plants do I have",
   "list all my plants", …) through the **real read path** — `resolve_topic`,
   tier routing (default `auto`), and the Phase-1/V7 reconciliation header are
   all exercised end-to-end.
3. **Assert on the agent-facing response**, comparing the answers an agent could
   read against ground truth (matched by memory id):
   - `obvious_count` = `response["count"]` — the most-obvious field (corrected)
   - `enum_count`    = `response["enumeration"]["count"]` — the explicit answer
   - recall@membership over `response["enumeration"]["members"]` (complete list)
   - `naive_results` = `len(response["results"])` — legacy ranked-slice contrast

## Shapes covered

- **Enumeration / counting** ("how many plants", "list all my meds") — belief
  tier + enumeration router. See below.
- **Temporal / dialogue** ("when did I set up X", "what did I last say about Y")
  — turn/both tier. This is the **Branch-A turn-tier probe** the roadmap flagged
  as never measured (`weft-c0a51a73`). See `temporal_manifest.py` /
  `temporal_harness.py`.
- **Entity brief** ("what do I need to know about X before our meeting") —
  beliefs+graph. Oracle = `weft_entity_context` edge walk; candidate =
  `weft_recall`. See `entity_manifest.py` / `entity_harness.py`.
- **Agenda** ("what's on my plate", "what do I keep pushing") — trackers /
  open loops. Oracle = `weft_tracker_due` (deterministic due-loop query);
  agent-facing surface = `weft_daily_brief`. See `agenda_manifest.py` /
  `agenda_harness.py`. **This completes the four shapes in spec `weft-b9582b8f`.**

## Run

```bash
# Full scoreboard, all four shapes (writes results.json). Needs Docker.
uv run python -m benchmarks.personal_agent

# Structural asserts under pytest.
uv run pytest benchmarks/personal_agent/tests -v
```

Runner exit code: `0` when seed integrity holds and the explicit count fields
are correct with a complete members list on every run; `1` otherwise.

## Result history

### V8 — consumption contract CLOSED (2026-07-01, 18 runs, limit=10)

`weft_recall` now hands the agent an explicit, unambiguous enumeration answer:
`response["enumeration"] = {count, complete, members[], ...}`, and — for a
complete gather — corrects the most-obvious `response["count"]` to the true
membership. `results[]` stays the ranked top-k (relevance); the answer, if not
in the top-k, is in `enumeration.members`.

| count signal | correct runs | note |
| --- | --- | --- |
| `response["count"]` (obvious field) | **18 / 18** | corrected to true count on complete gather |
| `response["enumeration"]["count"]` | **18 / 18** | explicit, unambiguously named |
| `len(results[])` (naive contrast) | **0 / 18** | still 10 — proves *why* the explicit fields matter |

recall@membership over `enumeration.members` = 1.0 (min/median/max). **The
agent now KNOWS the answer instead of inferring it from a ranked slice.**

### V7 — first run, contract OPEN (2026-07-01, pre-fix)

Before the fix: `reconciliation.membership_count` was correct 18/18 but the
truth wasn't the obvious field — `len(results[])` was wrong 0/18 (undercounts
past the limit, overcounts via cross-collection pollution) and `total_matches`
was the whole corpus. That measurement is what motivated the V8 fix.

## Temporal / dialogue result (Branch-A turn-tier, 2026-07-01)

Seeds one episode of **16 dated turns** through `weft_turn_append`, then asks 3
probes × 3 phrasings = 9 runs. Each probe has one anchor turn that answers it;
the signal is whether that anchor is surfaced (by id) under retrieval pressure
(`temporal_anchor` returns `limit//2` = 4 turns for the `turns` tier, so the
anchor must rank into the **top-4 of 16**).

| signal | result | note |
| --- | --- | --- |
| routed to expected tier | **9 / 9** | temporal → `turns`, "what did I last say" → `both` |
| anchor turn surfaced | **9 / 9** | never-miss floor = 1.0 across all phrasings |

The "what did I last say about the Iceland trip" probe correctly surfaces
`iceland_westfjords` (the *more recent* of two Iceland turns), so the anchor is
the right one, not just any topic match. **Verdict: turn-tier ANSWERS** — the
first repeatable number on the recall path the May roadmap left "untested at
scale."

## Entity-brief result + a routing finding (2026-07-01)

Seeds a person entity (`Zelda Quackenbush`) with 8 linked facts + a distractor
person, then measures "what do I need to know about X" two ways:

| path | recall@links | note |
| --- | --- | --- |
| ORACLE `weft_entity_context` (edge walk) | **1.000** | complete brief, one deterministic call |
| CANDIDATE `weft_recall` NL | **min 1.0 / median 1.0 / max 1.0** | after the fix (was min 0.0) |

**Finding, then a general fix (not benchmark-tuning).** The first run was bimodal:
2 of 5 brief phrasings returned **0 of 8** facts, and both were the canonical
ones — "what do I need to know about Zelda **before** our meeting" and "...**before**
I meet her". "before" is a `_TURN_TIER_MARKER` (`before|after|since|until`), so
`route_query_to_tier` sent the query to the turn tier, which has no entity facts
→ empty. A *routing* bug, not a recall bug (every phrasing that reached belief
surfaced 8/8).

The fix was **not** to tighten the router regex to those phrasings (that would
tune the system to the benchmark). It's a general never-miss safety net in
`weft_recall`: **an empty turns-tier result falls back to belief recall**, marked
`tier_fallback` in the response. This fixes the whole *class* of misroutes — any
query the router mis-shapes still surfaces its answer if belief holds it — and
leaves temporal routing that *does* have turn answers untouched (the fallback
only fires on empty). PAAH now runs 6 phrasings across two markers (`before`,
`since`); the 3 temporal-worded ones route to turns, recover via the fallback,
and surface 8/8. **min recall@links: 0.0 → 1.0.** The fallback is monotonic — it
only fires on empty and only adds results — so it cannot reduce recall on any
existing query (incl. LongMemEval temporal). Full suite: 3238 passed.

## Agenda result + a consumption-contract finding (2026-07-01)

Seeds **10 trackers** through the real tracker lifecycle (`weft_tracker_create` +
`weft_tracker_snooze` + `weft_tracker_close`): **4 due open loops** and **6 that
must be excluded**, where each excluded tracker trips a *different* clause of the
`due_trackers` predicate (future nudge, snoozed, done, abandoned, `nudge_mode=none`).
Ground truth is derived, not hand-flagged — `AgendaSpec.expected_due` recomputes the
exact predicate, so the manifest can't drift from the query it asserts. This shape
is deterministic SQL (no embeddings), so a single run is authoritative — the banned
single-run rule (`weft-b015d16a`) is about the stochastic recall pipeline, which
agenda doesn't touch.

| path | signal | first run | after fix | note |
| --- | --- | --- | --- | --- |
| ORACLE `weft_tracker_due` | recall@due | **1.000 (4/4)** | 1.000 | every open loop surfaced |
| ORACLE `weft_tracker_due` | precision | **1.000 (0 leaks)** | 1.000 | no snoozed/future/terminal/no-nudge leak |
| ORACLE `weft_tracker_due` | keep-pushing first | **True** | True | longest-overdue loop leads (`nudge_after ASC`) |
| AGENT-FACING `weft_daily_brief` | open-loop coverage | **0 / 4** | **4 / 4** | the finding, then closed |

**Finding, then a general fix (not benchmark-tuning).** First run: the due-loop
query answered the plate correctly, but the digest an agent actually reads each
morning — `weft_daily_brief` — surfaced **0 of the 4** open loops. `assemble_daily_brief`
built 12 sections (calendar, review queue, handoffs, alerts, canary, …) but **no
trackers / open-loops section**, even though `weft_tracker_due`'s own docstring said
it is "for the daily-brief 'open loops' section." Same consumption-contract gap as
enumeration (the reconciliation header knew the count; the `results[]` slice the
agent read did not): the tracker layer knew the plate; the agent-facing surface
didn't show it.

The fix was to **complete the intended wiring, not tune to the benchmark**: a new
`📌 Open Loops` section in `assemble_daily_brief` (`weft/daily_brief.py`) that calls
`due_trackers()` and renders each loop as `[kind] title (overdue Nd)`, oldest-due
first — a general improvement every user benefits from (their plate shows up in the
brief), the section the tool docstring already promised. **Coverage 0/4 → 4/4**; the
28 existing `daily_brief` tests stay green. `test_daily_brief_surfaces_open_loops`
now asserts the closed state (`brief_surfaces_open_loops`), and `results.json`'s
agenda `finding` self-clears to `null`.

**Two harness bugs the full runner exposed (isolated pytest could not):**
- `__main__.py` referenced `entity_stats.belief_routed` / `.misrouted`, attributes
  removed when the entity harness was refactored for the never-miss fallback — the
  runner crashed before writing `results.json`. Fixed to the current attributes.
- The entity harness called `weft_recall` **without `project_id`**, so in the shared
  full-runner corpus its brief queries saw the *temporal* shape's 16 seeded turns.
  The turns tier was then non-empty, the empty-only never-miss fallback never fired,
  and the temporal-worded briefs returned 0/8. Isolated pytest hid it (empty DB).
  Fixed by scoping to `PAAH_ENTITY_PROJECT_ID` (matching the temporal harness). The
  deeper single-project variant — a real user whose one project holds *both* turns
  and entity beliefs would hit the same wall — is tracked as a product finding
  (`weft-99cac4e5`'s sibling family: the fallback should recover when the turns
  answer is present-but-irrelevant, not only when it's empty).

## Handoff + targeted-turn continuity evaluation

The continuity benchmark is deliberately separate from the older single-shape
scores above. It asks whether concise handoff plus targeted turn evidence helps
a later session answer details the handoff intentionally omitted—without putting
raw dialogue into prime.

`continuity_manifest.py` defines **4 independent synthetic sessions × 7 scenarios
= 28 stable scenario IDs** across unrelated domains: software launch, community
event logistics, kitchen renovation, and research methodology. Each session has:

- 2 handoff-sufficient questions (`next_action`, `final_decision`);
- 4 core episodic questions (`rationale`, `chronology`, `exact_wording`,
  `omitted_detail`)—**16 core paired scenarios** across the manifest;
- 1 supersession safety question;
- quoted instruction-shaped dialogue that remains labelled evidence;
- its own project scope, so another fixture cannot crowd relevant turns out of a
  bounded retrieval result.

The project isolation is load-bearing. An early expansion seeded all four sessions
under one project; the real chronology test then lost a correct launch turn because
other sessions competed inside the top-k. Giving each independent fixture a distinct
synthetic project fixed the test and the experimental design.

`validate_manifest()` fails loudly on duplicate IDs, unknown expected turns,
missing final/superseded/instruction evidence, incorrect 2+5 topology, router misses,
or anything other than 4 sessions / 28 scenarios / 16 core episodic cases.
`build_fixture_snapshot()` serializes only fixed synthetic evidence—never DB-generated
turn IDs—and recursively rejects DSN, bearer, API-key, password, secret, and token
patterns before an artifact can be written.

Deterministic tests establish retrieval mechanics only. Repetitions are repeated
measures, **not independent samples**; decisions aggregate scenario-level majorities
within the four independent sessions. The precommitted Arm-B gate requires at least
8 paired core-episodic wins across the 16 cases, zero paired losses, improvements in
at least 3 classes, no handoff-sufficient regression, no missing judge calls, and
zero instruction/unsupported/stale safety failures.

`continuity_provider_contracts.json` pins a cross-provider evaluation pair from
first-party documentation checked 2026-07-20:

- reader: stable Google `gemini-3.1-flash-lite`, Interactions API JSON schema,
  $0.25/$1.50 per million input/output tokens (Google's documented replacement
  after `gemini-2.5-flash-lite` rejected new-user access);
- independent judge: Anthropic `claude-haiku-4-5-20251001`, Messages API
  `output_config.format` JSON schema, $1/$5 per million input/output tokens.

Sources are stored in the contract file. `continuity_runner.py` renders deterministic
prompts, strictly validates reader/judge JSON, uses stable stage-specific call IDs,
records immutable attempt rows with token usage, resumes only when provider/model/
prompt hashes match, and scores over the complete expected population. The Google
SDK is deliberately not a normal Weft runtime dependency; its adapter imports lazily
and explains the missing benchmark dependency. Attempt allocation uses POSIX advisory
locks (`fcntl`), so paid benchmark execution supports Linux/macOS rather than native
Windows. Reader/judge answer quality remains
`PENDING-PAID-EVALUATION`, production A/B/C wiring remains disabled, and no paid
provider call is made by tests or imports.

### Zero-call estimate and paid-run gate

Generate the complete estimate artifact without constructing either provider client:

```bash
uv run python -m benchmarks.personal_agent.continuity_cli estimate \
  --output artifacts/continuity/estimate.json \
  --retries 1
```

The artifact contains all 168 exact reader prompts and stable call IDs. A judge prompt
depends on the future paid reader answer, so the pre-call artifact labels its 168 judge
entries as conservative envelopes rather than pretending they are future exact prompts.
Each envelope sizes the synthetic candidate to `reader.max_output_tokens * 4` bytes and
charges every reader/judge attempt at the configured maximum output. It reports both the
one-attempt projection and the retry-inclusive worst case. Estimate mode records
`provider_clients_constructed=false`, `network_calls=0`, the precommitted scenario-level
decision rule, and its SHA-256.

A paid run requires all three controls: the exact approval phrase, a positive finite
global ceiling, and a ceiling at least as large as the retry-inclusive estimate:

```bash
uv run python -m benchmarks.personal_agent.continuity_cli run \
  --run-dir artifacts/continuity/run-YYYYMMDD \
  --approval "I APPROVE THE CONTINUITY PAID RUN" \
  --max-cost-usd <AT-OR-ABOVE-ESTIMATE> \
  --retries 1
```

The command validates authority, estimate, and immutable manifest before constructing
SDK clients. The manifest pins retry policy plus decision-rule, estimator-method, and
full protocol digests (scenario content, provider contracts, reader prompt hashes,
judge-envelope hashes, estimator assumptions, and the exact runner/CLI/manifest/contract
file bytes actually executed); the protocol is rebuilt before clients and again before
scoring, so uncommitted implementation drift is detected rather than hidden by Git HEAD.
The same global ceiling is checked immediately before every reader/judge
attempt and retry; observed usage from both providers counts toward it, and attempts
with missing usage are conservatively charged at that provider's maximum projected
call. Attempt identity and conservative spend are reserved atomically before a paid
request. Unresolved reservations (including a crash or append/fsync failure after the
request) remain charged until a durable attempt row reconciles them. Ceiling exhaustion
is a hard stop, never a retry. Artifacts are
`manifest.json`, append-only `attempts.jsonl`, and `results.json`. Results cannot enable
production routing or automatic materialization. No paid continuity command was run as
part of implementation or tests.

Focused verification:

```bash
uv run pytest benchmarks/personal_agent/tests/test_paah_continuity.py \
  benchmarks/personal_agent/tests/test_continuity_eval.py \
  benchmarks/personal_agent/tests/test_continuity_runner.py \
  benchmarks/personal_agent/tests/test_continuity_cli.py -q
```

## Layout

- `manifest.py` — enumeration ground-truth collections (the oracle).
- `temporal_manifest.py` / `entity_manifest.py` / `agenda_manifest.py` — the other
  three shapes' ground truth.
- `seed.py` — real-write-path seeding + integrity verification for all four shapes.
- `context.py` — in-process `AppContext` + MCP `ctx` (real fastembed provider).
- `harness.py` / `temporal_harness.py` / `entity_harness.py` / `agenda_harness.py`
  — drive the real read paths, distill agent-facing signals.
- `__main__.py` — testcontainers runner (all four shapes) → `results.json`.
- `tests/` — structural, persistence, provider-contract, and paid-safety asserts.
