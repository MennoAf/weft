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

Still to add (per spec `weft-b9582b8f`): agenda.

## Run

```bash
# Full scoreboard, both shapes (writes results.json). Needs Docker.
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

## Layout

- `manifest.py` — ground-truth collections (the oracle).
- `seed.py` — real-write-path seeding + cardinality verification.
- `context.py` — in-process `AppContext` + MCP `ctx` (real fastembed provider).
- `harness.py` — drives `weft_recall`, distills agent-facing signals.
- `__main__.py` — testcontainers runner → `results.json`.
- `tests/` — structural pytest asserts.
