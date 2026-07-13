# Weft Board — Unified Open-Items Contract + Triage Overlay

## Summary

Weft has no single call that answers "what's open and needs me?" Actionable state is fragmented across five surfaces — trackers, alerts, triggers, taREDACTED, and the memory review queue — each with its own read call and shape. `weft_daily_brief` reaches most of them but flattens everything to display strings for a morning digest, so it can't be triaged off of and an agent can't reason over it. This PRD defines `weft_board`: one structured call that fans out across all five sources, normalizes each into a common item shape, buckets by urgency (overdue / due_soon / pending / no_date), and returns JSON that is the single contract both a human UI and an agent consume. A thin localhost web overlay renders the board and fires triage writes back through *existing* Weft write tools — no new write surface. The board call is the product; the UI is a visual overlay on the call.

## Goals

- Ship `weft_board` — one MCP call returning all open items across the five sources as structured, normalized JSON (not display strings).
- Normalize every source into a single `Item` shape so a UI and an agent read one contract.
- Bucket and rank items by urgency, reusing the existing `up_next` overdue/due/no-date logic generalized across sources.
- Make each item self-describing for triage: carry the exact existing Weft write tool + args needed to close / snooze / dismiss / re-prioritize it.
- Subsume `weft_up_next` — the board becomes the canonical open-items call; `up_next` becomes one internal source adapter (its MCP tool kept as a thin delegate for back-compat).
- Ship a localhost web overlay that renders the board and issues triage writes through the named tools, refetching after each write.
- Fail soft — one source erroring never blanks the board (mirrors `daily_brief._safe`).

## Non-Goals

- Building a hosted/multi-user web app in v1. Owner is a single user on their own machine(s); hosting + identity scoping is a later phase once the local overlay proves the contract. The board call is designed to be host-agnostic so this is additive, not a rewrite.
- A new write/mutation endpoint for triage. All triage actions route through *existing* Weft write tools (`weft_tracker_close`, `weft_tracker_snooze`, `weft_alert_dismiss`, etc.); inventing a board-specific write path would duplicate validated logic and split the source of truth.
- Write actions on taREDACTED and the review queue in v1. Those two sources are read-only on the board initially — their edits (`weft_revise`, `weft_forget`) carry more blast radius and are deferred to v2 to keep the first triage surface small. Tracker/alert/trigger writes cover the "triage without CLI" pain.
- Replacing `weft_daily_brief`. The brief stays as the narrative morning digest; the board is the live, structured, triage-oriented surface. They serve different altitudes and may later share adapters.
- A general Loom task board. This is Weft's open-items surface (trackers/alerts/triggers/taREDACTED/review), explicitly not Loom's task queue — a distinction the owner called out directly.
- Activating the L1 feedback loop in v1. The feedback engine ships but runs in shadow mode (records proposals, applies nothing); activation is deferred behind a mechanical data gate so the loop is proven against real triage history before it mutates anything (see Compounding Loops §Phasing). Shipping it active-on-day-one would act on faith, not evidence.
- A plugin framework for external/other systems' feedback rules or alert types. There are no other systems today; building a loader for hypothetical ones is speculative abstraction (the "Hoard" anti-pattern). The extensibility contract is deliberately minimal — a data-described rule registry plus the existing Weft alert system's `payload` dict — so new rules/alerts are additive data, not new code, without over-building for a need that doesn't yet exist.

## Behavior: the board contract

`weft_board` fans out to five source adapters concurrently. Each adapter reads its source's open/actionable rows and maps every row into one normalized `Item`. Adapters run independently and each is wrapped so a single source's failure yields an empty contribution and a recorded warning rather than sinking the whole board — the same isolation `assemble_daily_brief` uses for its sections.

Once collected, every item is assigned an `urgency` bucket from its `due_at` relative to now: `overdue` (due_at in the past), `due_soon` (due_at within the configured horizon, default 7 days), `pending` (open but no due pressure yet — has state but due_at beyond horizon), and `no_date` (no due_at at all). Items are then sorted within each bucket oldest-due-first, so the loop you keep pushing leads; ties break on `age_days` descending (older items first), then title. Snoozed items — trackers whose `snooze_until` is still in the future — are hidden from the active board by default and exposed only when the caller passes `include_snoozed`.

The response groups items by bucket and also returns a flat `items` list plus per-bucket counts, so a caller can render grouped or reason over the flat list. The board is a pure read: calling it never mutates state.

## Behavior: triage without a new write path

Every item carries an `actions` list. Each action is a self-describing descriptor — `{verb, tool, args}` — naming an *existing* Weft write tool and the arguments to invoke it with for that specific item. A tracker item, for example, carries actions for `weft_tracker_close`, `weft_tracker_snooze`, and `weft_tracker_dismiss`, each pre-filled with that tracker's id. The UI (or an agent) performs triage by firing the named tool with the given args, then refetching the board. The board call itself stays a read; the write surface is exactly the set of tools that already exist and already enforce their own validation. This is how "triage without CLI" is delivered without a single new mutation endpoint.

## Behavior: the overlay is thin

The localhost overlay speaks only the board contract. A small local web server exposes `GET /board` (returns `weft_board` JSON) and `POST /act` (receives `{tool, args}` from an item's action and dispatches to the corresponding Weft write function, then returns the refreshed board). A single static page renders the buckets and, on an action click, posts to `/act`. The overlay never queries the database directly and holds no business logic beyond rendering and dispatch — which is precisely what makes it the seam an app or agent plugs into later, since both would speak the same `weft_board` / action-descriptor contract.

## Ground Truth

CONFIRMED:
- Open/actionable state is split across five sources with distinct calls: trackers (`weft_tracker_due`/`weft_tracker_list`), alerts (`weft_alert_list`), triggers (`weft_trigger_due`), taREDACTED (`weft_up_next`), review queue (memories with `review_after <= now`). Source: `weft/daily_brief.py` sections + tool list.
- `weft_daily_brief` aggregates most sources but emits display strings, not structured data. Source: `weft/daily_brief.py:475` `format_markdown` / `:501` `format_slack_blocks`; sections built as `list[str]`.
- `daily_brief` isolates each source so one failure can't block others via a `_safe` wrapper. Source: `weft/daily_brief.py:621`.
- `up_next` already implements overdue/due/no_date bucketing and reads the `memories` table (`'tasks' = ANY(topic)`, due-date + priority topics), not Obsidian directly. Source: `weft/skills.py:474`.
- Tracker model carries `kind`, `title`, `state`, `nudge_after`, `snooze_until`, `created_at`, `project_id`, `entity_id`; open-state semantics via `is_open()` / `TrackerState.open_states()`. Source: `weft/models.py:234`.
- `due_trackers(pool, now, limit)` returns open, non-snoozed trackers past `nudge_after`, oldest-due-first. Source: `weft/trackers.py:356`, used at `weft/daily_brief.py:428`.
- Alert model carries `alert_type`, `title`, `trigger_at`, `status`; `list_alerts(pool, status, limit)` filters by status. Source: `weft/models.py:607`, `weft/alerts.py:68`.
- Trigger condition types are time/threshold/event/absence with statuses enabled/disabled/fired. Source: `weft/models.py:634`.
- Existing write tools available to route triage through: `weft_tracker_close`, `weft_tracker_snooze`, `weft_tracker_dismiss`, `weft_tracker_update`, `weft_alert_dismiss`, `weft_trigger_delete`, `weft_trigger_fire`. Source: deferred MCP tool list.

ASSUMED:
- A due-soon horizon default of 7 days is right for the board (matches `up_next` default `days=7`). Unverified — promote to Research Item if a `done_when` hard-codes it as the only supported value.
- Triggers are worth surfacing on the human board (some are system-internal, e.g. canary/check-in triggers). Unverified — which trigger kinds are user-actionable vs noise needs a filter decision; promote to Research Item.
- A localhost web server is an acceptable new runtime component in the Weft repo (vs a TUI). Unverified — depends on whether the owner wants a browser surface; owner indicated app-path intent, which favors web. Listed as Research Item R4.

## Constraints Touched

The board fans out to source reads that each carry an enforced result cap. Left unaddressed, a caller with more open items than a cap silently gets a truncated board — a "looks complete but isn't" failure. The board resolves this rather than inheriting it:

- `list_alerts` default `limit=50` at `weft/alerts.py:72` — IN SCOPE: the alert adapter passes an explicit board cap and, when the cap is hit, emits a per-source truncation marker (see V7).
- `due_trackers` default `limit=100` at `weft/trackers.py:357` — IN SCOPE: same board-cap + truncation-marker treatment.
- `up_next` default `limit=50` at `weft/skills.py:479` — IN SCOPE: same treatment via the task adapter.
- `BRIEF_MAX_ITEMS_PER_SECTION=10` at `weft/daily_brief.py:42` — OUT OF SCOPE: that cap belongs to the daily brief's rendering, which the board does not modify (it is a sibling surface, not a consumer).

The board-level contract: each source adapter reads up to a configurable `per_source_cap` (default well above realistic single-user volume, e.g. 200), and any source whose read hits its cap contributes a `truncated: true` marker in `warnings` naming the source and the cap. Truncation is surfaced, never silent.

## Validation

Effective board output must satisfy:
- V1: Every returned item conforms to the `Item` schema — `id`, `source`, `kind`, `title`, `state`, `due_at`, `snoozed_until`, `age_days`, `urgency`, `project_id`, `entity_id`, `actions[]` — with `source` one of the five known values and `urgency` one of {overdue, due_soon, pending, no_date}.
- V2: An item's `urgency` is derived solely from its `due_at` and the configured horizon: past → overdue; within horizon → due_soon; open with due_at beyond horizon → pending; null due_at → no_date.
- V3: A tracker whose `snooze_until` is in the future is excluded from the default board and included only when `include_snoozed=true`.
- V4: Every action descriptor names a tool that exists in the current MCP tool set, and its `args` are sufficient to invoke that tool for that item (at minimum the item's id under the tool's id parameter).
- V5: If any single source adapter raises, the board still returns with the other four sources' items and a `warnings` entry naming the failed source; the call never raises for a single-source failure.
- V6: The board call performs no writes — invoking it leaves all five sources' rows unchanged.
- V7: Each source adapter reads up to `per_source_cap` (configurable, default 200); if a source's read reaches the cap, the board emits a `warnings` entry `{source, truncated: true, cap}`. Truncation is never silent (per Constraints Touched).
- V8: In `board_feedback_mode=shadow` (the v1 default), the L1 feedback pass performs no state mutation — it writes only to the proposals log; tracker rows and alert counts are unchanged by a shadow pass. Mutation occurs only in `active` mode.

## Interfaces / Schema

`weft_board(days: int = 7, include_snoozed: bool = false, sources: list[str] | None = None) -> dict`

Returns:
```
{
  "generated_at": ISO8601,
  "horizon_days": int,
  "buckets": { "overdue": [Item], "due_soon": [Item], "pending": [Item], "no_date": [Item] },
  "items": [Item],            // flat, same items, bucket-sorted
  "counts": { "overdue": int, "due_soon": int, "pending": int, "no_date": int, "total": int },
  "warnings": [ { "source": str, "error": str } ]
}
```

Item:
```
{
  "id": str,
  "source": "tracker" | "alert" | "trigger" | "task" | "review",
  "kind": str,                // tracker kind / alert_type / condition_type / task priority / "review"
  "title": str,
  "state": str | null,
  "due_at": ISO8601 | null,
  "snoozed_until": ISO8601 | null,
  "age_days": number,
  "urgency": "overdue" | "due_soon" | "pending" | "no_date",
  "project_id": str | null,
  "entity_id": str | null,
  "actions": [ { "verb": str, "tool": str, "args": object } ]
}
```

L1 feedback engine (v1 ships in shadow mode):
- `board_feedback_mode: "off" | "shadow" | "active"` — config, default `shadow`.
- Rule registry — a list of `{ name, signal_predicate, proposed_action }` entries evaluated over `board_triage_events`. Two rules ship (repeat-snooze → extend nudge_interval; repeat-dismiss → add to hidden_kinds). New rules are registry entries, not engine changes.
- Proposals log — every rule firing records `{ rule, target_id, proposed_change, mode, ts }`; in shadow mode this is the only write the engine makes.
- Loop-emitted alerts reuse the existing alert system: a single `board_feedback` alert type carrying the specifics in `Alert.payload` (`{ rule, target, proposed_change }`) — no new AlertType per rule.

Overlay HTTP surface (localhost only):
- `GET /board?days=&include_snoozed=` → board JSON (calls `assemble_board`).
- `POST /act` body `{ "tool": str, "args": object }` → dispatches to the named Weft write function, returns refreshed board JSON. Rejects any `tool` not in an allowlist of the known triage write tools.

## Testing

### Unit Tests
- Each of the five source adapters maps a representative source row to a valid `Item` (schema-complete, correct `source`).
- Urgency bucketing: a due_at 1 day in the past → overdue; within horizon → due_soon; open with due_at beyond horizon → pending; null due_at → no_date (boundary at exactly `now` and exactly `now + horizon` asserted).
- Snooze hiding: a tracker with `snooze_until` in the future is absent by default and present when `include_snoozed=true`.
- Action descriptors: a tracker item's actions include `weft_tracker_close`/`weft_tracker_snooze`/`weft_tracker_dismiss` each carrying the tracker's id; an alert item's actions include `weft_alert_dismiss` with the alert id.
- Ranking: within a bucket, items sort oldest-due-first, ties by `age_days` desc then title.

### Integration Tests
- Against a seeded test DB (testcontainers, per project convention), `assemble_board` returns items from all five sources in one call, each item's `urgency` bucket matching the Validation §V2 rule for its `due_at`.
- Truncation surfacing (V7): seed one source past `per_source_cap` and assert the board returns a `warnings` entry `{source, truncated: true, cap}` and still returns the capped source's items up to the cap.
- One-source-fails isolation: monkeypatch one adapter to raise; assert the board still returns the other four sources' items and a `warnings` entry for the failed source, and does not raise (V5).
- Read-purity: snapshot all five sources' row counts/states before and after a `weft_board` call; assert unchanged (V6).
- Overlay dispatch: `POST /act` with a tracker-close descriptor closes exactly that tracker (verified via `weft_tracker_get`) and the returned refreshed board no longer lists it; `POST /act` with a tool not in the allowlist is rejected.

### Acceptance Criteria
- `weft_board` is registered as an MCP tool and returns a schema-valid response satisfying V1–V6, verified by the unit + integration suite passing (`uv run pytest tests/test_board.py -v`).
- `weft_up_next`'s MCP tool still returns its documented shape, now delegating to the board's task adapter (existing `up_next` callers — `daily_brief`, PAAH — unaffected; their tests stay green).
- The localhost overlay's HTTP surface passes a scripted end-to-end check (httpx, no browser required): `GET /board` returns JSON whose `buckets` has all four keys; `POST /act` with a real tracker's close descriptor returns 200, and a subsequent `GET /board` no longer lists that tracker id; `POST /act` with a tool outside the allowlist returns a 4xx and performs no write.
- No new write endpoint exists: grep confirms `/act` dispatches only to the allowlisted existing Weft write functions.

## Technical Decisions

- `weft_up_next` as the canonical open-items call — superseded by this PRD. `up_next` is demoted to one internal source adapter; `weft_board` is canonical. The `weft_up_next` MCP tool is retained as a thin delegate for back-compat (daily_brief + PAAH depend on it).
- `weft_daily_brief` as the aggregation surface — retained, not affected. The brief remains the narrative digest; the board is the structured triage surface. They may later share adapter code but neither replaces the other.

## Compounding Loops

The board emits rich exhaust — every triage action is a datapoint about what actually deserved attention — and the base design discards it. Without the loops below, the board is The Dashboard anti-pattern: signal + store (mutated rows) with zero feedback into what surfaces next. L1's SIGNAL + STORE + feedback engine (in SHADOW mode) is a build-v1 requirement; the engine ships recording proposals but applying nothing, and is flipped to ACTIVE on a mechanical data threshold (this is "record now, prove the theorem, then activate" — not "loop later"). L2 is designed-and-seeded but gated on triage volume.

```
LOOP BLUEPRINT — Triage Correction Ratchet          (BUILD v1 — flagship)
══════════════════════════════
Family:   correction pattern
SIGNAL:   every board triage action — (item_id, source, kind, urgency_at_surface,
          age_days_at_surface, verb, snooze_duration|null, ts). Emitted as a side
          effect of /act (and of an agent firing a weft_board action).
          NEEDS instrumentation — one append in the action-dispatch path. Size S.
STORE:    board_triage_events table (append-only, windowed retention 90d),
          queryable by (item_id) and by (source, kind).
MODE:     ships in SHADOW mode in v1 — the feedback engine runs, computes every
          proposal it WOULD make, and records it to a proposals log WITHOUT applying
          anything (no tracker write, no alert fired). A config flag
          board_feedback_mode (off | shadow | active) flips it to ACTIVE on a named
          data threshold (see Phasing). Shadow mode is what proves the theorem: it
          produces the counterfactual (proposed action vs. what you actually did)
          before the loop is allowed to touch anything.
FEEDBACK: the engine evaluates a RULE REGISTRY — each rule is a data entry
          {name, signal_predicate, proposed_action}, so adding a rule (or a future
          system's rule) is a registry entry, not new engine code. Two rules ship:
          • ROBUSTNESS GAP (reversible): same tracker snoozed >=3 times →
            proposed_action = extend that tracker's nudge_interval one step. In
            ACTIVE mode, auto-applied via existing tracker-update internals.
          • FEATURE SIGNAL (human-as-judge, never auto): a source+kind dismissed
            across >=3 distinct items (threshold ASSUMED — mirrors the snooze
            threshold; a tuning question, see R7) → proposed_action = add that kind
            to hidden_kinds. In ACTIVE mode, emitted as a generic board_feedback
            alert (existing Weft alert system; specifics in the alert payload dict,
            no new AlertType-per-rule) that the human approves. This is how R1 gets
            ANSWERED — learned, not hard-coded.
PROOF:    (shadow) the proposals log accumulates and its agreement rate is measurable —
          e.g. proposed-hide kinds that you had in fact been dismissing. (active)
          chronic re-snooze rate (fraction of snooze actions that are the Kth+ snooze
          of the same item) declines as triage volume grows.
          Dead tell: proposals log grows but no activation threshold is ever documented
          or met; or, once active, an item accrues 5+ snooze events with nudge_interval
          unchanged, or a dismissed-across-3 kind keeps re-surfacing.
Payback:  shadow proof after ~10-20 triage actions; activation value immediately after.
Cost:     S. Deterministic, no LLM. No Pinch flag.
Phasing:  v1 = SIGNAL + STORE + feedback engine in SHADOW (records proposals, applies
          nothing). Activation = flip board_feedback_mode to ACTIVE once the proposals
          log holds >=1 human-reviewed batch AND >=20 accumulated triage events — a
          mechanical gate, not "later".
done_when: with 12 seeded board_triage_events (one tracker snoozed 3x, one source+kind
          dismissed across 3 distinct items):
          (shadow) running the feedback pass writes 2 proposals to the proposals log,
          mutates NO tracker and fires NO alert (assert tracker rows + alert count
          unchanged) — the shadow no-write gate, Validation §V8.
          (active) after flipping board_feedback_mode=active and re-running: that
          tracker's nudge_interval is strictly greater than pre-pass AND its next
          computed nudge_after strictly later than pre-pass (so due_trackers surfaces
          it less — the downstream effect, not just the field), and a board_feedback
          alert proposing the dismissed kind for hidden_kinds is created; after
          approving it a subsequent weft_board call omits that kind.
          Both asserted in tests/test_board_triage_loop.py.
══════════════════════════════

LOOP BLUEPRINT — Salience Self-Calibration          (SEEDED — build after L1)
══════════════════════════════
Family:   self-calibration
SIGNAL:   pair (item's bucket + in-bucket rank at surface time) with its outcome —
          closed = hit, dismissed within T of surfacing = false positive,
          snoozed = deferred. Same board_triage_events store + a surfaced_rank field.
STORE:    board_triage_events (shared with L1) + per-(source,kind) salience weight.
FEEDBACK: recompute a per-(source,kind) salience weight from hit/false-positive ratio;
          the weight adjusts ONLY the in-bucket sort tiebreak, never the urgency
          bucket itself (bucket stays due_at-derived per Validation V2). Kinds you
          dismiss fast sink; kinds you act on rise.
PROOF:    top-of-board false-positive rate (top-K items dismissed within T of
          surfacing) declines over weeks.
          Dead tell: top-K dismiss rate flat/rising while all salience weights == default.
Payback:  ~50+ triage actions — HONESTLY slow for one user (weeks). Build only once
          L1 proves the event volume exists; do not front-load it.
Cost:     S/M. Deterministic weight, no LLM. No Pinch flag.
done_when: given a seeded event history where source+kind X is dismissed-fast 8/10 and
          kind Y closed 8/10, the salience recompute assigns weight(X) < weight(Y) and
          two otherwise-equal-due items sort Y-before-X — asserted in tests.
══════════════════════════════
```

**Killed out loud:**
- *Triage → proactive-agent memory loop* (log "you keep snoozing X" as a Weft belief so an agent nudges "want to kill it?"). Real, but it overlaps L1's feedback and needs an agent consumer that doesn't exist yet. Deferred, not blueprinted.
- *Board-analytics dashboard* (counts/trends over time, rendered, consumed by nobody). That IS The Dashboard anti-pattern — named and refused.

## Research Items

- R1: Which trigger condition-kinds are user-actionable on the human board vs system-internal noise (canary, check-in, absence-watchdogs)? **Partially retired by Compounding Loop L1** — the hidden_kinds filter learns this from dismiss patterns rather than requiring a hard-coded list. A conservative cold-start hide-list (obvious system kinds) may still be worth seeding; the loop tunes from there.
- R2: Should `pending` (open, no near due pressure) be shown by default or collapsed behind a toggle? Affects default board size; needs a look at real tracker volume to avoid a wall of low-signal items. (L2's salience weight also mitigates this once volume exists.)
- R3: Does the due-soon horizon need to differ per source (e.g. alerts are precise timestamps, trackers are soft nudges)? Investigate whether one horizon over-/under-surfaces any source before hard-coding `days=7` as the sole path.
- R6: L1's auto-applied nudge_interval extension is a backend write outside the "no new write path" triage non-goal (which scopes UI triage, not the feedback job). CONFIRMED the internals exist (`update_tracker` w/ `nudge_interval` at `weft/trackers.py:130`, `close_tracker`/`dismiss_tracker`/`snooze_tracker` at :209/:226/:253); confirm the feedback pass reuses them rather than adding a second mutation surface.
- R7: L1 tuning values — snooze-count 3, dismiss-across-count 3, and the shadow→active activation gate (>=1 reviewed batch AND >=20 triage events) — are all ASSUMED. Right values need real triage volume; the shadow-mode proposals log is precisely the instrument that will inform them, and L2's calibration store refines them later. All are config constants, not hard-coded, so they move without a schema change.
- R8: `board_triage_events` 90d retention should follow the existing retention-config convention (cf. `_CANARY_AUDIT_RETENTION_DAYS`) rather than a bare literal — consistency with how Weft already windows event tables.
- R4: Overlay delivery mechanism — localhost web server vs TUI. Owner indicated app-path intent (favors web), but the new-runtime-component cost (a served process in the Weft repo) needs a nod. Blocks the overlay tasks, not the board call.
- R5: How does the board scope to a user in a single-user deploy — does it reuse `WEFT_DEFAULT_USER_ID` like the canary audit loop, and what changes when it becomes multi-user? Investigate the user-scoping seam so v1 doesn't bake in a single-user assumption that a later hosted phase must unwind.
