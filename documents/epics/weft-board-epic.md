**LLM GROUNDING:** Implements `documents/prds/weft-board.md` for Weft (project `weft-public`). Downstream Tasks must be merge-ready: each Task leaves the repo compiling, `uv run pytest` green, and safe to merge independently. Open Questions (PRD Research Items R1–R8) must NOT survive into implementation as invented facts — where a value is unsettled it ships as a named config constant, never a hard-coded literal. This Epic pins: (1) one `weft_board` call over five fail-soft source adapters producing a single normalized `Item`; (2) urgency buckets derived solely from `due_at` + a configurable horizon; (3) triage writes route through EXISTING Weft write tools — no new mutation endpoint; (4) `weft_up_next` is subsumed to a thin delegate; (5) L1 feedback ships in SHADOW mode by default (records proposals, mutates nothing) behind a data-gated activation flag, with feedback expressed as a data-described rule registry and loop alerts riding the existing alert system via one generic `board_feedback` type.

## Summary

Ship `weft_board` — a new module (`weft/board.py`) + MCP tool that unifies five open-item sources into one structured, bucketed contract; subsume `weft_up_next`; add the L1 triage-correction loop in shadow mode (new `board_triage_events` store + feedback engine that only records proposals in v1); and a localhost overlay (`weft/board_server.py` + static page) that renders the board and fires triage through existing write tools.

## Core Decisions

- One `weft_board(days=7, include_snoozed=False, sources=None)` call fans out to five adapters concurrently, each wrapped so a single failure yields an empty contribution + a `warnings` entry, never a raise. [CONFIRMED: PRD §Behavior/Validation V5; isolation pattern `weft/daily_brief.py:621` `_safe`]
- Normalized `Item` schema is the single contract for UI + agents: `id, source, kind, title, state, due_at, snoozed_until, age_days, urgency, project_id, entity_id, actions[]`. [CONFIRMED: PRD §Interfaces]
- `urgency` ∈ {overdue, due_soon, pending, no_date} derived only from `due_at` vs configurable horizon (default 7d); in-bucket sort oldest-due-first, ties by `age_days` desc then title. [CONFIRMED: PRD V2; reuses `up_next` bucketing at `weft/skills.py:474`]
- Triage = existing write tools only. Each `Item.actions[]` entry is `{verb, tool, args}` naming an existing tool (`weft_tracker_close/snooze/dismiss/update`, `weft_alert_dismiss`, `weft_trigger_delete/fire`). No new mutation endpoint. [CONFIRMED: PRD Non-Goals + §Behavior; tools exist — `close_tracker/dismiss_tracker/snooze_tracker/update_tracker` at `weft/trackers.py:209/226/253/130`]
- `weft_up_next` becomes one internal source adapter; its MCP tool is retained as a thin delegate so `daily_brief` + PAAH callers stay green. [CONFIRMED: PRD §Technical Decisions]
- Each adapter reads up to `per_source_cap` (config, default 200); hitting the cap emits `warnings:{source,truncated:true,cap}`. Never silent truncation. [CONFIRMED: PRD V7 + §Constraints Touched; underlying caps `weft/alerts.py:72`=50, `weft/trackers.py:357`=100, `weft/skills.py:479`=50]
- L1 loop ships with `board_feedback_mode` defaulting to `shadow`: the feedback engine computes proposals and writes ONLY the proposals log; zero tracker/alert mutation in shadow. Activation to `active` is gated on ≥1 reviewed batch AND ≥20 triage events. [CONFIRMED: PRD §Compounding Loops + Validation V8; decision weft-770121ea]
- Feedback is a data-described rule registry `{name, signal_predicate, proposed_action}`; loop alerts use ONE new `board_feedback` AlertType with specifics in `Alert.payload` — not a new type per rule. [CONFIRMED: PRD §Interfaces + Non-Goals]
- Overlay is a localhost web server with an httpx-testable HTTP surface (`GET /board`, `POST /act` with a write-tool allowlist). [ASSUMED — web-vs-TUI is PRD R4; owner favors the app path. Also a Critical Implementation Note.]

## Source-Semantics Cluster

The five sources carry `due_at` in five different shapes, and this is where an adapter will silently mis-bucket if written carelessly. Trackers use `nudge_after`; alerts use `trigger_at`; triggers carry `trigger_at` nested inside the condition JSON and only for `condition_type=time` (threshold/event/absence triggers have no due → `no_date`); taREDACTED parse a due-date from a topic string via the existing `_DUE_DATE_RE`; review-queue uses `memories.review_after`. The tracker adapter must source *open + due* rows without re-implementing snooze logic — `due_trackers` already excludes snoozed rows (`weft/trackers.py:368`) — and must not double-count a tracker that appears both in the due set and the open-list set.

## Critical Implementation Notes

- SHADOW NO-WRITE is a hard gate, not a convention: in `shadow` mode the feedback engine must evaluate rules and write proposals WITHOUT calling any tracker/alert mutation. The foot-gun is applying the `proposed_action` inline during rule eval; guard the apply behind a `mode == "active"` check. Tested by asserting tracker rows + alert count unchanged after a shadow pass (V8).
- `board_feedback` is a NEW `AlertType` enum value (`weft/models.py:551`). Verify whether `alert_type` is DB-enforced (enum type or CHECK constraint) — if so, adding the value needs a migration, not just an enum edit.
- User scoping: the board must scope to the deployment user (reuse `WEFT_DEFAULT_USER_ID`, as the canary audit loop does) and set the RLS/GUC user context before adapter reads. An adapter that reads without the user filter risks cross-user bleed. This is PRD R5 — do not bake a single-user assumption that a later hosted phase must unwind; take the user id as a parameter with the env default.
- `POST /act` must reject any `tool` not on the write-tool allowlist BEFORE dispatch and perform no write on rejection (returns 4xx). The overlay is localhost but the allowlist is the security boundary.
- Trigger adapter: enumerate live trigger kinds during build (PRD R1). Ship a conservative cold-start hide-list for obviously-internal kinds (canary, check-in) so v1 isn't noisy; the L1 `hidden_kinds` loop tunes from there. Do not hard-code the full taxonomy — seed a config list.
- `weft_up_next` subsumption: the task adapter and the retained `weft_up_next` MCP tool must call the SAME underlying logic (currently `weft/skills.py:474`) so behavior can't fork. Refactor the shared core; the MCP tool delegates.

## Merge & Validation

Build order, each step merge-ready and independently testable:
1. `Item` model + bucketing/ranking pure functions (unit-testable, no DB).
2. Five source adapters (reuse `due_trackers`, `list_alerts`, `up_next` core, review-queue query; new trigger adapter). Each returns `list[Item]`.
3. `assemble_board()` — concurrent fan-out with `_safe` isolation, `per_source_cap` + truncation warnings, bucket + rank.
4. `weft_board` MCP tool registration; `weft_up_next` refactored to delegate to the task adapter core.
5. Migration: `board_triage_events` table (append-only, retention per existing convention) + proposals log; `board_feedback` AlertType (+ migration if DB-enforced).
6. L1 feedback engine in SHADOW: rule registry, proposals-log writes, `board_feedback_mode` config, `/act` instrumentation appending triage events. No mutation.
7. L1 ACTIVE-mode apply path (nudge_interval extend + `board_feedback` alert), gated behind the mode flag; `hidden_kinds` filter applied in the board read.
8. Overlay: `board_server.py` (`GET /board`, `POST /act` + allowlist) + static page.

Validation is PRD §Validation V1–V8 verbatim — the single source of truth. Tasks reference V-numbers; they do not restate rules.

## Task Plan

1. `Item` model + urgency-bucketing + ranking pure functions.
2. Tracker + alert source adapters.
3. Trigger + task(memory) + review-queue source adapters (incl. conservative trigger cold-start hide-list).
4. `assemble_board()` fan-out with fail-soft isolation, `per_source_cap`, truncation warnings.
5. `weft_board` MCP tool + `weft_up_next` refactor-to-delegate.
6. Migration: `board_triage_events` + proposals log + `board_feedback` AlertType.
7. L1 feedback engine — SHADOW mode (rule registry, proposals log, `board_feedback_mode`, `/act` triage-event append).
8. L1 ACTIVE apply path + `hidden_kinds` board filter (mode-gated).
9. Localhost overlay server (`GET /board`, `POST /act` + allowlist) + static render page.
10. Test suites: `tests/test_board.py` + `tests/test_board_triage_loop.py`.

## Testing Standard

"Tests green" = `uv run pytest tests/test_board.py tests/test_board_triage_loop.py -v` passes, and the existing `daily_brief` + PAAH suites stay green (subsumption regression gate). Integration tests run against a testcontainers Postgres per project convention (real DB, not mocks). Every PRD acceptance gate (V1–V8, the shadow no-write gate, truncation surfacing, the scripted overlay httpx check) maps to a named test. No task closes on "code landed" — each closes on its behavioral assertion.

## Technical Decisions

- `weft_up_next` as canonical open-items call — superseded by this Epic (demoted to a source adapter; MCP tool retained as delegate).
- `weft_daily_brief` as the aggregation surface — retained, not affected (narrative digest vs structured triage board; may later share adapter code).

## Out Of Scope

Per PRD Non-Goals: hosted/multi-user web app; a new triage write endpoint; write actions on taREDACTED + review-queue (read-only in v1); replacing `daily_brief`; a Loom task board; ACTIVATING the L1 loop in v1 (ships shadow); a plugin framework for external systems' rules/alert-types (the rule-registry + `Alert.payload` seam is the whole extensibility contract). L2 salience calibration is designed but deferred behind triage volume.
