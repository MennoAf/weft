# Findings — `weft_prime` hang + recent commit review

**Date:** 2026-07-16
**Trigger:** `weft_prime` hung at session start (>60s, never returned, twice — once at `disclosure="full"`, once at `"progressive"`).
**Original incident status:** Root cause **confirmed by reproduction**. No code changed during the incident session.
**Current status (2026-07-20):** **Fixed in repository; deployed rerun pending.** The initial two-second timeout proved insufficient on deployed Streamable HTTP because cancellation can remain coupled to FastMCP's reverse-RPC response stream. `weft/mcp/tools.py::_detect_project_id` now avoids `roots/list` entirely on Streamable HTTP/SSE and retains bounded roots discovery on stdio. Unit, registered-tool, and real loopback TCP/Uvicorn regressions pass, including clean shutdown after a client advertises but never answers roots. See [`validation-findings-2026-07.md`](validation-findings-2026-07.md).

---

## Headline

`weft_prime` hangs forever when called without an explicit `project_id`. The cause is a single
unguarded line — `await ctx.list_roots()` at **`weft/mcp/tools.py:164`** — which sends a *reverse
JSON-RPC request to the MCP client* with **no timeout**. If the client never answers, the await
never returns.

This is not corpus size, not the Fly deployment, not the recent commits. The server is healthy.

**31 tools are exposed**, not just prime. Prime is simply the one you hit first at session start.

---

## Evidence (how this was confirmed, not just theorized)

| Observation | What it rules out |
|---|---|
| `weft_status(topic="weft")` returned **349k chars / 396 memories**, `complete: true`, promptly | Server down, wedged DB, bad token, client wiring. Scenarios A/B/C of `connection-recovery.md` are all **empirically excluded** — a tool call succeeded. |
| `weft_prime(disclosure="progressive")` hung identically to `"full"` | Corpus size, tier-2 section assembly, response payload size. Both modes share the broken path. |
| **`weft_prime(project_id="weft")` returned instantly** — full primer, 1574 tokens | Everything else. This is the confirmation: an explicit `project_id` short-circuits at `tools.py:196` and never reaches `list_roots`. |

That last row is the whole proof. Same tool, same server, same session — the only variable is
whether `_detect_project_id` runs.

---

## Mechanism

```python
# weft/mcp/tools.py:156-173
async def _detect_project_id(ctx: Context) -> str | None:
    """Auto-detect project_id from MCP client roots.
    ...
    Returns None if roots are unavailable or empty.
    """
    try:
        roots = await ctx.list_roots()      # <-- line 164: unbounded await
        ...
    except Exception as e:
        logger.debug("detect_project_id failed: %s", e, exc_info=True)
    return None
```

Two things make this bite:

**1. The docstring is misleading.** "Auto-detect from the client's working directory" reads like a
local filesystem operation. It isn't. `ctx.list_roots()` asks the *client* over the wire and blocks
on the reply:

- `fastmcp/server/context.py:766` → `await self.session.list_roots()`
- `mcp/server/session.py:350` → `send_request(...)` — **passes no `request_read_timeout_seconds`**
- `mcp/server/session.py:88` → `ServerSession.__init__` never passes `read_timeout_seconds`, so
  `self._session_read_timeout_seconds is None`
- `mcp/shared/session.py:284` → `with anyio.fail_after(None)` — **a no-op scope**

`anyio.fail_after(None)` does not time out. It waits forever.

**2. The `try/except` is decorative.** It catches an exception that is never raised. **A hang is not
an exception.** The docstring's promise — "Returns None if roots are unavailable" — holds only for
clients that actively *reply* or *error*. A client that stays silent (no roots capability, an
HTTP/SSE transport with no open server→client channel, or one that drops unknown reverse requests)
produces exactly the >60s never-returns signature we saw.

**Why `weft_status` is immune:** it takes no `project_id`, never calls `_resolve_project_id`, and
goes straight to local DB work on a bounded pool. Prime's *first* await is a round-trip to the
client. Status has none. That is the entire asymmetry.

---

## Blast radius — this is bigger than prime

`_resolve_project_id` has **35 call sites across 31 tools**. Every one hangs identically when
`project_id` is omitted and the client doesn't answer `roots/list`:

```
weft_alert_create        weft_calibration_summary  weft_degradation_set   weft_learn
weft_autonomy_set        weft_capability_lookup    weft_entity_create     weft_mode_set
weft_behavior_add        weft_cost_record          weft_entity_search     weft_prime
weft_behavior_list       weft_cost_summary         weft_episode_create    weft_project_status
weft_behavior_match      weft_degradation_check    weft_episode_timeline  weft_recall
weft_calibrate           weft_degradation_list     weft_focus             weft_remember
weft_calibration_history                           weft_handoff           weft_trigger_create
                                                   weft_ingest            weft_trigger_due
                                                                          weft_trigger_list
                                                                          weft_weekly_recap
```

`weft_remember` and `weft_recall` are on that list. **The core memory loop can hang.**

Timeout audit of prime's path: every external call is bounded **except the one that matters.**
DB `command_timeout=30.0`, `acquire_timeout=10.0` (`weft/config/__init__.py:81,85`); embeddings have
an explicit timeout (`weft/embeddings/openai.py:35-54` — someone already caught the SDK's 600s
default). But `grep -rn "wait_for\|asyncio.timeout" weft/mcp/tools.py weft/primer.py` returns
**nothing**, and `list_roots` appears at exactly one site with no guard.

---

## Recommended fix (not applied — your call)

The one-line version, at `tools.py:164`:

```python
async with asyncio.timeout(2.0):          # or asyncio.wait_for(..., timeout=2.0)
    roots = await ctx.list_roots()
```

`TimeoutError` is an `Exception`, so the existing handler catches it and the function returns `None`
— which is already the documented "roots unavailable" contract. **The failure path already exists;
it's just unreachable today.** Two seconds is generous for a live client round-trip.

Worth doing alongside it:

1. **Fix the docstring.** "Auto-detect from the client's working directory" caused me to spend the
   first half of this investigation looking at filesystem walks and DB queries. Say it's a reverse
   RPC to the client that may not answer.
2. **Decide the fallback.** Returning `None` means memories get written with `project_id=None`.
   Confirm that's the intended degradation and not a silent mis-homing risk — `connection-recovery.md`
   already documents how bad silent re-homing gets (§"which identity are you reissuing under").
3. **Consider caching the resolved root per session** so 31 tools don't each pay a round-trip.
4. **Consider an env override** (`WEFT_PROJECT_ID`) so a misbehaving client can't wedge the memory
   loop at all.

For public-ready, #1 and #4 matter as much as the timeout — an unknown third-party client is
*exactly* the thing that won't implement `roots`.

---

## Why the recent commits are innocent (I was wrong twice)

I twice suspected **441922a "Add MCP usage telemetry"** — a telemetry middleware that wraps tool
calls and blocks on a write would explain a universal hang. **It doesn't.**
`weft/mcp/tool_usage.py:31-43` is genuinely fire-and-forget: it `create_task`s the write and
immediately `return await call_next(context)`, never awaiting it. It's also registered globally
(`server.py:635`), so it wraps `weft_status` identically — **it cannot produce prime-only behavior.**
The asymmetry disproves it regardless of the code.

`_detect_project_id` last changed in **7dfcd8a**, well before any of the four commits reviewed. The
likelier trigger for "why now" is **client- or transport-side** — an HTTP/SSE deployment, or a change
in whether the client advertises the `roots` capability. Not your recent work.

**Reviewed and clean:**

| Commit | Verdict |
|---|---|
| `33bace4` Include capability registry in production image | Clean. One-line Dockerfile `COPY`, fixes the prod-only import failure. |
| `8150405` Fix canary owner wiring and health visibility | Clean, and **verified working in production** — see below. |
| `441922a` Add MCP usage telemetry, deprecate up_next | Clean re: the hang. One latent issue below. |
| `2363535` Add capability lookup MCP tool | Clean. Additive; new tool + tests. |

---

## Separate finding — latent GC bug in the telemetry middleware

**Severity: LOW-MEDIUM. Not the hang. Worth a follow-up.**

`weft/mcp/tool_usage.py:37-42`:

```python
task = asyncio.create_task(
    self._recorder(pool, context.message.name),
    name=f"weft-tool-usage-{context.message.name}",
)
task.add_done_callback(self._report_task_failure)
return await call_next(context)
```

The event loop keeps only a **weak** reference to tasks. Per the asyncio docs: *"Save a reference to
the result of this function, to avoid a task disappearing mid-execution."* `task` is a local that
goes out of scope immediately, and `add_done_callback` does not create a strong reference. Under GC
pressure the telemetry write can be **collected mid-flight and silently vanish**.

Symptom if it bites: `tool_usage` counts in `weft_check_health` read *low* rather than wrong — and
you'd have no way to tell. That matters because the commit message and the `weft_check_health`
docstring both state this data is **"the evidence used before removing or internalizing a tool"**,
and there's a decision gate on it (handoff: *"Reassess unused tools after the agreed 30-day window
on 2026-08-12"*). Undercounting biases that decision toward deleting tools that are actually in use.

Fix: hold module-level strong refs.

```python
_background_tasks: set[asyncio.Task] = set()
...
_background_tasks.add(task)
task.add_done_callback(_background_tasks.discard)
task.add_done_callback(self._report_task_failure)
```

Same pattern likely applies to the other fire-and-forget sites (`log_memory_access`, consolidation
at `tools.py:1506-1524`) — worth one sweep rather than four fixes.

---

## Resolved: canary open question from the last handoff

The 2026-07-13 handoff asked: *"Is the production canary active with probes and a completed audit
cycle, or still dark/no_probes?"*

**Answer: active and healthy.** From the successful prime call:

```json
"recall_canary": {
  "arms": {"active": {"probes": 196, "audited": 187, "misses": 0,
                      "checks": 2364, "miss_rate": 0, "trustworthy": true, "tripped": false}},
  "last_audit_at": "2026-07-16T08:52:17Z", "audit_age_hours": 17.8,
  "dark": false, "dark_reason": null
}
```

196 probes, 187 audited, **0 misses**, audit cycle completed 17.8h ago. Commit `8150405` is working
in production. That question can be closed.

---

## Still open (carried forward, not investigated today)

- **`memory_hygiene_alerts` inserts failing with `alerts.user_id` NULL.** From the 2026-07-13
  handoff, untouched. Likely the *same class* of bug `8150405` fixed for recall telemetry
  (fire-and-forget losing request-scoped `app.user_id` GUC). The handoff itself asks whether it
  should use the same explicit-user propagation pattern — based on today's reading of `8150405`,
  **probably yes**. Worth checking whether other fire-and-forget writers share it.
- **Loom binding.** Not investigated — you said you'd sit with it. One datapoint: this session's
  `loom_inbox()` / `loom_status()` were rejected and never retried, so Loom is **unverified**, not
  known-broken. `connection-recovery.md` §Scenario C already documents a Loom-specific gotcha
  (global `settings.json` declares `loom` with **no auth header** while a project `.mcp.json`
  declares it *with* one) and points at `~/.claude/loom-binding.md`. **Read that before debugging
  from scratch** — the note explicitly warns against fighting the binding.
- **`docs/connection-recovery.md` is untracked.** It's good — verified against prod, and it saved
  time here by ruling out three scenarios fast. It should be committed before it gets lost.
  Consider adding a fifth scenario: *"a tool hangs while other tools work"* → suspect an unbounded
  reverse RPC, and pass `project_id` explicitly to confirm. The existing triage assumes tools fail
  outright and doesn't cover this.

---

## Immediate workaround

Until the timeout lands, **pass `project_id` explicitly**:

```
weft_prime(project_id="weft", disclosure="progressive")   # returns instantly
```

Your global `CLAUDE.md` boot sequence says `weft_prime(disclosure="full")` with no `project_id` —
that will hang every session until either the fix lands or the boot instruction is updated. The
`house-style:prime` skill has the same gap.

---

## Confidence

**Root cause: confirmed.** Reproduced both directions — hangs without `project_id`, returns
instantly with it. The mechanism is traced end-to-end through installed SDK source.

The one thing not directly observed is the client's actual `roots` capability handshake (no server
log access this session). That would explain *why the client stays silent*, but it doesn't change
the fix: the server should not wait forever on a client that may never answer, regardless of why.
