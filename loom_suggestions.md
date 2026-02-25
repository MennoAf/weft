# Loom Feedback & Suggestions

Collected during Weft project setup, Phase 1, and Phase 2 builds.

## Bugs

### 1. `loom_status()` with no project context gives cryptic error
- **Observed:** Calling `loom_status()` before `loom_switch_project()` returns a Postgres error about invalid UUID `''`
- **Expected:** Clear message like "No project selected. Use loom_switch_project first." or a global overview

### 2. Dead-lettered tasks appear in `loom_ready` results
- **Observed:** Tasks marked via `loom_batch_done(dead_letter=True)` still show up in `loom_ready` output
- **Expected:** Dead-lettered tasks should be excluded from the ready queue entirely

### 3. Dead-lettered tasks retain `pending` status
- **Observed:** After `loom_batch_done(dead_letter=True)`, tasks have `dead_letter: true` but `status: "pending"`
- **Expected:** Status should transition to a terminal state (e.g., `dead_letter` or `cancelled`) so they don't pollute status filters

### 4. Epic `depends_on` doesn't propagate to child tasks on decompose
- **Observed:** When decomposing an epic that has `depends_on` other epics, all child tasks are born `pending` with no dependencies
- **Expected:** Child tasks should either inherit the parent epic's blockers or be born `blocked`. Without this, `loom_ready` returns tasks from later phases as immediately available, which is incorrect and dangerous for multi-agent builds
- **Impact:** Had to manually wire up ~18 cross-epic dependency edges after decomposition

## UX Improvements

### 5. Decompose auth error is opaque
- **Observed:** When no API key is configured, `loom_decompose` returns: `"Could not resolve authentication method. Expected either api_key or auth_token to be set..."`
- **Expected:** Actionable error like: `"No API key found. Set ANTHROPIC_API_KEY env var, or add skills.api_key to ~/.loom/config.yaml"`
- The raw Anthropic SDK error leaks through without context about Loom's own config resolution chain

### 6. Config changes require MCP server restart with no indication
- **Observed:** Updated `~/.loom/config.yaml` while the MCP server was running. Changes had no effect until full restart.
- **Suggestion:** Either hot-reload config on file change, provide a `loom_reload_config` tool, or at minimum document that restart is required

### 7. `LOOM_PROJECT_DIR` env var purpose is unclear
- **Observed:** `.mcp.json` sets `LOOM_PROJECT_DIR` but it doesn't auto-resolve to a Loom project. Still need to call `loom_switch_project` with an explicit UUID.
- **Question:** Is this env var actually used? If so, for what? If it's meant to set the default project, it doesn't seem to work that way.

## Feature Requests

### 8. Decompose should support dependency hints
- When decomposing an epic, allow passing dependency context so the LLM can generate proper inter-task `depends_on` edges rather than flat lists
- Example: `loom_decompose(epic_id="...", dependency_hints={"after": ["loom-xxx"], "before": ["loom-yyy"]})`

### 9. `loom_ready` should filter dead-lettered tasks by default
- Even if the status bug is fixed, `loom_ready` should have an explicit `exclude_dead_letter=True` default

## Process Observations (Phase 1 Build)

Context: Phase 1 was built by a single orchestrator agent (Warp) working sequentially rather than the intended multi-agent pattern where Warp dispatches to subagents. Some observations are specific to this solo-agent usage, but others apply to multi-agent too.

### 10. Auto-close parent epics when all children complete
- All 5 Phase 1 epics remain in `epic` status despite every child task being done
- The orchestrator shouldn't have to manually track and close epics — when the last child completes, the epic should auto-transition to `done`
- In a multi-agent workflow, the orchestrator may not even know when the last subagent finishes a child — auto-close prevents orphaned epics

### 11. Decompose generates flat dependency-free lists within epics
- Every sub-task decomposition produced tasks with `depends_on: []` — no inter-task ordering
- In a multi-agent build, this is dangerous: all tasks appear ready simultaneously, and a subagent could claim "Postgres CRUD" before "Pydantic Models" is done
- **The decompose LLM has enough context to infer ordering** (e.g., task B mentions files from task A). It should wire up `depends_on` edges automatically within the epic
- This is the same underlying issue as bug #4, but at the intra-epic level rather than cross-epic

### 12. `loom_done` requires claim-first — no shortcut for orchestrator cleanup
- When the orchestrator builds something that covers multiple tasks, it has to claim → done each one sequentially. `loom_batch_done` exists but requires them to be claimed first
- **Suggestion:** Allow `loom_done` on `pending` tasks directly (or add a `loom_batch_resolve` that claims+completes atomically), so the orchestrator can close out work without the ceremony

### 13. Decompose granularity could be configurable
- Decompose produced 23 leaf tasks for Phase 1 — appropriate granularity for dispatching to subagents, but verbose for solo work
- A `granularity` parameter (e.g., `coarse` / `fine` / `subagent-sized`) would let the orchestrator tune task size to the execution model
- For multi-agent: fine-grained is correct (one task per subagent unit of work)
- For single-agent or orchestrator-does-it-all: coarser tasks reduce bookkeeping

### 14. Subagent dispatch guidance in decomposed tasks
- The decomposed tasks include `context.files` and `context.description`, which is great for subagents
- Missing: **explicit instructions for the subagent** — what to build, what constraints to follow, what to test
- The `done_when` field partially covers this, but a dedicated `instructions` or `prompt` field in task context would make it trivial for the orchestrator to pass the task directly to a subagent via the Task tool
- Currently the orchestrator has to synthesize context + done_when + project knowledge into a subagent prompt manually

### 15. No "wave" or "batch dispatch" concept
- The natural multi-agent pattern is: find all ready tasks → dispatch N subagents in parallel → wait for completion → repeat
- Loom supports the primitives (`loom_ready` → `loom_batch_claim`) but there's no higher-level "dispatch a wave" operation
- **Suggestion:** A `loom_orchestrate` or `loom_dispatch_wave` tool that returns the current ready set grouped by parallelizability, making it easy for the orchestrator to know which tasks can run simultaneously vs. which should be sequenced

## Process Observations (Phase 2 Build)

Context: Phase 2 was again built by a single orchestrator (Warp) working sequentially. 18 tasks across 6 epics, completed in 5 waves.

### 16. Decompose still generates 0 dependency edges (Phase 2 confirms)
- Phase 2 had 6 epics decomposed into 18 leaf tasks — every single decompose produced `dependency_edges: 0`
- This is the same issue as #4 and #11, now confirmed across two full phases and 11 total decompositions
- Had to manually wire 12 cross-task dependency edges again
- **This is the single biggest friction point with Loom.** The orchestrator spends more time designing the DAG than building the features.

### 17. Decompose generates incorrect file paths
- **Observed:** Decompose produced `context.files` entries like `src/store.py`, `src/weft/mcp_tools.py`, `src/tools/weft_recall.py`
- **Actual paths:** `weft/store.py`, `weft/mcp/tools.py`
- **Impact:** In a multi-agent setup, subagents receiving these tasks would waste time looking for files that don't exist
- **Root cause:** The decompose LLM guesses file paths without access to the actual codebase structure
- **Suggestion:** Either pass the file tree to decompose (e.g., `loom_decompose(context="file tree: ...")`) or add a `loom_verify_paths` post-processing step that validates/corrects paths against the repo

### 18. Task granularity mismatch — work naturally groups differently than decompose predicts
- **Observed:** Several "Wave 2" tasks (Token Store Integration, MCP Filter Integration) were already done as part of Wave 1 because they were trivial 2-line extensions of the same file edit
- **Impact:** Had to claim → immediate-done on tasks that were already complete, which is bookkeeping overhead
- **Suggestion:** Either allow the orchestrator to merge tasks (`loom_merge [task1, task2]`) or make decompose smarter about grouping related edits to the same file into a single task

### 19. Phase 1 epics still not auto-closed (confirming #10)
- After Phase 2 completion, there are now 3 epics from Phase 1 still in `epic` status despite all children being done for the entire Phase 2 build cycle
- Combined with 3 dead-lettered tasks showing as `pending`, the `loom_status` overview shows misleading numbers: `pending: 3, epic: 3` when the real state is `0 pending, 0 open epics`
