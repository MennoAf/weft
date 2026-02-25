# Loom Feedback & Suggestions

Collected during Weft project setup and Phase 1 decomposition.

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
