# Changelog

Notable changes to Weft are documented here.

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the project version in `pyproject.toml`.

## [Unreleased]

## [1.0.0rc2] - 2026-11-01

### Fixed

- `weft up` waits for container health checks before running migrations, reports an actionable error instead of a raw connection-reset traceback when PostgreSQL is still initializing, surfaces the Redis `vm.overcommit_memory` prerequisite on Linux hosts with an opt-in fix, and names Claude Code as the `--global` registration target (new `--no-register` flag skips registration).
- MCP shutdown no longer misclassifies Redis clients as subprocesses, which crashed cleanup with `ManagementCommands.wait() missing required arguments` on redis-py 8.x.
- The quarantine-review background loop preflights the configured text-generation provider (package, credential, and model) and skips with one actionable log line instead of crashing with `ModuleNotFoundError` that only surfaced at MCP shutdown.
- `weft doctor` D004 accepts the local owner role `weft` used by the documented quickstart while still expecting the restricted `weft_app` role in managed deployments, and reports the connected role plus the expected roles.
- `weft doctor` D008 names the configured provider, its missing package, and its missing credential variable separately instead of suggesting a nonexistent "local provider".
- Belief-detector and topic-synthesis Anthropic clients import lazily with actionable errors, so base installs no longer fail at module import time.

### Changed

- Text generation defaults to the `openai` provider, and models are bring-your-own: `text_generation.model` (env `WEFT_TEXT_MODEL`) points every role at one user-chosen model, and `text_generation.models.<role>` (env `WEFT_TEXT_MODEL_<ROLE>`) overrides individual roles. Missing model configuration raises an actionable error instead of assuming a model name; the Anthropic provider keeps its legacy defaults.
- `weft config show` displays `text_generation.provider`, `text_generation.model`, and per-role model overrides.
- Quarantine review (`llm_review_pending`, its background loop, and the `weft quarantine review-llm` command) runs on any configured text-generation provider through the standard provider pipeline instead of requiring the Anthropic SDK.
- The bundled Docker Compose file ships `restart: unless-stopped` so local services come back after a reboot whenever Docker starts.

### Added

- Agent Skills-compatible skill packages at `skills/prime/SKILL.md` and `skills/handoff/SKILL.md` for harnesses that load `<skill-name>/SKILL.md` directories (command templates remain under `templates/commands/`).
- `weft.text_generation.text_generation_preflight`, a side-effect-free package/credential check shared by Doctor and background-task gating.

## [1.0.0rc1] - 2026-09-28

### Added

- An MCP server and Python package for persistent agent memory.
- PostgreSQL/pgvector-backed memory storage and retrieval, with authority scoping.
- Replay and consolidation capabilities for memory maintenance.

### Changed

- Hardened session priming to report degraded operation, isolate section failures, bound its context budget, and warn about near-miss project selection.
- Added explicit startup and shutdown deadlines for deployment lifecycle operations.
