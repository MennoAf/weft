# Wiring an agent to Weft

Weft exposes persistent-memory tools over MCP. Any agent harness that speaks MCP can connect; the steps for registering a server vary by harness, so use its own documentation for the harness-specific settings format. This guide describes the Weft side of the connection, plus tested examples for common harnesses.

## 1. Install and start Weft

Install Weft as described in the [README quickstart](../README.md#quickstart). For a local install, start its PostgreSQL and Redis services and apply migrations with:

```bash
weft up
```

`weft up` waits until both containers report healthy before running migrations, so a nominal first start does not need a retry. The bundled containers use `restart: unless-stopped`: as long as Docker itself is enabled at boot, services come back after a reboot without running `weft up` again. A globally registered MCP server depends on this infrastructure running — registration alone is not enough.

If you use externally managed services, configure `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` in the environment where the MCP server will run instead. See [Configuration](configuration.md) for supported variables and defaults.

## 2. Verify readiness

Before registering the MCP server, run:

```bash
weft doctor
```

Doctor is read-only and reports ten checks. Expectations for a local install:

- **D004 (database)**: the local quickstart connects as the container owner role `weft`, which passes. Managed deployments are expected to use the restricted `weft_app` role; Doctor reports the connected role either way.
- **D008 (text generation)**: Weft defaults to the `openai` text-generation provider. If the `openai` package or `OPENAI_API_KEY` is missing, D008 fails and names exactly what is missing. **Core memory operations work without text generation** — it powers optional features (ingest classification, quarantine review). Configure it when you need those features; see [Configuration](configuration.md#text-generation-provider-routing).
- **D003 (Docker)** and **D005 (Redis)** warn when the engines are not running.

On Linux hosts, `weft up` also warns if `vm.overcommit_memory` is not `1` (Redis background persistence can fail under memory pressure). It suggests an opt-in sysctl fix and never applies it.

## 3. Register the MCP server with your harness

For an installed local copy, register a stdio MCP server with:

- Name: `weft`
- Command: `weft`
- Argument: `mcp`

When using a source checkout rather than an installed command, run `uv run --directory /absolute/path/to/weft python -m weft.mcp` (replace the example path with the checkout's absolute path). Follow your harness's MCP documentation to enter these values; no one configuration-file format applies to every harness.

Note that `weft up` also writes MCP registrations as a convenience: without flags it registers the project-scoped server in `<cwd>/.mcp.json`, and with `--global` it registers in `~/.claude.json` — both are **Claude Code** destinations. Pass `--no-register` to keep infrastructure startup separate from client registration; every other harness is configured by hand as described below.

If connecting to a hosted Weft server, configure the endpoint and authentication supported by that deployment. See [User Identity](user-identity.md) for token and caller-mode details.

## 4. Harness-specific configuration examples

### Claude Code

`weft up` registers the server for you (project `.mcp.json`, or `~/.claude.json` with `--global`). Copy the memory protocol from [`CLAUDE.md`](../CLAUDE.md) into your global or project `CLAUDE.md`.

### Polytoken

Add the stdio server to the user config at `~/.config/polytoken/config.yaml`:

```yaml
mcp_servers:
  weft:
    transport: stdio
    command: weft
    args:
      - mcp
    pass_env:
      - HOME
      - PATH
```

Then:

- Validate the config: `polytoken config validate --user`
- Install the session skills (next section) into `~/.config/polytoken/skills/` — each skill is a directory named after the skill containing `SKILL.md`
- Run `polytoken doctor` to confirm the server is discovered, and start a new session (`polytoken new`) so the daemon picks up the registration

### Source checkouts

Point the harness at your checkout instead of an installed command (for example, command `uv` with arguments `run --directory /absolute/path/to/weft python -m weft.mcp`). The checkout's own project id and memory instructions live in [`AGENTS.md`](../AGENTS.md).

## 5. Install the session skills

The repo ships each session skill as an Agent Skills-compatible package: a directory named after the skill containing a `SKILL.md` with a `description` frontmatter field. Tool names appear in prose, not in harness-specific `allowed-tools` fields.

- [`skills/prime/SKILL.md`](../skills/prime/SKILL.md) — session boot: load memory with `weft_prime` and report status
- [`skills/handoff/SKILL.md`](../skills/handoff/SKILL.md) — session close: `weft_learn`, behaviors, and `weft_handoff`

Copy the directories into your harness's skill location (for Polytoken: `~/.config/polytoken/skills/`). Harnesses that load slash-command templates can use [`templates/commands/prime.md`](../templates/commands/prime.md) and [`templates/commands/handoff.md`](../templates/commands/handoff.md) instead; those use a command-template layout with `allowed-tools` frontmatter and may need adaptation per harness.

## 6. Point the agent at Weft's tools

Give the agent repository-level instructions or equivalent context appropriate to your harness. When using Weft for persistent memory, the basic workflow is:

- At session start, call `weft_prime(disclosure="progressive")` and read the returned context.
- Use `weft_recall` to find relevant saved information and `weft_remember` to store durable information.
- At the end of a non-trivial session, call `weft_handoff` with the summary and any follow-up context.
- If Weft is unavailable, report that clearly and do not claim that information was saved or retrieved.

The repository's [`AGENTS.md`](../AGENTS.md) provides a copy-paste memory protocol for your own agent instructions. For Claude Desktop or Claude Code-specific setup, see [`CLAUDE.md`](../CLAUDE.md).
