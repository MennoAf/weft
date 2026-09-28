# Wiring an agent to Weft

Weft exposes persistent-memory tools over MCP. Any agent harness that speaks MCP can connect; the steps for registering a server vary by harness, so use its own documentation for the harness-specific settings format. This guide describes the Weft side of the connection without assuming a particular client.

## 1. Install and start Weft

Install Weft as described in the [README quickstart](../README.md#quickstart). For a local install, start its PostgreSQL and Redis services and apply migrations with:

```bash
weft up
```

If you use externally managed services, configure `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` in the environment where the MCP server will run instead. See [Configuration](configuration.md) for supported variables and defaults.

## 2. Register the MCP server with your harness

For an installed local copy, register a stdio MCP server with:

- Name: `weft`
- Command: `weft`
- Argument: `mcp`

When using a source checkout rather than an installed command, run `uv run --directory /absolute/path/to/weft-memory python -m weft.mcp` (replace the example path with the checkout's absolute path). Follow your harness's MCP documentation to enter these values; no one configuration-file format applies to every harness.

If connecting to a hosted Weft server, configure the endpoint and authentication supported by that deployment. See [User Identity](user-identity.md) for token and caller-mode details.

## 3. Point the agent at Weft's tools

Give the agent repository-level instructions or equivalent context appropriate to your harness. When using Weft for persistent memory, the basic workflow is:

- At session start, call `weft_prime(disclosure="progressive")` and read the returned context.
- Use `weft_recall` to find relevant saved information and `weft_remember` to store durable information.
- At the end of a non-trivial session, call `weft_handoff` with the summary and any follow-up context.
- If Weft is unavailable, report that clearly and do not claim that information was saved or retrieved.

The repository's [`AGENTS.md`](../AGENTS.md) contains general agent and local server guidance. For Claude Desktop or Claude Code-specific setup, see [`CLAUDE.md`](../CLAUDE.md).
