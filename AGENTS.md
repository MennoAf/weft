# Using Weft with your agents

Weft is a persistent memory service that gives agents tools for recalling and saving useful context.

Copy the block below into your project's `AGENTS.md` (or your harness's equivalent instructions file):

```text
Weft is my persistent memory MCP server.

At session start, call `weft_prime(disclosure="progressive")` and read its returned context. Before making a decision that may depend on prior work, retrieve relevant information with `weft_recall`. Save durable facts and decisions with `weft_remember`. At the end of a non-trivial session, close the loop with `weft_handoff`.

If Weft is unavailable, say so plainly and continue without claiming that Weft memory was saved or recalled. Any separate fallback memory mechanism must be described as a fallback, not as a Weft operation.
```

To connect, register an MCP server named `weft` with command `weft` and argument `mcp` in your harness's configuration.
See [the agent wiring guide](docs/wiring-your-agent.md) for full setup instructions and source-checkout options.
Start local services with `weft up` before connecting.
Claude users: see [CLAUDE.md](CLAUDE.md) for Claude-specific setup and memory instructions.

Working on Weft itself? See [CONTRIBUTING.md](CONTRIBUTING.md) and [README.md](README.md).
