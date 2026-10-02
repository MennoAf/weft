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

## Deployment-repo constraints (this checkout)

The rules below apply when working inside this private deployment repository (`MennoAf/weft-memory`). They do not apply to the public canonical source at [MennoAf/weft](https://github.com/MennoAf/weft).

- **Source of truth.** Canonical app source lives at `MennoAf/weft` (the `public` remote). This repository (`origin`) is the source of truth for what is approved to run in production. `public/main` content is app source only: it is not production-approved by itself, and must not be deployed merely because it is canonical.
- **Deploy approval and guard.** Production deploys require explicit owner approval and must run through `./scripts/deploy_production.sh`, which enforces `scripts/deploy_production_guard.py` (branch `main` synchronized with `origin/main`, a clean checkout, no generated datasets/snapshots/runs or oversized blobs, and the required production Fly settings). Never deploy from a dirty checkout, a recovery branch, or any ref the guard would refuse. The private production `fly.toml` stays local and excluded from Git; the tracked sanitized examples under `deploy/examples/fly/` must never be used for production.
- **Clean-tag CI gate.** A push to `main` — or a merged PR — is not successful until CI completes green; verify the run for the pushed SHA before claiming success. Production deploys must reference an explicitly approved production tag cut from a clean, CI-green, synchronized ref, not an arbitrary `main` commit.
