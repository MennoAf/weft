# Weft Connection Recovery Runbook

**Use this when Weft tools stop working.** Sibling to [`disaster-recovery.md`](disaster-recovery.md) — that one is for *losing data* (Supabase gone, restore from backup). This one is for *losing the connection* while the data is fine: `unauthorized`, tools missing, timeouts, `weft_prime` erroring at session start.

Verified against production on 2026-07-16.

## The 60-second triage

Everything hinges on one question: **is the server down, or is my credential bad?** `/healthz` answers it, because it needs no auth.

```bash
# 1. Is the server alive? (no auth required)
curl -s -o /dev/null -w "%{http_code}\n" https://weft-mcp.fly.dev/healthz

# 2. Is my credential good? (401 = auth problem, 200 = auth fine)
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://weft-mcp.fly.dev/mcp \
  -H "Authorization: Bearer $(python3 -c "import json;print(json.load(open('$HOME/.claude/settings.json'))['mcpServers']['weft']['headers']['Authorization'].split()[1])")" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}'
```

Read the pair:

| `/healthz` | `/mcp` | Diagnosis | Go to |
|---|---|---|---|
| 200 | 200 | Server and token both fine — it's your **client config** | [Scenario C](#scenario-c-server-and-token-fine-client-still-broken) |
| 200 | 401 | Server up, **credential rejected** | [Scenario B](#scenario-b-401-unauthorized) |
| fail | — | **Server down / unreachable** | [Scenario A](#scenario-a-healthz-fails) |
| 200 | 200 but writes vanish | Token is **agent-mode**, writes quarantined | [Scenario D](#scenario-d-writes-succeed-but-dont-come-back) |

Known-good baseline (measured 2026-07-16): `/healthz` → 200, `/mcp` with real token → 200, bad token → 401, no token → 401, `GET /` → 404 (expected — there is no root route), `GET /health` → 404 (**the path is `/healthz`, not `/health`** — easy 10 minutes to lose).

A ready-made probe that prints status codes and never echoes the token lives in this repo's scratch history; the inline curl above is equivalent.

## Scenario A: `/healthz` fails

The Fly app is down, sleeping, or DNS is broken.

```bash
fly status --app weft-mcp            # machine state
fly logs --app weft-mcp              # why it died
fly machine list --app weft-mcp
```

- **Machines stopped** → `fly machine start <id> --app weft-mcp`, or just re-probe: Fly autostarts on request.
- **Crash-looping** → read `fly logs`. Most common cause is the DB being unreachable at lifespan startup; check `DATABASE_URL` in Fly secrets (`fly secrets list --app weft-mcp` shows names, not values).
- **DNS error** (`nodename nor servname provided`) → your network, not Fly. This exact error hit the Discord bridge on 2026-07-13 and self-healed.
- Deploy config: [`fly.toml`](../fly.toml) (app `weft-mcp`, region `iad`, internal port 8000, healthcheck `GET /healthz` every 15s). Staging is [`fly.staging.toml`](../fly.staging.toml) (app `weft-mcp-staging`).

**While it's down**, see [Working offline](#working-offline).

## Scenario B: 401 unauthorized

The server is up and rejecting the bearer. Auth model (full detail in [`user-identity.md`](user-identity.md)):

```
Authorization: Bearer <token>
  → weft.credentials.lookup_token  (sha256 → weft_tokens row)
  → row.user_id + row.caller_mode  → ContextVars
  → SET LOCAL app.user_id
```

**A token row is the identity.** No JWT decode in the hot path. So a 401 means: the SHA-256 of your bearer matches no active row. Causes, in likelihood order:

1. **Token revoked or expired** — reissue (below).
2. **Client config lost the header** — see Scenario C; a *missing* header and a *wrong* token both give 401.
3. **Token was rotated elsewhere** and this client wasn't updated.
4. **`WEFT_OAUTH_ENABLED=1` and you're sending a Supabase JWT** whose `sub` isn't the sole user — JWT is only a *fallback* when the bearer matches no token row. A token row always wins.

### ⚠️ First: which identity are you reissuing under?

**Do not run `weft tokens issue --user-id $(cat ~/.weft/user_id.json)` reflexively.** As of 2026-07-16 there were **three** distinct identities, and the local one is *not* the one your sessions write as:

| user | who | where |
|---|---|---|
| `d445dd9f` | **The real memory identity.** Everything Claude Code remembers. | `~/.claude/settings.json` |
| `e53adf9b` | Local CLI identity only | `~/.weft/user_id.json` (token `codex-desktop`) |
| `goose-tr…` | Brandon shared trust-dial dogfood, **agent mode** | was wrongly in the bridge plist |

Issuing under the wrong `user_id` **silently re-homes your memory** — writes succeed, recall returns someone else's world, and nothing errors. Always derive the user from the *credential you're replacing*:

```sql
SELECT user_id, label, caller_mode FROM weft_tokens WHERE token_hash = <sha256-of-current-token>;
```

### Reissue a token

```bash
weft tokens list --user-id <the-user-from-above>
weft tokens issue --user-id <same-user> --mode supervisor --label "claude-code-<host>"
weft tokens revoke <full-sha256-hash>         # kill the old one AFTER the new one works
```

Or via MCP if another client still has a working connection: `weft_token_issue`, `weft_token_list`, `weft_token_revoke`.

**Use `--mode supervisor`** for Claude Code with you in the loop. `--mode agent` is for autonomous containers (Wick) and its writes get quarantined — see Scenario D.

**Never let the plaintext hit a terminal you don't control.** `issue_token` returns it exactly once and it's never stored. If an agent is doing the rotation, script it so the value goes DB → config file directly and only status codes get printed — a redaction regex is not sufficient (see the warning below).

Order that can't strand you: **issue → probe the new token → write configs → re-probe → revoke old**. Never revoke first.

Then update every consumer ([Where credentials live](#where-credentials-live)) and re-probe.

### Telling a real token from the legacy key

- **Issued token** — 48 chars, starts with `weft-` (`TOKEN_PREFIX` + `secrets.token_urlsafe(32)`).
- **Legacy `WEFT_API_KEY`** — bare 43 chars, no prefix. If your `settings.json` holds one of these, you are on the legacy path and the value equals the Fly `WEFT_API_KEY` secret.

### Revoking a `legacy-env-key` row

Counterintuitive but verified 2026-07-16: **revoking the row is permanent and sufficient.** You do *not* need to rotate the Fly secret to kill it, because:

- `lookup_token` filters `WHERE revoked_at IS NULL` → revocation takes effect instantly, no redeploy.
- `bootstrap_legacy_api_key` inserts with `ON CONFLICT (token_hash) DO NOTHING` → it will **never** resurrect a revoked row, even with `WEFT_API_KEY` still set.

So the zero-downtime move is: migrate the client to an issued token, then revoke the legacy row. `fly secrets unset WEFT_API_KEY --app weft-mcp` afterwards is hygiene (retires the path; restarts the app), not urgency — the secret is inert once its row is revoked. Bootstrap no-ops when the key is empty, so unsetting is safe.

### Chicken-and-egg: reaching the DB when MCP is down

`weft tokens issue` talks to Postgres directly — it needs the DB, not a working MCP connection. Two traps, both of which cost real time on 2026-07-16:

1. **The CLI reads `WEFT_DATABASE_URL`, but `~/.weft/.env` defines `DATABASE_URL`.** Sourcing the env file is not enough; without the `WEFT_`-prefixed name the CLI silently falls back to its `localhost:5433` default.
2. **The Supabase host `db.<ref>.supabase.co` is IPv6-only** (`AAAA` resolves, `A` does not). Python's `getaddrinfo` with `AI_ADDRCONFIG` — which asyncio/asyncpg use by default — intermittently returns nothing, giving `socket.gaierror: nodename nor servname provided`. That error means *"no IPv4 and the v6 lookup got filtered"*, **not** "host doesn't exist." The Fly-hosted server is unaffected; this only bites local clients.

Workaround — pin the resolved v6 literal and skip hostname verification:

```python
v6 = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)[0][4][0]
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
conn = await asyncpg.connect(host=v6, port=port, user=user, password=pw, database=db, ssl=ctx)
```

`psql` is not installed on this Mac; use the repo venv's `asyncpg` (`Weft/.venv/bin/python`).

### Legacy `WEFT_API_KEY`

Deployments that ship the bearer via the `WEFT_API_KEY` env var still work: at startup Weft hashes the env value and inserts a matching token row labeled `legacy-env-key`, bound to `WEFT_DEFAULT_USER_ID` + supervisor mode. Idempotent across restarts, fires a deprecation warning on each resolve. If a 401 appears right after a deploy that dropped that env var, this is why. Migrate to issued tokens.

## Scenario C: server and token fine, client still broken

Both probes green but the agent has no `weft_*` tools, or `weft_prime` errors.

Config precedence, nearest wins:

| Where | What it does |
|---|---|
| `~/.claude/settings.json` → `mcpServers.weft` | **The global HTTP+Bearer wiring.** `url: https://weft-mcp.fly.dev/mcp`, `REDACTEDAuthorization: Bearer <token>` |
| `<project>/.mcp.json` | Per-project servers. **Overrides global for the same name.** |

Checks:

```bash
# Is weft actually declared, and does it have an auth header?
python3 -c "
import json;d=json.load(open('$HOME/.claude/settings.json'))
for n,c in d.get('mcpServers',{}).items():
    print(n, c.get('url','(stdio)'), 'auth_hdr=' + str('Authorization' in c.get('headers',{})))"

# Does a project .mcp.json shadow it?
cat ./.mcp.json 2>/dev/null
```

Gotchas seen in the wild:

- **A project `.mcp.json` declaring the same server name shadows the global one.** As of 2026-07-16, `boiler_room/.mcp.json` declares only `loom`, so `weft` correctly falls through to global. If you ever add `weft` there, it must carry its own `headers`.
- **Global `settings.json` declares `loom` with no auth header** while `boiler_room/.mcp.json` declares `loom` *with* one — and on 2026-07-16 Loom returned `unauthorized: missing or invalid bearer token` anyway. If Loom specifically is the broken one, read `~/.claude/loom-binding.md`; it's a documented binding problem, not a Weft issue. Don't fix it by fighting the binding.
- **Restart the client** after editing `settings.json` — MCP servers connect at startup.

## Scenario D: writes succeed but don't come back

`weft_remember` returns fine, `weft_recall` never surfaces it. Your token is **agent-mode**.

Every token binds a trust tier at issuance:

- **`supervisor`** — full write authority (Claude Code with a human in the loop, your CLI).
- **`agent`** — autonomous containers. Writes that Layer 3 flags as instruction-shaped land in `review_status='pending_review'` and stay **invisible to recall** until approved.

Confirm by checking `write_provenance` on a returned memory (`supervisor` is what you want), then `weft_quarantine_review` to approve stuck writes, or reissue with `--mode supervisor`. Note supervisor tokens can *downgrade* per-request via `X-Weft-Caller-Mode: agent`; agent tokens can never escalate.

## Where credentials live

Update **all** of these on rotation. Verified 2026-07-16:

| Path | Holds | Mode | Notes |
|---|---|---|---|
| `~/.claude/settings.json` | `mcpServers.weft.REDACTEDAuthorization` — the Bearer for Claude Code | `0644` | **The main one.** Token `claude-code-mac`, supervisor, user `d445dd9f`. Must stay working through Route 1 (backward-compat mandate). Restart Claude Code to pick up a change. |
| `~/Library/LaunchAgents/local.claude-discord-bridge.plist` | `WEFT_MCP_TOKEN` — passed to sessions the Discord bridge spawns | `0600` | A *different* token (`discord-bridge`, supervisor, `d445dd9f`). Takes effect only on daemon restart: `launchctl bootout gui/$UID/local.claude-discord-bridge && launchctl bootstrap gui/$UID ~/Library/LaunchAgents/local.claude-discord-bridge.plist` — **kills live panes**, so check `state.db` for running tasks first. |
| `~/.weft/.env` | `DATABASE_URL`, `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` | `0600` | **Far more sensitive than any MCP token** — `DATABASE_URL` is the source of truth for all memory. Was `0644` until 2026-07-16. |
| `~/.weft/.oauth-weft-mcp-secrets.env` | `OAUTH_JWT_PRIVATE_KEY_PEM`, `OAUTH_SESSION_SECRET` | `0600` | Route 1 signing material. |
| Fly secrets | server-side `DATABASE_URL`, `WEFT_API_KEY` (legacy) | — | `fly secrets list --app weft-mcp` (names only) |

> **Never `cat` these files, even through a redaction filter.** A regex tuned to one token format leaks the others — the two tokens here have different shapes (`weft--…` 48 chars vs. a bare 43-char string), and exactly that mistake leaked one on 2026-07-16. Print key names, lengths, and equality instead.

## Working offline

When Weft is unreachable and you need to keep going:

1. **`~/.weft/fallback.md`** — text-only mirror. No embeddings, no relationships, grep-only. ⚠️ **As of 2026-07-16 it was last written 2026-06-02 — 44 days stale.** Treat it as a partial archive, not current state. Worth re-checking whether the writer still runs.
2. **Flat-file memory** — `~/.claude/CLAUDE.md` designates the harness's `MEMORY.md` / `.claude/projects/*/memory/` system as the **fallback-only** path. Use it while Weft is down, then migrate entries back with `weft_remember` once restored. Do not leave memories stranded there.
3. **Don't fabricate continuity.** If `weft_prime` fails at session start, say so rather than proceeding as if primed.

## Escalation ladder

1. `/healthz` + `/mcp` probe → identifies the layer (60s)
2. Scenario A / B / C / D above
3. Still broken → `fly logs --app weft-mcp` and check Supabase directly: `psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM memories;"`
4. Data itself looks wrong (not just unreachable) → [`disaster-recovery.md`](disaster-recovery.md)

## Related

- [`disaster-recovery.md`](disaster-recovery.md) — data loss, backups, `weft restore`
- [`user-identity.md`](user-identity.md) — token/identity model, caller modes, JWT fallback
- [`configuration.md`](configuration.md) — env vars (`WEFT_API_KEY`, `WEFT_OAUTH_ENABLED`, `WEFT_DATABASE_URL`)
- [`cli.md`](cli.md) — `weft tokens issue|list|revoke`
- [`wiring-your-agent.md`](wiring-your-agent.md) — first-time setup (not recovery)
