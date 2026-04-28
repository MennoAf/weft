# Route 1 — OAuth 2.1 Authorization Server Merge Checklist

Branch: `feat/oauth-authz-server`
Scope doc: `boiler_room/weft-route1-scope.md`
Phases 1–4 complete on this branch. Phase 5 (staging deploy) is a
Jason-executed step — every `fly secrets set` / `fly deploy` command
is denied to automation by policy.

---

## Pre-merge checks

- [ ] All OAuth tests green: `uv run pytest tests/oauth/ -q`
  (~177 tests on this branch).
- [ ] Non-OAuth suite unchanged: the same 19 pre-existing primer flakes
  from `5e8132a` still fail; no new failures introduced.
  Verify with: `uv run pytest --ignore=tests/oauth/ -q`.
- [ ] Branch is rebased onto `origin/main` (or at least clean-mergeable).
- [ ] `WEFT_OAUTH_ENABLED` defaults to `0`. Confirm prod deploy would
  be byte-identical to `c965d49`:
  ```bash
  grep -n "oauth_enabled" weft/config.py
  # Expect: default False.
  grep -n "WEFT_OAUTH_ENABLED" fly.toml
  # Expect: no match (the production fly.toml must NOT set this).
  ```
- [ ] `fly.staging.toml` exists and pins `WEFT_OAUTH_ENABLED = "1"` —
  staging is the only place OAuth runs live at merge time.
- [ ] `.oauth-*.env` is gitignored (prevents accidental secret leaks
  from the secret-generation helper):
  ```bash
  git check-ignore .oauth-weft-mcp-secrets.env && echo OK
  ```
- [ ] `KNOWN_GOOD_CONFIG.md` §11 describes the OAuth known-good
  (§§1–10 preserved for API-key path).
- [ ] `DEPLOY.md` has a "Route 1 staging rollout" section with the
  full runbook.

---

## Merge approach

**Recommended: rebase merge** (not squash).

Each commit on this branch is deliberately self-contained and green
(count noted in the commit message). Rebase preserves the phase
boundaries so anyone debugging Route 1 later can `git blame` their way
to the right phase.

```bash
git checkout main
git pull
git rebase main feat/oauth-authz-server   # resolve if needed
git checkout main
git merge --ff-only feat/oauth-authz-server
git push
```

If the team prefers squash, the commit body should list each phase
with its test count so `git log` still tells the story.

Do **not** push `fly deploy` as part of the merge. Staging deploy is a
separate step; prod cutover waits on [claude-ai-mcp#134](https://github.com/anthropics/claude-ai-mcp/issues/134).

---

## Post-merge sequence

1. **Deploy to staging** using `fly.staging.toml`:
   ```bash
   fly apps create weft-mcp-staging   # one-time
   ./scripts/generate_oauth_secrets.sh weft-mcp-staging
   fly secrets import < "$HOME/.weft/.oauth-weft-mcp-staging-secrets.env" \
       -a weft-mcp-staging
   fly secrets set OAUTH_ISSUER="https://weft-mcp-staging.fly.dev" \
       SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
       -a weft-mcp-staging
   # Add https://weft-mcp-staging.fly.dev/oauth/callback to Supabase
   # Dashboard → Authentication → URL Configuration → Redirect URLs.
   fly deploy -c fly.staging.toml
   ```

2. **Smoke test the metadata endpoints** (see DEPLOY.md →
   "Route 1 staging rollout" step 6). Expect:
   - `/.well-known/oauth-authorization-server` returns issuer +
     endpoints.
   - `/.well-known/jwks.json` returns exactly one `keys` entry.
   - `/healthz` returns 200.

3. **End-to-end OAuth dance against Claude Desktop.** Add
   `https://weft-mcp-staging.fly.dev/mcp` as a custom connector
   choosing the OAuth option. Complete the browser flow, watch the
   consent page (if `OAUTH_SOLE_USER_SUB` is unset), and invoke
   `weft_recall` / `weft_prime` — verify latency is normal (~<2 s)
   since verification is local RS256.

4. **Wait on [anthropics/claude-ai-mcp#134](https://github.com/anthropics/claude-ai-mcp/issues/134).**
   Until that lands, claude.ai web can't consume
   `BearerAuthProvider` MCP servers. Keep staging running; keep
   prod on the API-key path.

5. **Prod cutover (when #134 lands):**
   ```bash
   ./scripts/generate_oauth_secrets.sh weft-mcp         # separate key from staging
   fly secrets import < "$HOME/.weft/.oauth-weft-mcp-secrets.env" -a weft-mcp
   fly secrets set OAUTH_ISSUER="https://weft-mcp.fly.dev" \
       SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
       -a weft-mcp
   # Add https://weft-mcp.fly.dev/oauth/callback to Supabase allowlist.
   fly secrets set WEFT_OAUTH_ENABLED=1 -a weft-mcp
   fly deploy
   ```

6. **Update KNOWN_GOOD_CONFIG.md §11** to reflect what prod actually
   runs (commit pin, issuer, consent or sole-user mode, kid value).

---

## Rollback — every level

- **Post-deploy issue on `weft-mcp` with OAuth enabled:**
  ```bash
  fly secrets unset WEFT_OAUTH_ENABLED -a weft-mcp
  fly deploy
  ```
  OAuth route handlers stop registering, middleware reverts to
  API-key-only. Claude Code is unaffected — its Bearer token is
  still the same `WEFT_API_KEY`.

- **Post-deploy issue on staging:** same command, replace
  `-a weft-mcp` with `-a weft-mcp-staging`. Or just ignore — staging
  is not a critical path.

- **Post-merge, pre-deploy regression (API-key path):** revert the
  merge commit on `main`. The branch tests prove the API-key
  behaviour is byte-identical when `WEFT_OAUTH_ENABLED=0`, so the
  blast radius is the extra imports + the dormant code paths. A
  revert is still safe.

- **Key rotation goes wrong:** the rotation helper
  (`python -m weft.rotate_key`) only prints instructions — it doesn't
  mutate anything. A bad PEM blocks token minting but nothing worse;
  unset `OAUTH_JWT_PRIVATE_KEY_PEM_NEXT` to back out of an overlap
  window, or redeploy with the previous primary key restored from
  the last good value.

---

## What this branch does **not** include

- Any `fly deploy` / `fly secrets set` execution. All such commands
  are Jason-run.
- A prod-side `fly.toml` change. `WEFT_OAUTH_ENABLED` stays unset in
  `fly.toml`; only `fly.staging.toml` turns it on.
- Multi-process rate-limit backend. The limiter is in-process (scope
  §6). If we scale past one Fly machine, revisit.
- Explicit consent-logging audit trail. Approval decisions live in
  the session cookie and the minted authorization code — there's no
  separate `oauth_consent_events` table today.
