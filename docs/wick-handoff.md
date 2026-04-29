# Wick credential handoff

Phase 2.5 closes the door Phase 2 left open: agents running inside Wick
containers now have their own credentials, bound at issuance to
`caller_mode = 'agent'`. An agent token cannot escalate to supervisor
by lying in the `X-Weft-Caller-Mode` header — the row's caller_mode is
the floor, enforced by `weft.mcp.server.UserIdentityMiddleware`. This
doc is the brief for whoever wires Wick to use those credentials
instead of the supervisor `WEFT_API_KEY`.

## Why this changes anything

Before Phase 2.5 a Wick container shipped with a copy of the human
operator's `WEFT_API_KEY`, which authenticated as supervisor. The Phase
2 caller-mode header (`X-Weft-Caller-Mode: agent`) downgraded the
request at the boundary, but the credential itself was supervisor-level
— a bug, a stripped header, or a malicious in-container process could
write under the human's full trust tier. The four-layer write defense
(provenance, instruction quarantine, prompt-injection scoring,
budget caps) only works if the *credential* says agent.

Phase 2.5 makes the credential authoritative. A token issued
`--mode agent` writes as agent, period — even if the in-container code
forgets to send the header, even if it sends `caller_mode: supervisor`,
even if the request originates from the supervisor's IP.

## What Wick should do

### At provisioning (one-time per container identity)

Mint an agent token bound to the container's user_id:

```bash
weft tokens issue \
  --user-id <container-user-uuid> \
  --mode agent \
  --label "wick-<container-id>" \
  --expires-in 30d
```

Or via MCP tool from a supervisor session:

```python
weft_token_issue(
    user_id="<container-user-uuid>",
    caller_mode="agent",
    label="wick-<container-id>",
    expires_in="30d",
)
```

Capture the plaintext `token` field — it's returned **once**. Store it
inside the container's secrets surface (Docker secret, K8s secret,
Fly.io secret, whatever Wick uses). Never bake it into the image.

### At container startup

Read the token from the secret store and set it as the `Authorization`
header on every `/mcp` request:

```
Authorization: Bearer <wick-agent-token>
```

That's the entire integration. The middleware resolves the token to
`(user_id, caller_mode='agent')` and stamps every memory write
accordingly. The container does not need to send `X-Weft-Caller-Mode`
— the row's binding wins.

### Rotation

Mint a new token, deploy it to the container, restart, then revoke the
old hash:

```bash
weft tokens revoke <old-64-char-hash>
```

`weft tokens list --user-id <container-user-uuid>` shows the live set.

## What NOT to do

- **Don't reuse the supervisor `WEFT_API_KEY`.** That's the bug Phase
  2.5 closes. The whole point is that Wick's identity is structurally
  agent.
- **Don't try to send `X-Weft-Caller-Mode: supervisor` from inside
  Wick.** The header on an agent token is ignored by the middleware
  (verified by `tests/test_user_identity_middleware.py
  ::test_agent_token_resolves_to_agent_regardless_of_header` and
  `tests/test_phase2_5_synthetic_agents.py`). It does nothing useful
  and signals confusion.
- **Don't store the plaintext token outside the container's secret
  surface.** Only the SHA-256 hash is persisted in `weft_tokens` —
  there is no recovery path for the plaintext. If it leaks, revoke
  and re-issue.

## What changes for the supervisor flow

Nothing. The Face (Claude Code with the human in the loop) keeps using
its supervisor token — either the auto-bootstrapped legacy row (from
`WEFT_API_KEY`) or a freshly-minted supervisor credential. Quarantine
review (`weft_quarantine_review approve / reject`) and token management
tools (`weft_token_issue / list / revoke`) remain supervisor-only at
the trust-tier level — agent tokens calling them get a structured 403
shape (the `_supervisor_gate` error envelope in `weft.mcp.tools`).

## Verification

A Wick container is correctly wired when:

1. `weft tokens list --user-id <container-user-uuid>` shows a row with
   `caller_mode=agent` and a recent `last_used_at`.
2. Memories written by the container land with
   `write_provenance='agent'` and instruction-shaped writes get
   `review_status='pending_review'` (visible via
   `weft_quarantine_review action='list'`).
3. Attempting `weft_token_issue` from inside the container returns
   the supervisor-only error envelope, not a token.

If any of these don't hold, the container is still authenticating as
supervisor and Phase 2.5's defense isn't engaged.
