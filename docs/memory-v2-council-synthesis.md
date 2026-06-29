# Weft Memory v2 — Full Council Synthesis (2026-06-27)

> Raw synthesis output from the 9-agent Finch council. TLDR lives in memory-v2-redesign.md.

All load-bearing facts confirmed. Synthesizing now.

---

# Weft Memory Architecture — Council Synthesis & Design Recommendation

## 1. Verdict

The partial-JD direction **half-holds**, and the half that fails is the half it leads with. The instinct underneath it is correct and survives every lens: **catalog vs associative is the right top-level cut**, different memory kinds genuinely need different machinery, and the associative half barely needs addresses. But the **headline mechanism — numeric Johnny-Decimal codes (21.11.11) as the primary address — is self-defeating and must be cut.** Decimals are insertion-order- and corpus-relative, so the same object written months apart lands at a different number and any sibling insert renumbers its neighbors. That violates Charge-Q1's own determinism requirement *by construction*, and every reference that points at a decimal (alts, cross_reference, a Loom task, a human citation) silently rots on each renumber — manufacturing the exact drift the charge exists to kill.

**The single most important reframe the council surfaced is bigger than addressing, though.** Six independent lenses plus the adversary converged on it: **never-miss is a write-time + reconciliation property, not a read-side ranking property.** The north-star miss — "I know I told it that and it didn't surface" — is the one event where *nothing* surfaced and *no failed query was logged*. It is invisible by construction in any pull-only pipeline. You cannot organize your way out of a probabilistic top-k cutoff with a tidier filing scheme. Shipping JD and declaring never-miss solved would itself be a confident-miss at the architecture level. **The missing primitive is reconciliation (trial-balance / fsck / canary / write-time push), not addressing.** Build the meter before you build the machine.

So: keep the catalog/associative split, **replace numeric addresses with deterministic AST/path-derived natural keys for the catalog half, ship collections+temporal as shapes (not addresses) for the associative half, and lead the whole program with a reconciliation layer that can observe its own misses.**

---

## 2. Recommended design per area

The unifying principle: an object earns an address **only if a pure `key(object)→stable_slug` extractor can produce a confident, collision-free identity with zero comparison to other items.** That litmus — not taste, not area-wide fiat — draws the deep/associative line, and it draws it cleanly because "unaddressable" becomes a *safe* outcome (auto-file to associative) rather than a forced wrong shelf.

| Area | Becomes | Deep / Associative | Why |
|---|---|---|---|
| **00 Weft Memories** (anti-patterns, behaviors, decisions) | Associative + soft-boost facet | Associative | Beliefs have no intrinsic key. This is the home of the *cross-area alt* (a decision pointing into 20/30). |
| **10 Jason's Brain** (Discord, reminders, Obsidian/mail) | Associative + collections + temporal | Associative | The enumeration + temporal failures live here. Needs shapes, not numbers. |
| **20 Code Blocks** ("box of legos") | Deep catalog, **content-addressed by symbol path**, DEPENDENCY-by-reference | Deep | Functions have one true identity, machine-derivable from AST. The queryable-dataset payoff lives here. |
| **30 Projects** (project map) | **Derived view**, not authored memory — recomputed from the git tree, AST graph-centrality for importance | Deep-*derived* | Per-object addressing is a pure function of repo structure; nobody hand-authors decimals. **Note: this tool does not exist yet** (see §7). |
| **40 WKTW** (shared/company) | Tenancy axis (`workspace_id`), fully orthogonal | Neither | "Who can see it" is never "where in the tree." Defend this against the numbered-tree temptation. |

**Per-area consistency rule: reject it.** Real areas are mixed at the *member* level — "10 Jason's Brain" holds a pasted Discord snippet (catalog-shaped) next to a fuzzy belief. The all-or-nothing rule forces either deep-addressing a belief with no identity, or denying a real catalog object an address because its drawer is mostly prose. **Draw the line per-object via the litmus; let areas earn deep density empirically.**

**Areas are FACETS, not address prefixes.** The "soft-boost router" the brief wants is `weft_focus` generalized: a named set of topic tags + a weight profile applied as a soft boost at recall time over the existing tag/entity graph. No numbered tree is needed to soft-boost — vector scores are already additive. This makes areas first-class, user-editable, and keeps tenancy orthogonal.

---

## 3. The four charge questions, answered decisively

### Q1 — Address-assignment procedure + cross-area alt seam

**Derive, don't classify.** Three mechanisms are corpus-relative similarity gambles that drift (embedding-NN-to-existing-addresses, a learned classifier, a rotting human decision tree) and one is deterministic by construction (AST/path extraction). Embedding-NN is the *seductive wrong answer* — it inherits the 37% flip and re-homes identical objects across sessions. **Catalog placement must be exact-match on an extracted slug.**

**Split the column into two layers — this is the load-bearing schema decision the sketch is missing:**

- **`loc_key` (TEXT, immutable)** = content/structure-derived natural key: `code:repo/mod.py#fully.qualified.symbol`, `file:path`, `project:weft`. This is the *real* primary identity; agents query it by exact match. Identical every run.
- **`loc` (TEXT, derived view)** = the human-facing JD decimal (21.11.11), lazily rendered as a sort order over the `loc_key` set. **Never stored as truth, never a reference target.** It may renumber freely because nothing load-bearing points at it.

**Assignment pipeline for a new catalog item:** (1) route to area — for code the area is implied by source location, no decision needed; for the rare ambiguous case a *tiny frozen* decision tree (branching ~5). (2) Run the area's extractor → slug. (3) Exact-match lookup in a `loc_registry(loc_key → loc)` append-mostly table: present → reuse (this is exactly how an item written months later lands at the same address); absent → mint and append. **The registry — not the embedding space — is the anti-drift artifact.** This is a compiler symbol table / git object store.

**Identity-vs-version:** key on the stable symbol path, carry the **content hash as a version, not the identity** — otherwise a typo-fix changes the hash and resets every link and the popularity count. The fingerprint diff emits `rename` edges (old-key `supersedes→` new-key) so identity survives refactors.

**The cross-area alt seam:** reject the denormalized `weft.alts` array (no referential integrity; rots on renumber). Express alts as **typed edges on the existing `memory_relationships` graph (ON DELETE CASCADE), anchored on immutable `loc_key`** — so a decimal renumber requires zero alt updates. Minting rules, by case:
- **catalog→catalog** (a block imports another): machine-derived from AST imports in the same extraction pass. Automatic, deterministic.
- **associative→catalog** (a decision in 00 names a function in 30): minted **only when content names a symbol resolvable to a concrete `loc_key`.** Ambiguous mention ("the normalize function" when two repos define `normalize`) → **fail closed, no edge.** A missed alt is recoverable; a wrong alt is a confident-miss.
- **associative→associative:** never an alt — that's what soft-boost recall and collections are for.

Re-derive the edge set idempotently on every re-ingest; never hand-maintain.

### Q2 — LEGOS vs DEPENDENCY

**DEPENDENCY-by-reference, decisively. This question is not close.** LEGOS was already self-diagnosed as rot ("fixes don't propagate; library rots") — shipping it ships the failure you named. Worse, it **silently kills the queryable-dataset payoff**: a copied block carries frozen `requires`, so "which repos are exposed to a `pandas<2` CVE" returns a *confidently wrong* answer set — a confident-miss in a security context.

Once identity is a natural key, DEPENDENCY falls out for free: "use this lego" is a **reference to the `loc_key` node**, not a copy of its bytes; a fix to the canonical node is seen by every reference. No package-registry plumbing required. **`requires` must be machine-derived from AST imports (tree-sitter) and normalized to canonical package names — never hand-typed, never LLM-extracted** — which makes "pandas vs pyarrow across the portfolio" an exact `GROUP BY`/graph traversal, not a similarity gamble. `popularity` reuses the existing `access_count`/usefulness machinery.

**Guardrails:** pin references to a content-hash and surface "N references behind" rather than auto-upgrading (avoid silent dependency-hell). For dangling pointers (repo moved/deleted/history rewritten), keep a content-hash **snapshot fallback** so a referenced block survives its source repo — accepting that this reintroduces a frozen copy as a *last resort only*. Keep LEGOS solely as an explicit **`vendored`/`frozen` subtype**, never the default.

### Q3 — Collections + temporal vs addresses for the associative half

**Collections + a real temporal axis. Addresses are the wrong tool for beliefs — and collections are ~80% already built.** Do not build a new shape from scratch:

- **Enumeration engine exists:** `weft/topic_gather.py` does an unbounded `= ANY(topic)` gather with `ORDER BY created_at ASC` (deterministic — no embedding, so immune to the 37% flip) and carries a `truncated` flag. `topic_digests` (v57) materializes the set with stale-invalidation. `entity_mentions` and `episode_memories` are already polymorphic membership join tables.
- **The miss is not a missing table — it's a missing *router*, a missing *contract*, and unreliable *membership*.** The single highest-leverage, smallest-code fix: **add an enumeration-intent classifier to `weft_recall`** (cheap regex/keyword: "list all | every | how many | enumerate" + small gate) that resolves the query noun to a topic/entity/collection and routes to `gather_topic_memories` **instead of** (or, better, **in parallel with**) the top-k-bounded hybrid search. Today a NL "list all the plants" goes through `search_hybrid` (LIMIT-bounded, similarity-ranked) and members below the cutoff silently drop, while the deterministic complete path sits unused behind topic-digest only.

But the adversary is right that collections shipped as a *bare join table relocates the miss to write-time, where it's harder to see.* Ship them **only** with the full contract:
1. **Membership is rule-based or confirmed (candidate→member), never silent similarity-attach** — otherwise you've re-imported the confident-miss one layer deeper.
2. **A taxonomic `is_a` edge layer** (`entity:basil --is_a--> entity:plant`) on the existing `memory_relationships` graph — because extraction names "basil," never "plant," so "list all plants" has no entity to enumerate without it. Hypernym assignment is a cheap async batch pass over the entity table, cacheable.
3. **Async backfill/consolidation** (reuse `consolidation.py`) so a collection created today retro-claims historical members via its rule predicate.
4. **Coverage telemetry** (`members_by_rule` vs `members_by_explicit_link`) so under-population is *observable*.
5. **`truncated=True` plumbed end-to-end to the agent** — never swallowed. The `_ENTITY_MEMORIES_LIMIT = 100` cap is confirmed real; a user with 200 plant memories gets a confident 100 unless truncation surfaces.
6. **Audit and tighten `resolve_entities` threshold=0.6 FIRST** (confirmed at `ingest_pipeline.py:359`) — it false-splits ("basil" vs "basil plant" → two entities, neither set complete) and false-merges. *Everything joins through it; collections built on it are building on sand.* Move to two-tier (>0.85 auto-merge, 0.6–0.85 candidate/review).

**Temporal axis** is a genuine *second, orthogonal* missing axis (neither subsumes the other). Generalize the bitemporal model already shipped in `belief_claims` (v48: `occurred_at` valid-time vs `created_at` transaction-time + supersession chains) to all memories: nullable `valid_from`/`valid_to` (cheap ALTER; null = always-valid), plus a `since`/`until`/`as_of` filter plumbed through `search_*` and `gather`. Default `valid_from = created_at` when undated; populate valid-time only when `intent.dates` is present. This answers "where did Jason live in March" and "plants mentioned before the move" — bi-temporal supersession (Zep/Graphiti's model) is the principled form, and Weft's own multi-anchor-temporal LongMemEval failures are largely supersession failures.

### Q4 — The never-miss recall metric

**Split it on the catalog/associative line, and respect the 37%-flip constraint by never measuring the stochastic path single-run.**

**Deterministic half (hard pass/fail, run on every commit — free):**
- **Catalog/import recall is EXACT:** an import either appears in the AST `requires` set or it doesn't. "pandas vs pyarrow across portfolio" is a SQL predicate, not a draw.
- **Collection/enumeration recall is EXACT:** `recall@membership = |returned ∩ members(C)| / |members(C)|` over seeded known-membership sets (synthetic "Jim Boblaw" fixtures per the persona rule). The deterministic `gather` is the **oracle** (assert it returns all M, run once); the **candidate** is the NL query a real agent phrases ("list all the plants") through `weft_recall`. Because that candidate path is stochastic, run it **k≥5 and report min/median/max** — the *spread is itself the never-miss signal*. **North-star operationalized: `min recall@membership → 1.0`.** A shape that scores 0.6 median but 0.2 min is failing the north star.

**Stochastic half (confidence band, nightly only — never per-write):**
- Freeze materialization (FastEmbed is local/deterministic; add a **deterministic tie-break to `ORDER BY embedding <=> $1`** so even a single run is reproducible) and measure `recall@∞` over the frozen candidate set across multiple runs. `benchmarks/longmemeval/materialize.py` already does this — there is no excuse to A/B on the live path.

**The compounding production meter (the real north star):**
- **`is_reask_miss` is real and shipped** (confirmed: `v52_reask_miss_signal`, `weft/reask.py`, `replay.py`). Drive the rolling re-ask-miss rate → 0. Every production miss auto-mints a known-answer eval case (query → satisfying_memory_id), so the eval set **grows from real usage**, not synthetic personas.
- **Critical caveat the adversary is right about:** `is_reask_miss` only fires when the user *re-asks*. A user who gives up silently never logs it, so the meter reads cleaner than reality. **Pair it with periodic `recall@∞` audits over the materialized set + the canary set (§5) to catch silent give-ups the telemetry can't see.**

---

## 4. Forge unlocks worth seeding & Lodestar frontier framing

**What Weft becomes if this lands:** it stops being a memory *store* and becomes a **queryable engineering knowledge graph over the whole portfolio** — a personal package index, a rationale layer for code, a dependency early-warning radar, and a coordination surface (DNS) for the Loom/Warp/Weft agent fleet. The unlocks, ranked by leverage:

1. **Portfolio dependency analytics / personal SBOM** — once `requires` is AST-derived and normalized, Weft answers questions no tool in the stack answers: "everywhere I use pandas," "which projects break if I drop 3.10," "a CVE just dropped for `requests<2.31` — which repos are exposed." Smallest seed: one `requires text[]` column + one `weft_portfolio_query(import='pandas')` containment scan. **Conditional entirely on DEPENDENCY** — copy-at-fork makes the SBOM lie.
2. **Self-curating package index** — loop the dead `popularity`/`cross_reference` counter: a snippet reused across N projects auto-nominates itself for promotion from associative scratch → catalog address, and from copied LEGO → shared reference. Weft learns your actual personal stdlib from your own build behavior. A Groundhog-shaped compounding loop; it also dissolves part of Q1 (an area "earns" deep status empirically).
3. **Cross-area alt as a rationale layer — git-blame for *intent*.** `weft_why(loc='code:...#fn')` returns the decisions whose alts target that symbol; open a file, see the decisions elsewhere that targeted it. Git blame says who/when; this says *why*. The most valuable edge in the system, not a seam to fear.
4. **Collections as saturating sets** — set algebra impossible against an embedding cloud: temporal diffs ("plants this month vs last"), set arithmetic ("ideas tagged AI MINUS ideas shipped"), an auto-maintained personal changelog ("what changed in my world since Tuesday" across *both* halves — the temporal axis the associative half wants is the *same* axis that makes fingerprint diffs queryable: build once, serve both).

**Lodestar verdict: NEAR FRONTIER on recall quality; 10x GAP on the north star.** The gap is not retrieval — it's **miss-detection**. Every cross-domain system that achieves never-miss at scale solves it *structurally*, not with better search: double-entry has the **trial balance** (an imbalance proves a missing entry exists before you know which), git has **`fsck`** (dangling = unreachable), Amazon runs **cycle counts** (bin scan vs ledger). Weft has no trial balance, so it can promise "pretty good recall," not "never miss." **The category-defining secret (the Zero-to-One "truth few see"): never-miss is a write+reconcile property, and the entire field is optimizing read-side ranking.** No competing agent-memory product makes a *detectability* claim. "Weft can PROVE it didn't lose anything" is a potential category — and it's the most likely thing to get cut for the visible JD tree, because instrumentation produces no demo. **Don't cut it.**

---

## 5. Adversary's fatal flaws and how the design answers them

| Fatal flaw | How this design answers it |
|---|---|
| **Numeric JD decimals violate Charge-Q1 by construction** (insertion-order-relative; rot every reference on renumber). | **Fully answered.** Decimals demoted to a lazily-rendered view over an immutable AST/path-derived `loc_key`. Determinism holds by construction. The "hard assignment problem" the brief sized is largely *self-inflicted* — trivial for natural keys, intractable only for numbers. |
| **Never-miss rests on a pull-only, read-side architecture that cannot observe its own misses.** | **Answered by reframe, but requires net-new work.** Build the reconciliation layer *first*: (a) **recall-canary set** — every `weft_remember` enrolls the fact; a daily job re-issues the originating context on *fixed materialization* and logs misses as first-class events (converts an unmeasurable aspiration into a tracked defect rate); (b) **`weft_fsck`** — lists memories reachable *only* by vector cosine (orphans = leading indicator of future misses), giving `loc_key`/alts a real job as reachability edges; (c) **write-time PUSH** (Anki-style scheduled resurfacing of durable facts into the primer) for the "didn't think to ask" class no retrieval can fix. **This is where the design is honest about its limits: a single reconciliation layer is the unbuilt long pole, and it's the one that actually delivers the north star.** |
| **Collections fix enumeration only at read-time; a bare join table relocates the miss to write-time where it's invisible.** | **Answered conditionally** — collections ship *only* with the full contract (§3 Q3): rule/confirmed membership, taxonomic `is_a` layer, async backfill, coverage telemetry, surfaced truncation, expected-cardinality watchdog, and a tightened entity threshold. The join table alone is explicitly **not** the deliverable. |
| **Stale project-map trust (area 30).** Confirmed: the commit-diffing `weft_fingerprint` **does not exist yet** (`identity.py` is federation hashing, unrelated). A map lagging HEAD is confidently *wrong*, not just incomplete. | **Answered by policy, but the tool is greenfield.** Stamp every map entry with the commit SHA it was derived from; treat the map as a **cache with explicit invalidation, never truth**; on a stale/missing entry **fail SAFE to reading source**, never fail-confident on a stale summary. Derive importance from AST graph-centrality (tree-sitter + personalized PageRank, à la Aider) recomputed per commit-diff. |
| **Soft-boost router can still evict a correct hit below top-k; single-run A/B is 37% noise.** | **Answered by measurement discipline:** every recall claim gated on fixed-materialization, multi-run (n≥5) `recall@k` deltas; deterministic tie-break added to the vector ORDER BY. Catalog/collection recall measured exactly; only associative stays a band. |
| **Numbered tree tempts tenancy-as-prefix (40 WKTW); RLS just hardened in v61/v62.** | **Answered:** `workspace_id` is the *sole* tenancy axis, orthogonal to any facet/address. Assert with a test that visibility is **invariant under any `loc` change.** |
| **Content-hash identity churn; DEPENDENCY dangling pointers.** | **Answered:** key on symbol path, hash as version (rename edges from fingerprint diff); content-hash snapshot fallback for dangling refs + "N refs behind" surfacing. |

**Where the design *can't yet* answer:** the reconciliation layer (canary + fsck + push) is net-new and unglamorous; until it ships, "never-miss" remains a *claim*, not a checkable property. And `valid_at` extraction from natural language ("since the divorce," "last spring") is error-prone — a wrong validity window silently hides a fact, so `created_at` must stay the always-correct fallback axis.

---

## 6. Left-field options to keep alive (do not foreclose)

These are roads the convergent recommendation doesn't take but Jason should explicitly preserve:

1. **Trial-balance / reconciliation as Weft's *headline product feature*** — "Weft can prove it didn't lose anything," not "Weft has good search." Potentially category-defining. Keep `weft_fsck` on the roadmap as a **shipped, user-facing CLI** that prints orphan memories — an external trust artifact, not just an internal invariant.
2. **Address QUESTIONS, not memories** (Reconceive's inversion) — the thing that must never-miss is the *recurring query*, not the stored fact. Register canonical questions ("what plants do I have?", "what's my deploy command?") as first-class nodes; memories bind as answers; never-miss becomes "every registered question has a complete answer set." This *subsumes* collections (a question is a membership predicate) and the router (a question is a saved facet). Strongest single unifying idea on the table — keep it alive as a possible v-next reframe.
3. **One `node(kind, identity?, …)` primitive** — entities, episodes, collections, areas, code-blocks, trackers are six tables of the same shape: a typed node with edges to memories. Collapsing them gives every one set-algebra + temporal-diff for free. The move that makes a senior engineer pause then nod. Don't force the migration now, but don't build a *seventh* parallel table either.
4. **Coverage retrieval (set-cover stop condition)** replacing fixed top-k for enumeration queries — drain the cluster until marginal novelty < threshold. Fixes the plant-drop with *zero new memory shapes*, as a pure retrieval-policy change. A cheap alternative/complement to the collections route.
5. **Expected-cardinality watchdog** — a collection carries "I take 3 medications"; when enumeration falls below expectation Weft *proactively* flags "you mentioned 3 meds, I have 2." The strongest possible expression of never-miss: the system notices *its own* gaps.
6. **Hilbert-curve-derived JD numbers** — if a human-browsable numeric code is ever genuinely wanted, derive it from the embedding via a space-filling curve so numerically-adjacent codes are semantically-adjacent, computed lazily, zero hand-assignment, zero renumber-on-insert.
7. **Proactive security/staleness radar** — when a CVE or deprecation drops for a library in any project's `requires`, the existing `weft_alert_*`/`weft_trigger_*` machinery fires "repos X,Y,Z exposed." The SBOM becomes an early-warning system as a side effect.

---

## 7. Sequenced plan (cheap+certain now vs hard+research-shaped — measure first)

Everything below is an **additive ALTER** — consistent with Weft's boot-time migration model. Nothing requires a destructive migration; the catalog/associative split, the new columns, and the new tables all land alongside the existing schema.

### Phase 0 — Build the meter before the machine (cheap, certain, highest-leverage)
Unglamorous, no demo, but it gates the honesty of every later claim. **Do not let it get cut for the visible JD tree.**
- **Tighten `resolve_entities` 0.6 → two-tier (0.85 auto / 0.6–0.85 candidate).** Everything joins through entities; this is a silent correctness bug, fix first.
- **Plumb `truncated=True` end-to-end** from `topic_gather` to the agent. Never swallow a cap.
- **Deterministic tie-break on the vector `ORDER BY`** so single runs are reproducible.
- **Seed known-membership eval fixtures** ("Jim Boblaw": 12 plants, 8 meds…) + ship `benchmarks/enumeration_eval/` reporting min/median/max `recall@membership`. Free, deterministic, immune to the 37% flip.
- **Stand up the recall-canary enrollment + daily fixed-materialization audit**, and **`weft_fsck` (orphan = vector-only-reachable).** This is the reconciliation layer; it is the north star.

### Phase 1 — Cheap + certain wins (ship next, low risk, directly measurable)
- **Enumeration-intent router in `weft_recall`** → fire `gather_topic_memories` in **parallel** as a fallback contract, returning a reconciliation header ("similarity surfaced 7; membership knows 12; 5 not shown: [ids]"). Reuses shipped, tested, deterministic code. Single change most likely to move never-miss. Validate against the Phase-0 eval.
- **`loc_key` (nullable TEXT) + `loc_registry` table.** Fill `loc_key` synchronously for code via AST (free, deterministic); leave NULL for associative. A NULL-`loc_key` memory is fully recallable via embeddings — `loc_key` is a hint, never a gate.
- **AST-derived normalized `requires` + `weft_portfolio_query`.** Proven against real repos with the brief's own example. Exact, deterministic, dodges the 37% flip.
- **Collections as self-maintaining saved-query predicates** seeded from the existing topic-gather path, with the full §3-Q3 contract. Membership deterministic (rule/confirmed), backfill async, coverage telemetry on.

### Phase 2 — Hard + research-shaped (measure first; do not commit on a single run)
- **Bi-temporal `valid_from`/`valid_to` + `since`/`until`/`as_of` filters.** Cheap ALTER; the *correctness* of valid-time extraction is the research risk — gate on the eval, keep `created_at` as fallback.
- **Taxonomic `is_a` layer** for hypernym enumeration. Async batch hypernym assignment; measure over-/under-enumeration on the eval set.
- **`weft_fingerprint` (area 30) — greenfield, confirmed not to exist.** Build as a *derived view*: tree-sitter AST + personalized PageRank, recomputed per commit-diff, every entry SHA-stamped, fail-safe to source on staleness. This is the largest net-new build; **measure token-savings-vs-staleness-risk before trusting it.**
- **Soft-boost facet router** (`weft_focus` generalized). Every boost claim gated on fixed-materialization multi-run `recall@k`. If the ambiguity band fires >20%, the areas are mis-cut — fix the taxonomy, not the model tier.
- **Self-curating promotion loop** (popularity → catalog nomination) — validate by replaying git history against helpers already manually de-duplicated.

### Cost guardrail across all phases (Pinch)
The write path is **free today** (regex + local FastEmbed + one insert). **Never put an LLM address-classifier on the `weft_remember` hot path.** Route associative-area assignment by **argmax-cosine over 5 precomputed area centroids** using the FastEmbed vector already computed at write time (zero marginal cost); escalate to a model **only inside a measured ambiguity band** (top-2 within ~0.05) and **defer that escalation to the nightly consolidation pass.** Backfill `loc`/membership in batch, never block a write. `weft_fingerprint` must summarize **only files whose content hash changed** (diff-and-hash gated), and pay Haiku **only for files with no docstring/exported symbols** — most well-written files self-describe via AST for free.

---

## 8. Open decisions for Jason (only he can make these)

1. **Sequencing priority: meter-first or wins-first?** The council's strong recommendation is **Phase 0 (reconciliation) before the visible JD/collections work** — because it's the actual north star and it's the thing most likely to get cut. But it produces no demo. This is a discipline call only you can enforce against your own desire to see the catalog tree light up.

2. **How far to take the "one primitive" reconception (left-field #2/#3).** Ship collections as a pragmatic shape now, or hold for the bigger "address questions, not memories" / unified-`node` reframe? Building collections now is additive and doesn't foreclose the reframe — but it does spend design budget. Your call on appetite for the bolder architecture.

3. **Do you want human-facing JD decimals *at all*?** The agent path only needs `loc_key` exact-match. Decimals are purely for *your* browsing. If you rarely browse by number, drop them entirely (git-object-store model) and save the rendering machinery. If you want them, accept that they renumber and you must cite `loc_key`, not decimals.

4. **`valid_at` ambition for the temporal axis.** Best-effort NL date extraction (richer queries, silent-hide risk on extraction errors) vs `created_at`-only (always correct, less expressive). How much do you trust the extractor with "since the divorce"-class phrasing?

5. **Is `weft_fingerprint`/area 30 in scope this cycle at all?** It's the largest net-new build, it's greenfield, and its payoff (token savings) is real but its risk (confidently-stale map) is the sharpest. It may be the right thing to *defer* behind the cheaper, more measurable wins — your call on whether the project-map dream is this-cycle or next.