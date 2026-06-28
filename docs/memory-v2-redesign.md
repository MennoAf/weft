# Weft Memory v2 — TLDR

*Status: design resolved 2026-06-27, not yet built. Next step: PRD. Full council synthesis in [`memory-v2-council-synthesis.md`](./memory-v2-council-synthesis.md). Canonical decision record: Weft memory `weft-5276bf05`.*

## The one-paragraph version

We started toward a Johnny-Decimal "address every memory with a number" scheme. A 9-agent design council pressure-tested it and we changed two things. **(1)** We dropped the *numbers* but kept the *navigable path* — catalog items (code, files, projects) get a stable, self-describing key like `code:weft/store.py#gather_topic_memories` instead of an arbitrary `21.11.11`. **(2)** We reframed the real goal: "never lose anything" is a **write-and-reconcile** property, not a better-search property. A miss you can't see is the whole problem, so we build the *meter that detects misses* before we build anything else.

## Why the old model was wrong (and what replaces it)

| | Old idea (JD numbers) | New idea (v2) | Why it's better |
|---|---|---|---|
| **Catalog identity** | Hand-/auto-assigned decimal `21.11.11` | `loc_key` = AST/path-derived natural key (`code:repo/mod.py#symbol`) | Numbers are *positional* — they renumber when you insert a sibling, and the same function written months apart gets a different number. A natural key is **identical every run** and needs no lookup table. |
| **Navigation** | Descend a numbered tree, consult an index | Prefix-walk the key: `code:weft/` → `code:weft/store.py#` → symbol | Same "narrow down fast" Dewey feel, but the branch labels *are* the real names — self-describing, never renumber, agent-native (exact-match + prefix). |
| **What gets a key** | Whole areas declared "deep" or "associative" | Per-object litmus: a key only if a pure `key(object)→slug` extractor gives a confident, collision-free identity | "Unaddressable" becomes a *safe* outcome (auto-file to associative) instead of forcing a belief onto a wrong shelf. |
| **Personal memories** | Also get addresses | **No addresses.** Navigate by facets + collections + the entity graph | Beliefs have no intrinsic key; forcing one is the drift trap. A memory palace *is* associative, not a numbered shelf. |
| **Code library** | Open question: copy-paste "legos" vs shared dependency | **Dependency-by-reference.** "Use this block" = a reference to its `loc_key` node, not a copy | Copies freeze their dependency list and rot; a fix to the canonical node is seen everywhere. Unlocks exact portfolio queries ("which repos use pandas / are exposed to this CVE"). |
| **"Never miss" goal** | Achieved by tidier filing | Achieved by **reconciliation** (a trial-balance / `fsck` / canary that detects misses) | You can't organize your way out of a probabilistic top-k cutoff. The miss that matters is the one where *nothing surfaced and nothing was logged* — invisible unless you actively reconcile. |

## How the new system works, by half

### Catalog half — code, files, projects (has an intrinsic identity)
- Every item gets an **immutable `loc_key`** derived from its structure (AST symbol path / file path). A `loc_registry` table is the anti-drift source of truth (think compiler symbol table / git object store).
- **Content hash = version, not identity** — a typo-fix doesn't reset a function's links or reuse-count; refactors emit `rename` edges so identity survives.
- Code blocks are **referenced, not copied**; `requires` (dependencies) is **machine-extracted from the AST and normalized**, making the library a queryable dataset.

### Associative half — your brain, decisions, beliefs (no intrinsic identity)
- **No addresses.** The fix for "list all the X drops some" isn't a new table — it's a **router**: an enumeration-intent classifier in `weft_recall` that sends "list all / every / how many" queries to the deterministic *complete* gather (already built in `topic_gather.py`) instead of the top-k similarity search that silently truncates.
- **Collections** = first-class sets with **rule-based or confirmed membership** (never silent similarity-attach). A taxonomic `is_a` layer lets "list all plants" find "basil."
- **Areas are facets** (a soft boost over the tag/entity graph), not address prefixes. Tenancy (`workspace_id`) stays a completely separate axis from relevance.

### Time (applies to both)
- Keep what exists: `created_at`, `updated_at` (modified), `accessed_at`, `review_after` (= "re-verify this," trust-expiry).
- **Add a world-validity window** `valid_from` / `valid_to` (= "was true *until* the move"). Kept separate from `review_after`; `created_at` stays the always-correct fallback. Only built if it can be populated cleanly.

## The thing we build first: the meter

Before any of the visible work, **Phase 0** builds miss-*detection* — because a never-miss claim you can't check is just a hope:
- A **recall canary**: every memory enrolls a known-answer probe; a daily fixed-materialization audit logs any that fail to surface as first-class defects.
- **`weft_fsck`**: lists memories reachable *only* by vector similarity (orphans = the leading indicator of a future miss).
- Plus the cheap correctness fixes everything else rides on (entity-merge threshold, surfaced truncation, deterministic ranking tie-break, known-membership eval fixtures).

This is unglamorous and produces no demo, which is exactly why it's the thing most likely to get cut — so it goes first.

## What Weft becomes if this lands
Not a memory *store* but a **queryable engineering knowledge graph over the whole portfolio**: a personal dependency index / SBOM, a "why does this code exist" rationale layer (git-blame for *intent*), set-algebra over your own ideas, and — the category-defining bet — a memory system that can **prove it didn't lose anything**.

## Sequencing
- **Phase 0 — the meter** (cheap, certain, first): reconciliation canary, `weft_fsck`, entity-threshold fix, truncation plumbing, deterministic tie-break, eval fixtures.
- **Phase 1 — cheap wins**: enumeration router, `loc_key` + registry (AST-filled for code, null for beliefs), AST-derived `requires` + portfolio query, collections-as-predicates.
- **Phase 2 — research-shaped** (measure first, never single-run): bi-temporal validity, taxonomic `is_a` layer, the project-map tool (`weft_fingerprint`, deferred until the ingest path can support it), facet router, self-curating promotion.

## Explicitly deferred
- **The "one primitive" collapse** (merging entities/episodes/collections/areas/code-blocks/trackers into one node table) — too risky to migrate now; instead new shapes ride a shared substrate, and we revisit after dogfooding v2 (memory `weft-96b067f3`, review ~2026-08).
- **`weft_fingerprint` / project-map** — framework now (the `loc_key` + registry groundwork), build the map view once ingestion supports it.

## Cost guardrail
The write path is free today (regex + local embeddings + one insert). **No LLM classifier on the `weft_remember` hot path** — route by cosine over precomputed area centroids using the embedding we already compute; escalate to a model only in a narrow ambiguity band, deferred to the nightly pass.
