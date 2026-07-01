#!/usr/bin/env python3
"""
manifest.py — PAAH ground-truth manifest for the enumeration shape.

The manifest is the ORACLE. Each collection is a topic tag plus a set of
distinct member facts with KNOWN cardinality, seeded under a dedicated
synthetic persona (Jim Boblaw — never a real name, per house rule). Every
downstream assertion measures the recall path against these counts.

Two design constraints matter for the enumeration end-to-end test:

1. **At least one collection exceeds the recall limit.** ``PLANTS`` has 14
   members; ``weft_recall``'s default ``limit`` is 10. That gap is the whole
   point: a naive agent that counts ``len(results)`` sees at most 10 and
   undercounts, while the enumeration reconciliation header knows all 14.
   The consumption-contract question — does the agent read the header? — is
   only observable when membership > limit.

2. **Members are semantically distinct.** ``weft_remember`` runs a pre-insert
   dedup check (``check_dedup_on_store``); near-duplicate facts collapse to a
   single row and silently corrupt the known cardinality. Distinct plants /
   meds / projects keep each write a genuine new member. The seeder still
   verifies the ACTUAL stored count against these intents, so any collapse is
   surfaced as a seeding-integrity failure rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────
# Dedicated synthetic identity — isolated from Jason's real corpus AND from
# the enumeration_eval Jim Boblaw sandbox (distinct user_id/project_id so the
# two harnesses never share seeded rows even inside one container).
# ─────────────────────────────────────────────────────────────────────────

PAAH_USER_ID = "paah-jim-boblaw-synthetic"
PAAH_PROJECT_ID = "paah-personal-agent-eval"


@dataclass(frozen=True)
class Collection:
    """A known-membership enumerable collection.

    Attributes:
        name: Human-readable collection key (also the JSON report key).
        topic_tag: The single topic tag every member is stored under. The
            enumeration router resolves the query noun to a tag list via
            ``resolve_topic``; naive normalization lowercases the noun, so the
            tag must be the lowercase plural head noun the query will yield
            (e.g. query "how many plants" → noun "plants" → tag "plants").
        noun: The plural head noun an enumeration query carries. Kept explicit
            so query phrasings and the resolve target stay in lock-step.
        members: The distinct member facts. ``len(members)`` is the ground
            truth cardinality.
    """

    name: str
    topic_tag: str
    noun: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(f"Collection {self.name!r} has zero members")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"Collection {self.name!r} has duplicate member text")

    @property
    def cardinality(self) -> int:
        return len(self.members)


# PLANTS — 14 members, intentionally > weft_recall default limit (10).
# This is the collection that forces the consumption gap into the open.
_PLANTS = Collection(
    name="plants",
    topic_tag="plants",
    noun="plants",
    members=(
        "Jim grows heirloom tomatoes on his back porch.",
        "Jim keeps a golden pothos trailing over his bookshelf.",
        "Jim grows Thai basil in a kitchen window box.",
        "Jim planted a row of sunflowers by the shed.",
        "Jim repotted a moth orchid last spring.",
        "Jim propagates jade succulents on the windowsill.",
        "Jim trained climbing roses along the back fence.",
        "Jim nurtures a Japanese maple bonsai on his desk.",
        "Jim sows rainbow carrots in the raised bed.",
        "Jim mists a Boston fern in the bathroom each day.",
        "Jim dried English lavender from the front border.",
        "Jim collects barrel cactus varieties in the office.",
        "Jim keeps a fiddle-leaf fig in the living room corner.",
        "Jim grows peppermint in a hanging planter.",
    ),
)

# MEDICATIONS — 6 members, < limit. Control collection: membership fits inside
# top-k, so a naive results[] count and the header should AGREE here.
_MEDICATIONS = Collection(
    name="medications",
    topic_tag="medications",
    noun="medications",
    members=(
        "Jim takes low-dose aspirin each morning for his heart.",
        "Jim manages blood sugar with metformin twice daily.",
        "Jim takes a vitamin D3 supplement in winter.",
        "Jim rubs an ibuprofen gel on his left knee.",
        "Jim takes omeprazole before breakfast for reflux.",
        "Jim takes a cetirizine tablet during allergy season.",
    ),
)

# PROJECTS — 9 members, < limit. Second control collection, different noun so
# the resolve_topic step is exercised on more than one token.
_PROJECTS = Collection(
    name="projects",
    topic_tag="projects",
    noun="projects",
    members=(
        "Jim is renovating the upstairs bathroom this year.",
        "Jim is building a cedar deck in the backyard.",
        "Jim is restoring a 1970s road bike in the garage.",
        "Jim is writing a mystery novel set in Maine.",
        "Jim is learning to bake sourdough on weekends.",
        "Jim is digitizing his family's old photo albums.",
        "Jim is planning a solo trip to Iceland next fall.",
        "Jim is teaching himself the upright bass.",
        "Jim is coaching his kid's little-league team.",
    ),
)


def get_collections() -> list[Collection]:
    """Return all enumerable collections in the manifest (the ground truth)."""
    return [_PLANTS, _MEDICATIONS, _PROJECTS]
