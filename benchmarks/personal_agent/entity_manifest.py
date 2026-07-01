#!/usr/bin/env python3
"""
entity_manifest.py — PAAH ground truth for the entity-brief shape (beliefs+graph).

The personal-agent query: "what do I need to know about X before I meet them."
The answer is the COMPLETE set of facts linked to that person — a brief. The
manifest seeds a person entity with a known set of linked facts, plus a
DISTRACTOR person with their own facts, so a recall path has to discriminate
(otherwise recall@links is trivially complete because there's nothing else to
confuse it with).

Two read paths, mirroring the enumeration oracle/candidate split:
  * ORACLE   = weft_entity_context(entity_id) — the graph edge walk; returns the
    linked memories directly. Should be complete (recall@links == 1.0).
  * CANDIDATE = weft_recall("brief me on X") — natural-language top-k. The spread
    across phrasings is the never-miss signal; the gap vs the oracle is the
    finding (does an agent need the entity path, or does NL recall suffice?).

Synthetic personas only (house rule): the user is Jim Boblaw; the colleague is
the clearly-fabricated "Zelda Quackenbush". No real names.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.personal_agent.manifest import PAAH_USER_ID  # re-export identity

PAAH_ENTITY_PROJECT_ID = "paah-personal-agent-entity"

# Recall limit for the candidate NL path. Below the total seeded fact count
# (brief + distractor = 8 + 6 = 14) so the brief facts must rank into the top-k
# against the distractor's facts — real completeness pressure.
ENTITY_LIMIT = 10

__all__ = [
    "PAAH_USER_ID",
    "PAAH_ENTITY_PROJECT_ID",
    "ENTITY_LIMIT",
    "EntitySpec",
    "get_brief_entity",
    "get_distractor_entity",
    "get_brief_phrasings",
]


@dataclass(frozen=True)
class EntitySpec:
    """A person entity with a known set of linked facts (the brief)."""

    name: str
    entity_type: str
    description: str
    facts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.facts)) != len(self.facts):
            raise ValueError(f"Entity {self.name!r} has duplicate fact text")

    @property
    def cardinality(self) -> int:
        return len(self.facts)


# The person to brief on — 8 distinct facts an agent would want before a meeting.
_ZELDA = EntitySpec(
    name="Zelda Quackenbush",
    entity_type="person",
    description="Jim Boblaw's new colleague on the co-op rooftop-garden project.",
    facts=(
        "Zelda Quackenbush leads the rooftop-garden project at the co-op.",
        "Zelda Quackenbush is allergic to peanuts, so avoid peanut snacks at meetings.",
        "Zelda Quackenbush prefers morning meetings before 10am.",
        "Zelda Quackenbush previously worked as a landscape architect in Portland.",
        "Zelda Quackenbush is wary of drip-irrigation vendors after a bad contract.",
        "Zelda Quackenbush takes handwritten notes and dislikes slide decks.",
        "Zelda Quackenbush co-authored a guide on urban composting.",
        "Zelda Quackenbush's daughter just started college in Vermont.",
    ),
)

# A second person whose facts must NOT bleed into Zelda's brief — discrimination
# pressure for the candidate NL recall path.
_BARTLEBY = EntitySpec(
    name="Bartleby Quench",
    entity_type="person",
    description="Jim Boblaw's contact at the hardware cooperative.",
    facts=(
        "Bartleby Quench runs the tool-lending library at the hardware co-op.",
        "Bartleby Quench is restoring a vintage sailboat on weekends.",
        "Bartleby Quench prefers email over phone calls.",
        "Bartleby Quench used to coach high-school debate.",
        "Bartleby Quench is gluten-free.",
        "Bartleby Quench collects antique hand planes.",
    ),
)


def get_brief_entity() -> EntitySpec:
    return _ZELDA


def get_distractor_entity() -> EntitySpec:
    return _BARTLEBY


def get_brief_phrasings() -> tuple[str, ...]:
    """NL brief queries — none trigger the enumeration or turn-tier routers."""
    return (
        "what do I need to know about Zelda Quackenbush before our meeting",
        "brief me on Zelda Quackenbush",
        "tell me everything about Zelda Quackenbush",
        "what should I remember about Zelda Quackenbush before I meet her",
        "give me the background on Zelda Quackenbush",
    )
