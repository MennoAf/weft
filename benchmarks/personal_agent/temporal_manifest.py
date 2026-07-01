#!/usr/bin/env python3
"""
temporal_manifest.py — PAAH ground truth for the temporal/dialogue shape.

This is the turn-tier probe the roadmap (weft-c0a51a73) flags as never closed:
turn-tier recall is "untested at scale." The manifest seeds a dated dialogue
trace with KNOWN anchor turns, then each probe asks a temporal/dialogue
question whose answer lives in exactly one seeded turn. The structural signal is
whether the agent-facing recall response SURFACES that anchor turn (by id) —
if it doesn't, the turn tier silently failed to answer.

Two routing families are exercised (see weft/turn_recall.py markers):
  * temporal ("when did X", "how long since X") → tier='turns'
  * dialogue/self-referential ("what did I last say about Y") → tier='both'

The Iceland trip is mentioned in TWO turns so "what did I last say" has a
non-trivial answer — the anchor is the MORE RECENT of the two, which a naive
"any Iceland turn" match would get wrong.

Seeded under PAAH_USER_ID (proven-scoping identity) but a DEDICATED temporal
project so it never shares rows with the enumeration collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.personal_agent.manifest import PAAH_USER_ID  # re-export identity

PAAH_TEMPORAL_PROJECT_ID = "paah-personal-agent-temporal"

# Recall limit for the temporal shape: below the seeded turn count (16) so an
# anchor must RANK into the top-k against the whole trace — otherwise "anchor
# present" would be trivially true because every turn fits under the cutoff.
TEMPORAL_LIMIT = 8

__all__ = [
    "PAAH_USER_ID",
    "PAAH_TEMPORAL_PROJECT_ID",
    "TEMPORAL_LIMIT",
    "TurnSpec",
    "TurnProbe",
    "get_turn_specs",
    "get_turn_probes",
]


@dataclass(frozen=True)
class TurnSpec:
    """One seeded dialogue turn with a known timestamp.

    key: stable handle used by probes to name their anchor turn.
    occurred_at: ISO-8601 (UTC) — the temporal anchor value the agent needs.
    content: the user utterance (all turns are 'user' role — self-referential
        "what did I last say" queries are about the user's own utterances).
    """

    key: str
    occurred_at: str
    content: str


@dataclass(frozen=True)
class TurnProbe:
    """A temporal/dialogue question with a known anchor turn.

    expected_tier: the tier route_query_to_tier should pick ('turns'|'both').
        A misroute is itself a finding (the router never fires the right path).
    anchor_key: the single seeded turn that answers the probe.
    topic_keys: all seeded turns about the probe's topic. For a "last" probe
        this is >1 and the anchor must be the most recent of them.
    phrasings: k phrasings, all preserving the routing marker + topic terms.
        The spread across phrasings is the never-miss signal.
    """

    key: str
    expected_tier: str
    anchor_key: str
    phrasings: tuple[str, ...]
    topic_keys: tuple[str, ...] = field(default_factory=tuple)


# ── The dialogue trace: 16 dated turns, distinct topics, two about Iceland ──
def get_turn_specs() -> list[TurnSpec]:
    return [
        TurnSpec("espresso", "2026-06-01T09:00:00+00:00",
                 "I finally set up the new espresso machine in the kitchen."),
        TurnSpec("running", "2026-06-03T07:30:00+00:00",
                 "I started the couch-to-5k running program this morning."),
        TurnSpec("iceland_booked", "2026-06-05T20:00:00+00:00",
                 "I booked the flights for the Iceland trip in October."),
        TurnSpec("beagle", "2026-06-07T14:00:00+00:00",
                 "We adopted a rescue beagle named Biscuit today."),
        TurnSpec("bank", "2026-06-09T11:00:00+00:00",
                 "I switched my main bank over to a local credit union."),
        TurnSpec("garden", "2026-06-11T16:00:00+00:00",
                 "I planted the fall garden beds with garlic and kale."),
        TurnSpec("woodworking", "2026-06-13T18:30:00+00:00",
                 "I signed up for a woodworking class downtown."),
        TurnSpec("office_paint", "2026-06-15T13:00:00+00:00",
                 "I repainted the home office a deep forest green."),
        TurnSpec("bass", "2026-06-17T19:00:00+00:00",
                 "I started teaching myself the upright bass."),
        TurnSpec("faucet", "2026-06-19T10:00:00+00:00",
                 "I finally fixed the leaky kitchen faucet."),
        TurnSpec("hiking", "2026-06-21T08:00:00+00:00",
                 "I joined a Sunday morning hiking group."),
        TurnSpec("standing_desk", "2026-06-23T12:00:00+00:00",
                 "I bought a standing desk for the office."),
        TurnSpec("iceland_westfjords", "2026-06-25T21:00:00+00:00",
                 "I changed the Iceland trip to add three days in the Westfjords."),
        TurnSpec("timing_belt", "2026-06-27T15:00:00+00:00",
                 "I replaced the car's timing belt this weekend."),
        TurnSpec("sourdough", "2026-06-29T09:30:00+00:00",
                 "I started a sourdough starter and named it Clint."),
        TurnSpec("solar_lights", "2026-06-30T17:00:00+00:00",
                 "I installed solar-powered lights along the garden path."),
    ]


def get_turn_probes() -> list[TurnProbe]:
    return [
        TurnProbe(
            key="when_espresso",
            expected_tier="turns",
            anchor_key="espresso",
            phrasings=(
                "when did I set up the espresso machine",
                "when did I get the espresso machine set up",
                "how long ago did I set up the espresso machine",
            ),
            topic_keys=("espresso",),
        ),
        TurnProbe(
            key="since_beagle",
            expected_tier="turns",
            anchor_key="beagle",
            phrasings=(
                "how long since I adopted Biscuit the beagle",
                "when did I adopt Biscuit",
                "how long ago did I adopt the beagle Biscuit",
            ),
            topic_keys=("beagle",),
        ),
        TurnProbe(
            key="last_iceland",
            expected_tier="both",
            anchor_key="iceland_westfjords",  # the MORE RECENT Iceland turn
            phrasings=(
                "what did I last say about the Iceland trip",
                "what did I last mention about the Iceland trip",
                "what did I last tell you about Iceland",
            ),
            topic_keys=("iceland_booked", "iceland_westfjords"),
        ),
    ]
