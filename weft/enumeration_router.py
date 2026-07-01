"""Enumeration-intent router (Phase 1, V7).

Natural-language enumeration asks — "list all the plants", "how many
medications", "every project I'm on", "enumerate the open issues" — flow through
``weft_recall``'s limit-bounded top-k search, where members below the cutoff
silently drop. Meanwhile the deterministic *complete* gather
(``gather_topic_memories``) already knows the full membership set but sits behind
the topic-digest surface, unused by recall.

This module supplies the cheap intent classifier ``weft_recall`` uses to detect
those asks and resolve the noun to enumerate. ``weft_recall`` then fires the
gather in parallel with the top-k search and returns a reconciliation header —
*"similarity surfaced 7; membership knows 12; 5 not shown: [ids]"* — so the gap
between what similarity showed and what membership knows is visible, not silent.

Design notes:
  - **The gather is the false-positive filter.** The intent regexes are
    deliberately generous ("every X" is broad). A spurious match resolves a noun
    that is not a real topic → the gather returns zero members → ``weft_recall``
    adds no header. So over-triggering costs one cheap indexed query, never a
    wrong answer.
  - **No LLM on this path** (Vcost). Detection is pure regex; resolution is the
    existing alias/naive ``resolve_topic``; the gather is one indexed SQL query.
  - Temporal "how many days/weeks…" asks never reach here: ``route_query_to_tier``
    routes them to the turns tier before the belief path runs the router.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg

    from weft.topic_gather import TopicGatherResult

logger = logging.getLogger(__name__)


# --- Intent gate -----------------------------------------------------------
# Any hit marks the query as enumeration-shaped. Generous by design: the gather
# is the real filter (a non-topic noun yields an empty membership set, so no
# header is emitted). Lowercased before matching.
_INTENT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\blist\s+(?:all|every|out|of|the|my|our)\b"),
    re.compile(r"\benumerate\b"),
    re.compile(r"\bhow many\b"),
    re.compile(r"\bevery\s+(?:single\s+)?[a-z]"),
    re.compile(r"\ball\s+(?:of\s+)?(?:my|the|your|our)\b"),
    re.compile(r"\bwhat\s+(?:are|were)\s+all\b"),
    re.compile(r"\b(?:give|show)\s+me\s+all\b"),
)


# --- Noun extraction -------------------------------------------------------
# Each pattern captures a generous noun tail in group ``noun``; the tail is then
# trimmed at the first clause-boundary token (see ``_BOUNDARY``). Ordered most-
# to least specific; first match wins.
_NOUN_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bhow many\s+(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\benumerate\s+(?:all\s+|every\s+)?(?:the\s+|my\s+|our\s+)?(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\bwhat\s+(?:are|were)\s+all\s+(?:the\s+|my\s+)?(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\b(?:give|show)\s+me\s+all\s+(?:the\s+|my\s+)?(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\blist\s+(?:all|every|out)?\s*(?:of\s+)?(?:the\s+|my\s+|our\s+)?(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\bevery\s+(?:single\s+)?(?P<noun>[a-z0-9\- ]+)"),
    re.compile(r"\ball\s+(?:of\s+)?(?:my|the|your|our)\s+(?P<noun>[a-z0-9\- ]+)"),
)

# Leading determiners/quantifiers stripped off a captured noun.
_LEADING_STOP: frozenset[str] = frozenset(
    {"the", "my", "our", "your", "of", "all", "some", "any", "those", "these", "every"}
)

# Clause-boundary tokens — the noun run ends at the first of these. "plants i
# have", "medications do i take", "projects that are open" → "plants",
# "medications", "projects".
_BOUNDARY: frozenset[str] = frozenset(
    {
        "i", "we", "you", "do", "did", "does", "have", "has", "had",
        "that", "which", "who", "are", "is", "was", "were", "am",
        "in", "on", "for", "with", "about", "from", "to", "so", "far",
        "currently", "right", "now", "there", "and", "or", "but",
    }
)


def _clean_noun(raw: str) -> str | None:
    """Trim a captured noun run to its head noun phrase.

    Strips leading determiners, then keeps words up to the first clause-boundary
    token. Returns the cleaned phrase, or ``None`` if nothing usable remains.
    """
    words = [w for w in re.split(r"\s+", raw.strip()) if w]
    # Drop leading determiners/quantifiers.
    while words and words[0] in _LEADING_STOP:
        words.pop(0)
    # Keep words until a boundary token.
    kept: list[str] = []
    for w in words:
        if w in _BOUNDARY:
            break
        kept.append(w)
    phrase = " ".join(kept).strip(" -?.,!")
    return phrase or None


def detect_enumeration_intent(query: str) -> tuple[bool, str | None]:
    """Classify a query's enumeration intent and extract the noun to enumerate.

    Returns ``(is_enumeration, noun)``:
      - ``(False, None)`` — not an enumeration ask.
      - ``(True, "plants")`` — enumeration ask with an extracted target noun.
      - ``(True, None)`` — enumeration-shaped but no noun could be extracted
        (caller should fall back to an explicit ``topic`` argument, if any).

    Pure regex, no I/O — safe to call on the hot path.
    """
    if not query or not query.strip():
        return (False, None)
    q = query.lower()

    if not any(p.search(q) for p in _INTENT_PATTERNS):
        return (False, None)

    for pat in _NOUN_PATTERNS:
        m = pat.search(q)
        if m:
            noun = _clean_noun(m.group("noun"))
            if noun:
                logger.debug(
                    "enumeration intent: %r → noun=%r (via %s)",
                    query[:60], noun, pat.pattern,
                )
                return (True, noun)

    logger.debug("enumeration intent: %r matched but no noun extracted", query[:60])
    return (True, None)


async def gather_enumeration(
    pool: "asyncpg.Pool",
    target: str,
    user_id: str,
    budget_tokens: int = 2000,
) -> tuple[list[str], "TopicGatherResult | None"]:
    """Resolve ``target`` to canonical tags and run the deterministic gather.

    Returns ``(resolved_tags, gather_result)``. Swallows its own errors and
    returns ``([], None)`` on failure — the enumeration reconciliation is a
    best-effort augmentation and must NEVER break ``weft_recall``.
    """
    try:
        from weft.topic_gather import gather_topic_memories
        from weft.topic_resolution import resolve_topic

        resolved_tags = await resolve_topic(target, user_id, pool)
        gather_result = await gather_topic_memories(
            pool, tags=resolved_tags, user_id=user_id, budget_tokens=budget_tokens
        )
        return (resolved_tags, gather_result)
    except Exception as exc:  # noqa: BLE001 - best-effort augmentation
        logger.warning("enumeration gather failed for target=%r: %s", target, exc, exc_info=True)
        return ([], None)
