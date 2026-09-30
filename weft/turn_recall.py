"""Turn-tier query planner + multi-anchor recall.

Two responsibilities:

1. ``route_query_to_tier`` — decide whether a query should hit the belief
   tier (semantic facts), the turn tier (raw dialogue trace), or both.
   Regex-first; cheap; deterministic. The driver: LongMemEval temporal-
   reasoning questions ("how many days between A and B", "when did I last
   ...") fail on belief-tier extraction because dates collapse during
   classification. Routing those to turns avoids the lossy step.

2. ``temporal_anchor`` — for queries with multiple named anchors
   ("between launch and demo", "after I started X but before Y"), split
   the query into per-anchor sub-queries and run ``recall_turns`` for
   each. Returns a mapping ``anchor → [turns]`` so the Reader can do
   anchored arithmetic without conflating the two halves.

This module is intentionally thin: heavy lifting lives in
``weft.episode_turns.recall_turns``. Adding a tiny LLM-based fallback
planner is on the roadmap but the cost/latency hit is real and the regex
covers the LongMemEval failure patterns we measured.

Spec: weft-d3a2ef78. Loom task: loom-d9ac7e18.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Literal

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.episode_turns import _RRF_K, list_recent_turns, recall_turns
from weft.models import EpisodeTurn, MemoryRecall, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)


Tier = Literal["belief", "turns", "both", "auto"]


# --- Routing ---


# Lowercased regexes — case folding happens before match. Each pattern
# captures a temporal-reasoning marker that belief-tier extraction tends
# to lose. Order doesn't matter; first hit returns.
_TURN_TIER_MARKERS: tuple[re.Pattern, ...] = (
    re.compile(r"\bhow many times\b"),
    re.compile(r"\bhow many (?:days|weeks|months|hours|minutes|years)\b"),
    re.compile(r"\bhow long (?:since|ago|before|after|between)\b"),
    re.compile(r"\bwhen (?:did|was)\b"),
    re.compile(r"\b(?:before|after|between|since|until)\b"),
    re.compile(r"\blast (?:time|week|month|year|tuesday|wednesday|thursday|friday|saturday|sunday|monday)\b"),
    re.compile(r"\b(?:earlier|later) (?:than|that)\b"),
    re.compile(r"\bwhat (?:day|date|time)\b"),
    re.compile(r"\bwhat (?:was|is) the chronology\b"),
)


# Queries that benefit from RRF across belief and turns: explicit
# episodic asks ("do you remember", "have we discussed"), self-referential
# fact queries that reach for prior dialogue ("what did I last say",
# "my decision on"), and remind-me prompts. The _TURN_TIER_MARKERS list
# above catches pure temporal-arithmetic queries that need raw turns;
# this list catches queries where the user wants the canonical fact
# (belief tier) AND the dialogue evidence (turn tier) fused together.
#
# These patterns are checked BEFORE _TURN_TIER_MARKERS in
# route_query_to_tier so a query like "what did I last say about X"
# routes to 'both' instead of falling through to 'turns'.
#
# Starter set is deliberately narrow (PRECISION over recall on first
# ship) — broaden based on Oracle data once Tier 1.5 RRF lands the
# 'both' dispatch path. Today the auto-path caller (weft_recall) only
# handles 'belief' and 'turns'; 'both' returns fall through to belief
# until P1.B2 wires the RRF fusion.
_BOTH_TIER_MARKERS: tuple[re.Pattern, ...] = (
    # Explicit episodic recall asks.
    re.compile(r"\b(?:do you|did we|have we) (?:recall|remember|discuss)"),
    # "have we / I (talked|discussed|mentioned)" + topic.
    re.compile(r"\bhave (?:we|i) (?:discussed|talked|mentioned)"),
    # Self-referential fact retrieval that reaches for prior dialogue.
    re.compile(
        r"\bwhat did i (?:last )?(?:say|tell|mention|decide|think|conclude)"
    ),
    # Possessive opinion / decision queries — fact + history together.
    re.compile(
        r"\b(?:my|our) (?:decision|opinion|view|stance|take|position) (?:on|about)\b"
    ),
    # Remind-me prompts — factual answer grounded in prior dialogue.
    re.compile(r"\bremind me (?:about|of|what|when)\b"),
    # Decision rationale is often omitted from concise handoffs/beliefs; retrieve
    # the quoted dialogue evidence rather than fabricating a reason.
    re.compile(r"\bwhy did (?:we|i) (?:reject|choose|pick|decide|change)\b"),
)


def route_query_to_tier(query: str) -> Tier:
    """Decide which retrieval tier a query should hit.

    Returns:
        ``'both'`` for queries that benefit from RRF across belief and
        turns (explicit episodic asks, self-referential fact queries
        that need dialogue evidence). ``'turns'`` for queries with
        explicit temporal markers (LongMemEval temporal-reasoning
        patterns). ``'belief'`` otherwise.

    Priority order is ``both`` → ``turns`` → ``belief``. A query like
    "what did I last say about the launch" matches the ``_BOTH_TIER_MARKERS``
    self-referential pattern AND would also match no turn marker today,
    but the ordering matters for queries that hit both lists (e.g.
    "do you recall when did I ship" hits ``do you recall`` first and
    routes to 'both' rather than falling through to 'turns').
    """
    q = query.lower()
    for pattern in _BOTH_TIER_MARKERS:
        if pattern.search(q):
            logger.debug("route_query_to_tier: %r matched %s → both",
                         query[:60], pattern.pattern)
            return "both"
    for pattern in _TURN_TIER_MARKERS:
        if pattern.search(q):
            logger.debug("route_query_to_tier: %r matched %s → turns",
                         query[:60], pattern.pattern)
            return "turns"
    return "belief"


# --- Anchor extraction ---


# Patterns that imply two named anchors. Capture groups carry the anchor
# strings so the caller can run a sub-query per anchor.
_ANCHOR_PATTERNS: tuple[re.Pattern, ...] = (
    # Three-stage chronology must be checked before the two-stage ``from``
    # pattern or the middle event is swallowed into the first anchor.
    re.compile(
        r"\bfrom\s+(.+?)\s+through\s+(.+?)\s+to\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
    # "between X and Y", "from X to Y"
    re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.,]|$)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[?.,]|$)", re.IGNORECASE),
    # "after X but before Y" / "before X and after Y"
    re.compile(
        r"\bafter\s+(.+?)\s+(?:but|and)\s+before\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbefore\s+(.+?)\s+(?:but|and)\s+after\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """A UTC calendar window used by the benchmark temporal probe."""

    since: datetime
    until: datetime


_RELATIVE_QUANTITY_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_RELATIVE_DATE_PATTERN = re.compile(
    r"\b(?P<quantity>\d+|(?:a\s+)?couple\s+of|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<unit>days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)
_LAST_WEEKDAY_PATTERN = re.compile(
    r"\blast\s+(?P<weekday>monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_RECURRING_WEEKDAY_PATTERN = re.compile(
    r"\b(?:every|each)\s+(?:monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)s?\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
    r"\s*(?:,|and|&)\s*(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)s?\b",
    re.IGNORECASE,
)


def _parse_question_date(value: str | date | datetime) -> date | None:
    """Parse the date formats used by LongMemEval without guessing."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("/", "-")
    if not normalized:
        return None
    try:
        if "T" in normalized:
            return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _relative_quantity(value: str) -> int | None:
    normalized = " ".join(value.casefold().split())
    if normalized.isdigit():
        return int(normalized)
    if normalized in _RELATIVE_QUANTITY_WORDS:
        return _RELATIVE_QUANTITY_WORDS[normalized]
    if normalized in {"couple of", "a couple of"}:
        return 2
    return None


def _subtract_calendar_months(value: date, months: int) -> date:
    """Subtract calendar months, clamping an end-of-month day."""
    month_index = value.year * 12 + value.month - 1 - months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _window_for_center(center: date, margin_days: int) -> TemporalWindow:
    if margin_days < 0:
        raise ValueError("margin_days must be non-negative")
    start = center - timedelta(days=margin_days)
    end = center + timedelta(days=margin_days)
    return TemporalWindow(
        since=datetime.combine(start, time.min, tzinfo=timezone.utc),
        until=datetime.combine(end, time.max, tzinfo=timezone.utc),
    )


def parse_temporal_window(
    query: str,
    question_date: str | date | datetime,
    *,
    margin_days: int = 1,
) -> TemporalWindow | None:
    """Infer one conservative UTC calendar window from a relative-time query.

    The helper is deliberately pure and conservative. It recognizes one
    unambiguous relative expression (``N days/weeks/months/years ago``,
    ``a couple of days ago``, or ``last <weekday>``) relative to the supplied
    benchmark ``question_date``. Recurring weekday expressions, multiple time
    expressions, malformed dates, and unsupported wording return ``None`` so
    callers can leave the normal unfiltered retrieval path unchanged.
    """
    reference = _parse_question_date(question_date)
    if reference is None or not isinstance(query, str):
        return None
    if _RECURRING_WEEKDAY_PATTERN.search(query):
        return None

    matches: list[tuple[str, re.Match[str]]] = [
        ("relative", match) for match in _RELATIVE_DATE_PATTERN.finditer(query)
    ]
    matches.extend(
        ("weekday", match) for match in _LAST_WEEKDAY_PATTERN.finditer(query)
    )
    if len(matches) != 1:
        return None

    kind, match = matches[0]
    if kind == "relative":
        quantity = _relative_quantity(match.group("quantity"))
        if quantity is None:
            return None
        unit = match.group("unit").casefold()
        if unit.startswith("day"):
            center = reference - timedelta(days=quantity)
        elif unit.startswith("week"):
            center = reference - timedelta(weeks=quantity)
        elif unit.startswith("month"):
            center = _subtract_calendar_months(reference, quantity)
        else:
            center = _subtract_calendar_months(reference, quantity * 12)
    else:
        weekday = _WEEKDAY_NAMES[match.group("weekday").casefold()]
        days_ago = (reference.weekday() - weekday) % 7 or 7
        center = reference - timedelta(days=days_ago)

    return _window_for_center(center, margin_days)


def temporal_query_variants(
    query: str,
    *,
    include_embedded_temporal_variant: bool = False,
) -> list[str]:
    """Return conservative event-focused variants for a temporal query.

    The historical variants remove temporal scaffolding at the *start* of a
    query (``how many weeks ago did I ...`` / ``when did we ...``).  The
    benchmark miss matrix also contains questions where the time expression
    is embedded in, or trails, the event clause (``what did I buy 10 days
    ago?``).  Those are opt-in because removing a time phrase can discard a
    useful retrieval signal for ordinary callers.

    Variants are deduplicated case-insensitively.  The embedded variant is
    placed first when requested so it is actually probed even when the
    original query already fills the per-probe output budget; the original
    query remains immediately available as the fallback variant.
    """
    original = query.strip()
    variants = [original]

    def append_variant(value: str) -> None:
        value = re.sub(r"\s+([?.!,])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip(" ?.,")
        if len(value) < 4:
            return
        if value.casefold() in {variant.casefold() for variant in variants}:
            return
        variants.append(value)

    cleaned = re.sub(
        r"^\s*how\s+many\s+times\s+(?:have|has)\s+(?:i|we|you|we\s+all)\s+",
        "",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:how\s+many\s+(?:days|weeks|months|hours|minutes|years)\s+ago|"
        r"how\s+long\s+ago|when\s+did|when\s+was|"
        r"what\s+(?:day|date|time)\s+was)\s+",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:did\s+)?(?:i|we|you)\s+", "", cleaned,
        count=1, flags=re.IGNORECASE,
    )
    append_variant(cleaned)

    chronology = re.sub(
        r"^\s*(?:what\s+is\s+the\s+order\s+of|order\s+of|what\s+was\s+the\s+order\s+of)\s+"
        r"(?:the\s+)?(?:three|four|five|six|several|multiple)\s+",
        "",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    chronology = re.sub(
        r"\s+from\s+earliest\s+to\s+latest\s*\??$", "", chronology,
        flags=re.IGNORECASE,
    )
    # A chronology rewrite that only removes terminal punctuation is not a
    # distinct probe. Preserve the historical contract: the punctuation-
    # stripped original is useful for the legacy scaffolding case, but do not
    # add it as a redundant third variant for ordinary event questions.
    if chronology.strip(" ?.,").casefold() != original.strip(" ?.,").casefold():
        append_variant(chronology)

    if include_embedded_temporal_variant:
        event_focused = query
        # Relative durations: ``10 days ago``, ``four weeks ago``, and
        # ``a couple of days ago``.  Keep the surrounding event wording.
        event_focused = re.sub(
            r"\b(?:a\s+couple\s+of|a\s+few|one|two|three|four|five|six|"
            r"seven|eight|nine|ten|\d+)\s+"
            r"(?:days?|weeks?|months?|years?|hours?|minutes?)\s+ago\b",
            " ",
            event_focused,
            flags=re.IGNORECASE,
        )
        # A weekday plus its relative duration is one temporal unit; remove
        # it together so ``on the Wednesday two months ago`` does not leave
        # malformed residue in the event query.
        event_focused = re.sub(
            r"\b(?:on\s+(?:the\s+)?)?(?:last\s+)?"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            r"(?:s\s+and\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))?"
            r"(?:\s+(?:a\s+couple\s+of|a\s+few|one|two|three|four|five|six|"
            r"seven|eight|nine|ten|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago)?\b",
            " ",
            event_focused,
            flags=re.IGNORECASE,
        )
        # Relative named days/weeks and recurring weekday qualifiers.
        event_focused = re.sub(
            r"\b(?:last|this|next)\s+(?:monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday|week|month|year)\b",
            " ",
            event_focused,
            flags=re.IGNORECASE,
        )
        event_focused = re.sub(
            r"\bon\s+(?:mondays?|tuesdays?|wednesdays?|thursdays?|"
            r"fridays?|saturdays?|sundays?)\s+and\s+"
            r"(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|"
            r"saturdays?|sundays?)\b",
            " ",
            event_focused,
            flags=re.IGNORECASE,
        )
        event_focused = re.sub(r"\s+", " ", event_focused).strip()
        if event_focused.casefold() != original.casefold():
            event_focused = re.sub(r"\s+([?.!,])", r"\1", event_focused)
            event_focused = event_focused.strip(" ?.,")
            if len(event_focused) >= 4 and event_focused.casefold() not in {
                variant.casefold() for variant in variants
            }:
                variants.insert(0, event_focused)

    return variants


# Stopwords for the benchmark-only multi-session representation. These words
# express question framing, aggregation, or temporal scaffolding rather than
# the entity/event being enumerated. Keep this list deliberately small and
# deterministic: the treatment is opt-in and must not rewrite production
# queries or silently remove named entities.
_MULTI_SESSION_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "about", "after", "am", "and", "are", "as", "at",
        "be", "been", "before", "between", "but", "by", "can", "could",
        "count", "days", "day", "different", "did", "do", "does", "during",
        "each", "few", "for", "from", "had", "has", "have", "how", "i",
        "in", "into", "is", "it", "last", "long", "many", "me", "months",
        "month", "much", "my", "next", "number", "of", "on", "or", "our",
        "out", "over", "past", "please", "should", "the", "their", "them",
        "these", "this", "those", "three", "through", "times", "to", "total",
        "two", "typical", "us", "was", "we", "were", "what", "whats", "when",
        "where", "which", "with", "would", "years", "year", "you", "your",
        "weeks", "week", "one", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve",
    }
)
_MULTI_SESSION_QUERY_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
)


def multi_session_query_variant(query: str) -> str | None:
    """Build one entity/event-focused representation for multi-session recall.

    This is a conservative, benchmark-only query transformation. It removes
    aggregation framing (``how many``, ``total``), first-person glue, and
    obvious relative-time scaffolding while retaining the nouns and event
    verbs that identify the sessions to enumerate. It returns ``None`` when
    the query is not a string, has no usable content token, or would be
    unchanged. Callers must keep the original query as the primary probe.
    """
    if not isinstance(query, str):
        return None
    original = " ".join(query.strip().split())
    if not original:
        return None

    tokens = _MULTI_SESSION_QUERY_TOKEN_PATTERN.findall(original)
    kept = [
        token for token in tokens
        if token.casefold() not in _MULTI_SESSION_QUERY_STOPWORDS
        and not token.isdigit()
    ]
    if not kept:
        return None

    variant = " ".join(kept).strip()
    if len(variant) < 3 or variant.casefold() == original.casefold():
        return None
    return variant


def extract_anchors(query: str) -> list[str]:
    """Extract named anchor phrases from a temporal query.

    Returns a list of anchor strings, longest-first. Empty list when no
    multi-anchor pattern fires — the caller should fall back to a single
    ``recall_turns(query)`` call in that case.

    Example:
        >>> extract_anchors("how many days between the demo and the retro?")
        ['the demo', 'the retro']
    """
    for pattern in _ANCHOR_PATTERNS:
        m = pattern.search(query)
        if m:
            anchors = [g.strip() for g in m.groups() if g and g.strip()]
            # Filter out trivially short anchors that are probably stop
            # words spilled in by greedy capture (``after I``, ``before he``).
            anchors = [a for a in anchors if len(a.split()) >= 1 and len(a) >= 2]
            if anchors:
                return anchors
    return []


def event_focused_anchor_query(anchor: str) -> str | None:
    """Return one conservative event-focused representation of an anchor.

    This is deliberately deterministic and lexical. It removes only temporal
    scaffolding that can crowd the event out of a per-anchor probe, while
    retaining the anchor's event words and named entities. ``None`` means the
    representation would be unchanged (or too short to be useful).
    """
    if not isinstance(anchor, str):
        return None
    original = " ".join(anchor.strip().split())
    if not original:
        return None
    def remove_first_person_glue(value: str) -> str:
        # Only remove a leading pronoun. Internal ``I``/``we`` tokens can be
        # part of the event wording and are intentionally preserved.
        return re.sub(
            r"^\s*(?:i|we|you)\s+", "", value,
            count=1, flags=re.IGNORECASE,
        ).strip(" ?.,")

    variants = temporal_query_variants(
        original, include_embedded_temporal_variant=True,
    )
    for variant in variants:
        focused = remove_first_person_glue(variant)
        if focused.casefold() != original.casefold() and len(focused) >= 4:
            return focused
    stripped = re.sub(
        r"^\s*(?:the|a|an)\s+(?:time|day|week|month|year)\s+",
        "", original, count=1, flags=re.IGNORECASE,
    ).strip(" ?.,")
    focused = remove_first_person_glue(stripped)
    if len(focused) >= 4 and focused.casefold() != original.casefold():
        return focused
    return None


# --- Multi-anchor recall ---


async def temporal_anchor(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    top_k_per_anchor: int = 5,
    embedder: EmbeddingProvider | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    executor: asyncpg.Connection | None = None,
    as_of: datetime | None = None,
    diag_callback: Callable[[str, list, list, list[EpisodeTurn]], None] | None = None,
    probe_diag_callback: Callable[[dict[str, Any]], None] | None = None,
    sql_diag_callback: Callable[[dict[str, Any]], None] | None = None,
    candidate_sql_limit: int | None = None,
    fusion_candidate_limit: int | None = None,
    use_anchor_local_variant: bool = True,
    anchor_candidate_sql_limit: int | None = None,
    anchor_result_limit: int | None = None,
    include_embedded_temporal_variant: bool = False,
    use_event_focused_anchor_representation: bool = False,
    use_stored_search_tsv: bool = False,
    vector_weight: float = 1.0,
    keyword_weight: float = 0.3,
) -> dict[str, list[EpisodeTurn]]:
    """Per-anchor turn recall for multi-anchor temporal questions.

    For queries like "how many days between the launch and the demo", we
    pull the top-K turns matching ``"the launch"`` separately from the
    top-K turns matching ``"the demo"``. The Reader then sees two
    grounded sets and can do the arithmetic without conflating evidence.

    When no multi-anchor pattern fires, falls back to a single
    ``recall_turns(query)`` keyed under the original query string. That
    way the caller always gets a populated dict and never has to special-
    case the no-anchor path.

    Args:
        embedder: required when callers want vector recall in the
            sub-queries. If None, anchor sub-queries are keyword-only
            (still useful — anchor strings tend to contain the rare
            keywords that BM25 hits cleanly).
        executor: optional connection shared by every probe and fallback.
        as_of: fixed timestamp forwarded to relevance reranking.
        diag_callback: observational callback receiving ``(anchor, vector,
            keyword, final_turns)`` for each probe. Callback failures are
            swallowed so diagnostics cannot alter retrieval.
        candidate_sql_limit: optional raw candidate width for each probe.
        anchor_candidate_sql_limit: optional temporal-only raw candidate width;
            applied independently to each anchor and variant probe. When None,
            ``candidate_sql_limit`` is used.
        anchor_result_limit: optional per-anchor fused/merged output cap. This
            widens only the output/merge budget; ``top_k_per_anchor`` remains
            the SQL-width basis and the caller still owns the final Reader cap.
        fusion_candidate_limit: optional RRF absent-half penalty width.
        include_embedded_temporal_variant: opt-in removal of embedded or
            trailing relative-time wording before the original query probe.
        use_stored_search_tsv: benchmark-only opt-in forwarded to
            ``recall_turns`` for the generated FTS column.
        vector_weight / keyword_weight: RRF fusion weights forwarded to
            ``recall_turns`` for every probe. The turn tier weights the
            vector half above the keyword half so a populous keyword half
            cannot demote vector-only gold turns through double RRF
            contributions.

    Returns:
        ``{anchor_text: [EpisodeTurn, ...]}`` ordered as anchors appear
        in the query. The original query is included as a key when
        anchor extraction failed.
    """
    anchors = extract_anchors(query)
    requested_anchor_result_limit = (
        top_k_per_anchor if anchor_result_limit is None else int(anchor_result_limit)
    )
    effective_anchor_result_limit = max(
        top_k_per_anchor, requested_anchor_result_limit,
    )
    if not anchors:
        # No multi-anchor pattern — search the original question and, for
        # temporal scaffolding, an event-focused lexical variant. Unioning
        # these result sets prevents words like "how many weeks ago" from
        # crowding the actual event (e.g. "friends and family sale at
        # Nordstrom") out of the candidate pool.
        variants = temporal_query_variants(
            query,
            include_embedded_temporal_variant=include_embedded_temporal_variant,
        )
        seen: set[str] = set()
        merged: list[EpisodeTurn] = []
        for variant in variants:
            embedding = await embedder.embed(variant) if embedder else None
            vector_rows: list = []
            keyword_rows: list = []

            def _probe_diag(vec: list, kw: list) -> None:
                vector_rows.extend(vec)
                keyword_rows.extend(kw)

            def _sql_diag(event: dict[str, Any]) -> None:
                if sql_diag_callback is None:
                    return
                payload = dict(event)
                payload.update({
                    "anchor": query,
                    "query_variant": variant,
                    "representation": (
                        "original" if variant.casefold() == query.casefold()
                        else "event_focused"
                    ),
                })
                try:
                    sql_diag_callback(payload)
                except Exception:
                    logger.warning(
                        "temporal SQL diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )

            turns = await recall_turns(
                pool, variant,
                project_id=project_id, since=since, until=until,
                top_k=top_k_per_anchor, embedding=embedding,
                executor=executor, as_of=as_of,
                candidate_sql_limit=candidate_sql_limit,
                fusion_candidate_limit=fusion_candidate_limit,
                result_limit=effective_anchor_result_limit,
                diag_callback=_probe_diag,
                sql_diag_callback=(
                    _sql_diag if sql_diag_callback is not None else None
                ),
                use_stored_search_tsv=use_stored_search_tsv,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            if diag_callback is not None:
                try:
                    diag_callback(variant, vector_rows, keyword_rows, turns)
                except Exception:
                    logger.warning(
                        "temporal anchor diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )
            if probe_diag_callback is not None:
                try:
                    probe_diag_callback({
                        "anchor": query,
                        "query": variant,
                        "representation": "original" if variant.casefold() == query.casefold() else "event_focused",
                        "vector_candidate_count": len(vector_rows),
                        "keyword_candidate_count": len(keyword_rows),
                        "candidate_count": len({row["id"] for row in vector_rows} | {row["id"] for row in keyword_rows}),
                        "candidate_ids": sorted({row["id"] for row in vector_rows} | {row["id"] for row in keyword_rows}),
                        "retrieved_turn_ids": [turn.id for turn in turns],
                        "requested_output_cap": requested_anchor_result_limit,
                        "effective_output_cap": effective_anchor_result_limit,
                    })
                except Exception:
                    logger.warning(
                        "temporal probe diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )
            for turn in turns:
                if turn.id in seen:
                    continue
                seen.add(turn.id)
                merged.append(turn)
                if len(merged) >= effective_anchor_result_limit:
                    break
            if len(merged) >= effective_anchor_result_limit:
                break
        if not merged:
            # Empty hybrid hit — try a temporal-only fallback so the
            # Reader at least sees recent dialogue. Keep it on the same
            # executor when benchmark diagnostics provide one.
            merged = await list_recent_turns(
                pool, project_id=project_id, since=since, until=until,
                limit=effective_anchor_result_limit, executor=executor,
            )
        return {query: merged[:effective_anchor_result_limit]}

    out: dict[str, list[EpisodeTurn]] = {}
    probe_candidate_sql_limit = (
        anchor_candidate_sql_limit
        if anchor_candidate_sql_limit is not None
        else candidate_sql_limit
    )
    for anchor in anchors:
        # Probe the original anchor first. Optionally add one conservative
        # event-focused local variant and union locally, preserving the
        # per-anchor budget and output order. The benchmark A/B harness can
        # disable this exact variant to provide a genuine baseline arm;
        # production callers retain the historical default (enabled).
        anchor_variants = [(anchor, "original")]
        if use_event_focused_anchor_representation:
            focused = event_focused_anchor_query(anchor)
            if focused is not None and focused.casefold() not in {
                variant.casefold() for variant, _ in anchor_variants
            }:
                anchor_variants.append((focused, "event_focused"))
        if use_anchor_local_variant:
            stripped = re.sub(
                r"^\s*(?:the|a|an)\s+(?:time|day|week|month|year)\s+",
                "", anchor, count=1, flags=re.IGNORECASE,
            ).strip(" ?.,")
            if len(stripped) >= 4 and stripped.casefold() != anchor.casefold():
                anchor_variants.append((stripped, "anchor_local"))

        if include_embedded_temporal_variant:
            # ``extract_anchors`` can capture an event together with an
            # embedded/trailing relative-time phrase.  Reuse the same
            # conservative event-focused transform as the no-anchor path,
            # but add only its first (event-focused) variant here.  Put it
            # first so the opt-in treatment is exercised even when the
            # original anchor fills the per-anchor output budget.
            embedded_variants = temporal_query_variants(
                anchor, include_embedded_temporal_variant=True,
            )
            if embedded_variants and (
                embedded_variants[0].casefold() != anchor.casefold()
                and embedded_variants[0].casefold() not in {
                    variant.casefold() for variant, _ in anchor_variants
                }
            ):
                anchor_variants.insert(0, (embedded_variants[0], "embedded_event_focused"))

        merged_anchor: list[EpisodeTurn] = []
        seen_anchor: set[str] = set()
        for anchor_variant, representation in anchor_variants:
            embedding = await embedder.embed(anchor_variant) if embedder else None
            vector_rows: list = []
            keyword_rows: list = []

            def _probe_diag(vec: list, kw: list) -> None:
                vector_rows.extend(vec)
                keyword_rows.extend(kw)

            def _sql_diag(event: dict[str, Any]) -> None:
                if sql_diag_callback is None:
                    return
                payload = dict(event)
                payload.update({
                    "anchor": anchor,
                    "query_variant": anchor_variant,
                    "representation": representation,
                })
                try:
                    sql_diag_callback(payload)
                except Exception:
                    logger.warning(
                        "temporal SQL diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )

            turns = await recall_turns(
                pool, anchor_variant,
                project_id=project_id, since=since, until=until,
                top_k=top_k_per_anchor, embedding=embedding,
                executor=executor, as_of=as_of,
                candidate_sql_limit=probe_candidate_sql_limit,
                fusion_candidate_limit=fusion_candidate_limit,
                result_limit=effective_anchor_result_limit,
                diag_callback=_probe_diag,
                sql_diag_callback=(
                    _sql_diag if sql_diag_callback is not None else None
                ),
                use_stored_search_tsv=use_stored_search_tsv,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            if diag_callback is not None:
                try:
                    diag_callback(anchor_variant, vector_rows, keyword_rows, turns)
                except Exception:
                    logger.warning(
                        "temporal anchor diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )
            if probe_diag_callback is not None:
                try:
                    vector_ids = [str(row["id"]) for row in vector_rows]
                    keyword_ids = [str(row["id"]) for row in keyword_rows]
                    candidate_ids = sorted(set(vector_ids) | set(keyword_ids))
                    probe_diag_callback({
                        "anchor": anchor,
                        "query": anchor_variant,
                        "representation": representation,
                        "vector_candidate_count": len(vector_rows),
                        "keyword_candidate_count": len(keyword_rows),
                        "candidate_count": len(candidate_ids),
                        "candidate_ids": candidate_ids,
                        "retrieved_turn_ids": [str(turn.id) for turn in turns],
                        "requested_output_cap": requested_anchor_result_limit,
                        "effective_output_cap": effective_anchor_result_limit,
                    })
                except Exception:
                    logger.warning(
                        "temporal probe diagnostic callback failed; preserving retrieval result",
                        exc_info=True,
                    )
            for turn in turns:
                if turn.id in seen_anchor:
                    continue
                seen_anchor.add(turn.id)
                merged_anchor.append(turn)
                if len(merged_anchor) >= effective_anchor_result_limit:
                    break
            # The benchmark event-focused treatment must execute every
            # declared representation probe so diagnostics measure the arm,
            # even when the original probe already fills the output cap. Keep
            # the historical short-circuit for the baseline and production
            # callers that do not opt into that treatment.
            if (
                len(merged_anchor) >= effective_anchor_result_limit
                and not use_event_focused_anchor_representation
            ):
                break
        out[anchor] = merged_anchor[:effective_anchor_result_limit]
    return out


# --- Both-tier RRF fusion ---


async def recall_both(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    top_k: int = 10,
    embedder: EmbeddingProvider | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    use_stored_search_tsv: bool = False,
) -> list[dict[str, Any]]:
    """Run belief + turn recall in parallel, fuse via RRF, return unified list.

    The driver: ``route_query_to_tier`` returns 'both' for explicit episodic
    asks ("do you remember", "what did I last say") where the user wants
    the canonical fact (belief tier) AND the dialogue evidence (turn tier)
    fused together. This function runs both halves in parallel and fuses
    them via Reciprocal Rank Fusion using the same K=60 constant as
    ``weft.store.search_hybrid`` and ``weft.episode_turns.recall_turns``
    (imported as ``_RRF_K``).

    Heterogeneous fusion: belief items and turn items are disjoint — no
    single item appears in both lists — so RRF degenerates to
    ``rrf_score = 1 / (K + rank_in_originating_list)``. Sort descending
    and take ``top_k``.

    Scoping is applied to BOTH halves identically so cross-project context
    cannot bleed in:

    * ``project_id`` → both halves
    * ``since`` / ``until`` → turn half only (memories don't carry an
      occurred_at; their lifecycle is created_at/updated_at, not the
      semantic "when did this happen" the temporal filter implies)
    * ``agent_id`` / ``user_id`` → belief half only (turn-tier rows don't
      carry agent_id; user_id on episode_turns is RLS-enforced via the
      session GUC at acquire-time, not a query param)

    Each half's ``top_k`` is set to ``2 * top_k`` so the fuser has
    material — RRF on two N-item lists with no overlap is just sort by
    rank, but oversampling lets us pick a richer mix when one side dwarfs
    the other.

    Returns a list of unified entries:
        ``{"kind": "memory" | "turn", "payload": {...}, "rank": int,
           "rrf_score": float}``

    where ``payload`` is the same dict the existing belief / turn paths
    return (``MemoryRecall.to_dict()`` for belief, ``EpisodeTurn.to_dict()``
    for turns), ``rank`` is the 1-indexed position within the originating
    list, and ``rrf_score`` is the per-item RRF contribution.
    """
    from weft.store import search_hybrid

    half_k = max(1, top_k * 2)

    # Both halves need an embedding. Compute once; share across the gather.
    embedding: list[float] | None = None
    if embedder is not None:
        embedding = await embedder.embed(query)

    async def _belief_half() -> list[MemoryRecall]:
        if embedding is None:
            # Belief search_hybrid requires an embedding. Without one we
            # can't fuse a vector signal — fall back to keyword-only by
            # returning an empty list so RRF degrades gracefully.
            from weft.store import search_by_keyword
            return await search_by_keyword(
                pool, query,
                limit=half_k,
                status=status,
                memory_type=memory_type,
                topic=topic,
                project_id=project_id,
                agent_id=agent_id,
                user_id=user_id,
                sources=sources,
                include_agent_provenance=include_agent_provenance,
            )
        return await search_hybrid(
            pool, query, embedding,
            limit=half_k,
            status=status,
            memory_type=memory_type,
            topic=topic,
            project_id=project_id,
            agent_id=agent_id,
            user_id=user_id,
            sources=sources,
            include_agent_provenance=include_agent_provenance,
        )

    # WEFT_HIERARCHICAL=1 swaps the flat recall_turns for the
    # hierarchical episode→turn descent. Same env-var convention as
    # the MCP `_weft_recall_turns` dispatch site so a single env-var
    # toggle covers both turn-only and both-tier paths.
    import os as _os
    _hierarchical = _os.environ.get("WEFT_HIERARCHICAL") == "1"

    async def _turn_half() -> list[EpisodeTurn]:
        if _hierarchical:
            from weft.episode_turns import recall_turns_hierarchical
            return await recall_turns_hierarchical(
                pool, query,
                project_id=project_id,
                since=since,
                until=until,
                top_k_episodes=10,
                top_k_turns=half_k,
                embedding=embedding,
                use_stored_search_tsv=use_stored_search_tsv,
            )
        return await recall_turns(
            pool, query,
            project_id=project_id,
            since=since,
            until=until,
            top_k=half_k,
            embedding=embedding,
            use_stored_search_tsv=use_stored_search_tsv,
        )

    # Parallel by default. When the caller has activated an RLS-scoped
    # connection via ``acquire()`` (contextvar), both halves race on the
    # same single connection — asyncpg refuses concurrent queries, so we
    # serialize. The check is cheap and keeps both call paths correct:
    #
    #   * MCP tool path: outer ``async with acquire(pool)`` sets contextvar
    #     → run sequentially on the bound conn (preserves user_id RLS).
    #   * Direct call path (tests, internal callers): no contextvar →
    #     each half pulls its own pool connection, gather races them.
    from weft.db.connection import _current_conn
    if _current_conn.get(None) is not None:
        belief_results = await _belief_half()
        turn_results = await _turn_half()
    else:
        belief_results, turn_results = await asyncio.gather(
            _belief_half(), _turn_half(),
        )

    # Disjoint RRF: each item gets one term, 1 / (K + rank_in_its_list).
    fused: list[dict[str, Any]] = []
    for i, recall in enumerate(belief_results):
        rank = i + 1
        fused.append({
            "kind": "memory",
            "payload": recall.to_dict(),
            "rank": rank,
            "rrf_score": 1.0 / (_RRF_K + rank),
        })
    for i, turn in enumerate(turn_results):
        rank = i + 1
        fused.append({
            "kind": "turn",
            "payload": turn.to_dict(),
            "rank": rank,
            "rrf_score": 1.0 / (_RRF_K + rank),
        })

    fused.sort(key=lambda e: e["rrf_score"], reverse=True)
    return fused[:top_k]
