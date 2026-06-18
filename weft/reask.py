"""Re-ask detection: identify when an agent re-asks a similar query (retrieval miss).

Given a session's recent recall query rows, detects near-duplicate query recurrence
within a bounded time/turn window using text similarity. Pure module — no database
access, no network calls, fully testable with fixtures.

Design:
* Text similarity via difflib.SequenceMatcher ratio (simple, dependency-light)
* Within-window detection: recent queries grouped by (time_delta, turn_delta)
* Returns (original_query, reask_query) pairs where the reask is a near-duplicate
  of an earlier query within the window.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class QueryRow:
    """A single weft_recall or weft_search_all query row.

    Fields from weft_recall_queries table. In practice, callers will pass
    dicts or ORM objects; this class is the canonical shape for type hints.
    """
    query_id: str
    query_text: str
    created_at: datetime
    tool_name: str
    project_id: str | None = None
    tier: str | None = None
    mode: str | None = None
    retrieval_mode: str | None = None
    result_count: int | None = None
    turn_index: int | None = None  # Optional: agent turn number if available

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueryRow:
        """Construct from a dict (e.g., asyncpg Record, ORM row)."""
        kwargs = {}
        for field in cls.__dataclass_fields__:
            if field in d:
                kwargs[field] = d[field]
        return cls(**kwargs)


def _text_similarity(text1: str, text2: str) -> float:
    """Compute text similarity ratio using difflib.

    Args:
        text1: First query text.
        text2: Second query text.

    Returns:
        Similarity ratio in [0.0, 1.0]. 1.0 = identical, 0.0 = completely different.
    """
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def detect_reasked_queries(
    query_rows: list[QueryRow | dict[str, Any]],
    *,
    similarity_threshold: float = 0.75,
    time_window_minutes: int = 30,
    max_turn_delta: int | None = None,
) -> list[tuple[QueryRow, QueryRow]]:
    """Detect near-duplicate queries (re-asks) within a session window.

    Given a list of recent recall query rows ordered by time (oldest first),
    identifies pairs where a later query is a near-duplicate of an earlier
    query within the time/turn window. Returns the (original, reask) pairs.

    Args:
        query_rows: List of query rows (QueryRow objects or dicts). Should be
                   sorted by created_at ascending (oldest first).
        similarity_threshold: Text similarity ratio (0-1) above which queries
                             are considered duplicates. Default 0.75.
        time_window_minutes: Only flag re-asks within this many minutes of
                            the original query. Default 30.
        max_turn_delta: Optional maximum turn number difference. If set,
                       only flag re-asks within this many turns of the original.
                       Requires turn_index field in rows.

    Returns:
        List of (original_query, reask_query) tuples. Empty if no re-asks detected.
    """
    # Normalize input: convert dicts to QueryRow if needed.
    normalized_rows: list[QueryRow] = []
    for row in query_rows:
        if isinstance(row, dict):
            normalized_rows.append(QueryRow.from_dict(row))
        else:
            normalized_rows.append(row)

    if not normalized_rows:
        return []

    # Ensure rows are sorted by created_at (caller should do this, but be defensive).
    normalized_rows = sorted(normalized_rows, key=lambda r: r.created_at)

    reasked_pairs: list[tuple[QueryRow, QueryRow]] = []

    # Compare each row to all prior rows in the window.
    for i, current in enumerate(normalized_rows):
        for j in range(i):
            prior = normalized_rows[j]

            # Time window check.
            if current.created_at - prior.created_at > timedelta(minutes=time_window_minutes):
                # This prior is too old; all earlier ones are older, so break.
                break

            # Turn window check (if max_turn_delta is set).
            if max_turn_delta is not None:
                if (
                    prior.turn_index is None
                    or current.turn_index is None
                    or abs(current.turn_index - prior.turn_index) > max_turn_delta
                ):
                    continue

            # Text similarity check.
            similarity = _text_similarity(prior.query_text, current.query_text)
            if similarity >= similarity_threshold:
                reasked_pairs.append((prior, current))

    return reasked_pairs


def compute_reask_rate(
    query_rows: list[QueryRow | dict[str, Any]],
    *,
    similarity_threshold: float = 0.75,
    time_window_minutes: int = 30,
    max_turn_delta: int | None = None,
    group_by_session: bool = False,
) -> float:
    """Compute the same-session re-ask rate: the fraction of queries containing a re-ask.

    Pure PROOF metric for the re-ask correction loop. Over time, as the loop boosts
    the right memories, this rate should decline. A flat or rising rate while query
    volume grows is the dead tell.

    The rate is computed as the fraction of queries that are part of a re-ask pair
    (``group_by_session=False``, the only supported mode).

    Args:
        query_rows: List of query rows (QueryRow objects or dicts), typically from
                   a single session or a bounded time window.
        similarity_threshold: Text similarity ratio (0-1) for detection. Default 0.75.
        time_window_minutes: Time window for detection. Default 30.
        max_turn_delta: Optional turn window for detection.
        group_by_session: Must be False (the default). Session-based rate is not yet
                         supported — ``weft_recall_queries`` has no ``session_id`` column.
                         Passing True raises NotImplementedError.

    Returns:
        Re-ask rate in [0.0, 1.0]. 0.0 = no re-asks, 1.0 = all queries are re-asks.

    Examples:
        >>> rows = [Q1, Q2_reask, Q3, Q4]  # Q2 duplicates Q1, rest distinct
        >>> compute_reask_rate(rows)  # 0.5 (2 of 4 queries in re-ask pairs)
        0.5
        >>> rows = [Q1, Q2_reask]
        >>> compute_reask_rate(rows)  # 1.0 (both queries in re-ask pair)
        1.0
    """
    if group_by_session:
        raise NotImplementedError(
            "session-based re-ask rate needs a session_id on weft_recall_queries"
            " (not in schema yet)"
        )

    if not query_rows:
        return 0.0

    # Detect re-ask pairs using the L1 detection logic.
    reasked_pairs = detect_reasked_queries(
        query_rows,
        similarity_threshold=similarity_threshold,
        time_window_minutes=time_window_minutes,
        max_turn_delta=max_turn_delta,
    )

    if not reasked_pairs:
        return 0.0

    # Query-based rate: count unique queries involved in re-ask pairs.
    queries_in_reasked_pairs = set()
    for original, reask in reasked_pairs:
        queries_in_reasked_pairs.add(original.query_id)
        queries_in_reasked_pairs.add(reask.query_id)

    # Normalize input to QueryRow for consistent query_id access.
    normalized_rows: list[QueryRow] = []
    for row in query_rows:
        if isinstance(row, dict):
            normalized_rows.append(QueryRow.from_dict(row))
        else:
            normalized_rows.append(row)

    total_queries = len(normalized_rows)
    queries_with_reasks = len(queries_in_reasked_pairs)

    return queries_with_reasks / total_queries
