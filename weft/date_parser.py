"""Natural language date parsing — converts relative/absolute date strings to datetimes.

Uses python-dateutil for parsing. All returned datetimes are timezone-aware.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# Patterns for relative date expressions
_RELATIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "next_weekday"),
    (re.compile(r"\b(this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "weekday"),
    (re.compile(r"\bin\s+(\d+)\s+(day|week|month)s?\b", re.I), "in_delta"),
    (re.compile(r"\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "by_weekday"),
    (re.compile(r"\bnext\s+week\b", re.I), "next_week"),
    (re.compile(r"\btoday\b", re.I), "today"),
]

_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _next_weekday(reference: datetime, weekday: int) -> datetime:
    """Find the next occurrence of a weekday (0=Monday) after reference."""
    days_ahead = weekday - reference.weekday()
    if days_ahead <= 0:  # already passed this week → next week
        days_ahead += 7
    return reference + timedelta(days=days_ahead)


def parse_dates(
    text: str,
    *,
    reference_time: datetime | None = None,
    tz_name: str = "UTC",
) -> list[datetime]:
    """Extract dates from text and return as timezone-aware datetimes.

    Args:
        text: natural language text potentially containing date references.
        reference_time: anchor for relative dates. Defaults to now.
        tz_name: IANA timezone for the returned datetimes.

    Returns:
        List of timezone-aware datetimes found in the text. May be empty.
    """
    if not text or not text.strip():
        return []

    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("date_parser.invalid_tz: %s, falling back to UTC", tz_name)
        tz = timezone.utc

    if reference_time is None:
        reference_time = datetime.now(tz)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=tz)

    ref = reference_time.astimezone(tz)
    results: list[datetime] = []

    # Try relative patterns first — stop after first match to avoid duplicates
    for pattern, kind in _RELATIVE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        if kind == "today":
            results.append(ref.replace(hour=9, minute=0, second=0, microsecond=0))
        elif kind == "tomorrow":
            dt = ref + timedelta(days=1)
            results.append(dt.replace(hour=9, minute=0, second=0, microsecond=0))
        elif kind == "next_weekday" or kind == "by_weekday":
            weekday_name = match.group(1) if kind == "by_weekday" else match.group(1)
            wd = _WEEKDAY_MAP.get(weekday_name.lower())
            if wd is not None:
                dt = _next_weekday(ref, wd)
                results.append(dt.replace(hour=9, minute=0, second=0, microsecond=0))
        elif kind == "weekday":
            # "Saturday" without "next" — treat as next occurrence
            weekday_name = match.group(2)
            wd = _WEEKDAY_MAP.get(weekday_name.lower())
            if wd is not None:
                dt = _next_weekday(ref, wd)
                results.append(dt.replace(hour=9, minute=0, second=0, microsecond=0))
        elif kind == "in_delta":
            amount = int(match.group(1))
            unit = match.group(2).lower()
            if unit == "day":
                dt = ref + timedelta(days=amount)
            elif unit == "week":
                dt = ref + timedelta(weeks=amount)
            elif unit == "month":
                dt = ref + relativedelta(months=amount)
            else:
                continue
            results.append(dt.replace(hour=9, minute=0, second=0, microsecond=0))
        elif kind == "next_week":
            dt = ref + timedelta(weeks=1)
            results.append(dt.replace(hour=9, minute=0, second=0, microsecond=0))

        # Return after first successful match to avoid duplicate detections
        if results:
            return results

    # Fall back to dateutil for absolute dates
    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=ref)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        # Only return if it looks like a real date was found (not just the default)
        # Check if any date-like tokens exist in the text
        if re.search(r"\b\d{1,4}[-/]\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d", text, re.I):
            results.append(parsed)
    except (ValueError, OverflowError):
        pass

    return results
