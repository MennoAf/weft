"""Shared check-in parsing logic.

Lifted from weft.mcp.slack_commands so both the Slack handler and the
Discord slash command can reuse the same parser + range-validation helpers.
"""

from __future__ import annotations

import re


def parse_checkin_text(text: str) -> dict:
    """Parse slash command text into check-in fields.

    Supports flexible formats:
      mood 3 sleep 7 energy 4 feeling good today
      mood:3 sleep:7.5 energy:4
      m3 s7 e4 feeling tired
      3 7 4  (positional: mood sleep energy)

    Returns dict with keys: mood, sleep_hours, energy, notes.
    """
    text = text.strip()
    if not text:
        return {}

    result: dict = {}

    # Named patterns: mood/m 3, sleep/s 7.5, energy/e 4
    mood_match = re.search(r'\b(?:mood|m)[:\s]?\s*(\d)', text, re.IGNORECASE)
    sleep_match = re.search(r'\b(?:sleep|sleep_hours|s)[:\s]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
    energy_match = re.search(r'\b(?:energy|e)[:\s]?\s*(\d)', text, re.IGNORECASE)

    if mood_match:
        result["mood"] = int(mood_match.group(1))
    if sleep_match:
        result["sleep_hours"] = float(sleep_match.group(1))
    if energy_match:
        result["energy"] = int(energy_match.group(1))

    # If no named patterns found, try positional: "3 7 4"
    if not result:
        nums = re.findall(r'\b(\d+\.?\d*)\b', text)
        if len(nums) >= 1:
            result["mood"] = int(float(nums[0]))
        if len(nums) >= 2:
            result["sleep_hours"] = float(nums[1])
        if len(nums) >= 3:
            result["energy"] = int(float(nums[2]))

    # Extract notes: everything that isn't a recognized field
    notes_text = text
    for pattern in [
        r'\b(?:mood|m)[:\s]?\s*\d',
        r'\b(?:sleep|sleep_hours|s)[:\s]?\s*\d+\.?\d*',
        r'\b(?:energy|e)[:\s]?\s*\d',
    ]:
        notes_text = re.sub(pattern, '', notes_text, flags=re.IGNORECASE)
    notes_text = re.sub(r'\s+', ' ', notes_text).strip()
    if notes_text:
        result["notes"] = notes_text

    return result


def validate_checkin_ranges(parsed: dict) -> str | None:
    """Validate mood, energy, and sleep_hours ranges.

    Returns an error message string if any value is out of range, or None
    if all values are within acceptable bounds.
    """
    mood = parsed.get("mood")
    energy = parsed.get("energy")
    sleep_hours = parsed.get("sleep_hours")

    if mood is not None and not 1 <= mood <= 5:
        return f"Mood must be 1-5, got {mood}."
    if energy is not None and not 1 <= energy <= 5:
        return f"Energy must be 1-5, got {energy}."
    if sleep_hours is not None and not 0 <= sleep_hours <= 24:
        return f"Sleep hours must be 0-24, got {sleep_hours}."

    return None
