"""Brief delivery state — tracks last delivery date to prevent duplicates.

Simple file-based state stored at ~/.weft/brief_state.json.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_PATH = Path.home() / ".weft" / "brief_state.json"


def get_last_brief_date() -> date | None:
    """Return the last date a brief was delivered, or None."""
    try:
        if _STATE_PATH.exists():
            data = json.loads(_STATE_PATH.read_text())
            return date.fromisoformat(data["last_date"])
    except Exception:
        logger.warning("brief_state.read_error")
    return None


def set_last_brief_date(d: date) -> None:
    """Record that a brief was delivered on this date."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"last_date": d.isoformat()}))
    except Exception:
        logger.exception("brief_state.write_error")
