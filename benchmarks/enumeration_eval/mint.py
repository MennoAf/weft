"""Eval case minting for the CL1 compounding loop (loom-add4d5c8).

A canary miss or an is_reask_miss event auto-mints a known-answer eval case so
the enumeration harness can exercise it on subsequent runs.  The store is a
simple JSONL file — one case per line — kept separate from the hand-seeded
Jim Boblaw fixtures.

Case format (each line is a JSON object):
    {
        "query": "<query text, truncated to 512 chars>",
        "satisfying_memory_id": "<memory id>",
        "source": "canary" | "reask",
        "minted_at": "<ISO 8601 UTC timestamp>",
        "probe_id": "<probe_id, optional>"
    }

Dedup key: (query, satisfying_memory_id) — idempotent on repeated audits of
the same miss.  Appending to the JSONL is safe for concurrent single-writer
use (one audit process at a time).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default JSONL store — lives next to the benchmark package, separate from fixtures.py
DEFAULT_MINTED_CASES_PATH: Path = Path(__file__).parent / "minted_cases.jsonl"

# Must match PROBE_TEXT_MAX_CHARS in weft/canary.py (512) so the dedup key
# is consistent with what the canary uses when truncating probe_text.
_QUERY_MAX_CHARS = 512


def load_minted_cases(path: "Path | str | None" = None) -> list[dict]:
    """Load all minted eval cases from the JSONL store.

    Args:
        path: Path to the JSONL file.  Defaults to DEFAULT_MINTED_CASES_PATH.

    Returns:
        List of case dicts.  Empty list if the file does not exist or is empty.
    """
    p = Path(path) if path is not None else DEFAULT_MINTED_CASES_PATH
    if not p.exists():
        return []
    cases: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("load_minted_cases: skipping malformed line: %s", exc)
    return cases


def mint_eval_case(
    query: str,
    satisfying_memory_id: str,
    source: str,
    *,
    probe_id: "str | None" = None,
    path: "Path | str | None" = None,
) -> bool:
    """Mint a known-answer eval case into the JSONL store.

    Idempotent: if a case with the same (query[:512], satisfying_memory_id)
    already exists, the file is not modified and False is returned.

    Args:
        query: The query text (probe_text or original missed query).
            Truncated to 512 chars to match the canary's PROBE_TEXT_MAX_CHARS.
        satisfying_memory_id: The memory ID that should surface for this query.
        source: ``'canary'`` (from a recall canary miss) or ``'reask'`` (from an
            is_reask_miss event).
        probe_id: The recall_canary probe_id, if available (for traceability).
        path: Path to the JSONL store.  Defaults to DEFAULT_MINTED_CASES_PATH.

    Returns:
        True if a new case was appended; False if it was a duplicate (no-op).
    """
    p = Path(path) if path is not None else DEFAULT_MINTED_CASES_PATH
    query_key = query[:_QUERY_MAX_CHARS]

    # Dedup check — load existing cases and compare on the canonical key.
    existing = load_minted_cases(p)
    for case in existing:
        if (
            case.get("query") == query_key
            and case.get("satisfying_memory_id") == satisfying_memory_id
        ):
            logger.debug(
                "mint_eval_case: duplicate, skipping memory_id=%s", satisfying_memory_id
            )
            return False

    record: dict = {
        "query": query_key,
        "satisfying_memory_id": satisfying_memory_id,
        "source": source,
        "minted_at": datetime.now(timezone.utc).isoformat(),
    }
    if probe_id is not None:
        record["probe_id"] = probe_id

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.debug(
        "mint_eval_case: minted new case memory_id=%s source=%s path=%s",
        satisfying_memory_id,
        source,
        p,
    )
    return True
