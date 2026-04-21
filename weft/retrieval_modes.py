"""Retrieval mode presets.

Defines `retrieval_mode` → allowed `source` values mapping for memory retrieval.
Face-facing queries (daily brief, recall when Jason is the reader) default to
`face` mode which excludes codebase ingest noise. Code-context queries opt into
`code` mode to see ingest. `all` mode applies no filter.

See `weft_v2_spec.md` §6 (Daily Brief Refactor) and Q5 resolution for rationale.
"""

from __future__ import annotations

# Source whitelist per mode. `None` = no filter (all sources surface).
# Keep in sync with MemorySource enum in weft/models.py.
MODE_SOURCES: dict[str, list[str] | None] = {
    "face": ["conversation", "documentation", "inference", "seed"],
    "code": ["ingest", "code", "conversation", "documentation"],
    "all": None,
}

DEFAULT_MODE = "face"


def sources_for_mode(mode: str | None) -> list[str] | None:
    """Resolve a retrieval_mode string to a source allowlist.

    Returns None when no filter should apply (mode="all" or unknown mode).
    Unknown modes fall through to None rather than erroring — callers that
    require strict validation should check membership explicitly.
    """
    if mode is None:
        mode = DEFAULT_MODE
    return MODE_SOURCES.get(mode)
