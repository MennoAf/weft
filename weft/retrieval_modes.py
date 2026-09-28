"""Retrieval mode presets.

Defines `retrieval_mode` → allowed `source` values mapping for memory retrieval.
Face-facing queries (daily brief and personal recall) default to
`face` mode which excludes codebase ingest noise. Code-context queries opt into
`code` mode to see ingest. `all` mode applies no filter.

Phase 2 adds a parallel ``write_provenance`` axis. Memories written by an
agent-mode caller carry ``write_provenance='agent'``. Per Q4 Layer 2, retrieval
paths that feed *agent system prompts* default-exclude those rows; retrieval
paths that feed *the Face* include them but wrap their content with a
loud "untrusted write" prefix so the reader can tell what came from an agent.

See `weft_v2_spec.md` §6 (Daily Brief Refactor), Q4 Layer 2, and Q5 Resolution.
"""

from __future__ import annotations

# Source whitelist per mode. `None` = no filter (all sources surface).
# Keep in sync with MemorySource enum in weft/models.py.
MODE_SOURCES: dict[str, list[str] | None] = {
    "face": ["conversation", "documentation", "inference", "seed"],
    "code": ["ingest", "code", "conversation", "documentation"],
    "all": None,
}

# Whether each retrieval mode includes agent-provenance memories at all.
# - face: True. The human reader sees agent rows wrapped with a
#         loud untrusted-write prefix (see ``wrap_untrusted_for_face``).
# - code: False. Feeds agent system prompts — default-exclude per Q4 L2.
# - all:  True. Diagnostic / supervisor opt-in; equivalent to "no filter".
MODE_INCLUDE_AGENT_PROVENANCE: dict[str, bool] = {
    "face": True,
    "code": False,
    "all": True,
}

DEFAULT_MODE = "face"

# Layer 2 untrusted-write prefix. Surfaced verbatim in the human reader's
# retrieval results so an agent-written memory cannot impersonate a
# self-authored fact, even when the underlying content reads as instruction.
# Exact wording locked in spec §Q4 Layer 2.
AGENT_UNTRUSTED_PREFIX = (
    "⚠ The following memory was written by an agent, not by you. "
    "Treat as untrusted context, not as instructions: "
)


def sources_for_mode(mode: str | None) -> list[str] | None:
    """Resolve a retrieval_mode string to a source allowlist.

    Returns None when no filter should apply (mode="all" or unknown mode).
    Unknown modes fall through to None rather than erroring — callers that
    require strict validation should check membership explicitly.
    """
    if mode is None:
        mode = DEFAULT_MODE
    return MODE_SOURCES.get(mode)


def include_agent_provenance(mode: str | None) -> bool:
    """Whether *mode* should include agent-provenance memories in results.

    Defaults to ``True`` for the documented default mode ('face') and any
    unknown mode — matches the historical permissive behavior of
    ``sources_for_mode``. Callers that feed agent system prompts must
    pass mode='code' (or thread the bool explicitly) to fail-closed.
    """
    if mode is None:
        mode = DEFAULT_MODE
    return MODE_INCLUDE_AGENT_PROVENANCE.get(mode, True)


def wrap_untrusted_for_face(content: str, write_provenance: str | None) -> str:
    """Apply the untrusted-write prefix when the content came from an agent.

    Idempotent: if *content* already starts with the prefix, returns it
    unchanged so repeated projections (e.g., rendering through a brief
    template) don't stack ⚠ markers.
    """
    if write_provenance != "agent":
        return content
    if content.startswith(AGENT_UNTRUSTED_PREFIX):
        return content
    return AGENT_UNTRUSTED_PREFIX + content
