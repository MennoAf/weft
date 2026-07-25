"""Memory candidate extraction — heuristic identification of memorable content."""

from __future__ import annotations

import re
from typing import Any


# Pattern definitions: (regex, memory_type, base_confidence)
_PREFERENCE_PATTERNS = [
    (re.compile(r"^I prefer\s+(?!not\b)(.+?)(?:\s+but\s+I\s+prefer\s+.+)?$", re.IGNORECASE), "preference", 0.85, "positive", "soft"),
    (re.compile(r"^I always\s+(.+)$", re.IGNORECASE), "preference", 0.8, "constraint", "hard"),
    (re.compile(r"^I never\s+(?:like|want|use|prefer)\s+(.+)$", re.IGNORECASE), "preference", 0.8, "avoidance", "hard"),
    (re.compile(r"^I (?:avoid|don't|do not) (?:like|want|use|prefer)\s+(.+)$", re.IGNORECASE), "preference", 0.8, "avoidance", "hard"),
    (re.compile(r"^I (?:like|want) to\s+(?!fix|debug|implement|build|write|run|check|use\b)(.+)$", re.IGNORECASE), "preference", 0.7, "positive", "soft"),
]

_FACT_PATTERNS = [
    (re.compile(r"(?:(?:project|app|system|service|codebase) )?uses?\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"(?:runs?|running) on\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"version\s+(\d[\d.]*\S*)", re.IGNORECASE), "fact", 0.7),
    (re.compile(r"(?:deployed|hosted|stored) (?:on|in|at)\s+(.+)", re.IGNORECASE), "fact", 0.7),
    (re.compile(r"(?:database|db) is\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"(?:written|built|implemented) (?:in|with)\s+(.+)", re.IGNORECASE), "fact", 0.75),
]

_PATTERN_PATTERNS = [
    (re.compile(r"(?:the |our )?(?:pattern|convention|rule|practice) (?:is|for)\s+(.+)", re.IGNORECASE), "pattern", 0.7),
    (re.compile(r"(?:we |I )?(?:typically|usually|generally)\s+(.+)", re.IGNORECASE), "pattern", 0.65),
    (re.compile(r"(?:the |our )?(?:approach|workflow|process) (?:is|for)\s+(.+)", re.IGNORECASE), "pattern", 0.7),
]

_ARCHITECTURE_PATTERNS = [
    (re.compile(r"(?:the )?architecture\s+(?:is|uses|follows)\s+(.+)", re.IGNORECASE), "architecture", 0.8),
    (re.compile(r"(?:the )?(?:design|stack|infrastructure)\s+(?:is|uses|includes)\s+(.+)", re.IGNORECASE), "architecture", 0.8),
    (re.compile(r"(?:three|two|multi)[- ](?:tier|layer|stage)\s+(.+)", re.IGNORECASE), "architecture", 0.75),
]

# Patterns for lessons learned during task execution
_SOLUTION_PATTERNS = [
    (re.compile(r"(?:the )?(?:trick|fix|solution|workaround) (?:is|was)\s+(.+)", re.IGNORECASE), "solution", 0.8),
    (re.compile(r"(?:had to|needed to|must)\s+(.+?)(?:\s+(?:because|since|due to|otherwise)\s+.+)?$", re.IGNORECASE), "solution", 0.75),
    (re.compile(r"(?:watch out|be careful|careful with|gotcha|caveat)[:\s]+(.+)", re.IGNORECASE), "solution", 0.8),
    (re.compile(r"(?:turns? out|discovered|learned|realized)\s+(?:that\s+)?(.+)", re.IGNORECASE), "solution", 0.75),
    (re.compile(r"(?:the )?(?:issue|problem|bug) (?:is|was)\s+(.+)", re.IGNORECASE), "solution", 0.7),
    (re.compile(r"(?:don't|do not|avoid) (?:forget to|skip)\s+(.+)", re.IGNORECASE), "solution", 0.75),
    (re.compile(r"(?:important|critical|key)[:\s]+(.+)", re.IGNORECASE), "fact", 0.7),
]

_BEHAVIOR_PATTERNS = [
    # "when X, do/use/always Y"
    (re.compile(
        r"when(?:ever)?\s+(.+?),?\s+(?:always |you should |we should |I |we )?(?:do|use|run|call|check|make sure|ensure|start with|prefer)\s+(.+)",
        re.IGNORECASE,
    ), 0.75),
    # "if X, then Y" / "if X, Y"
    (re.compile(
        r"if\s+(.+?),\s*(?:then\s+)?(?:always |you should |we should )?(.+)",
        re.IGNORECASE,
    ), 0.7),
    # "before/after X, always Y" — comma required to anchor the trigger.
    # Without it, non-greedy (.+?) collapses to a single determiner ("a",
    # "every") and the action grabs the rest of the sentence — see
    # weft-715b809b / weft-c335c36f for the production failure mode.
    (re.compile(
        r"(before|after)\s+(.+?),\s+(?:always |you should |we should |I |we )?(.+)",
        re.IGNORECASE,
    ), 0.75),
    # "always X when Y"
    (re.compile(
        r"always\s+(.+?)\s+when\s+(.+)",
        re.IGNORECASE,
    ), 0.8),
    # "never X without Y" / "don't X without Y"
    (re.compile(
        r"(?:never|don't|do not)\s+(.+?)\s+without\s+(.+)",
        re.IGNORECASE,
    ), 0.8),
    # "make sure to X before/after Y"
    (re.compile(
        r"make sure (?:to )?(.+?)\s+(before|after)\s+(.+)",
        re.IGNORECASE,
    ), 0.75),
]

ALL_PATTERNS = (
    _PREFERENCE_PATTERNS + _SOLUTION_PATTERNS + _FACT_PATTERNS
    + _PATTERN_PATTERNS + _ARCHITECTURE_PATTERNS
)

# Trailing tokens that signal the trigger phrase got truncated mid-noun-phrase.
# A trigger ending in a determiner ("after a", "after every") almost always
# means the regex grabbed the head of a noun phrase but missed the noun.
_TRIGGER_TRAIL_DETERMINERS = frozenset({
    "a", "an", "the", "every", "each", "some", "any", "all",
    "this", "that", "these", "those", "my", "our", "your", "their",
})

# Minimum trigger/action lengths. Kept low so legitimate compact triggers
# like "writing tests" or "deploy" still pass; the load-bearing guard is
# the determiner-tail check below, which catches the over-match shapes.
_MIN_TRIGGER_LEN = 5
_MIN_ACTION_LEN = 5


# Minimum content length for a stored memory. Below this, the content is
# almost always a chunked-doc heading, a stray prefix, or a ghost write.
_MIN_MEMORY_CONTENT_LEN = 15


def validate_memory_content(content: str) -> tuple[bool, str | None]:
    """Gate for weft_remember. Returns (ok, reason).

    Catches the production fragment shapes seen in audit:
    - bare markdown headings stored as standalone memories
      (e.g. "### .gitignore pattern for env templates" — no body)
    - trailing-colon content where the body got truncated upstream
      (e.g. "**Bool-vs-int validation pattern is now canonical** (...):")
    - sub-15-char strings that can't carry useful context

    Returns (False, reason_code) on rejection so callers can surface a
    specific error to the writing agent. Reason codes are stable:
    content_too_short, heading_only, trailing_colon.
    """
    stripped = content.strip() if content else ""
    if len(stripped) < _MIN_MEMORY_CONTENT_LEN:
        return False, "content_too_short"
    non_empty_lines = [ln for ln in stripped.split("\n") if ln.strip()]
    if len(non_empty_lines) == 1 and re.match(r"^#{1,6}\s+\S", non_empty_lines[0]):
        return False, "heading_only"
    if stripped.endswith(":"):
        return False, "trailing_colon"
    return True, None


def _is_valid_behavior_pair(trigger: str, action: str) -> bool:
    """Reject behavior candidates that look like regex over-match artifacts.

    Production failure mode: the "before/after" pattern produced triggers
    like "after a" with the rest of the sentence as the action. These slip
    past a naive length check because the action half is long. Validate the
    trigger is a plausible noun-phrase boundary instead.
    """
    if len(trigger) < _MIN_TRIGGER_LEN or len(action) < _MIN_ACTION_LEN:
        return False
    trigger_words = trigger.split()
    if not trigger_words:
        return False
    if trigger_words[-1].lower().strip(".,;:") in _TRIGGER_TRAIL_DETERMINERS:
        return False
    return True


def _extract_topics(text: str) -> list[str]:
    """Extract likely topic keywords from text."""
    # Common tech terms that make good topics
    tech_terms = [
        "python", "javascript", "typescript", "rust", "go", "java",
        "postgres", "postgresql", "redis", "mongodb", "mysql", "sqlite",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "react", "vue", "angular", "fastapi", "django", "flask",
        "git", "ci", "cd", "testing", "deployment",
        "api", "rest", "graphql", "grpc",
        "cache", "queue", "logging", "monitoring",
        "frontend", "backend", "database", "infrastructure",
    ]
    text_lower = text.lower()
    found = [t for t in tech_terms if t in text_lower]
    return found[:3]  # Cap at 3 topics


def extract_behaviors(
    text: str,
    *,
    min_confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """Extract behavior candidates (trigger → action rules) from text.

    Returns a list of dicts with: trigger_pattern, action, confidence, source_line.
    These are NOT stored — caller decides what to do with them.
    """
    if not text or not text.strip():
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 15:
            continue

        clean = re.sub(r"^[-*]\s+", "", line)
        clean = re.sub(r"^\d+\.\s+", "", clean)

        for pattern, confidence in _BEHAVIOR_PATTERNS:
            match = pattern.search(clean)
            if match and confidence >= min_confidence:
                groups = match.groups()
                if len(groups) == 2:
                    trigger, action = groups[0].strip(), groups[1].strip()
                elif len(groups) == 3:
                    # "before/after X, Y" or "make sure X before/after Y"
                    trigger = f"{groups[0].strip()} {groups[1].strip()}"
                    action = groups[2].strip()
                else:
                    continue

                if not _is_valid_behavior_pair(trigger, action):
                    continue

                key = (trigger.lower(), action.lower())
                if key in seen:
                    continue
                seen.add(key)

                candidates.append({
                    "trigger_pattern": trigger,
                    "action": action,
                    "confidence": round(confidence, 2),
                    "source_line": line,
                })
                break  # One match per line

    return candidates


def extract_candidates(
    text: str,
    *,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Extract memory candidates from a text block.

    Scans each sentence/line for patterns that suggest memorable content.
    Returns a list of candidate dicts (NOT stored — for agent review).

    Each candidate has: content, type, confidence, topic, source_line
    """
    if not text or not text.strip():
        return []

    candidates: list[dict[str, Any]] = []
    seen_content: set[str] = set()

    # Split into lines/sentences
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Strip markdown bullets
        clean = re.sub(r"^[-*]\s+", "", line)
        clean = re.sub(r"^\d+\.\s+", "", clean)

        for pattern_entry in ALL_PATTERNS:
            pattern, mem_type, confidence = pattern_entry[:3]
            if mem_type == "preference" and not re.match(r"^I\b", clean, re.IGNORECASE):
                continue
            match = pattern.search(clean)
            if match:
                if mem_type == "preference" and re.search(
                    r"\b[A-Z][a-z]+\s+(?:prefers?|likes?|wants?|avoids?)\b",
                    match.group(1),
                ):
                    continue
                # Use the full clean line as content (more context than just the capture group)
                content = clean.strip()

                # Skip duplicates
                content_key = content.lower()
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

                if confidence >= min_confidence:
                    candidate = {
                        "content": content,
                        "type": mem_type,
                        "confidence": round(confidence, 2),
                        "topic": _extract_topics(content),
                        "source_line": line,
                    }
                    if mem_type == "preference" and len(pattern_entry) >= 5:
                        candidate["preference_metadata"] = {
                            "polarity": pattern_entry[3],
                            "strength": pattern_entry[4],
                            "value": match.group(1).strip(),
                        }
                    candidates.append(candidate)
                break  # One match per line

    return candidates
