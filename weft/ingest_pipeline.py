"""Smart ingestion pipeline — LLM-powered intent classification and routing.

Source-agnostic core: takes raw text + metadata, classifies intent via LLM,
extracts entities and dates, and routes to appropriate Weft subsystems.

Usage:
    result = await process(IngestItem(text="Bob is the CEO of Acme"), pool)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from anthropic import AsyncAnthropic

from weft.date_parser import parse_dates

logger = logging.getLogger(__name__)

# --- Constants ---

INTENT_TYPES = (
    "reminder",
    "person_fact",
    "company_fact",
    "follow_up",
    "decision",
    "action_item",
    "general_note",
)
IntentType = Literal[
    "reminder", "person_fact", "company_fact", "follow_up",
    "decision", "action_item", "general_note",
]

ENTITY_TYPES = ("person", "company", "tool", "concept", "location", "project")

_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
_MAX_TEXT_LENGTH = 12000
_MIN_TEXT_LENGTH = 5

_SYSTEM_PROMPT = """You are an intent classifier for a personal knowledge system. Analyze the input text and return a JSON array of intents found.

Each intent object must have:
- "type": one of ["reminder", "person_fact", "company_fact", "follow_up", "decision", "action_item", "general_note"]
- "content": a clean summary of the intent (not the raw text)
- "confidence": float 0.0-1.0
- "entities": array of {"name": "...", "entity_type": "person|company|tool|concept|location|project"}
- "dates": array of date strings found (e.g., "Saturday", "next Friday", "2025-03-15")

Rules:
- Return [] for empty/meaningless text
- One input can have multiple intents
- "reminder" = user wants to be reminded of something at a specific time
- "person_fact" = fact about a specific person (role, relationship, preference)
- "company_fact" = fact about a company/organization
- "follow_up" = someone needs to do something, or user needs to follow up with someone
- "decision" = a decision was made
- "action_item" = user needs to do something (no specific person mentioned)
- "general_note" = anything else worth remembering

Examples:

Input: "I need to remember to buy eggs Saturday"
Output: [{"type": "reminder", "content": "Buy eggs", "confidence": 0.9, "entities": [], "dates": ["Saturday"]}]

Input: "Bob is the CEO of TechCorp"
Output: [{"type": "person_fact", "content": "Bob is the CEO of TechCorp", "confidence": 0.95, "entities": [{"name": "Bob", "entity_type": "person"}, {"name": "TechCorp", "entity_type": "company"}], "dates": []}]

Input: "Bob at TechCorp wants to see the demo next Tuesday"
Output: [{"type": "follow_up", "content": "Bob at TechCorp wants to see the demo", "confidence": 0.9, "entities": [{"name": "Bob", "entity_type": "person"}, {"name": "TechCorp", "entity_type": "company"}], "dates": ["next Tuesday"]}]

Return ONLY valid JSON. No markdown, no explanation."""


# --- Data models ---


@dataclass
class EntityRef:
    """A reference to an entity extracted from text."""
    name: str
    entity_type: str = "concept"

    def __post_init__(self):
        if self.entity_type not in ENTITY_TYPES:
            self.entity_type = "concept"


@dataclass
class Intent:
    """A classified intent extracted from text."""
    type: IntentType
    content: str
    confidence: float = 0.7
    entities: list[EntityRef] = field(default_factory=list)
    dates: list[datetime] = field(default_factory=list)
    raw_text: str = ""

    def __post_init__(self):
        if self.type not in INTENT_TYPES:
            self.type = "general_note"
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class IngestItem:
    """A piece of text to be processed by the pipeline."""
    text: str
    source: str = "unknown"  # slack, obsidian, cli, etc.
    author: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IngestResult:
    """Result of processing an IngestItem through the pipeline."""
    intents: list[Intent] = field(default_factory=list)
    memories_created: int = 0
    entities_created: int = 0
    entities_linked: int = 0
    alerts_created: int = 0
    errors: list[str] = field(default_factory=list)


# --- LLM client ---

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    """Lazy singleton for the Anthropic client."""
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


def _normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def _should_skip(text: str, metadata: dict[str, Any]) -> bool:
    """Pre-filter: skip bot messages, very short text, emoji-only."""
    if metadata.get("is_bot") or metadata.get("bot_id"):
        return True
    clean = text.strip()
    if len(clean) < _MIN_TEXT_LENGTH:
        return True
    # Emoji-only (common Slack reactions like 👍, :thumbsup:, +1)
    if re.fullmatch(r"[\U0001F000-\U0001FFFF\U00002600-\U000027BF\s]+", clean):
        return True
    # Slack emoji shortcodes like :thumbsup: or +1
    if re.fullmatch(r"(:[a-z0-9_+-]+:\s*)+", clean) or clean in ("+1", "-1"):
        return True
    return False


# --- Classifier ---


async def classify(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    tz_name: str = "America/New_York",
) -> list[Intent]:
    """Classify text into structured intents via LLM.

    Returns a list of Intent objects. On any error, falls back to a single
    general_note intent. Returns [] for empty/bot/very-short text.
    """
    if metadata is None:
        metadata = {}

    if _should_skip(text, metadata):
        return []

    normalized = _normalize_text(text)
    if len(normalized) > _MAX_TEXT_LENGTH:
        normalized = normalized[:_MAX_TEXT_LENGTH]
        logger.warning("classify.truncated text to %d chars", _MAX_TEXT_LENGTH)

    try:
        client = _get_client()
        response = await client.messages.create(
            model=_CLASSIFIER_MODEL,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": normalized}],
        )
        raw_json = response.content[0].text.strip()
        logger.debug("classify.response: %s", raw_json)

        # Parse JSON — handle both array and single object
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed)}")

        intents: list[Intent] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            # Extract entities
            entities = []
            for e in item.get("entities", []):
                if isinstance(e, dict) and e.get("name"):
                    entities.append(EntityRef(
                        name=e["name"],
                        entity_type=e.get("entity_type", "concept"),
                    ))

            # Extract and resolve dates
            date_strings = item.get("dates", [])
            resolved_dates: list[datetime] = []
            for ds in date_strings:
                if isinstance(ds, str) and ds.strip():
                    parsed_dates = parse_dates(ds, tz_name=tz_name)
                    resolved_dates.extend(parsed_dates)

            intent_type = item.get("type", "general_note")
            if intent_type not in INTENT_TYPES:
                logger.warning("classify.unknown_type: %s", intent_type)
                intent_type = "general_note"

            intents.append(Intent(
                type=intent_type,
                content=item.get("content", normalized),
                confidence=float(item.get("confidence", 0.7)),
                entities=entities,
                dates=resolved_dates,
                raw_text=normalized,
            ))

        return intents

    except json.JSONDecodeError as e:
        logger.warning("classify.json_error: %s", e)
    except Exception:
        logger.exception("classify.error")

    # Fallback: return as general_note
    return [Intent(
        type="general_note",
        content=normalized,
        confidence=0.0,
        raw_text=normalized,
    )]


__all__ = ["EntityRef", "Intent", "IngestItem", "IngestResult", "classify"]
