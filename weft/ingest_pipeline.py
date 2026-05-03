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
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from anthropic import AsyncAnthropic

from weft.date_parser import parse_dates
from weft.db.connection import acquire

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

        # Strip markdown code fences. Despite "Return ONLY valid JSON. No
        # markdown" in the system prompt, Claude wraps responses in
        # ```json ... ``` for long conversational inputs (multi-turn
        # dialogues). Without this strip, json.loads sees the leading
        # backticks and throws JSONDecodeError, falling back to a
        # single conf=0.0 general_note — losing all extracted intents.
        if raw_json.startswith("```"):
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json).strip()

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


# --- Intent-to-MemoryType mapping ---

_INTENT_MEMORY_TYPE = {
    "reminder": "fact",
    "person_fact": "fact",
    "company_fact": "fact",
    "follow_up": "fact",
    "decision": "decision",
    "action_item": "fact",
    "general_note": "fact",
}

# Intent types that should create an alert
_INTENT_ALERT_TYPES = {
    "reminder": "follow_up",
    "action_item": "due_task",
    "follow_up": "follow_up",
}

# Default alert offset when no date is provided
_DEFAULT_ALERT_HOURS = 24


# --- Entity resolution ---


async def resolve_entities(
    entities: list[EntityRef],
    pool: "asyncpg.Pool",
    embedding_provider,
    *,
    project_id: str | None = None,
    _cache: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve entity references to entity IDs.

    Searches for existing entities by embedding similarity. Creates new ones
    if no match is found. Uses a within-batch cache to deduplicate entities
    that appear multiple times in the same batch.

    Returns:
        Dict mapping entity name → entity ID.
    """
    from weft.entities import search_entities, store_entity
    from weft.models import EntityCreate, EntityType

    if _cache is None:
        _cache = {}

    resolved: dict[str, str] = {}

    for entity_ref in entities:
        name = entity_ref.name.strip()
        if not name or len(name) < 2:
            logger.debug("resolve_entities.skip_short: %r", name)
            continue

        # Check batch cache (normalized lowercase)
        cache_key = name.lower()
        if cache_key in _cache:
            resolved[name] = _cache[cache_key]
            continue

        # Search by embedding similarity
        embedding = None
        try:
            embedding = await embedding_provider.embed(name)
            etype = (
                EntityType(entity_ref.entity_type)
                if entity_ref.entity_type in [e.value for e in EntityType]
                else None
            )
            matches = await search_entities(
                pool,
                embedding,
                entity_type=etype,
                project_id=project_id,
                limit=3,
                threshold=0.6,
            )

            if matches:
                # Pick highest similarity match
                best_entity, best_sim = max(matches, key=lambda x: x[1])
                resolved[name] = best_entity.id
                _cache[cache_key] = best_entity.id
                logger.debug(
                    "resolve_entities.found: %s → %s (sim=%.3f)",
                    name, best_entity.id, best_sim,
                )
                continue
        except Exception:
            logger.exception("resolve_entities.search_error: %s", name)

        # Create new entity
        try:
            etype = (
                EntityType(entity_ref.entity_type)
                if entity_ref.entity_type in [e.value for e in EntityType]
                else EntityType.concept
            )
            entity = await store_entity(
                pool,
                EntityCreate(
                    name=name,
                    entity_type=etype,
                    project_id=project_id,
                ),
                embedding=embedding,
            )
            resolved[name] = entity.id
            _cache[cache_key] = entity.id
            logger.debug("resolve_entities.created: %s → %s", name, entity.id)
        except Exception:
            logger.exception("resolve_entities.create_error: %s", name)

    return resolved


# --- Router ---


async def route(
    intents: list[Intent],
    pool: "asyncpg.Pool",
    embedding_provider=None,
    *,
    project_id: str | None = None,
    source: str = "ingest",
) -> IngestResult:
    """Route classified intents to appropriate Weft subsystems.

    Each intent is processed independently — a failure in one intent
    is caught, logged, and does not abort the rest.
    """
    from weft.alerts import create_alert
    from weft.entities import link_mention
    from weft.models import (
        AlertChannel,
        AlertCreate,
        AlertType,
        MemoryCreate,
        MemorySource,
        MemoryType,
    )
    from weft.store import store_memory

    result = IngestResult(intents=intents)
    entity_cache: dict[str, str] = {}

    for intent in intents:
        try:
            # --- Build memory inputs (HTTP / pure logic — outside acquire) ---
            mem_type_str = _INTENT_MEMORY_TYPE.get(intent.type, "fact")
            topics = [f"intent:{intent.type}"]
            if intent.entities:
                for e in intent.entities:
                    topics.append(f"entity:{e.name}")

            embedding = None
            if embedding_provider:
                try:
                    embedding = await embedding_provider.embed(intent.content)
                except Exception:
                    logger.warning("route.embed_failed for intent: %s", intent.type)

            # --- DB writes — must run inside acquire() so SET LOCAL
            # app.user_id fires; without this the migration-34 NOT NULL on
            # memories.user_id (and the equivalent on entities) trips.
            async with acquire(pool):
                # --- Resolve entities ---
                entity_ids: dict[str, str] = {}
                if intent.entities and embedding_provider:
                    entity_ids = await resolve_entities(
                        intent.entities,
                        pool,
                        embedding_provider,
                        project_id=project_id,
                        _cache=entity_cache,
                    )
                    result.entities_created += sum(
                        1 for _ in entity_ids.values()
                    )  # approximate; cache hits counted too

                create = MemoryCreate(
                    type=MemoryType(mem_type_str),
                    content=intent.content,
                    topic=topics,
                    source=MemorySource(source) if source in MemorySource._value2member_map_ else MemorySource.ingest,
                    confidence=intent.confidence,
                    project_id=project_id,
                )
                memory = await store_memory(pool, create, embedding=embedding)
                result.memories_created += 1

                # --- Link entities to memory ---
                for entity_name, entity_id in entity_ids.items():
                    try:
                        await link_mention(pool, entity_id, memory.id)
                        result.entities_linked += 1
                    except Exception:
                        logger.exception(
                            "route.link_error: entity=%s memory=%s",
                            entity_id, memory.id,
                        )

                # --- Create alert if applicable ---
                alert_type_str = _INTENT_ALERT_TYPES.get(intent.type)
                if alert_type_str:
                    if intent.dates:
                        trigger_at = intent.dates[0]
                    else:
                        trigger_at = datetime.now(timezone.utc) + timedelta(
                            hours=_DEFAULT_ALERT_HOURS
                        )
                    if trigger_at.tzinfo is None:
                        trigger_at = trigger_at.replace(tzinfo=timezone.utc)

                    alert_create = AlertCreate(
                        alert_type=AlertType(alert_type_str),
                        title=intent.content[:200],
                        body=intent.raw_text[:500] if intent.raw_text else None,
                        trigger_at=trigger_at,
                        channel=AlertChannel.log,
                        project_id=project_id,
                    )
                    await create_alert(pool, alert_create)
                    result.alerts_created += 1

        except Exception as exc:
            logger.exception(
                "route.intent_error: type=%s content=%s",
                intent.type, intent.content[:100],
            )
            result.errors.append(f"{intent.type}: {exc}")

    return result


# --- Public entry point ---


async def process(
    item: IngestItem,
    pool: "asyncpg.Pool",
    embedding_provider=None,
    *,
    project_id: str | None = None,
    tz_name: str = "America/New_York",
) -> IngestResult:
    """Process a single IngestItem through the full pipeline.

    classify → route → return IngestResult.
    This is the only public entry point for the ingestion pipeline.
    """
    intents = await classify(
        item.text,
        metadata=item.metadata,
        tz_name=tz_name,
    )

    if not intents:
        return IngestResult()

    return await route(
        intents,
        pool,
        embedding_provider,
        project_id=project_id,
        source=item.source,
    )


__all__ = ["EntityRef", "Intent", "IngestItem", "IngestResult", "classify", "process", "route", "resolve_entities"]
