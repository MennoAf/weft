"""Smart ingestion pipeline — LLM-powered intent classification and routing.

Source-agnostic core: takes raw text + metadata, classifies intent via LLM,
extracts entities and dates, and routes to appropriate Weft subsystems.

Usage:
    result = await process(IngestItem(text="Bob is the CEO of Acme"), pool)
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import asyncpg

from weft.date_parser import parse_dates
from weft.db.connection import acquire
from weft.text_generation import (
    GenerationRequest,
    TextGenerationProvider,
    managed_provider_for_role,
    model_for_role,
)

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
    # Source-supplied overrides — when present, take precedence over the
    # LLM-inferred mapping in route(). Set by process() from IngestItem.metadata
    # (e.g. Discord channel mapping). None means "no override; use defaults".
    memory_type_hint: str | None = None
    extra_topics: list[str] = field(default_factory=list)
    preference_metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.type not in INTENT_TYPES:
            self.type = "general_note"
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.preference_metadata is None:
            self.preference_metadata = _preference_metadata_for_intent(self)


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


def _normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def _preference_metadata_for_intent(intent: Intent) -> dict[str, Any] | None:
    """Extract conservative preference polarity from an already-classified intent.

    This reuses the deterministic candidate extractor; it does not add an LLM
    call and returns None when the text does not contain a clear preference
    expression. The human-readable intent content remains the source of truth.
    """
    if intent.type not in {"person_fact", "general_note"}:
        return None
    from weft.extract import extract_candidates

    text = intent.content or intent.raw_text
    for candidate in extract_candidates(text, min_confidence=0.7):
        if candidate.get("type") == "preference":
            return candidate.get("preference_metadata")
    return None


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


def _strip_json_fences(raw: str) -> str:
    """Strip markdown fences without changing the enclosed payload."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    return raw


def _salvage_partial_intent_array(raw: str) -> list[Any] | None:
    """Recover only complete top-level values from a truncated JSON array."""
    start = raw.find("[")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    idx = start + 1
    recovered: list[Any] = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(raw) or raw[idx] == "]":
            break
        try:
            value, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            break
        recovered.append(value)
        idx = end
    return recovered or None


async def classify(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    tz_name: str = "America/New_York",
    generation_provider: TextGenerationProvider | None = None,
) -> list[Intent]:
    """Classify text with an injected or per-operation managed provider."""
    effective_metadata = metadata or {}
    if _should_skip(text, effective_metadata):
        return []
    if generation_provider is not None:
        return await _classify_with_provider(
            text,
            metadata=metadata,
            tz_name=tz_name,
            generation_provider=generation_provider,
        )
    async with managed_provider_for_role("ingest_classifier") as provider:
        return await _classify_with_provider(
            text,
            metadata=metadata,
            tz_name=tz_name,
            generation_provider=provider,
        )


async def _classify_with_provider(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    tz_name: str = "America/New_York",
    generation_provider: TextGenerationProvider | None = None,
) -> list[Intent]:
    """Classify text into structured intents via LLM.

    Returns a list of Intent objects. Complete items may be salvaged from a
    truncated JSON array; unrecoverable model/transport failures abstain with
    ``[]``. Raw input is never fabricated into a successful-looking intent.
    Returns [] for empty/bot/very-short text.
    """
    if metadata is None:
        metadata = {}

    if _should_skip(text, metadata):
        return []

    normalized = _normalize_text(text)
    if len(normalized) > _MAX_TEXT_LENGTH:
        normalized = normalized[:_MAX_TEXT_LENGTH]
        logger.warning("classify.truncated text to %d chars", _MAX_TEXT_LENGTH)

    provider = generation_provider
    if provider is None:
        raise RuntimeError("classifier provider was not supplied")
    request = GenerationRequest(
        model=model_for_role("ingest_classifier", _CLASSIFIER_MODEL),
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=({"role": "user", "content": normalized},),
    )

    try:
        response = await provider.generate(request)
        content_text = response.text
        if not content_text.strip():
            logger.warning("classify.abstained reason=empty_content")
            return []
        content = [type("TextBlock", (), {"text": content_text})()]
        if not content:
            logger.warning("classify.abstained reason=empty_content")
            return []
        text = getattr(content[0], "text", None)
        if not isinstance(text, str) or not text.strip():
            logger.warning("classify.abstained reason=empty_content")
            return []
        raw_json = _strip_json_fences(text)
        logger.debug("classify.response: %s", raw_json)

        stop_reason = getattr(response, "stop_reason", None)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            if stop_reason != "max_tokens":
                logger.warning(
                    "classify.abstained reason=malformed_json error=%s",
                    exc,
                )
                return []
            parsed = _salvage_partial_intent_array(raw_json)
            if parsed is None:
                logger.warning(
                    "classify.abstained reason=max_tokens error=%s",
                    exc,
                )
                return []
            logger.warning(
                "classify.salvaged reason=max_tokens recovered=%d",
                len(parsed),
            )

        if stop_reason == "max_tokens" and not isinstance(parsed, list):
            logger.warning("classify.abstained reason=max_tokens_unbounded_shape")
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            logger.warning(
                "classify.abstained reason=unexpected_shape type=%s",
                type(parsed).__name__,
            )
            return []

        intents: list[Intent] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.warning("classify.item_rejected index=%d reason=not_object", index)
                continue

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                logger.warning(
                    "classify.item_rejected index=%d reason=invalid_content",
                    index,
                )
                continue
            content = content.strip()

            raw_confidence = item.get("confidence", 0.7)
            if isinstance(raw_confidence, bool):
                logger.warning(
                    "classify.item_rejected index=%d reason=invalid_confidence",
                    index,
                )
                continue
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                logger.warning(
                    "classify.item_rejected index=%d reason=invalid_confidence",
                    index,
                )
                continue
            if not math.isfinite(confidence):
                logger.warning(
                    "classify.item_rejected index=%d reason=nonfinite_confidence",
                    index,
                )
                continue

            raw_entities = item.get("entities", [])
            raw_dates = item.get("dates", [])
            if not isinstance(raw_entities, list) or not isinstance(raw_dates, list):
                logger.warning(
                    "classify.item_rejected index=%d reason=invalid_collection_shape",
                    index,
                )
                continue

            entities = []
            for entity in raw_entities:
                if not isinstance(entity, dict):
                    continue
                name = entity.get("name")
                if isinstance(name, str) and name.strip():
                    entities.append(EntityRef(
                        name=name.strip(),
                        entity_type=entity.get("entity_type", "concept"),
                    ))

            resolved_dates: list[datetime] = []
            for date_string in raw_dates:
                if isinstance(date_string, str) and date_string.strip():
                    resolved_dates.extend(parse_dates(date_string, tz_name=tz_name))

            intent_type = item.get("type", "general_note")
            if intent_type not in INTENT_TYPES:
                logger.warning("classify.unknown_type: %s", intent_type)
                intent_type = "general_note"

            intents.append(Intent(
                type=intent_type,
                content=content,
                confidence=confidence,
                entities=entities,
                dates=resolved_dates,
                raw_text=normalized,
            ))

        return intents

    except Exception:
        logger.exception("classify.abstained reason=classifier_error")
        return []


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
    """Resolve entity references to entity IDs (two-tier: auto-merge / candidate / new).

    Searches for existing entities by embedding similarity. Implements two-tier matching:
    - cosine ≥0.85: auto-merge/link (uses existing entity)
    - 0.6 ≤ cosine <0.85: record as candidate (creates candidate entity, NOT auto-linked)
    - cosine <0.6: create new entity

    Uses a within-batch cache to deduplicate entities that appear multiple times
    in the same batch.

    Returns:
        Dict mapping entity name → entity ID. Includes both active and candidate
        entities; the route() function decides whether to link based on entity status.
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

        # Search by embedding similarity (lower threshold to catch all candidates)
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
                threshold=0.4,  # Lower threshold to capture candidates
            )

            if matches:
                # Pick highest similarity match
                best_entity, best_sim = max(matches, key=lambda x: x[1])

                if best_sim >= 0.85:
                    # Auto-merge: high confidence match to existing entity
                    resolved[name] = best_entity.id
                    _cache[cache_key] = best_entity.id
                    logger.debug(
                        "resolve_entities.auto_merge: %s → %s (sim=%.3f)",
                        name, best_entity.id, best_sim,
                    )
                    continue
                elif best_sim >= 0.6:
                    # Candidate: possible match, requires review before linking
                    # Create candidate entity but do NOT link to memory
                    try:
                        etype_for_create = (
                            EntityType(entity_ref.entity_type)
                            if entity_ref.entity_type in [e.value for e in EntityType]
                            else EntityType.concept
                        )
                        candidate = await store_entity(
                            pool,
                            EntityCreate(
                                name=name,
                                entity_type=etype_for_create,
                                project_id=project_id,
                            ),
                            embedding=embedding,
                            status="candidate",
                        )
                        resolved[name] = candidate.id
                        _cache[cache_key] = candidate.id
                        logger.debug(
                            "resolve_entities.candidate: %s → %s (sim=%.3f, existing: %s)",
                            name, candidate.id, best_sim, best_entity.id,
                        )
                        continue
                    except Exception:
                        logger.exception("resolve_entities.candidate_create_error: %s", name)
                        # Fall through to create new entity as fallback
                # else: best_sim < 0.6, fall through to create new entity
        except Exception:
            logger.exception("resolve_entities.search_error: %s", name)

        # Create new entity (no match found, or failed to create candidate)
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
                status="active",
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
    from weft.entities import get_entity, link_mention
    from weft.models import (
        AlertChannel,
        AlertCreate,
        AlertType,
        MemoryCreate,
        MemorySource,
        MemoryType,
        PreferenceMetadata,
    )
    from weft.store import store_memory

    result = IngestResult(intents=intents)
    entity_cache: dict[str, str] = {}

    for intent in intents:
        try:
            # --- Build memory inputs (HTTP / pure logic — outside acquire) ---
            # Source-supplied hint (e.g. Discord channel mapping) wins over the
            # LLM-derived default. Validate against MemoryType; on invalid hint
            # log + fall back to lookup so a bad config can't crash ingest.
            mem_type_str = _INTENT_MEMORY_TYPE.get(intent.type, "fact")
            if intent.preference_metadata is not None:
                mem_type_str = MemoryType.preference.value
            if intent.memory_type_hint:
                try:
                    MemoryType(intent.memory_type_hint)
                    mem_type_str = intent.memory_type_hint
                except ValueError:
                    logger.warning(
                        "route.invalid_memory_type_hint: hint=%r, falling back to %s",
                        intent.memory_type_hint, mem_type_str,
                    )
            topics = [f"intent:{intent.type}"]
            if intent.entities:
                for e in intent.entities:
                    # Canonical lowercase entity tag: the topic-gather match
                    # (weft/topic_gather.py) is case-sensitive exact, and the
                    # resolver (weft/topic_resolution._naive_normalize) emits a
                    # lowercase entity tag. Writing the raw (mixed-case) e.name
                    # would make a lowercase-resolved query miss this memory.
                    # The entity's display name is preserved in the entities
                    # table; only the tag string is canonicalized.
                    topics.append(f"entity:{e.name.lower()}")
            # Source-supplied topics (e.g. ["discord", "brain-dump"]) — append
            # after auto-topics so they're easy to spot in recall queries.
            if intent.extra_topics:
                topics.extend(intent.extra_topics)

            embedding = None
            if embedding_provider:
                try:
                    from weft.store import embed_text_for_memory
                    embedding = await embedding_provider.embed(
                        embed_text_for_memory(intent.content, topics)
                    )
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
                    preference_metadata=(
                        PreferenceMetadata.model_validate(intent.preference_metadata)
                        if mem_type_str == MemoryType.preference.value
                        and intent.preference_metadata is not None
                        else None
                    ),
                )
                memory = await store_memory(pool, create, embedding=embedding)
                result.memories_created += 1

                # --- Link entities to memory ---
                # Only link active entities; skip candidates (status='candidate')
                # which require review before being merged into mentions.
                for entity_name, entity_id in entity_ids.items():
                    try:
                        entity = await get_entity(pool, entity_id)
                        if entity and entity.status == "active":
                            await link_mention(pool, entity_id, memory.id)
                            result.entities_linked += 1
                        elif entity and entity.status == "candidate":
                            logger.debug(
                                "route.skip_candidate_link: candidate=%s memory=%s",
                                entity_id, memory.id,
                            )
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
    generation_provider: TextGenerationProvider | None = None,
) -> IngestResult:
    """Process a single IngestItem through the full pipeline.

    classify → route → return IngestResult.
    This is the only public entry point for the ingestion pipeline.
    """
    intents = await classify(
        item.text,
        metadata=item.metadata,
        tz_name=tz_name,
        generation_provider=generation_provider,
    )

    if not intents:
        return IngestResult()

    # Stamp source-supplied overrides onto every intent. Channel-mapped sources
    # (e.g. Discord adapter) put memory_type_hint + topics in metadata; route()
    # honors them over the LLM-derived defaults.
    hint = item.metadata.get("memory_type_hint")
    extra_topics_raw = item.metadata.get("topics") or []
    extra_topics = [t for t in extra_topics_raw if isinstance(t, str) and t]
    if hint or extra_topics:
        for intent in intents:
            if hint and isinstance(hint, str):
                intent.memory_type_hint = hint
            if extra_topics:
                intent.extra_topics = list(extra_topics)

    return await route(
        intents,
        pool,
        embedding_provider,
        project_id=project_id,
        source=item.source,
    )


__all__ = ["EntityRef", "Intent", "IngestItem", "IngestResult", "classify", "process", "route", "resolve_entities"]
