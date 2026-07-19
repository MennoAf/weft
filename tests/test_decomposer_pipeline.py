"""Tests for the smart ingestion pipeline (weft/ingest_pipeline.py).

Covers data model construction, pre-filtering, and the classify() function
with fully mocked LLM calls.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.ingest_pipeline import (
    EntityRef,
    IngestItem,
    IngestResult,
    Intent,
    _should_skip,
    classify,
    process,
    resolve_entities,
    route,
)


@pytest.fixture(autouse=True)
def _patch_acquire_for_mock_pools():
    """Phase-2.5 ingest_pipeline.route now wraps DB writes in
    weft.db.connection.acquire(), which expects a real asyncpg pool.
    These unit tests use AsyncMock pools that don't model the acquire
    contract; patch acquire to a no-op context manager so they keep
    exercising the classification + routing logic. Full-stack
    coverage lives in tests/test_ingest_pipeline_acquire.py against
    the real pool fixture."""
    @asynccontextmanager
    async def _noop_acquire(_pool):
        yield None

    with patch("weft.ingest_pipeline.acquire", _noop_acquire):
        yield


# --- Mock LLM response helpers ---


def _mock_anthropic_response(json_payload: list | dict) -> MagicMock:
    """Build a mock Anthropic Messages response with the given JSON payload."""
    text_block = MagicMock()
    text_block.text = json.dumps(json_payload)
    response = MagicMock()
    response.content = [text_block]
    return response


# --- Data model tests ---


class TestEntityRef:
    def test_basic_construction(self):
        e = EntityRef(name="Bob", entity_type="person")
        assert e.name == "Bob"
        assert e.entity_type == "person"

    def test_defaults_to_concept(self):
        e = EntityRef(name="Something")
        assert e.entity_type == "concept"

    def test_invalid_type_defaults_to_concept(self):
        e = EntityRef(name="X", entity_type="invalid_type")
        assert e.entity_type == "concept"


class TestIntent:
    def test_basic_construction(self):
        i = Intent(type="reminder", content="Buy eggs")
        assert i.type == "reminder"
        assert i.content == "Buy eggs"
        assert i.confidence == 0.7  # default
        assert i.entities == []
        assert i.dates == []

    def test_confidence_clamped_high(self):
        i = Intent(type="reminder", content="test", confidence=1.5)
        assert i.confidence == 1.0

    def test_confidence_clamped_low(self):
        i = Intent(type="reminder", content="test", confidence=-0.5)
        assert i.confidence == 0.0

    def test_unknown_type_defaults_to_general_note(self):
        i = Intent(type="unknown_type", content="test")
        assert i.type == "general_note"

    def test_with_entities_and_dates(self):
        dt = datetime(2025, 3, 1, tzinfo=timezone.utc)
        i = Intent(
            type="follow_up",
            content="Meet Bob",
            entities=[EntityRef(name="Bob", entity_type="person")],
            dates=[dt],
        )
        assert len(i.entities) == 1
        assert i.entities[0].name == "Bob"
        assert len(i.dates) == 1


class TestIngestItem:
    def test_basic_construction(self):
        item = IngestItem(text="hello")
        assert item.text == "hello"
        assert item.source == "unknown"
        assert item.author is None
        assert item.metadata == {}

    def test_missing_text_raises(self):
        with pytest.raises(TypeError):
            IngestItem()

    def test_metadata_defaults_to_dict(self):
        item = IngestItem(text="test", metadata=None)
        assert item.metadata == {}


class TestIngestResult:
    def test_defaults(self):
        r = IngestResult()
        assert r.intents == []
        assert r.memories_created == 0
        assert r.entities_created == 0
        assert r.alerts_created == 0
        assert r.errors == []


# --- Pre-filter tests ---


class TestShouldSkip:
    def test_bot_message_by_flag(self):
        assert _should_skip("hello", {"is_bot": True}) is True

    def test_bot_message_by_bot_id(self):
        assert _should_skip("hello", {"bot_id": "B123"}) is True

    def test_short_text(self):
        assert _should_skip("hi", {}) is True

    def test_emoji_only_unicode(self):
        assert _should_skip("👍", {}) is True

    def test_emoji_shortcode(self):
        assert _should_skip(":thumbsup:", {}) is True

    def test_plus_one(self):
        assert _should_skip("+1", {}) is True

    def test_normal_text_passes(self):
        assert _should_skip("This is a real message", {}) is False

    def test_empty_metadata(self):
        assert _should_skip("Hello world", {}) is False


# --- Classify tests ---


class TestClassify:
    @pytest.mark.asyncio
    async def test_skips_bot_message(self):
        result = await classify("hello", metadata={"is_bot": True})
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_short_text(self):
        result = await classify("hi")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_reminder_intent(self):
        payload = [
            {
                "type": "reminder",
                "content": "Buy eggs",
                "confidence": 0.9,
                "entities": [],
                "dates": ["Saturday"],
            }
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("remind me to buy eggs Saturday")

        assert len(result) == 1
        assert result[0].type == "reminder"
        assert result[0].content == "Buy eggs"
        assert 0.8 <= result[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_person_fact_with_entities(self):
        payload = [
            {
                "type": "person_fact",
                "content": "Bob is the CEO of TechCorp",
                "confidence": 0.95,
                "entities": [
                    {"name": "Bob", "entity_type": "person"},
                    {"name": "TechCorp", "entity_type": "company"},
                ],
                "dates": [],
            }
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("Bob is the CEO of TechCorp")

        assert len(result) == 1
        assert result[0].type == "person_fact"
        assert len(result[0].entities) == 2
        assert result[0].entities[0].name == "Bob"
        assert result[0].entities[0].entity_type == "person"
        assert result[0].entities[1].name == "TechCorp"
        assert result[0].entities[1].entity_type == "company"

    @pytest.mark.asyncio
    async def test_multi_intent(self):
        payload = [
            {
                "type": "person_fact",
                "content": "Bob is the CEO",
                "confidence": 0.9,
                "entities": [{"name": "Bob", "entity_type": "person"}],
                "dates": [],
            },
            {
                "type": "follow_up",
                "content": "Meet Bob next Tuesday",
                "confidence": 0.85,
                "entities": [{"name": "Bob", "entity_type": "person"}],
                "dates": ["next Tuesday"],
            },
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("Bob is the CEO. Meet Bob next Tuesday.")

        assert len(result) == 2
        assert result[0].type == "person_fact"
        assert result[1].type == "follow_up"

    @pytest.mark.asyncio
    async def test_decision_with_no_entities(self):
        payload = [
            {
                "type": "decision",
                "content": "We will use PostgreSQL",
                "confidence": 0.8,
                "entities": [],
                "dates": [],
            }
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("We decided to use PostgreSQL for the backend")

        assert len(result) == 1
        assert result[0].type == "decision"
        assert result[0].entities == []

    @pytest.mark.asyncio
    async def test_malformed_json_abstains_without_raw_fallback(self):
        """Invalid JSON must not become a persistable raw-input note."""
        text_block = MagicMock()
        text_block.text = "not valid json {{"
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=response)

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("some valid input text here")

        assert result == []

    @pytest.mark.asyncio
    async def test_llm_exception_abstains_without_raw_fallback(self):
        """Transport failures abstain rather than fabricating durable state."""
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=RuntimeError("API timeout")
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("some valid input text here")

        assert result == []

    @pytest.mark.asyncio
    async def test_single_object_response_wrapped_in_list(self):
        """LLM returns a single dict instead of a list — should still work."""
        payload = {
            "type": "action_item",
            "content": "Fix the bug",
            "confidence": 0.88,
            "entities": [],
            "dates": [],
        }
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("I need to fix the authentication bug")

        assert len(result) == 1
        assert result[0].type == "action_item"

    @pytest.mark.asyncio
    async def test_unknown_intent_type_becomes_general_note(self):
        payload = [
            {
                "type": "some_new_type",
                "content": "Test",
                "confidence": 0.5,
                "entities": [],
                "dates": [],
            }
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("Something with unknown intent type")

        assert len(result) == 1
        assert result[0].type == "general_note"

    @pytest.mark.asyncio
    async def test_dates_are_resolved_to_datetimes(self):
        payload = [
            {
                "type": "reminder",
                "content": "Buy eggs",
                "confidence": 0.9,
                "entities": [],
                "dates": ["tomorrow"],
            }
        ]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(payload)
        )

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("remind me to buy eggs tomorrow")

        assert len(result) == 1
        assert len(result[0].dates) == 1
        assert isinstance(result[0].dates[0], datetime)


# --- Shared fixtures for route/resolve/process tests ---

FUTURE_DT = datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc)


def _mock_memory(memory_id="weft-mem-1"):
    m = MagicMock()
    m.id = memory_id
    return m


def _mock_entity(entity_id="weft-ent-1", name="Bob", status="active"):
    m = MagicMock()
    m.id = entity_id
    m.name = name
    m.status = status
    return m


def _mock_embedding_provider():
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[0.1] * 768)
    return provider


# --- Resolve entities tests ---


class TestResolveEntities:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()

        result = await resolve_entities([], pool, provider)

        assert result == {}
        provider.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_finds_existing_entity(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            mock_search.return_value = [(existing, 0.85)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool, provider,
            )

        assert result == {"Alice": "weft-existing"}
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_new_entity(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        new_entity = _mock_entity("weft-new-1", "Charlie")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            mock_search.return_value = []
            mock_store.return_value = new_entity

            result = await resolve_entities(
                [EntityRef(name="Charlie", entity_type="person")],
                pool, provider,
            )

        assert result == {"Charlie": "weft-new-1"}
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplicates_within_batch(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        new_entity = _mock_entity("weft-dedup", "Bob")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            mock_search.return_value = []
            mock_store.return_value = new_entity

            result = await resolve_entities(
                [
                    EntityRef(name="Bob", entity_type="person"),
                    EntityRef(name="bob", entity_type="person"),  # same, different case
                ],
                pool, provider,
            )

        # Should only create once thanks to cache
        assert mock_store.call_count == 1
        assert "Bob" in result

    @pytest.mark.asyncio
    async def test_skips_short_names(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()

        with patch("weft.entities.search_entities", new_callable=AsyncMock), \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            result = await resolve_entities(
                [EntityRef(name="X"), EntityRef(name="")],
                pool, provider,
            )

        assert result == {}
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_picks_highest_similarity(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        low = _mock_entity("weft-low", "Alic")
        high = _mock_entity("weft-high", "Alice")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock):
            mock_search.return_value = [(low, 0.65), (high, 0.92)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool, provider,
            )

        assert result == {"Alice": "weft-high"}


# --- Route tests ---


class TestRoute:
    @pytest.mark.asyncio
    async def test_empty_intents(self):
        pool = AsyncMock()
        result = await route([], pool)
        assert result.memories_created == 0
        assert result.alerts_created == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_reminder_creates_memory_and_alert(self):
        pool = AsyncMock()
        intent = Intent(
            type="reminder", content="Buy eggs",
            confidence=0.9, dates=[FUTURE_DT],
        )

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()
            mock_alert.return_value = MagicMock()

            result = await route([intent], pool)

        assert result.memories_created == 1
        assert result.alerts_created == 1

    @pytest.mark.asyncio
    async def test_person_fact_creates_memory_with_entity(self):
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        intent = Intent(
            type="person_fact", content="Alice is the CTO",
            confidence=0.95,
            entities=[EntityRef(name="Alice", entity_type="person")],
        )
        new_entity = _mock_entity("weft-alice", "Alice", status="active")

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_create, \
             patch("weft.entities.get_entity", new_callable=AsyncMock) as mock_get, \
             patch("weft.entities.link_mention", new_callable=AsyncMock) as mock_link:
            mock_store.return_value = _mock_memory()
            mock_search.return_value = []
            mock_create.return_value = new_entity
            mock_get.return_value = new_entity  # Mock get_entity to return active entity
            mock_link.return_value = True

            result = await route([intent], pool, provider)

        assert result.memories_created == 1
        assert result.entities_linked == 1
        mock_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_item_creates_memory_and_alert(self):
        pool = AsyncMock()
        intent = Intent(type="action_item", content="Fix the auth bug", confidence=0.8)

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()
            mock_alert.return_value = MagicMock()

            result = await route([intent], pool)

        assert result.memories_created == 1
        assert result.alerts_created == 1
        # Alert type should be due_task
        alert_create_arg = mock_alert.call_args[0][1]
        assert alert_create_arg.alert_type.value == "due_task"

    @pytest.mark.asyncio
    async def test_decision_creates_memory_no_alert(self):
        pool = AsyncMock()
        intent = Intent(type="decision", content="Use PostgreSQL", confidence=0.8)

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()

            result = await route([intent], pool)

        assert result.memories_created == 1
        assert result.alerts_created == 0
        mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_general_note_creates_memory_only(self):
        pool = AsyncMock()
        intent = Intent(type="general_note", content="Just a note", confidence=0.5)

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()

            result = await route([intent], pool)

        assert result.memories_created == 1
        assert result.alerts_created == 0
        mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_follow_up_creates_memory_and_alert(self):
        pool = AsyncMock()
        intent = Intent(
            type="follow_up", content="Follow up with Bob",
            confidence=0.85, dates=[FUTURE_DT],
        )

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()
            mock_alert.return_value = MagicMock()

            result = await route([intent], pool)

        assert result.memories_created == 1
        assert result.alerts_created == 1

    @pytest.mark.asyncio
    async def test_company_fact_creates_memory(self):
        pool = AsyncMock()
        intent = Intent(
            type="company_fact", content="Acme raised Series B",
            confidence=0.9,
            entities=[EntityRef(name="Acme", entity_type="company")],
        )
        provider = _mock_embedding_provider()
        new_entity = _mock_entity("weft-acme", "Acme")

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_create, \
             patch("weft.entities.link_mention", new_callable=AsyncMock):
            mock_store.return_value = _mock_memory()
            mock_search.return_value = []
            mock_create.return_value = new_entity

            result = await route([intent], pool, provider)

        assert result.memories_created == 1


# --- Error handling tests ---


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_error_isolation_continues_after_failure(self):
        """If one intent fails, remaining intents still get processed."""
        pool = AsyncMock()
        intents = [
            Intent(type="general_note", content="Will fail", confidence=0.5),
            Intent(type="general_note", content="Will succeed", confidence=0.5),
        ]

        call_count = 0

        async def store_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB connection lost")
            return _mock_memory("weft-ok")

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store:
            mock_store.side_effect = store_side_effect

            result = await route(intents, pool)

        assert result.memories_created == 1
        assert len(result.errors) == 1
        assert "DB connection lost" in result.errors[0]

    @pytest.mark.asyncio
    async def test_reminder_without_date_uses_default(self):
        """Reminder with no dates should still create alert with default future time."""
        pool = AsyncMock()
        intent = Intent(type="reminder", content="Remember this", confidence=0.9)
        # No dates set

        with patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_store.return_value = _mock_memory()
            mock_alert.return_value = MagicMock()

            result = await route([intent], pool)

        assert result.alerts_created == 1
        # trigger_at should be in the future
        alert_create_arg = mock_alert.call_args[0][1]
        assert alert_create_arg.trigger_at > datetime.now(timezone.utc)


# --- Process (end-to-end) tests ---


class TestProcess:
    @pytest.mark.asyncio
    async def test_end_to_end(self):
        pool = AsyncMock()
        item = IngestItem(text="Bob is the CEO of TechCorp", source="slack")

        mock_intents = [
            Intent(
                type="person_fact", content="Bob is the CEO of TechCorp",
                confidence=0.95,
                entities=[EntityRef(name="Bob", entity_type="person")],
            )
        ]

        with patch("weft.ingest_pipeline.classify", new_callable=AsyncMock) as mock_classify, \
             patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_create, \
             patch("weft.entities.link_mention", new_callable=AsyncMock):
            mock_classify.return_value = mock_intents
            mock_store.return_value = _mock_memory()
            mock_search.return_value = []
            mock_create.return_value = _mock_entity()

            result = await process(item, pool, _mock_embedding_provider())

        assert isinstance(result, IngestResult)
        assert result.memories_created == 1
        mock_classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_classify_returns_empty_result(self):
        pool = AsyncMock()
        item = IngestItem(text="hi", source="slack")

        with patch("weft.ingest_pipeline.classify", new_callable=AsyncMock) as mock_classify, \
             patch("weft.ingest_pipeline.route", new_callable=AsyncMock) as mock_route:
            mock_classify.return_value = []

            result = await process(item, pool)

        assert result.memories_created == 0
        assert result.alerts_created == 0
        assert result.errors == []
        mock_route.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multi_intent_processing(self):
        pool = AsyncMock()
        item = IngestItem(text="Bob is CEO. Remind me to call him Friday.", source="slack")

        mock_intents = [
            Intent(type="person_fact", content="Bob is CEO", confidence=0.9),
            Intent(type="reminder", content="Call Bob Friday", confidence=0.85, dates=[FUTURE_DT]),
        ]

        with patch("weft.ingest_pipeline.classify", new_callable=AsyncMock) as mock_classify, \
             patch("weft.store.store_memory", new_callable=AsyncMock) as mock_store, \
             patch("weft.alerts.create_alert", new_callable=AsyncMock) as mock_alert:
            mock_classify.return_value = mock_intents
            mock_store.return_value = _mock_memory()
            mock_alert.return_value = MagicMock()

            result = await process(item, pool)

        assert result.memories_created == 2
        assert result.alerts_created == 1  # only the reminder gets an alert
