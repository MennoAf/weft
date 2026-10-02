"""Tests for the smart ingestion pipeline — data models, date parser, and classifier."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from weft.date_parser import parse_dates
from weft.ingest_pipeline import (
    EntityRef,
    IngestItem,
    IngestResult,
    Intent,
    classify,
)
from weft.text_generation import GenerationResponse


# --- Date parser ---


class TestParseDates:
    # Fixed reference: Monday Jan 15, 2024 at noon ET
    REF = datetime(2024, 1, 15, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    def test_empty_text(self):
        assert parse_dates("", reference_time=self.REF) == []

    def test_whitespace_only(self):
        assert parse_dates("   ", reference_time=self.REF) == []

    def test_no_dates(self):
        assert parse_dates("just some random text", reference_time=self.REF) == []

    def test_tomorrow(self):
        results = parse_dates("do this tomorrow", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].day == 16
        assert results[0].month == 1

    def test_today(self):
        results = parse_dates("do this today", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].day == 15

    def test_next_weekday(self):
        # Reference is Monday, "next Friday" should be Jan 19
        results = parse_dates("meet next Friday", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].weekday() == 4  # Friday

    def test_bare_weekday(self):
        # "Saturday" from Monday = next Saturday (Jan 20)
        results = parse_dates("buy eggs Saturday", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].weekday() == 5  # Saturday
        assert results[0] > self.REF

    def test_by_weekday(self):
        results = parse_dates("finish by Friday", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].weekday() == 4

    def test_in_days(self):
        results = parse_dates("remind me in 3 days", reference_time=self.REF)
        assert len(results) == 1
        expected = self.REF + timedelta(days=3)
        assert results[0].day == expected.day

    def test_in_weeks(self):
        results = parse_dates("check in 2 weeks", reference_time=self.REF)
        assert len(results) == 1
        expected = self.REF + timedelta(weeks=2)
        assert results[0].day == expected.day

    def test_next_week(self):
        results = parse_dates("do this next week", reference_time=self.REF)
        assert len(results) == 1
        expected = self.REF + timedelta(weeks=1)
        assert results[0].day == expected.day

    def test_absolute_date(self):
        results = parse_dates("meeting on March 15, 2025", reference_time=self.REF)
        assert len(results) == 1
        assert results[0].month == 3
        assert results[0].day == 15

    def test_all_results_timezone_aware(self):
        for text in ["tomorrow", "next Friday", "in 3 days", "March 15, 2025"]:
            results = parse_dates(text, reference_time=self.REF)
            for dt in results:
                assert dt.tzinfo is not None, f"Naive datetime for input: {text}"

    def test_custom_timezone(self):
        results = parse_dates("tomorrow", reference_time=self.REF, tz_name="US/Pacific")
        assert len(results) == 1
        assert results[0].tzinfo is not None


# --- Data models ---


class TestDataModels:
    def test_entity_ref_defaults(self):
        e = EntityRef(name="Bob")
        assert e.entity_type == "concept"

    def test_entity_ref_invalid_type_coerced(self):
        e = EntityRef(name="Bob", entity_type="alien")
        assert e.entity_type == "concept"

    def test_intent_defaults(self):
        i = Intent(type="reminder", content="test")
        assert i.confidence == 0.7
        assert i.entities == []
        assert i.dates == []

    def test_intent_invalid_type_coerced(self):
        i = Intent(type="foobar", content="test")
        assert i.type == "general_note"

    def test_intent_confidence_clamped(self):
        assert Intent(type="reminder", content="x", confidence=1.5).confidence == 1.0
        assert Intent(type="reminder", content="x", confidence=-0.5).confidence == 0.0

    def test_intent_extracts_positive_preference_metadata(self):
        intent = Intent(
            type="general_note",
            content="I prefer history podcasts on my commute",
            raw_text="I prefer history podcasts on my commute",
        )
        assert intent.preference_metadata == {
            "polarity": "positive",
            "strength": "soft",
            "value": "history podcasts on my commute",
        }

    def test_intent_extracts_avoidance_metadata(self):
        intent = Intent(
            type="general_note",
            content="I never want true crime podcasts",
            raw_text="I never want true crime podcasts",
        )
        assert intent.preference_metadata["polarity"] == "avoidance"
        assert intent.preference_metadata["strength"] == "hard"

    def test_intent_does_not_scan_other_raw_text_intents(self):
        intent = Intent(
            type="person_fact",
            content="Bob is the CEO of Acme",
            raw_text="I prefer history podcasts. Bob is the CEO of Acme",
        )
        assert intent.preference_metadata is None

    def test_ingest_item_defaults(self):
        item = IngestItem(text="hello")
        assert item.source == "unknown"
        assert item.metadata == {}

    def test_ingest_item_none_metadata_normalized(self):
        item = IngestItem(text="hello", metadata=None)
        assert item.metadata == {}

    def test_ingest_result_defaults(self):
        r = IngestResult()
        assert r.intents == []
        assert r.memories_created == 0
        assert r.errors == []

    def test_intent_hint_fields_default(self):
        """memory_type_hint defaults to None; extra_topics defaults to [].
        route() treats these defaults as 'no override'."""
        i = Intent(type="general_note", content="x")
        assert i.memory_type_hint is None
        assert i.extra_topics == []

    def test_intent_hint_fields_carry_values(self):
        i = Intent(
            type="general_note",
            content="x",
            memory_type_hint="preference",
            extra_topics=["discord", "brain-dump"],
        )
        assert i.memory_type_hint == "preference"
        assert i.extra_topics == ["discord", "brain-dump"]


class TestProcessMetadataStamping:
    """process() must lift source-supplied hints from IngestItem.metadata onto
    every Intent before route() runs. Without this plumbing, channel mappings
    silently no-op at write time — the substrate-stub gap that put L8 on HOLD."""

    @pytest.mark.asyncio
    async def test_metadata_hint_stamped_onto_intents(self):
        from weft.ingest_pipeline import process

        # Two intents come back from classify; both must carry the hint.
        fake_intents = [
            Intent(type="general_note", content="first note"),
            Intent(type="general_note", content="second note"),
        ]
        fake_route_result = IngestResult(intents=fake_intents, memories_created=2)

        with patch(
            "weft.ingest_pipeline.classify",
            new_callable=AsyncMock,
            return_value=fake_intents,
        ), patch(
            "weft.ingest_pipeline.route",
            new_callable=AsyncMock,
            return_value=fake_route_result,
        ) as mock_route:
            item = IngestItem(
                text="some content from a mapped channel",
                source="discord",
                metadata={
                    "memory_type_hint": "preference",
                    "topics": ["discord", "brain-dump"],
                    "channel": "1507757010857496618",
                },
            )
            await process(item, pool=AsyncMock(), embedding_provider=None)

        passed_intents = mock_route.call_args.args[0]
        for intent in passed_intents:
            assert intent.memory_type_hint == "preference"
            assert intent.extra_topics == ["discord", "brain-dump"]

    @pytest.mark.asyncio
    async def test_no_metadata_hint_leaves_intent_defaults(self):
        from weft.ingest_pipeline import process

        fake_intents = [Intent(type="general_note", content="bare note")]
        fake_route_result = IngestResult(intents=fake_intents, memories_created=1)

        with patch(
            "weft.ingest_pipeline.classify",
            new_callable=AsyncMock,
            return_value=fake_intents,
        ), patch(
            "weft.ingest_pipeline.route",
            new_callable=AsyncMock,
            return_value=fake_route_result,
        ) as mock_route:
            item = IngestItem(text="plain text", source="cli")
            await process(item, pool=AsyncMock(), embedding_provider=None)

        passed = mock_route.call_args.args[0][0]
        assert passed.memory_type_hint is None
        assert passed.extra_topics == []

    @pytest.mark.asyncio
    async def test_non_string_hint_ignored(self):
        """A non-string hint in metadata is treated as absent — never crashes."""
        from weft.ingest_pipeline import process

        fake_intents = [Intent(type="general_note", content="defensive note")]
        fake_route_result = IngestResult(intents=fake_intents, memories_created=1)

        with patch(
            "weft.ingest_pipeline.classify",
            new_callable=AsyncMock,
            return_value=fake_intents,
        ), patch(
            "weft.ingest_pipeline.route",
            new_callable=AsyncMock,
            return_value=fake_route_result,
        ) as mock_route:
            item = IngestItem(
                text="text with garbage metadata",
                metadata={"memory_type_hint": 42},  # int, not str
            )
            await process(item, pool=AsyncMock(), embedding_provider=None)

        passed = mock_route.call_args.args[0][0]
        assert passed.memory_type_hint is None


# --- Classifier ---


def _mock_llm_response(json_data):
    """Create a mock Anthropic response with the given JSON data."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(json_data))]
    return mock_response


def _anthropic_config():
    return SimpleNamespace(
        text_generation=SimpleNamespace(provider="anthropic", models={})
    )


@contextmanager
def _managed_anthropic_client(client):
    with patch("weft.config.load_config", return_value=_anthropic_config()), patch(
        "anthropic.AsyncAnthropic", return_value=client
    ):
        yield


class TestClassify:
    @pytest.mark.asyncio
    async def test_injected_provider_drives_classifier_without_anthropic_client(self):
        class FakeProvider:
            def __init__(self):
                self.requests = []

            async def generate(self, request):
                self.requests.append(request)
                return GenerationResponse(
                    text='[{"type":"general_note","content":"from fake provider",'
                    '"confidence":0.9,"entities":[],"dates":[]}]',
                    model=request.model,
                )

        provider = FakeProvider()
        with patch("anthropic.AsyncAnthropic", side_effect=AssertionError("client constructed")):
            result = await classify("a note for the fake provider", generation_provider=provider)

        assert [(item.type, item.content) for item in result] == [("general_note", "from fake provider")]
        assert provider.requests[0].model == "claude-haiku-4-5-20251001"
        assert provider.requests[0].max_tokens == 512

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        assert await classify("") == []

    @pytest.mark.asyncio
    async def test_short_text_returns_empty(self):
        assert await classify("hi") == []

    @pytest.mark.asyncio
    async def test_bot_message_returns_empty(self):
        assert await classify("some text", metadata={"is_bot": True}) == []

    @pytest.mark.asyncio
    async def test_reminder_classification(self):
        llm_response = [
            {
                "type": "reminder",
                "content": "Buy eggs",
                "confidence": 0.9,
                "entities": [],
                "dates": ["Saturday"],
            }
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("I need to remember to buy eggs Saturday")

        assert len(result) == 1
        assert result[0].type == "reminder"
        assert result[0].content == "Buy eggs"
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_person_fact_with_entities(self):
        llm_response = [
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
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("Bob is the CEO of TechCorp")

        assert len(result) == 1
        assert result[0].type == "person_fact"
        assert len(result[0].entities) == 2
        assert result[0].entities[0].name == "Bob"
        assert result[0].entities[0].entity_type == "person"
        assert result[0].entities[1].name == "TechCorp"
        assert result[0].entities[1].entity_type == "company"

    @pytest.mark.asyncio
    async def test_multiple_intents(self):
        llm_response = [
            {"type": "person_fact", "content": "Bob is CEO", "confidence": 0.9, "entities": [], "dates": []},
            {"type": "follow_up", "content": "Demo next week", "confidence": 0.8, "entities": [], "dates": []},
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("Bob is the CEO. He wants a demo next week.")

        assert len(result) == 2
        assert result[0].type == "person_fact"
        assert result[1].type == "follow_up"

    @pytest.mark.asyncio
    async def test_llm_returns_object_not_array(self):
        """Single object response is wrapped in a list."""
        llm_response = {"type": "general_note", "content": "test", "confidence": 0.7, "entities": [], "dates": []}

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("some note about something")

        assert len(result) == 1
        assert result[0].type == "general_note"

    @pytest.mark.asyncio
    async def test_classifier_transport_error_abstains(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

        with _managed_anthropic_client(mock_client):
            result = await classify("some important text here")

        assert result == []

    @pytest.mark.asyncio
    async def test_classifier_malformed_json_does_not_fabricate(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="{not valid json!!!")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("something worth remembering here")

        assert result == []

    @pytest.mark.asyncio
    async def test_classifier_malformed_end_turn_does_not_salvage_prefix(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text=(
            '[{"type":"decision","content":"Use Postgres",'
            '"confidence":0.9,"entities":[],"dates":[]} GARBAGE'
        ))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("we decided to use Postgres")

        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_item",
        [
            {},
            {"type": "decision", "content": None},
            {"type": "decision", "content": "   "},
            {"type": "decision", "content": "Use Postgres", "confidence": "nan"},
            {"type": "decision", "content": "Use Postgres", "entities": None},
            {"type": "decision", "content": "Use Postgres", "dates": None},
        ],
    )
    async def test_classifier_rejects_invalid_items_without_raw_fallback(
        self, invalid_item,
    ):
        mock_response = _mock_llm_response([invalid_item])
        mock_response.stop_reason = "end_turn"
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        raw = "the entire raw input must never become a fallback memory"
        with _managed_anthropic_client(mock_client):
            result = await classify(raw)

        assert result == []

    @pytest.mark.asyncio
    async def test_classifier_bad_item_does_not_discard_valid_sibling(self):
        mock_response = _mock_llm_response([
            {"type": "decision", "content": None},
            {
                "type": "decision",
                "content": "Use Postgres",
                "confidence": 0.9,
                "entities": [],
                "dates": [],
            },
        ])
        mock_response.stop_reason = "end_turn"
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("we decided to use Postgres")

        assert [(intent.type, intent.content) for intent in result] == [
            ("decision", "Use Postgres"),
        ]

    @pytest.mark.asyncio
    async def test_classifier_max_tokens_abstains_without_complete_item(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "max_tokens"
        mock_response.content = [MagicMock(text='[{"type": "decision",')]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("we decided something important")

        assert result == []

    @pytest.mark.asyncio
    async def test_classifier_salvages_complete_partial_items(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "max_tokens"
        mock_response.content = [MagicMock(text=(
            '[{"type":"decision","content":"Use Postgres",'
            '"confidence":0.9,"entities":[],"dates":[]},'
            '{"type":"follow_up","content":"unfinished"'
        ))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("We decided to use Postgres and follow up later")

        assert [(item.type, item.content) for item in result] == [
            ("decision", "Use Postgres"),
        ]

    @pytest.mark.asyncio
    async def test_classifier_empty_content_abstains(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("something worth remembering here")

        assert result == []

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_is_parsed(self):
        """Claude wraps long-input responses in ```json ... ``` fences despite
        the system prompt forbidding markdown. The classifier must strip the
        fence before parsing or every long multi-turn input degrades to a
        single conf=0.0 general_note (LongMemEval multi-session questions
        regressed silently because of this — see commit history)."""
        fenced = (
            '```json\n'
            '[{"type": "person_fact", "content": "Bob is CEO", '
            '"confidence": 0.9, "entities": [], "dates": []}]\n'
            '```'
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=fenced)]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("a long conversation about Bob the CEO")

        assert len(result) == 1
        assert result[0].type == "person_fact"
        assert result[0].content == "Bob is CEO"
        # Specifically NOT the conf=0.0 fallback.
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_bare_fence_json_is_parsed(self):
        """Some Claude responses use bare ``` (no language tag). Strip both."""
        fenced = (
            '```\n'
            '{"type": "general_note", "content": "noted", '
            '"confidence": 0.8, "entities": [], "dates": []}\n'
            '```'
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=fenced)]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with _managed_anthropic_client(mock_client):
            result = await classify("something worth noting")

        assert len(result) == 1
        assert result[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_unknown_intent_type_coerced(self):
        llm_response = [
            {"type": "mystery_type", "content": "test", "confidence": 0.7, "entities": [], "dates": []}
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("this is a test message for classification")

        assert len(result) == 1
        assert result[0].type == "general_note"

    @pytest.mark.asyncio
    async def test_dates_resolved_to_datetimes(self):
        llm_response = [
            {"type": "reminder", "content": "Do stuff", "confidence": 0.9, "entities": [], "dates": ["tomorrow"]}
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with _managed_anthropic_client(mock_client):
            result = await classify("remind me to do stuff tomorrow")

        assert len(result) == 1
        assert len(result[0].dates) == 1
        assert isinstance(result[0].dates[0], datetime)
        assert result[0].dates[0].tzinfo is not None
