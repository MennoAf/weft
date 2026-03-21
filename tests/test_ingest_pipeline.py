"""Tests for the smart ingestion pipeline — data models, date parser, and classifier."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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


# --- Classifier ---


def _mock_llm_response(json_data):
    """Create a mock Anthropic response with the given JSON data."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(json_data))]
    return mock_response


class TestClassify:
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

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
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
    async def test_multiple_intents(self):
        llm_response = [
            {"type": "person_fact", "content": "Bob is CEO", "confidence": 0.9, "entities": [], "dates": []},
            {"type": "follow_up", "content": "Demo next week", "confidence": 0.8, "entities": [], "dates": []},
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
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

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("some note about something")

        assert len(result) == 1
        assert result[0].type == "general_note"

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_general_note(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("some important text here")

        assert len(result) == 1
        assert result[0].type == "general_note"
        assert result[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="{not valid json!!!")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("something worth remembering here")

        assert len(result) == 1
        assert result[0].type == "general_note"
        assert result[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_unknown_intent_type_coerced(self):
        llm_response = [
            {"type": "mystery_type", "content": "test", "confidence": 0.7, "entities": [], "dates": []}
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(llm_response))

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
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

        with patch("weft.ingest_pipeline._get_client", return_value=mock_client):
            result = await classify("remind me to do stuff tomorrow")

        assert len(result) == 1
        assert len(result[0].dates) == 1
        assert isinstance(result[0].dates[0], datetime)
        assert result[0].dates[0].tzinfo is not None
