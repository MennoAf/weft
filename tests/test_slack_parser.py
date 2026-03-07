"""Tests for the Slack message parser."""

import pytest

from weft.slack.parser import (
    SlackMessage,
    build_memory_content,
    clean_slack_text,
    extract_mentions,
    extract_urls,
    parse_message,
)


class TestParseMessage:
    def test_basic_message(self):
        raw = {"text": "hello world", "user": "U123", "ts": "1709740800.000000"}
        msg = parse_message(raw)
        assert msg.text == "hello world"
        assert msg.user == "U123"
        assert msg.ts == "1709740800.000000"
        assert msg.thread_ts is None
        assert msg.edited_ts is None
        assert msg.reactions == []
        assert msg.files == []

    def test_thread_reply(self):
        raw = {
            "text": "reply",
            "user": "U456",
            "ts": "1709740900.000000",
            "thread_ts": "1709740800.000000",
        }
        msg = parse_message(raw)
        assert msg.is_thread_reply
        assert not msg.is_thread_parent

    def test_thread_parent(self):
        raw = {
            "text": "parent",
            "user": "U123",
            "ts": "1709740800.000000",
            "reply_count": 3,
        }
        msg = parse_message(raw)
        assert msg.is_thread_parent
        assert not msg.is_thread_reply

    def test_edited_message(self):
        raw = {
            "text": "edited text",
            "user": "U123",
            "ts": "1709740800.000000",
            "edited": {"user": "U123", "ts": "1709740900.000000"},
        }
        msg = parse_message(raw)
        assert msg.edited_ts == "1709740900.000000"

    def test_reactions(self):
        raw = {
            "text": "nice",
            "user": "U123",
            "ts": "1709740800.000000",
            "reactions": [
                {"name": "thumbsup", "users": ["U456"], "count": 1},
                {"name": "heart", "users": ["U789", "U012"], "count": 2},
            ],
        }
        msg = parse_message(raw)
        assert len(msg.reactions) == 2
        assert msg.reactions[0]["name"] == "thumbsup"

    def test_files(self):
        raw = {
            "text": "check this out",
            "user": "U123",
            "ts": "1709740800.000000",
            "files": [
                {"name": "report.pdf", "url_private": "https://files.slack.com/report.pdf"},
                {"name": "image.png", "url_private": "https://files.slack.com/image.png"},
            ],
        }
        msg = parse_message(raw)
        assert len(msg.files) == 2
        assert msg.files[0]["name"] == "report.pdf"

    def test_timestamp_property(self):
        raw = {"text": "test", "ts": "1709740800.000000"}
        msg = parse_message(raw)
        assert msg.timestamp.year == 2024
        assert msg.timestamp.month == 3

    def test_missing_fields(self):
        raw = {}
        msg = parse_message(raw)
        assert msg.text == ""
        assert msg.user is None
        assert msg.ts == ""


class TestExtractUrls:
    def test_url_with_label(self):
        urls = extract_urls("check <https://example.com|Example Site> out")
        assert len(urls) == 1
        assert urls[0]["url"] == "https://example.com"
        assert urls[0]["label"] == "Example Site"

    def test_url_without_label(self):
        urls = extract_urls("see <https://example.com>")
        assert len(urls) == 1
        assert urls[0]["url"] == "https://example.com"
        assert urls[0]["label"] == "https://example.com"

    def test_multiple_urls(self):
        urls = extract_urls("<https://a.com|A> and <https://b.com>")
        assert len(urls) == 2

    def test_no_urls(self):
        urls = extract_urls("plain text no urls")
        assert urls == []


class TestExtractMentions:
    def test_single_mention(self):
        mentions = extract_mentions("hey <@U0A8VD0U34H> check this")
        assert mentions == ["U0A8VD0U34H"]

    def test_multiple_mentions(self):
        mentions = extract_mentions("<@U123> and <@U456>")
        assert len(mentions) == 2

    def test_no_mentions(self):
        mentions = extract_mentions("no mentions here")
        assert mentions == []


class TestCleanSlackText:
    def test_url_replacement(self):
        result = clean_slack_text("<https://example.com|Example>")
        assert result == "Example (https://example.com)"

    def test_url_without_label(self):
        result = clean_slack_text("<https://example.com>")
        assert result == "https://example.com"

    def test_user_mention_with_names(self):
        result = clean_slack_text("<@U123> said hi", {"U123": "Jason"})
        assert result == "@Jason said hi"

    def test_user_mention_without_names(self):
        result = clean_slack_text("<@U123> said hi")
        assert result == "@U123 said hi"

    def test_channel_mention(self):
        result = clean_slack_text("post in <#C123|general>")
        assert result == "post in #general"

    def test_mixed_markup(self):
        result = clean_slack_text(
            "<@U123> shared <https://example.com|a link> in <#C456|random>",
            {"U123": "Jason"},
        )
        assert "@Jason" in result
        assert "a link (https://example.com)" in result
        assert "#random" in result


class TestBuildMemoryContent:
    def test_basic_message(self):
        msg = SlackMessage(text="hello world", user="U123", ts="1709740800.000000")
        content = build_memory_content(msg, "general", {"U123": "Jason"})
        assert "# Slack: #general" in content
        assert "From: @Jason" in content
        assert "hello world" in content

    def test_with_thread_replies(self):
        parent = SlackMessage(text="question?", user="U123", ts="1709740800.000000")
        reply = SlackMessage(text="answer!", user="U456", ts="1709740900.000000")
        parent.replies = [reply]
        content = build_memory_content(
            parent, "general", {"U123": "Jason", "U456": "Alice"}
        )
        assert "Thread (1 replies)" in content
        assert "@Alice" in content
        assert "answer!" in content

    def test_with_reactions(self):
        msg = SlackMessage(
            text="great idea",
            user="U123",
            ts="1709740800.000000",
            reactions=[{"name": "thumbsup", "count": 3}],
        )
        content = build_memory_content(msg, "general")
        assert "Reactions:" in content
        assert ":thumbsup: (3)" in content

    def test_with_files(self):
        msg = SlackMessage(
            text="here's the doc",
            user="U123",
            ts="1709740800.000000",
            files=[{"name": "report.pdf", "url": "https://files.slack.com/report.pdf"}],
        )
        content = build_memory_content(msg, "general")
        assert "Attachments:" in content
        assert "report.pdf" in content

    def test_with_urls(self):
        msg = SlackMessage(
            text="check <https://example.com|this> out",
            user="U123",
            ts="1709740800.000000",
        )
        content = build_memory_content(msg, "general")
        assert "Links:" in content
        assert "https://example.com" in content

    def test_deduplicates_urls(self):
        parent = SlackMessage(
            text="<https://example.com|link>",
            user="U123",
            ts="1709740800.000000",
        )
        reply = SlackMessage(
            text="<https://example.com|same link>",
            user="U456",
            ts="1709740900.000000",
        )
        parent.replies = [reply]
        content = build_memory_content(parent, "general")
        assert content.count("https://example.com") == 3  # in text, reply text, and links section

    def test_unknown_user(self):
        msg = SlackMessage(text="test", user="U999", ts="1709740800.000000")
        content = build_memory_content(msg, "general")
        assert "From: @U999" in content

    def test_no_user(self):
        msg = SlackMessage(text="test", ts="1709740800.000000")
        content = build_memory_content(msg, "general")
        assert "From: @unknown" in content
