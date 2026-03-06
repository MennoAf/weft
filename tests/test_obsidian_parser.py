"""Tests for the Obsidian markdown parser — pure unit tests, no I/O."""

import pytest

from weft.obsidian.parser import (
    ParsedNote,
    Section,
    Task,
    extract_callout_types,
    extract_date,
    extract_embeds,
    extract_inline_tags,
    extract_tasks,
    extract_wikilinks,
    normalize_tag,
    parse_note,
    split_by_headings,
)


class TestNormalizeTag:
    def test_strips_hash(self):
        assert normalize_tag("#python") == "python"

    def test_lowercases(self):
        assert normalize_tag("Python") == "python"

    def test_replaces_spaces(self):
        assert normalize_tag("meal prep") == "meal-prep"

    def test_combined(self):
        assert normalize_tag("#Meal Prep") == "meal-prep"


class TestExtractDate:
    def test_date_string(self):
        result = extract_date({"date": "2026-03-06"})
        assert result is not None
        assert "2026-03-06" in result

    def test_datetime_string(self):
        result = extract_date({"date": "2026-03-06T10:30:00"})
        assert result is not None
        assert "2026-03-06" in result

    def test_date_object(self):
        from datetime import date

        result = extract_date({"date": date(2026, 3, 6)})
        assert result is not None
        assert "2026-03-06" in result

    def test_datetime_object(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 6, 10, 30, tzinfo=timezone.utc)
        result = extract_date({"date": dt})
        assert result is not None
        assert "2026-03-06" in result

    def test_priority_order(self):
        result = extract_date({"created": "2020-01-01", "date": "2026-03-06"})
        assert "2026-03-06" in result

    def test_fallback_to_created(self):
        result = extract_date({"created": "2026-03-06"})
        assert result is not None
        assert "2026-03-06" in result

    def test_no_date(self):
        assert extract_date({}) is None

    def test_invalid_date(self):
        assert extract_date({"date": "not a date at all xyz"}) is None


class TestExtractWikilinks:
    def test_simple_link(self):
        text, links = extract_wikilinks("See [[Alice]] for details")
        assert text == "See Alice for details"
        assert links == [{"target": "Alice", "alias": None}]

    def test_aliased_link(self):
        text, links = extract_wikilinks("Talk to [[Alice Smith|Alice]]")
        assert text == "Talk to Alice"
        assert links == [{"target": "Alice Smith", "alias": "Alice"}]

    def test_multiple_links(self):
        text, links = extract_wikilinks("[[Alice]] and [[Bob]]")
        assert len(links) == 2
        assert text == "Alice and Bob"

    def test_no_links(self):
        text, links = extract_wikilinks("No links here")
        assert text == "No links here"
        assert links == []


class TestExtractEmbeds:
    def test_image_embed(self):
        text, embeds = extract_embeds("![[image.png]]")
        assert embeds == [{"target": "image.png"}]
        assert "image.png" not in text

    def test_note_embed(self):
        text, embeds = extract_embeds("See also: ![[other-note]]")
        assert embeds == [{"target": "other-note"}]

    def test_no_embeds(self):
        text, embeds = extract_embeds("No embeds")
        assert embeds == []
        assert text == "No embeds"


class TestExtractInlineTags:
    def test_simple_tags(self):
        tags = extract_inline_tags("This is #python and #rust")
        assert "python" in tags
        assert "rust" in tags

    def test_tag_with_slash(self):
        tags = extract_inline_tags("This is #lang/python")
        assert "lang/python" in tags

    def test_no_tags(self):
        assert extract_inline_tags("No tags here") == []

    def test_deduplication(self):
        tags = extract_inline_tags("#python #python #python")
        assert tags.count("python") == 1


class TestExtractCalloutTypes:
    def test_note_callout(self):
        types = extract_callout_types("> [!NOTE] Important\n> some content")
        assert "note" in types

    def test_warning_callout(self):
        types = extract_callout_types("> [!WARNING] Be careful")
        assert "warning" in types

    def test_multiple_callouts(self):
        text = "> [!NOTE] First\n> content\n\n> [!TIP] Second\n> more"
        types = extract_callout_types(text)
        assert "note" in types
        assert "tip" in types

    def test_no_callouts(self):
        assert extract_callout_types("Just normal text") == []


class TestParseNote:
    def test_full_frontmatter(self):
        content = """\
---
title: My Note
tags: [python, testing]
date: 2026-03-06
aliases: [my-note]
---

This is the body with #inline-tag and [[Some Link]].
"""
        result = parse_note(content)
        assert result.title == "My Note"
        assert result.date is not None
        assert "python" in result.frontmatter_tags
        assert "testing" in result.frontmatter_tags
        assert "inline-tag" in result.inline_tags
        assert result.aliases == ["my-note"]
        assert len(result.wikilinks) == 1
        assert result.wikilinks[0]["target"] == "Some Link"
        assert "[[" not in result.body  # wikilinks replaced with plain text

    def test_no_frontmatter(self):
        content = "Just plain markdown\n\nWith some paragraphs."
        result = parse_note(content)
        assert result.frontmatter == {}
        assert result.body == "Just plain markdown\n\nWith some paragraphs."
        assert result.title is None
        assert result.tags == []

    def test_malformed_frontmatter(self):
        content = """\
---
title: Bad YAML
tags: [unclosed
---

Body content here.
"""
        result = parse_note(content)
        assert result.body  # should still have body
        assert result.frontmatter == {}  # fallback to empty

    def test_tags_as_string(self):
        content = """\
---
tags: single-tag
---

Body.
"""
        result = parse_note(content)
        assert "single-tag" in result.tags

    def test_aliases_as_string(self):
        content = """\
---
aliases: my-alias
---

Body.
"""
        result = parse_note(content)
        assert result.aliases == ["my-alias"]

    def test_tag_deduplication(self):
        content = """\
---
tags: [python]
---

This has #python inline too.
"""
        result = parse_note(content)
        assert result.tags.count("python") == 1

    def test_embeds_stripped(self):
        content = """\
---
title: Test
---

Some text ![[image.png]] more text.
"""
        result = parse_note(content)
        assert len(result.embeds) == 1
        assert "![[" not in result.body

    def test_recipe_frontmatter_preserved(self):
        content = """\
---
title: Pasta Carbonara
cuisine: Italian
prep_time: 10 min
cook_time: 20 min
servings: 4
lissy_approved: true
tags: [pasta, quick]
source: https://example.com
---

## Ingredients
- Pasta
- Eggs
"""
        result = parse_note(content)
        assert result.title == "Pasta Carbonara"
        assert result.frontmatter["cuisine"] == "Italian"
        assert result.frontmatter["lissy_approved"] is True
        assert "pasta" in result.tags
        assert "quick" in result.tags

    def test_person_frontmatter(self):
        content = """\
---
name: Alice Smith
company: Acme Corp
role: Engineer
email: alice@acme.com
how_met: Conference 2025
tags: [engineering, friend]
---

Met at PyCon. Very knowledgeable about async Python.
"""
        result = parse_note(content)
        assert result.frontmatter["name"] == "Alice Smith"
        assert result.frontmatter["company"] == "Acme Corp"


class TestExtractTasks:
    def test_simple_open_task(self):
        tasks = extract_tasks("- [ ] Buy groceries")
        assert len(tasks) == 1
        assert tasks[0].description == "Buy groceries"
        assert tasks[0].done is False

    def test_completed_task(self):
        tasks = extract_tasks("- [x] Buy groceries")
        assert len(tasks) == 1
        assert tasks[0].done is True

    def test_cancelled_task(self):
        tasks = extract_tasks("- [-] Buy groceries")
        assert len(tasks) == 1
        assert tasks[0].cancelled is True

    def test_due_date(self):
        tasks = extract_tasks("- [ ] Buy groceries 📅 2026-03-10")
        assert tasks[0].due == "2026-03-10"
        assert "📅" not in tasks[0].description

    def test_scheduled_date(self):
        tasks = extract_tasks("- [ ] Review PR ⏳ 2026-03-08")
        assert tasks[0].scheduled == "2026-03-08"

    def test_start_date(self):
        tasks = extract_tasks("- [ ] Start project 🛫 2026-03-07")
        assert tasks[0].start == "2026-03-07"

    def test_created_date(self):
        tasks = extract_tasks("- [ ] New idea ➕ 2026-03-06")
        assert tasks[0].created == "2026-03-06"

    def test_done_date(self):
        tasks = extract_tasks("- [x] Finished task ✅ 2026-03-05")
        assert tasks[0].done_date == "2026-03-05"

    def test_cancelled_date(self):
        tasks = extract_tasks("- [-] Dropped task ❌ 2026-03-04")
        assert tasks[0].cancelled_date == "2026-03-04"

    def test_priority_highest(self):
        tasks = extract_tasks("- [ ] Urgent thing 🔺")
        assert tasks[0].priority == "highest"

    def test_priority_high(self):
        tasks = extract_tasks("- [ ] Important thing ⏫")
        assert tasks[0].priority == "high"

    def test_priority_medium(self):
        tasks = extract_tasks("- [ ] Normal thing 🔼")
        assert tasks[0].priority == "medium"

    def test_priority_low(self):
        tasks = extract_tasks("- [ ] Minor thing 🔽")
        assert tasks[0].priority == "low"

    def test_priority_lowest(self):
        tasks = extract_tasks("- [ ] Someday thing ⏬")
        assert tasks[0].priority == "lowest"

    def test_recurrence(self):
        tasks = extract_tasks("- [ ] Water plants 🔁 every week")
        assert tasks[0].recurrence == "every week"

    def test_multiple_fields(self):
        tasks = extract_tasks(
            "- [ ] #task Submit report ⏫ 📅 2026-03-10 ⏳ 2026-03-08 ➕ 2026-03-01"
        )
        assert len(tasks) == 1
        t = tasks[0]
        assert t.description == "Submit report"
        assert t.priority == "high"
        assert t.due == "2026-03-10"
        assert t.scheduled == "2026-03-08"
        assert t.created == "2026-03-01"

    def test_multiple_tasks(self):
        text = "- [ ] Task one\n- [x] Task two\n- [ ] Task three"
        tasks = extract_tasks(text)
        assert len(tasks) == 3
        assert tasks[1].done is True

    def test_no_tasks(self):
        assert extract_tasks("Just regular text\n\nNo checkboxes here") == []

    def test_strips_task_tag(self):
        tasks = extract_tasks("- [ ] #task Buy milk")
        assert tasks[0].description == "Buy milk"

    def test_task_in_parsed_note(self):
        content = """\
---
title: Shopping
---

## Groceries
- [ ] Milk 📅 2026-03-10
- [ ] Eggs
- [x] Bread ✅ 2026-03-05

## Hardware
- [ ] Nails ⏫
"""
        result = parse_note(content)
        assert len(result.tasks) == 4
        open_tasks = [t for t in result.tasks if not t.done]
        assert len(open_tasks) == 3
        milk = next(t for t in result.tasks if "Milk" in t.description)
        assert milk.due == "2026-03-10"
        nails = next(t for t in result.tasks if "Nails" in t.description)
        assert nails.priority == "high"


class TestSplitByHeadings:
    def test_small_file_no_split(self):
        body = "Short content"
        sections = split_by_headings(body, threshold_bytes=8192)
        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].content == body
        assert sections[0].index == 0
        assert sections[0].total == 1

    def test_split_by_headings(self):
        # Create content that exceeds threshold
        body = "# Section One\n\n" + ("Content A. " * 200) + "\n\n"
        body += "# Section Two\n\n" + ("Content B. " * 200)

        sections = split_by_headings(body, threshold_bytes=100)
        assert len(sections) == 2
        assert sections[0].heading == "Section One"
        assert sections[1].heading == "Section Two"
        assert sections[0].index == 0
        assert sections[1].index == 1
        assert sections[0].total == 2

    def test_content_before_first_heading(self):
        body = "Intro text\n\n# Section One\n\nContent here"
        sections = split_by_headings(body, threshold_bytes=10)
        assert sections[0].heading is None
        assert "Intro text" in sections[0].content
        assert sections[1].heading == "Section One"

    def test_no_headings_splits_by_paragraphs(self):
        body = ("Paragraph one. " * 50) + "\n\n" + ("Paragraph two. " * 50)
        sections = split_by_headings(body, threshold_bytes=100)
        assert len(sections) >= 2
        assert all(s.heading is None for s in sections)

    def test_preserves_heading_in_content(self):
        body = "# My Heading\n\nBody text here"
        sections = split_by_headings(body, threshold_bytes=10)
        assert "# My Heading" in sections[0].content
