"""Generate Obsidian vault folder structure and frontmatter templates."""

from __future__ import annotations

from pathlib import Path

VAULT_DIRS = [
    "inbox",
    "notes",
    "journal/daily",
    "people",
    "wktw/clients",
    "wktw/meetings",
    "wktw/ideas",
    "wktw/finances",
    "wktw/finances/income",
    "wktw/finances/expenses",
    "wktw/operations",
    "tools",
    "recipes",
    "media",
    "writing/ideas",
    "writing/blog",
    "assets",
    "templates",
]

TEMPLATES: dict[str, str] = {
    "templates/Inbox.md": """\
---
created: {{date}}
---

""",
    "templates/Note.md": """\
---
created: {{date}}
due:
tags: []
---

""",
    "templates/Daily Note.md": """\
---
date: {{date}}
mood:
tags: []
---

## What happened today


## Thoughts


## Tomorrow

""",
    "templates/Person.md": """\
---
name:
company:
role:
email:
phone:
how_met:
tags: []
---

## Notes

""",
    "templates/WKTW General.md": """\
---
created: {{date}}
category:
status: active
tags: []
---

""",
    "templates/WKTW Client.md": """\
---
created: {{date}}
client_name:
contact:
status: active
tags: []
---

## Overview


## Notes

""",
    "templates/Recipe.md": """\
---
title:
cuisine:
prep_time:
cook_time:
servings:
lissy_approved:
tags: []
source:
---

## Ingredients


## Instructions


## Notes

""",
    "templates/Media.md": """\
---
title:
type:
author:
date_finished:
rating:
tags: []
---

## Summary


## Thoughts


## Quotes

""",
    "templates/Writing Idea.md": """\
---
created: {{date}}
genre:
status: seed
tags: []
---

## Premise


## Notes

""",
    "templates/Blog Post.md": """\
---
title:
created: {{date}}
status: idea
published_url:
tags: []
---

""",
    "templates/Tool Note.md": """\
---
name: "{{title}}"
description:
readme:
created_by_me:
category:
development_status:
is_public:
tags: []
---
## What Is The Core Function


## Known Integrations


## Notes
""",
    "templates/WKTW Income.md": """\
---
created: {{date}}
client:
amount:
date_received:
invoice_id:
category: income
tags:
  - wktw/finances
---

## Notes

""",
    "templates/WKTW Expense.md": """\
---
created: {{date}}
vendor:
amount:
date_paid:
category:
recurring:
tags:
  - wktw/finances
---

## Notes

""",
}


def init_vault(vault_path: Path) -> dict:
    """Create vault folder structure and templates.

    Returns a summary of what was created.
    """
    vault_path = Path(vault_path)
    dirs_created = 0
    templates_created = 0

    for d in VAULT_DIRS:
        dir_path = vault_path / d
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            dirs_created += 1

    for rel_path, content in TEMPLATES.items():
        tmpl_path = vault_path / rel_path
        if not tmpl_path.exists():
            tmpl_path.write_text(content, encoding="utf-8")
            templates_created += 1

    return {
        "vault_path": str(vault_path),
        "dirs_created": dirs_created,
        "templates_created": templates_created,
        "total_dirs": len(VAULT_DIRS),
        "total_templates": len(TEMPLATES),
    }
