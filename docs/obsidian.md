# Obsidian Integration

Weft can sync an Obsidian vault into memories, making your personal notes, tasks, contacts, recipes, and more available to AI agents via semantic search.

## Quickstart

```bash
# Create a vault with Weft-optimized folder structure and templates
weft obsidian init ~/Documents/MyVault

# Sync vault contents into Weft as memories
weft obsidian sync ~/Documents/MyVault

# Preview what would be synced (safe, no writes)
weft obsidian sync ~/Documents/MyVault --dry-run
```

## Vault structure

`weft obsidian init` creates an opinionated folder structure with frontmatter templates:

```
vault/
  inbox/              Quick capture (lower confidence)
  notes/              Reminders, misc notes
  journal/daily/      Daily notes
  people/             Contacts (stored as user_model type)
  wktw/               Side business
    clients/
    meetings/
    ideas/
    finances/
    operations/
  recipes/            Meals (with lissy_approved field)
  media/              Book/show/movie reviews
  writing/
    ideas/            Creative writing concepts
    blog/             Blog posts and drafts
  assets/             Images, attachments (ignored)
  templates/          Frontmatter templates (ignored)
```

Folders map to Weft memory types and topics automatically. Frontmatter `type:` and `confidence:` fields override the defaults.

## Obsidian Tasks plugin

Weft parses [Obsidian Tasks](https://github.com/obsidian-tasks-group/obsidian-tasks) checkboxes with full emoji support:

| Emoji | Field |
|-------|-------|
| 📅 | Due date |
| ⏳ | Scheduled date |
| 🛫 | Start date |
| ➕ | Created date |
| ✅ | Done date |
| ❌ | Cancelled date |
| 🔁 | Recurrence |
| 🔺 ⏫ 🔼 🔽 ⏬ | Priority (highest to lowest) |

Open tasks are included in the memory content with their dates and priority, making them available for summary prompts and planning.

## Sync behavior

- **Hash-based change detection** — re-running sync skips unchanged files
- **Modified files** — old memories archived, new ones created
- **Deleted files** — memories archived on next sync
- **Large files** — split by heading hierarchy (falls back to paragraphs)
- **Frontmatter** — wikilinks, tags, dates, and type-specific fields (recipes, contacts, media) are extracted and included in memory content

## Custom vaults

The folder taxonomy above is opinionated, not required. You can sync any Obsidian vault — Weft will parse files with frontmatter and apply the default `fact` type to any folder it doesn't recognize. Add per-file overrides in frontmatter:

```markdown
---
type: decision
confidence: 0.9
topic: [architecture, auth]
pinned: true
---

# Why we picked Supabase

...
```
