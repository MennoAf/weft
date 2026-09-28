# Contributing

## Before you start

Read `AGENTS.md`, `pyproject.toml`, and the `tests/` layout before changing code or tests. For substantial changes, open an issue to discuss the proposal and scope before starting implementation.

## Development setup

Weft requires Python 3.12 or later. From the repository root, install the project and development dependencies with:

```bash
uv sync
```

Run the test suite with:

```bash
uv run pytest tests/ -v
```

Tests that require PostgreSQL or other external services must use local or test-only containers and disposable test data. Never point tests at production services, use production data, or use production secrets.

## Pull requests

Please describe:

- The user-visible behavior changed or added.
- Any configuration or compatibility impact.
- Security and privacy implications.
- The tests and checks you ran, including any that you could not run.

Update `CHANGELOG.md` for user-visible changes.

Do not commit `.env` files, personal MCP configuration, memory exports, datasets, or test/run outputs. Keep credentials and personal data out of commits, issue discussions, and pull requests.

## Conduct

Be respectful and constructive. Maintainers may moderate participation to keep the project and its discussions usable for everyone.
