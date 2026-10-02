# Changelog

Notable changes to Weft are documented here.

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the project version in `pyproject.toml`.

## [Unreleased]

## [1.0.0rc1] - release date TBD

### Added

- An MCP server and Python package for persistent agent memory.
- PostgreSQL/pgvector-backed memory storage and retrieval, with authority scoping.
- Replay and consolidation capabilities for memory maintenance.

### Changed

- Hardened session priming to report degraded operation, isolate section failures, bound its context budget, and warn about near-miss project selection.
- Added explicit startup and shutdown deadlines for deployment lifecycle operations.
