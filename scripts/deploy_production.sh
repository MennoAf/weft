#!/usr/bin/env bash
# Deploy the canonical production ref only after the repository guard passes.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
uv run python scripts/deploy_production_guard.py
exec fly deploy --app weft-mcp --config fly.toml "$@"
