#!/usr/bin/env bash
set -euo pipefail

# Detached, resumable LongMemEval runner.
# Usage:
#   ./benchmarks/longmemeval/run_detached.sh <dataset> <output-file> [adapter args...]
#
# The script intentionally loads only API keys from .env. It does not source the
# whole file because DATABASE_URL may contain shell metacharacters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:?dataset path required}"
OUTPUT_FILE="${2:?stable output JSONL path required}"
shift 2

if [[ ! -f "$DATASET" ]]; then
  printf 'dataset not found: %s\n' "$DATASET" >&2
  exit 2
fi

RUN_DIR="${ROOT_DIR}/benchmarks/longmemeval/runs"
mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_FILE")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${RUN_DIR}/run_${STAMP}.log"
PID_FILE="${RUN_DIR}/run_${STAMP}.pid"
META_FILE="${RUN_DIR}/run_${STAMP}.meta"

ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in
      OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENAI_ORGANIZATION)
        value="${value%$'\r'}"
        value="${value#\"}"; value="${value%\"}"
        export "${key}=${value}"
        ;;
    esac
  done < <(grep -E '^(OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENAI_ORGANIZATION)=' "$ENV_FILE" || true)
fi

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required (set it in .env or the environment)}"

CMD=(
  uv run python -m benchmarks.longmemeval.adapter
  --dataset "$DATASET"
  --mode turns
  --tier turns
  --output-file "$OUTPUT_FILE"
  --resume
  --log-level INFO
  "$@"
)
printf 'started_at=%s\noutput_file=%s\nlog_file=%s\n' \
  "$STAMP" "$OUTPUT_FILE" "$LOG_FILE" > "$META_FILE"
printf 'command=' >> "$META_FILE"
printf '%q ' "${CMD[@]}" >> "$META_FILE"
printf '\n' >> "$META_FILE"

cd "$ROOT_DIR"
if command -v setsid >/dev/null 2>&1; then
  setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
fi
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

printf 'pid=%s\nlog=%s\noutput=%s\nmeta=%s\n' \
  "$PID" "$LOG_FILE" "$OUTPUT_FILE" "$META_FILE"
printf 'monitor: tail -f %q\n' "$LOG_FILE"
printf 'status: kill -0 %s\n' "$PID"
