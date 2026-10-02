#!/usr/bin/env bash
# v3 compatibility adapter. The Python runner is the sole normal resource owner.
set -Eeuo pipefail

JOB_DIR="${WEFT_ACCEPTANCE_JOB_DIR:?set an isolated, non-existing job directory}"
SOURCE_DIR="${WEFT_ACCEPTANCE_SOURCE_DIR:?set an extracted source directory}"
IMAGE="${WEFT_ACCEPTANCE_IMAGE:?set a candidate image tag}"
TIMEOUT="${WEFT_ACCEPTANCE_TIMEOUT:-120}"
TOTAL_TIMEOUT="${WEFT_ACCEPTANCE_TOTAL_TIMEOUT:-2400}"
FINALIZATION_RESERVE="${WEFT_ACCEPTANCE_FINALIZATION_RESERVE:-240}"
COMPOSE_RESERVE="${WEFT_ACCEPTANCE_COMPOSE_CLEANUP_RESERVE:-120}"
PUBLICATION_RESERVE="${WEFT_ACCEPTANCE_PUBLICATION_RESERVE:-60}"
CLEANUP_TIMEOUT="${WEFT_ACCEPTANCE_CLEANUP_TIMEOUT:-30}"
CLEANUP_KILL_GRACE="${WEFT_ACCEPTANCE_CLEANUP_KILL_GRACE:-1}"
PYTHON="${WEFT_ACCEPTANCE_PYTHON:-$(command -v python3 || true)}"
DOCKER="${WEFT_ACCEPTANCE_DOCKER:-$(command -v docker || true)}"
COMPOSE_FILE="$SOURCE_DIR/docker-compose.local.yml"
RECEIPT="$JOB_DIR/receipt.json"
STATUS="$JOB_DIR/status.txt"
LOG="$JOB_DIR/acceptance.log"
RUNNER="$SOURCE_DIR/scripts/local_docker_acceptance.py"

[[ "$JOB_DIR" = /* && "$SOURCE_DIR" = /* ]] || { echo 'FAIL: absolute isolated paths required' >&2; exit 64; }
[[ ! -e "$JOB_DIR" ]] || { echo 'FAIL: refusing pre-existing job directory' >&2; exit 65; }
[[ -f "$COMPOSE_FILE" && -f "$RUNNER" ]] || { echo 'FAIL: acceptance source files missing' >&2; exit 65; }
[[ "$IMAGE" =~ ^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$ ]] || { echo 'FAIL: unsafe image tag' >&2; exit 66; }
[[ "$PYTHON" = /* && -f "$PYTHON" && -x "$PYTHON" ]] || { echo 'FAIL: absolute executable Python required' >&2; exit 68; }
[[ "$DOCKER" = /* && -f "$DOCKER" && -x "$DOCKER" ]] || { echo 'FAIL: absolute executable Docker required' >&2; exit 69; }
for value in "$TIMEOUT" "$TOTAL_TIMEOUT" "$FINALIZATION_RESERVE" "$COMPOSE_RESERVE" "$PUBLICATION_RESERVE" "$CLEANUP_TIMEOUT" "$CLEANUP_KILL_GRACE"; do
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'FAIL: numeric timeout/reserve required' >&2; exit 67; }
done
awk -v n="$TIMEOUT" -v t="$TOTAL_TIMEOUT" -v f="$FINALIZATION_RESERVE" -v c="$COMPOSE_RESERVE" -v w="$PUBLICATION_RESERVE" -v d="$CLEANUP_TIMEOUT" -v k="$CLEANUP_KILL_GRACE" 'BEGIN { exit !(n > 0 && t > 0 && f > c+w && f < t && c > 0 && w > 0 && d > 0 && k >= 0 && d+k+20 < c) }' || {
  echo 'FAIL: invalid aggregate envelope or cleanup compatibility mapping' >&2; exit 67;
}

mkdir -p "$JOB_DIR"
RUN_ID="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(6))')"
PROJECT="weft-rc-$RUN_ID"
DOCKER_DIR="${DOCKER%/*}"; DOCKER_DIR="${DOCKER_DIR:-/}"
# Adapter owns only validation, logging, and argv forwarding; cleanup is runner-owned.
exec env -i HOME=/root PATH="$DOCKER_DIR" WEFT_OUTBOUND_CONNECTOR=none WEFT_QUARANTINE_REVIEW_ENABLED=0 \
  "$PYTHON" "$RUNNER" \
  --image "$IMAGE" --run-id "$RUN_ID" --project-name "$PROJECT" \
  --compose-file "$COMPOSE_FILE" --receipt "$RECEIPT" --status "$STATUS" \
  --timeout "$TIMEOUT" --total-timeout "$TOTAL_TIMEOUT" \
  --finalization-reserve "$FINALIZATION_RESERVE" \
  --compose-cleanup-reserve "$COMPOSE_RESERVE" --publication-reserve "$PUBLICATION_RESERVE" \
  --compose-cleanup-timeout "$CLEANUP_TIMEOUT" --cleanup-kill-grace "$CLEANUP_KILL_GRACE" \
  --docker-executable "$DOCKER" >"$LOG" 2>&1
