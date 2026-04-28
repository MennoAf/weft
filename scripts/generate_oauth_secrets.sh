#!/usr/bin/env bash
# generate_oauth_secrets.sh — mint the OAuth secrets for a Weft Route 1
# deploy (staging or prod).
#
# Produces:
#   OAUTH_JWT_PRIVATE_KEY_PEM  — RSA-2048 PKCS8 PEM for access/refresh
#                                token signing (RS256).
#   OAUTH_SESSION_SECRET       — 48-byte base64url HMAC secret for the
#                                itsdangerous-signed /oauth/authorize
#                                pending cookie.
#
# Writes both to a temporary file under $HOME/.weft/ (gitignored via
# repo convention) AND prints the Fly `fly secrets set` lines you need
# to paste. This script does NOT call fly itself — Jason runs every
# secret mutation by hand.
#
# Usage:
#   ./scripts/generate_oauth_secrets.sh                    # staging app
#   ./scripts/generate_oauth_secrets.sh weft-mcp           # prod app
#   ./scripts/generate_oauth_secrets.sh --output <path>    # custom output
#
# The output file is a bash-sourceable env file:
#   source .oauth-staging-secrets.env
#   fly secrets import < .oauth-staging-secrets.env -a weft-mcp-staging
#
# Security: the file contains a private RSA key. `chmod 600` is applied.
# DO NOT commit it. The standard .gitignore entry for `.oauth-*.env` keeps
# it out of the repo but double-check before every commit.

set -euo pipefail

APP="${1:-weft-mcp-staging}"
OUTPUT_DIR="${HOME}/.weft"
OUTPUT_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            APP="$1"
            shift
            ;;
    esac
done

if [ -z "$OUTPUT_FILE" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="${OUTPUT_DIR}/.oauth-${APP}-secrets.env"
fi

# --- openssl check --------------------------------------------------------
if ! command -v openssl >/dev/null 2>&1; then
    echo "error: openssl is required" >&2
    exit 1
fi

# --- Generate the RSA-2048 private key ------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

openssl genrsa -out "${TMPDIR}/private.pem" 2048 2>/dev/null
# PKCS8 — what pyjwt / cryptography prefer.
openssl pkcs8 -topk8 -nocrypt \
    -in "${TMPDIR}/private.pem" \
    -out "${TMPDIR}/private.pkcs8.pem"

PRIVATE_PEM="$(cat "${TMPDIR}/private.pkcs8.pem")"

# --- Generate the session secret (48 bytes base64url, ~64 chars) ----------
SESSION_SECRET="$(openssl rand -base64 48 | tr -d '=' | tr '/+' '_-')"

# --- Write the env file ---------------------------------------------------
umask 077
cat > "$OUTPUT_FILE" <<EOF
# Weft OAuth 2.1 staging secrets — generated $(date -u +%Y-%m-%dT%H:%M:%SZ)
# APP: $APP
#
# Source this file to get the variables into your shell:
#     source "$OUTPUT_FILE"
# Or import all at once:
#     fly secrets import < "$OUTPUT_FILE" -a $APP
#
# DO NOT COMMIT THIS FILE.

OAUTH_JWT_PRIVATE_KEY_PEM='$PRIVATE_PEM'
OAUTH_SESSION_SECRET='$SESSION_SECRET'
EOF
chmod 600 "$OUTPUT_FILE"

# --- Print the runbook ----------------------------------------------------
cat <<EOF
=========================================================================
OAuth secrets generated for Fly app: $APP
=========================================================================

Output file:  $OUTPUT_FILE  (chmod 600)

Next steps:

1. Inspect the file:
       cat "$OUTPUT_FILE"

2. Load into Fly (preferred — single atomic import):
       fly secrets import < "$OUTPUT_FILE" -a $APP

   OR set individually:
       source "$OUTPUT_FILE"
       fly secrets set OAUTH_JWT_PRIVATE_KEY_PEM="\$OAUTH_JWT_PRIVATE_KEY_PEM" -a $APP
       fly secrets set OAUTH_SESSION_SECRET="\$OAUTH_SESSION_SECRET" -a $APP

3. Set the remaining OAuth env vars that DON'T need to be secret, but
   must still point at the right URLs for \$APP:
       fly secrets set OAUTH_ISSUER="https://${APP}.fly.dev" -a $APP
       fly secrets set SUPABASE_URL="https://YOUR_PROJECT.supabase.co" -a $APP
       # For single-user mode only (leave unset for multi-user):
       fly secrets set OAUTH_SOLE_USER_SUB="<your Supabase sub UUID>" -a $APP

4. Shred the local file after the secrets are live:
       rm -P "$OUTPUT_FILE"   # or use 'shred -u' on Linux

=========================================================================
EOF
