#!/usr/bin/env bash
# generate_oauth_secrets.sh — generate local OAuth secrets for a Weft
# self-hosted deployment.
#
# Produces:
#   OAUTH_JWT_PRIVATE_KEY_PEM  — RSA-2048 PKCS8 PEM for access/refresh
#                                token signing (RS256).
#   OAUTH_SESSION_SECRET       — 48-byte base64url HMAC secret for the
#                                itsdangerous-signed /oauth/authorize
#                                pending cookie.
#
# Writes both secrets to a chmod-600 env file under $HOME/.weft/ by default.
# The output path can be overridden for an operator-managed secret store. This
# script never contacts a deployment provider or uploads a secret.
#
# Usage:
#   ./scripts/generate_oauth_secrets.sh
#   ./scripts/generate_oauth_secrets.sh --output <path>
#
# The output file is a bash-sourceable env file for the operator's deployment.
#
# Security: the file contains a private RSA key. `chmod 600` is applied.
# DO NOT commit it. The standard .gitignore entry for `.oauth-*.env` keeps
# it out of the repo but double-check before every commit.

set -euo pipefail

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
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$OUTPUT_FILE" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="${OUTPUT_DIR}/.oauth-secrets.env"
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
# Weft OAuth 2.1 secrets — generated $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Source this file only in the operator's protected deployment environment.
# DO NOT COMMIT THIS FILE.

OAUTH_JWT_PRIVATE_KEY_PEM='$PRIVATE_PEM'
OAUTH_SESSION_SECRET='$SESSION_SECRET'
EOF
chmod 600 "$OUTPUT_FILE"

# --- Print the runbook ----------------------------------------------------
cat <<EOF
=========================================================================
Weft OAuth secrets generated
=========================================================================

Output file:  $OUTPUT_FILE  (chmod 600)

Next steps:

1. Load the variables into your protected deployment secret store.
2. Configure the OAuth issuer, redirect URLs, and deployment-specific values.
3. Delete the local file after the secrets are stored securely:
       rm -P "$OUTPUT_FILE"   # or use 'shred -u' on Linux

Never commit this file or pass its contents through logs, chat, or a shell
history that is shared with other users.
=========================================================================
EOF
