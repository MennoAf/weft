"""Google Calendar credential management and service builder.

Handles OAuth2 credential persistence, automatic token refresh,
and construction of an authenticated Calendar API v3 service.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

__all__ = [
    "CREDENTIALS_PATH",
    "SCOPES",
    "get_credentials",
    "build_service",
    "run_oauth_flow",
]

CREDENTIALS_PATH = Path.home() / ".weft" / "google_credentials.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def get_credentials() -> Credentials:
    """Load and return valid Google OAuth2 credentials.

    Handles three states:
    1. Credentials file missing — raises RuntimeError
    2. Credentials expired with refresh token — refreshes and persists
    3. Credentials valid — returns as-is
    """
    if not CREDENTIALS_PATH.exists():
        raise RuntimeError(
            f"Credentials not found at {CREDENTIALS_PATH}. "
            "Run `weft calendar-auth` to authenticate."
        )

    creds = Credentials.from_authorized_user_file(str(CREDENTIALS_PATH), SCOPES)

    if creds.expired:
        if not creds.refresh_token:
            raise RuntimeError(
                "Google credentials expired and no refresh token available. "
                "Run `weft calendar-auth` to re-authenticate."
            )
        creds.refresh(Request())
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CREDENTIALS_PATH.write_text(creds.to_json())

    return creds


def build_service():
    """Build an authenticated Google Calendar API v3 service."""
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def run_oauth_flow(client_secrets_path: str | Path) -> None:
    """Run the OAuth2 installed app flow and persist credentials.

    Args:
        client_secrets_path: Path to the Google OAuth client secrets JSON file.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), SCOPES)
    creds = flow.run_local_server(port=0)

    if not creds or not creds.valid:
        raise RuntimeError("OAuth flow did not complete successfully.")

    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(creds.to_json())
    CREDENTIALS_PATH.chmod(0o600)
