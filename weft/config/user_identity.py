"""User identity module for Weft V2.

Generates and persists a UUID to ~/.weft/user_id.json on first call, then
returns the same UUID on subsequent calls. This UUID is never hardcoded and
works identically for local and hosted installations.
"""

import json
import uuid
from pathlib import Path


def get_user_id() -> str:
    """Get or create the persistent user ID for this installation.

    Returns the same UUID across multiple calls (idempotent). UUID is stored
    in ~/.weft/user_id.json.
    """
    config_dir = Path.home() / ".weft"
    config_dir.mkdir(parents=True, exist_ok=True)
    user_id_file = config_dir / "user_id.json"

    if user_id_file.exists():
        return json.loads(user_id_file.read_text())["user_id"]

    new_id = uuid.uuid4().hex
    user_id_file.write_text(json.dumps({"user_id": new_id}, indent=2))
    return new_id
