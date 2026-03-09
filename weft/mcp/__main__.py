"""Entry point for the MCP server: python -m weft.mcp"""

import os

from weft.mcp import mcp
from weft.mcp.server import user_identity_middleware

transport = os.environ.get("WEFT_TRANSPORT", "stdio")
port = int(os.environ.get("PORT", "8000"))

if transport == "stdio":
    mcp.run(transport="stdio")
else:
    mcp.run(
        transport=transport,
        host="0.0.0.0",
        port=port,
        middleware=[user_identity_middleware],
    )
