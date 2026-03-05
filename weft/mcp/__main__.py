"""Entry point for the MCP server: python -m weft.mcp"""

import os

from weft.mcp import mcp

transport = os.environ.get("WEFT_TRANSPORT", "stdio")
port = int(os.environ.get("PORT", "8000"))

mcp.run(transport=transport, host="0.0.0.0", port=port)
