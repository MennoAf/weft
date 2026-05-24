"""Discord connector for outbound events (e.g. daily brief).

Importing this package registers the Discord outbound handler with
weft.scheduler._OUTBOUND_EVENT_REGISTRY. The handler dispatches via the
module-level bot reference set by `discord_bot_loop` in weft.scheduler.
"""

from weft.discord import connector as _connector  # noqa: F401  — registration side-effect
