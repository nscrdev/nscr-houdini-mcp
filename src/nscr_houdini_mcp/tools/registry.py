"""Every tool the server offers, in the order the tool list presents them.

The order is fixed and the list is the same for every client and every
connection, so a client can cache it. A new tool is added here and nowhere
else.
"""

from __future__ import annotations

from nscr_houdini_mcp.tools.base import ToolSpec
from nscr_houdini_mcp.tools.ping import HOU_PING

TOOLS: tuple[ToolSpec, ...] = (HOU_PING,)
