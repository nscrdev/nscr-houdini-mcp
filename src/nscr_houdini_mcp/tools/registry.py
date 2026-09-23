"""Every tool the server offers, in the order the tool list presents them.

The order is fixed and the list is the same for every client and every
connection, so a client can cache it. A new tool is added here and nowhere
else.
"""

from __future__ import annotations

from nscr_houdini_mcp.tools.base import ToolSpec
from nscr_houdini_mcp.tools.capture import HOU_CAPTURE
from nscr_houdini_mcp.tools.inspect import HOU_INSPECT
from nscr_houdini_mcp.tools.jobs import HOU_JOBS
from nscr_houdini_mcp.tools.ping import HOU_PING
from nscr_houdini_mcp.tools.python import HOU_PYTHON
from nscr_houdini_mcp.tools.scene import HOU_SCENE
from nscr_houdini_mcp.tools.sessions import HOU_SESSIONS

TOOLS: tuple[ToolSpec, ...] = (
    HOU_PING,
    HOU_SESSIONS,
    HOU_SCENE,
    HOU_INSPECT,
    HOU_PYTHON,
    HOU_JOBS,
    HOU_CAPTURE,
)
