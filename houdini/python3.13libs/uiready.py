"""Houdini runs this in a session with an interface, once that is up.

Starting a server before the interface exists is asking for trouble, so a GUI
session starts its bridge from here rather than from a startup script.
"""

import nscr_mcp_autostart

nscr_mcp_autostart.start_if_wanted()
