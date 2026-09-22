"""Houdini runs this once at startup, before any scene is loaded.

This is the route for a session with no interface, which is where a startup
script is the only hook there is. A session with an interface waits for the
interface instead, in `python3.13libs/uiready.py`.
"""

import hou
import nscr_mcp_autostart

if not hou.isUIAvailable():
    nscr_mcp_autostart.start_if_wanted()
