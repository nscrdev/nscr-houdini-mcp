"""Houdini runs this every time a scene is loaded.

A session that already has a bridge keeps it, so this only matters for a
session that opened a scene before anything else started one.
"""

import nscr_mcp_autostart

nscr_mcp_autostart.start_if_wanted()
