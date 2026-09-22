"""Houdini runs this every time a scene is loaded.

A second chance, not the route in: `python3.13libs/ready.py` is what starts a
bridge. This matters only for a session that is somehow up with no bridge and
then opens a scene. A session that already has one keeps it.
"""

import nscr_mcp_autostart

nscr_mcp_autostart.start_if_wanted()
