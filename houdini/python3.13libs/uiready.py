"""Houdini runs this in a session with an interface, once that is up.

`ready.py` next to this file is the hook that runs in every kind of session.
This one is here for a build where that runs before the interface: the start
is guarded, so whichever gets there first is the only one that starts a
bridge.
"""

import nscr_mcp_autostart

nscr_mcp_autostart.start_if_wanted()
