"""Houdini runs this once it is up, in every kind of session.

This is the hook to hang a startup on. Houdini runs it for every folder on
its path, with an interface or without, so adding this package takes nothing
away from anyone else's. A startup script in `scripts/` would: only the first
`123.py` on the path is ever run, so ours would quietly replace the artist's
or the studio's own.
"""

import nscr_mcp_autostart

nscr_mcp_autostart.start_if_wanted()
