"""Start a bridge when this Houdini opens, if the artist asked for that.

Houdini runs this file at startup for every entry on its path. It does nothing
unless NSCR_MCP_AUTOSTART is 1, so installing the package alone never opens a
port. Nothing here is allowed to stop Houdini from starting: a failure is
printed and the session carries on.

Start one by hand instead with:

    from nscr_houdini_mcp.bridge import Bridge
    hou.session.nscr_mcp_bridge = Bridge()
    hou.session.nscr_mcp_bridge.start()
"""

import os

import hou

AUTOSTART_VAR = "NSCR_MCP_AUTOSTART"
SESSION_ATTRIBUTE = "nscr_mcp_bridge"


def _wanted():
    return os.environ.get(AUTOSTART_VAR, "0").strip().lower() in ("1", "true", "yes", "on")


def _start():
    if getattr(hou.session, SESSION_ATTRIBUTE, None) is not None:
        return
    from nscr_houdini_mcp.bridge import Bridge

    bridge = Bridge()
    record = bridge.start()
    setattr(hou.session, SESSION_ATTRIBUTE, bridge)
    print(f"nscr bridge {record.alias} on port {bridge.port}")


if _wanted():
    try:
        _start()
    except Exception as error:  # noqa: BLE001 - a bridge is never worth a failed startup
        print(f"nscr bridge did not start: {type(error).__name__}: {error}")
