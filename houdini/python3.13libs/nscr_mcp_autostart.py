"""Start a bridge in this Houdini, if the artist asked for that.

The startup scripts next to this file all end up here. Installing the package
opens no port: nothing happens unless NSCR_MCP_AUTOSTART is 1. A session that
already has a bridge keeps the one it has, so it does not matter how many of
those scripts run.

Nothing here is allowed to stop Houdini from starting. A failure is printed
and the session carries on.

Start one by hand instead with `nscr-houdini-mcp bridge snippet`.
"""

import os

AUTOSTART_VAR = "NSCR_MCP_AUTOSTART"
PORT_VAR = "NSCR_MCP_PORT"
MAX_PORT_VAR = "NSCR_MCP_MAX_PORT"
SESSION_ATTRIBUTE = "nscr_mcp_bridge"

TRUE_WORDS = ("1", "true", "yes", "on")


def wanted():
    """Whether this session was asked to open a port at startup."""
    return os.environ.get(AUTOSTART_VAR, "0").strip().lower() in TRUE_WORDS


def port_range(default):
    """The ports this session may take, when the artist has named them."""
    first, last = default
    try:
        first = int(os.environ.get(PORT_VAR) or first)
        last = int(os.environ.get(MAX_PORT_VAR) or last)
    except ValueError:
        return default
    return (first, last)


def start():
    """Start a bridge and hang it on the session, or leave the one there."""
    import hou

    if getattr(hou.session, SESSION_ATTRIBUTE, None) is not None:
        return None
    from nscr_houdini_mcp.bridge import Bridge, BridgeConfig
    from nscr_houdini_mcp.bridge.net import DEFAULT_PORT_RANGE

    bridge = Bridge(BridgeConfig(port_range=port_range(DEFAULT_PORT_RANGE)))
    record = bridge.start()
    setattr(hou.session, SESSION_ATTRIBUTE, bridge)
    print(f"nscr bridge {record.alias} on port {bridge.port}")
    return bridge


def start_if_wanted():
    """The whole of what a startup script does."""
    if not wanted():
        return None
    try:
        return start()
    except Exception as error:  # noqa: BLE001 - a bridge is never worth a failed startup
        print(f"nscr bridge did not start: {type(error).__name__}: {error}")
        return None
