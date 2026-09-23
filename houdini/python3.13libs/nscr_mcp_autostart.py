"""Start a bridge in this Houdini, if the artist asked for that.

The startup files next to this one all end up here. Installing the package
opens no port: nothing happens unless NSCR_MCP_AUTOSTART is 1. A session that
already has a bridge keeps the one it has, so it does not matter how many of
those files run.

Nothing here is allowed to stop Houdini from starting. A failure goes to the
error stream in one line, and the whole traceback to a log file when there is
somewhere to put it, because a bridge that quietly did not start is the worst
of both.

Start one by hand instead with `nscr-houdini-mcp bridge snippet`.
"""

import os
import sys
import traceback

AUTOSTART_VAR = "NSCR_MCP_AUTOSTART"
# Set by this tool when it starts a hython with a bridge of its own. A package
# installed with autostart sets the variable above in every Houdini it loads
# into, workers included, so the worker says it here instead.
KEEP_OUT_VAR = "NSCR_MCP_NO_AUTOSTART"
PORT_VAR = "NSCR_MCP_PORT"
MAX_PORT_VAR = "NSCR_MCP_MAX_PORT"
HOME_VAR = "NSCR_MCP_HOME"
SESSION_ATTRIBUTE = "nscr_mcp_bridge"

TRUE_WORDS = ("1", "true", "yes", "on")

LOG_NAME = "autostart.log"

# Ports below this belong to the system, and there is nothing above the second.
LOWEST_PORT = 1024
HIGHEST_PORT = 65535


def wanted():
    """Whether this session was asked to open a port at startup."""
    if os.environ.get(KEEP_OUT_VAR, "0").strip().lower() in TRUE_WORDS:
        return False
    return os.environ.get(AUTOSTART_VAR, "0").strip().lower() in TRUE_WORDS


def port_range(default):
    """The ports this session may take, when the artist has named them.

    A range that is not a pair of numbers in order, inside the ports a user
    program may have, is refused and said so: taking the default quietly would
    put a session on a port nobody asked for.
    """
    first_text = os.environ.get(PORT_VAR)
    last_text = os.environ.get(MAX_PORT_VAR)
    if not first_text and not last_text:
        return default
    try:
        first = int(first_text) if first_text else default[0]
        last = int(last_text) if last_text else default[1]
    except ValueError:
        _complain(f"{PORT_VAR} and {MAX_PORT_VAR} have to be whole numbers, using {default}")
        return default
    if not (LOWEST_PORT <= first <= last <= HIGHEST_PORT):
        _complain(
            f"{PORT_VAR} to {MAX_PORT_VAR} has to run upwards inside"
            f" {LOWEST_PORT} to {HIGHEST_PORT}, using {default}"
        )
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
    """The whole of what a startup file does."""
    if not wanted():
        return None
    try:
        return start()
    except Exception as error:  # noqa: BLE001 - a bridge is never worth a failed startup
        _complain(f"nscr bridge did not start: {type(error).__name__}: {error}")
        _log(traceback.format_exc())
        return None


def _complain(line):
    """One line where a person running Houdini from a terminal will see it."""
    print(line, file=sys.stderr, flush=True)


def _log(text):
    """The whole story, when there is a folder to write it in."""
    home = os.environ.get(HOME_VAR)
    if not home:
        return
    try:
        folder = os.path.join(home, "logs")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, LOG_NAME), "a", encoding="utf-8") as log:
            log.write(text if text.endswith("\n") else text + "\n")
    except OSError as error:
        _complain(f"nscr bridge could not write its log: {error}")
